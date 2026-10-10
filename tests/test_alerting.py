import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from alerting import (
    send_alert_batch,
    init_alert_tables,
    record_delivery,
    reserve_new_high_risk_events,
)


class AlertDeduplicationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = str(Path(self.temp_dir.name) / "alerts.db")
        init_alert_tables(self.database_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def customer(self, level="High", score=82):
        return {
            "customer_label": "C-100",
            "source_label": "Uploaded CSV: customers.csv",
            "source_fingerprint": "stable-customer-source",
            "risk_score": score,
            "risk_level": level,
            "driver": "Payment difficulty",
            "next_action": "Payment reminder",
        }

    def test_high_risk_alert_is_not_resent_on_every_rerun(self):
        first = reserve_new_high_risk_events(
            self.database_path, [self.customer()], ["Slack"]
        )
        self.assertEqual(len(first["Slack"]), 1)
        record_delivery(
            self.database_path,
            [first["Slack"][0]["id"]],
            "Slack",
            success=True,
        )

        repeated = reserve_new_high_risk_events(
            self.database_path, [self.customer()], ["Slack"]
        )

        self.assertEqual(repeated["Slack"], [])

    def test_alert_retriggers_after_customer_leaves_and_reenters_high_risk(self):
        first = reserve_new_high_risk_events(
            self.database_path, [self.customer()], ["Email"]
        )
        record_delivery(
            self.database_path,
            [first["Email"][0]["id"]],
            "Email",
            success=True,
        )
        reserve_new_high_risk_events(
            self.database_path, [self.customer("Low", 25)], ["Email"]
        )

        reentered = reserve_new_high_risk_events(
            self.database_path, [self.customer()], ["Email"]
        )

        self.assertEqual(len(reentered["Email"]), 1)
        self.assertNotEqual(reentered["Email"][0]["id"], first["Email"][0]["id"])

    def test_risk_transitions_are_tracked_when_notifications_are_disabled(self):
        reserve_new_high_risk_events(
            self.database_path, [self.customer("Low", 25)], []
        )

        reentered = reserve_new_high_risk_events(
            self.database_path, [self.customer()], ["Slack"]
        )

        self.assertEqual(len(reentered["Slack"]), 1)

    def test_failed_delivery_waits_for_retry_cooldown(self):
        first = reserve_new_high_risk_events(
            self.database_path, [self.customer()], ["Slack"]
        )
        event_id = first["Slack"][0]["id"]
        record_delivery(
            self.database_path,
            [event_id],
            "Slack",
            success=False,
            detail="AlertDeliveryError",
        )

        before_cooldown = reserve_new_high_risk_events(
            self.database_path, [self.customer()], ["Slack"], retry_minutes=15
        )
        self.assertEqual(before_cooldown["Slack"], [])

        with closing(sqlite3.connect(self.database_path)) as conn, conn:
            conn.execute(
                """UPDATE alert_deliveries
                   SET attempted_at = '2000-01-01T00:00:00+00:00'
                   WHERE event_id = ? AND channel = 'Slack'""",
                (event_id,),
            )

        after_cooldown = reserve_new_high_risk_events(
            self.database_path, [self.customer()], ["Slack"], retry_minutes=15
        )
        self.assertEqual([event["id"] for event in after_cooldown["Slack"]], [event_id])


class AlertTransportTests(unittest.TestCase):
    def setUp(self):
        self.events = [{
            "customer_label": "C-100",
            "risk_score": 82,
            "driver": "Payment difficulty",
            "next_action": "Payment reminder",
        }]

    @patch("alerting.requests.post")
    def test_slack_delivery_uses_webhook_and_timeout(self, post):
        post.return_value.text = "ok"

        send_alert_batch("Slack", {"webhook_url": "https://example.test/hook"}, self.events)

        post.assert_called_once()
        self.assertEqual(post.call_args.kwargs["timeout"], (3.05, 10))
        self.assertIn("Payment difficulty", post.call_args.kwargs["json"]["text"])

    @patch("alerting.smtplib.SMTP")
    def test_email_delivery_uses_starttls_before_sending(self, smtp_class):
        smtp = smtp_class.return_value.__enter__.return_value
        config = {
            "host": "smtp.example.test",
            "port": 587,
            "username": "",
            "password": "",
            "sender": "alerts@example.test",
            "recipients": ["team@example.test"],
            "use_ssl": False,
            "use_starttls": True,
        }

        send_alert_batch("Email", config, self.events)

        smtp.starttls.assert_called_once()
        smtp.send_message.assert_called_once()
        self.assertIn("Payment difficulty", smtp.send_message.call_args.args[0].get_content())


if __name__ == "__main__":
    unittest.main()

import unittest

import pandas as pd

from retention_playbooks import build_retention_playbook


class RetentionPlaybookTests(unittest.TestCase):
    def test_maps_customer_signals_to_existing_intervention_actions(self):
        playbook = build_retention_playbook(
            pd.Series({
                "complaints_last_90_days": 2,
                "missed_payments_last_90_days": 1,
                "usage_previous": 20,
                "usage_recent": 8,
                "engagement_score": 2,
                "contract_type": "Month-to-month",
                "tenure_months": 4,
            }),
            risk_level="High",
            usage_decline_threshold=40,
        )

        drivers = {step["driver"] for step in playbook}
        actions = {step["action"] for step in playbook}
        self.assertEqual(
            drivers,
            {
                "Unresolved service friction",
                "Payment difficulty",
                "Declining product engagement",
                "Low engagement",
                "Flexible monthly contract",
                "Early customer lifecycle",
            },
        )
        self.assertTrue(actions.issubset({
            "Resolve complaint",
            "Payment reminder",
            "Schedule follow-up",
            "Offer retention plan",
        }))
        self.assertTrue(all(step["priority"] == "Urgent" for step in playbook))
        self.assertTrue(all(step["timing"] == "Within 24 hours" for step in playbook))

    def test_only_flags_usage_decline_at_configured_threshold(self):
        customer = pd.Series({"usage_previous": 20, "usage_recent": 13})

        playbook = build_retention_playbook(customer, "Medium", 40)

        self.assertEqual(
            [step["driver"] for step in playbook],
            ["No configured warning signal identified"],
        )

    def test_uses_transparent_fallback_when_no_signal_is_available(self):
        playbook = build_retention_playbook(
            pd.Series({"monthly_charges": 65}),
            risk_level="Low",
            usage_decline_threshold=40,
        )

        self.assertEqual(len(playbook), 1)
        self.assertEqual(playbook[0]["action"], "Schedule follow-up")
        self.assertEqual(playbook[0]["priority"], "Proactive")


if __name__ == "__main__":
    unittest.main()

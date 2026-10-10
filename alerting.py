"""Opt-in Slack and SMTP notifications with SQLite-backed deduplication."""

from __future__ import annotations

import hashlib
import os
import smtplib
import sqlite3
import ssl
from contextlib import closing
from datetime import datetime, timedelta
from email.message import EmailMessage
from email.utils import parseaddr
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

import pandas as pd
import requests


class AlertDeliveryError(RuntimeError):
    """A delivery failed; messages intentionally exclude secret-bearing details."""


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


def configured_channels() -> dict[str, dict[str, Any]]:
    """Read delivery configuration from environment variables, never the UI."""
    channels: dict[str, dict[str, Any]] = {}
    slack_url = os.getenv("SLACK_WEBHOOK_URL", "").strip()
    if slack_url:
        parsed = urlparse(slack_url)
        if parsed.scheme == "https" and parsed.netloc:
            channels["Slack"] = {"webhook_url": slack_url}
        else:
            raise ValueError("SLACK_WEBHOOK_URL must be a valid HTTPS URL.")

    smtp_host = os.getenv("SMTP_HOST", "").strip()
    email_to = [
        address.strip()
        for address in os.getenv("ALERT_EMAIL_TO", "").split(",")
        if address.strip()
    ]
    if smtp_host or email_to:
        required = {
            "SMTP_HOST": smtp_host,
            "SMTP_FROM": os.getenv("SMTP_FROM", "").strip(),
            "ALERT_EMAIL_TO": ",".join(email_to),
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError("Email alert configuration is incomplete: " + ", ".join(missing))
        sender = parseaddr(required["SMTP_FROM"])[1]
        recipients = [parseaddr(address)[1] for address in email_to]
        if not sender or any(not address for address in recipients):
            raise ValueError("SMTP_FROM and ALERT_EMAIL_TO must contain valid email addresses.")
        username = os.getenv("SMTP_USER", "").strip()
        password = os.getenv("SMTP_PASSWORD", "")
        if bool(username) != bool(password):
            raise ValueError("Set both SMTP_USER and SMTP_PASSWORD, or leave both unset.")
        port = int(os.getenv("SMTP_PORT", "587"))
        if not 1 <= port <= 65535:
            raise ValueError("SMTP_PORT must be between 1 and 65535.")
        use_ssl = env_flag("SMTP_USE_SSL", False)
        use_starttls = env_flag("SMTP_USE_STARTTLS", True)
        if use_ssl and use_starttls:
            raise ValueError("Set only one of SMTP_USE_SSL or SMTP_USE_STARTTLS.")
        channels["Email"] = {
            "host": smtp_host,
            "port": port,
            "username": username,
            "password": password,
            "sender": sender,
            "recipients": recipients,
            "use_ssl": use_ssl,
            "use_starttls": use_starttls,
        }
    return channels


def init_alert_tables(database_path: str) -> None:
    with closing(sqlite3.connect(database_path, timeout=30)) as conn:
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS risk_alert_state (
                    customer_key TEXT PRIMARY KEY,
                    risk_level TEXT NOT NULL,
                    risk_score INTEGER NOT NULL,
                    event_id TEXT,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alert_events (
                    id TEXT PRIMARY KEY,
                    customer_label TEXT NOT NULL,
                    source_label TEXT NOT NULL,
                    risk_score INTEGER NOT NULL,
                    risk_level TEXT NOT NULL,
                    driver TEXT NOT NULL,
                    next_action TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alert_deliveries (
                    event_id TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    attempted_at TEXT,
                    detail TEXT,
                    PRIMARY KEY (event_id, channel),
                    FOREIGN KEY (event_id) REFERENCES alert_events(id)
                )
            """)


def dataset_fingerprint(frame: Any, namespace: str = "") -> str:
    schema = "|".join(str(column) for column in frame.columns).encode("utf-8")
    return hashlib.sha256(namespace.encode("utf-8") + b"\0" + schema).hexdigest()


def _customer_key(source_fingerprint: str, customer_label: str) -> str:
    return hashlib.sha256(
        f"{source_fingerprint}\0{customer_label}".encode("utf-8")
    ).hexdigest()


def reserve_new_high_risk_events(
    database_path: str,
    customers: list[dict[str, Any]],
    channels: list[str],
    retry_minutes: int = 15,
    material_score_increase: int = 15,
) -> dict[str, list[dict[str, Any]]]:
    """Create transition events and reserve unsent deliveries atomically."""
    deliveries: dict[str, list[dict[str, Any]]] = {channel: [] for channel in channels}
    now = datetime.now().astimezone()
    now_text = now.isoformat(timespec="seconds")
    retry_before = (now - timedelta(minutes=retry_minutes)).isoformat(timespec="seconds")

    with closing(sqlite3.connect(database_path, timeout=30)) as conn:
        conn.row_factory = sqlite3.Row
        with conn:
            conn.execute("BEGIN IMMEDIATE")
            for customer in customers:
                customer_key = _customer_key(
                    str(customer["source_fingerprint"]),
                    str(customer["customer_label"])
                )
                previous = conn.execute(
                    "SELECT risk_level, risk_score, event_id FROM risk_alert_state WHERE customer_key = ?",
                    (customer_key,)
                ).fetchone()

                is_high_risk = str(customer["risk_level"]) == "High"
                new_event = (
                    is_high_risk
                    and (previous is None or previous["risk_level"] != "High")
                )
                if (
                    is_high_risk
                    and
                    previous is not None
                    and previous["risk_level"] == "High"
                    and int(customer["risk_score"]) - int(previous["risk_score"]) >= material_score_increase
                ):
                    new_event = True

                event_id = (
                    previous["event_id"]
                    if previous is not None and is_high_risk else None
                )
                if new_event:
                    event_id = str(uuid4())
                    conn.execute(
                        """INSERT INTO alert_events
                           (id, customer_label, source_label, risk_score, risk_level, driver, next_action, created_at)
                           VALUES (?, ?, ?, ?, 'High', ?, ?, ?)""",
                        (
                            event_id,
                            str(customer["customer_label"]),
                            str(customer["source_label"]),
                            int(customer["risk_score"]),
                            str(customer["driver"]),
                            str(customer["next_action"]),
                            now_text,
                        )
                    )

                conn.execute(
                    """INSERT INTO risk_alert_state
                       (customer_key, risk_level, risk_score, event_id, updated_at)
                       VALUES (?, ?, ?, ?, ?)
                       ON CONFLICT(customer_key) DO UPDATE SET
                           risk_level = excluded.risk_level,
                           risk_score = excluded.risk_score,
                           event_id = excluded.event_id,
                           updated_at = excluded.updated_at""",
                    (
                        customer_key,
                        "High" if is_high_risk else str(customer["risk_level"]),
                        int(customer["risk_score"]),
                        event_id,
                        now_text
                    )
                )
                if not is_high_risk or not event_id:
                    continue
                for channel in channels:
                    conn.execute(
                        """INSERT OR IGNORE INTO alert_deliveries
                           (event_id, channel, status) VALUES (?, ?, 'pending')""",
                        (event_id, channel)
                    )

            active_keys = [
                _customer_key(
                    str(customer["source_fingerprint"]),
                    str(customer["customer_label"])
                )
                for customer in customers
                if str(customer["risk_level"]) == "High"
            ]
            for key in active_keys:
                state = conn.execute(
                    "SELECT event_id FROM risk_alert_state WHERE customer_key = ? AND risk_level = 'High'",
                    (key,)
                ).fetchone()
                if not state or not state["event_id"]:
                    continue
                event = conn.execute(
                    """SELECT id, customer_label, source_label, risk_score, risk_level,
                              driver, next_action, created_at
                       FROM alert_events WHERE id = ?""",
                    (state["event_id"],)
                ).fetchone()
                for channel in channels:
                    delivery = conn.execute(
                        """SELECT status, attempted_at FROM alert_deliveries
                           WHERE event_id = ? AND channel = ?""",
                        (state["event_id"], channel)
                    ).fetchone()
                    if delivery is None:
                        continue
                    retryable = delivery["status"] in {"pending", "failed"}
                    stale_in_flight = (
                        delivery["status"] == "sending"
                        and delivery["attempted_at"]
                        and delivery["attempted_at"] <= retry_before
                    )
                    retry_due = (
                        delivery["status"] in {"failed", "sending"}
                        and delivery["attempted_at"]
                        and delivery["attempted_at"] <= retry_before
                    )
                    if delivery["status"] == "failed":
                        retryable = bool(retry_due)
                    elif delivery["status"] == "sending":
                        retryable = bool(stale_in_flight)
                    if not retryable:
                        continue
                    conn.execute(
                        """UPDATE alert_deliveries
                           SET status = 'sending', attempts = attempts + 1,
                               attempted_at = ?, detail = NULL
                           WHERE event_id = ? AND channel = ?""",
                        (now_text, state["event_id"], channel)
                    )
                    deliveries[channel].append(dict(event))
    return deliveries


def record_delivery(
    database_path: str,
    event_ids: list[str],
    channel: str,
    success: bool,
    detail: str | None = None,
) -> None:
    status = "sent" if success else "failed"
    with closing(sqlite3.connect(database_path, timeout=30)) as conn:
        with conn:
            conn.executemany(
                """UPDATE alert_deliveries SET status = ?, detail = ?
                   WHERE event_id = ? AND channel = ?""",
                [(status, detail, event_id, channel) for event_id in event_ids]
            )


def read_delivery_history(database_path: str) -> pd.DataFrame:
    with closing(sqlite3.connect(database_path, timeout=30)) as conn:
        return pd.read_sql_query(
            """SELECT
                   e.created_at AS timestamp,
                   e.customer_label AS customer_id,
                   e.risk_score,
                   e.risk_level,
                   e.driver,
                   e.next_action,
                   d.channel,
                   d.status,
                   d.attempts,
                   d.attempted_at AS last_attempt
               FROM alert_events e
               LEFT JOIN alert_deliveries d ON d.event_id = e.id
               ORDER BY e.created_at DESC, d.channel""",
            conn
        )


def _alert_digest(events: list[dict[str, Any]]) -> str:
    rows = [
        (
            f"• {event['customer_label']} — {event['risk_score']}/100 risk; "
            f"signal: {event['driver']}; recommended: {event['next_action']}"
        )
        for event in events[:50]
    ]
    if len(events) > 50:
        rows.append(f"…plus {len(events) - 50} additional customer alerts.")
    return "\n".join(rows)


def send_alert_batch(channel: str, config: dict[str, Any], events: list[dict[str, Any]]) -> None:
    if not events:
        return
    body = _alert_digest(events)
    subject = f"ChurnAI: {len(events)} new high-risk customer alert(s)"
    if channel == "Slack":
        try:
            response = requests.post(
                config["webhook_url"],
                json={"text": f"*{subject}*\n{body}"},
                timeout=(3.05, 10)
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise AlertDeliveryError("Slack delivery failed; verify webhook connectivity and configuration.") from exc
        if response.text.strip().lower() not in {"", "ok"}:
            raise AlertDeliveryError("Slack webhook did not acknowledge the alert.")
        return
    if channel == "Email":
        message = EmailMessage()
        message["Subject"] = subject
        message["From"] = config["sender"]
        message["To"] = ", ".join(config["recipients"])
        message.set_content(
            "Automated churn-risk alert. Risk scores are model estimates, not certainty.\n\n"
            + body
        )
        smtp_class = smtplib.SMTP_SSL if config["use_ssl"] else smtplib.SMTP
        try:
            with smtp_class(config["host"], config["port"], timeout=10) as smtp:
                if config["use_starttls"] and not config["use_ssl"]:
                    smtp.starttls(context=ssl.create_default_context())
                if config["username"]:
                    smtp.login(config["username"], config["password"])
                smtp.send_message(message)
        except (smtplib.SMTPException, OSError) as exc:
            raise AlertDeliveryError("Email delivery failed; verify SMTP connectivity and configuration.") from exc
        return
    raise ValueError(f"Unsupported alert channel: {channel}")

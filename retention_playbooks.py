"""Signal-based retention guidance for customer-level churn records."""

from __future__ import annotations

from typing import Any

import pandas as pd


def _number(value: Any) -> float | None:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return None if pd.isna(parsed) else float(parsed)


def _find_usage_columns(columns: list[str]) -> tuple[str | None, str | None]:
    previous_markers = ("previous", "prior", "prev")
    recent_markers = ("recent", "current", "latest")
    usage_columns = [
        column for column in columns
        if any(term in column.lower() for term in ("usage", "login", "activity"))
    ]
    previous = next(
        (
            column for column in usage_columns
            if any(marker in column.lower() for marker in previous_markers)
        ),
        None
    )
    recent = next(
        (
            column for column in usage_columns
            if any(marker in column.lower() for marker in recent_markers)
        ),
        None
    )
    return previous, recent


def build_retention_playbook(
    customer: pd.Series,
    risk_level: str,
    usage_decline_threshold: float,
) -> list[dict[str, str]]:
    """Map observed customer signals to practical, non-causal retention actions."""
    playbook: list[dict[str, str]] = []
    columns = [str(column) for column in customer.index]
    priority = "Urgent" if risk_level == "High" else "Proactive"
    target_time = "Within 24 hours" if risk_level == "High" else "Within 3 business days"

    for column in columns:
        normalized = column.lower()
        if "complaint" not in normalized:
            continue
        count = _number(customer[column])
        if count is not None and count > 0:
            playbook.append({
                "driver": "Unresolved service friction",
                "evidence": f"{column.replace('_', ' ').title()}: {count:g}",
                "action": "Resolve complaint",
                "owner": "Customer care",
                "timing": target_time,
                "priority": priority,
            })

    for column in columns:
        normalized = column.lower()
        if not any(term in normalized for term in ("missed_payment", "overdue", "delinquen")):
            continue
        count = _number(customer[column])
        if count is not None and count > 0:
            playbook.append({
                "driver": "Payment difficulty",
                "evidence": f"{column.replace('_', ' ').title()}: {count:g}",
                "action": "Payment reminder",
                "owner": "Billing support",
                "timing": target_time,
                "priority": priority,
            })

    previous_column, recent_column = _find_usage_columns(columns)
    if previous_column and recent_column:
        previous = _number(customer[previous_column])
        recent = _number(customer[recent_column])
        if previous is not None and recent is not None and previous > 0:
            decline = max(0.0, (previous - recent) / previous * 100)
            if decline >= usage_decline_threshold:
                playbook.append({
                    "driver": "Declining product engagement",
                    "evidence": f"Usage down {decline:.0f}% versus previous period",
                    "action": "Schedule follow-up",
                    "owner": "Customer success",
                    "timing": target_time,
                    "priority": priority,
                })

    for column in columns:
        if "engagement" not in column.lower():
            continue
        engagement = _number(customer[column])
        if engagement is not None and engagement <= 3:
            playbook.append({
                "driver": "Low engagement",
                "evidence": f"{column.replace('_', ' ').title()}: {engagement:g}",
                "action": "Schedule follow-up",
                "owner": "Customer success",
                "timing": target_time,
                "priority": priority,
            })

    for column in columns:
        if "contract" not in column.lower():
            continue
        contract = str(customer[column]).strip().lower().replace("_", " ")
        if contract in {"month-to-month", "month to month", "monthly"}:
            playbook.append({
                "driver": "Flexible monthly contract",
                "evidence": f"{column.replace('_', ' ').title()}: {customer[column]}",
                "action": "Offer retention plan",
                "owner": "Retention team",
                "timing": target_time,
                "priority": priority,
            })

    for column in columns:
        if "tenure" not in column.lower():
            continue
        tenure = _number(customer[column])
        if tenure is not None and 0 <= tenure <= 6:
            playbook.append({
                "driver": "Early customer lifecycle",
                "evidence": f"{column.replace('_', ' ').title()}: {tenure:g} months",
                "action": "Schedule follow-up",
                "owner": "Onboarding team",
                "timing": target_time,
                "priority": priority,
            })

    if not playbook:
        playbook.append({
            "driver": "No configured warning signal identified",
            "evidence": "The model indicates elevated risk, but available fields do not match a configured signal.",
            "action": "Schedule follow-up",
            "owner": "Customer success",
            "timing": target_time,
            "priority": priority,
        })

    return playbook

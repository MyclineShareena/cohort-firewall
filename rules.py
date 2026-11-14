"""Check results -> verdict. Deterministic; Claude never assigns the verdict.

Any hard fail -> Blocked; else any unknown -> Insufficient Evidence;
else any soft fail -> Conditional; else Ready.
"not_applicable" checks are neutral (they never drive a verdict) but are not
passes; callers must show and describe them separately.
"""
from __future__ import annotations

RULE_TEXT = {  # the rule that fired, in the order the rules are applied
    "Blocked": "Any hard fail -> Blocked",
    "Insufficient Evidence": "No hard fail, and any unknown -> Insufficient Evidence",
    "Conditional": "No hard fail or unknown, and any soft fail -> Conditional",
    "Ready": "No hard fail, unknown or soft fail -> Ready",
}


def _verdict(verdict: str, driving: list[dict]) -> dict:
    return {"verdict": verdict, "driving_checks": driving, "rule": RULE_TEXT[verdict]}


def assign_verdict(checks: list[dict]) -> dict:
    """Return {"verdict", "driving_checks", "rule"}: the checks that triggered the
    rule, and the rule's text (used verbatim by the evidence record)."""
    if not checks:
        return _verdict("Insufficient Evidence", [])
    hard = [c for c in checks if c["status"] == "fail" and c["severity"] == "hard"]
    if hard:
        return _verdict("Blocked", hard)
    unknown = [c for c in checks if c["status"] == "unknown"]
    if unknown:
        return _verdict("Insufficient Evidence", unknown)
    soft = [c for c in checks if c["status"] == "fail"]
    if soft:
        return _verdict("Conditional", soft)
    return _verdict("Ready", [])

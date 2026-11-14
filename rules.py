"""Check results -> verdict. Deterministic; Claude never assigns the verdict.

Any hard fail -> Blocked; else any unknown -> Insufficient Evidence;
else any soft fail -> Conditional; else Ready.
"not_applicable" checks are neutral (they never drive a verdict) but are not
passes; callers must show and describe them separately.
"""
from __future__ import annotations


def assign_verdict(checks: list[dict]) -> dict:
    """Return {"verdict", "driving_checks"} (the checks that triggered the rule)."""
    if not checks:
        return {"verdict": "Insufficient Evidence", "driving_checks": []}
    hard = [c for c in checks if c["status"] == "fail" and c["severity"] == "hard"]
    if hard:
        return {"verdict": "Blocked", "driving_checks": hard}
    unknown = [c for c in checks if c["status"] == "unknown"]
    if unknown:
        return {"verdict": "Insufficient Evidence", "driving_checks": unknown}
    soft = [c for c in checks if c["status"] == "fail"]
    if soft:
        return {"verdict": "Conditional", "driving_checks": soft}
    return {"verdict": "Ready", "driving_checks": []}

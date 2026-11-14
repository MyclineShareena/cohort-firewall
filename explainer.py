"""Claude call: check results -> explanation and next actions.

Claude sees only the question, the requirements and the check results (never the
data). Python validates the output, retries once with the errors, and otherwise
falls back to a deterministic explanation built from the checks' own messages.
"""
from __future__ import annotations

import re
from typing import Optional

import prompts
from parser import ConfigError, _Strict, call_structured, get_client, get_model

VERDICTS = ["Ready", "Conditional", "Blocked", "Insufficient Evidence"]
VERDICT_RE = re.compile(r"\b(Ready|Conditional|Blocked|Insufficient Evidence)\b")
PASSED_RE = re.compile(r"\bpass(ed|es|ing)?\b", re.IGNORECASE)
ALL_PASSED_RE = re.compile(r"\ball\b[^.]*\bchecks?\b[^.]*\bpass", re.IGNORECASE)

FALLBACK_ACTIONS = {
    "question_parsed": "Rephrase the question with an explicit outcome, time horizon and "
                       "predictors, then re-run.",
    "required_fields_present": "Add or correct the missing or invalid required columns, then re-run.",
    "question_terms_mappable": "Add a column for the unmapped terms, or rephrase the question "
                               "without them.",
    "required_field_missingness": "Reduce missingness in the listed fields through source-data "
                                  "review, or plan and justify an imputation strategy.",
    "outcome_derivable": "Provide vital_status, diagnosis_date and last_followup_date so the "
                         "outcome can be derived.",
    "early_censoring": "Extend follow-up or obtain linked vital status for patients censored "
                       "before the horizon.",
    "subgroup_counts": "Add patients to the small subgroups listed, or report the model's "
                       "limits for them.",
    "label_leakage": "Remove the outcome-encoding columns listed before modeling.",
    "sample_size": "Increase the sample to the Riley minimum, or reduce the number of "
                   "candidate predictor parameters.",
    "identifier_columns": "Remove or pseudonymize identifier-like columns; use synthetic data only.",
}


class DrivingNote(_Strict):
    name: str
    explanation: str


class NotApplicableNote(_Strict):
    name: str
    reason: str


class NextAction(_Strict):
    check_name: Optional[str]
    action: str


class ExplanationOut(_Strict):
    summary: str
    driving_checks: list[DrivingNote]
    not_applicable: list[NotApplicableNote]
    next_actions: list[NextAction]


def validate_explanation(out: dict, checks: list[dict], verdict: dict) -> list[str]:
    errs = []
    names = {c["name"] for c in checks}
    driving = {c["name"] for c in verdict["driving_checks"]}
    na = {c["name"] for c in checks if c["status"] == "not_applicable"}
    bad = {c["name"] for c in checks if c["status"] in {"fail", "unknown"}}

    other = sorted(set(VERDICT_RE.findall(out["summary"])) - {verdict["verdict"]})
    if other:
        errs.append(f"summary mentions verdict label(s) {other}; the verdict is "
                    f"{verdict['verdict']} and the summary may name only that one.")
    if na and ALL_PASSED_RE.search(out["summary"]):
        errs.append("summary says all checks passed, but some were not applicable.")

    got_driving = [d["name"] for d in out["driving_checks"]]
    if set(got_driving) != driving or len(got_driving) != len(driving):
        errs.append(f"driving_checks must cover exactly {sorted(driving)}; got {got_driving}.")

    got_na = [n["name"] for n in out["not_applicable"]]
    if set(got_na) != na or len(got_na) != len(na):
        errs.append(f"not_applicable must list exactly {sorted(na)}; got {got_na}.")
    for n in out["not_applicable"]:
        if PASSED_RE.search(n["reason"]):
            errs.append(f"not_applicable reason for {n['name']} calls it passed; it was not evaluated.")

    for a in out["next_actions"]:
        cn = a["check_name"]
        if cn is not None and cn not in names:
            errs.append(f"next_actions references unknown check '{cn}'.")
        elif cn is not None and cn not in bad:
            errs.append(f"next_actions check_name '{cn}' did not fail and is not unknown.")
    if bad and not any(a["check_name"] in bad for a in out["next_actions"]):
        errs.append(f"next_actions must address the failing/unknown checks {sorted(bad)}.")
    return errs


def fallback_explanation(checks: list[dict], verdict: dict, reason: str) -> dict:
    """Deterministic explanation from the checks' own messages."""
    driving = {c["name"] for c in verdict["driving_checks"]}
    return {
        "summary": (f"Verdict: {verdict['verdict']}, assigned by the rules engine. The "
                    "findings below are the checks' own messages."),
        "driving_checks": [{"name": c["name"], "explanation": c["message"]}
                           for c in checks if c["name"] in driving],
        "not_applicable": [{"name": c["name"], "reason": c["message"]}
                           for c in checks if c["status"] == "not_applicable"],
        "next_actions": [{"check_name": c["name"], "action": FALLBACK_ACTIONS.get(c["name"], "Review this finding.")}
                         for c in checks if c["status"] in {"fail", "unknown"}],
        "source": "fallback",
        "error": reason,
    }


def explain(question: str, requirements: dict | None, checks: list[dict], verdict: dict,
            client=None, model: str | None = None) -> dict:
    """Returns {summary, driving_checks, not_applicable, next_actions, source, error}."""
    try:
        model = model or get_model()
        client = client or get_client()
    except ConfigError as e:
        return fallback_explanation(checks, verdict, str(e))

    feedback, last_error = None, None
    for _ in (1, 2):
        user = prompts.explainer_user_message(question, requirements, checks, verdict, feedback)
        out, err, _ = call_structured(client, model, prompts.EXPLAINER_SYSTEM, user,
                                      ExplanationOut)
        if out is None:
            last_error, feedback = err, None
            continue
        errs = validate_explanation(out, checks, verdict)
        if not errs:
            return {**out, "source": "claude", "error": None}
        last_error, feedback = "validation failed: " + "; ".join(errs), errs
    return fallback_explanation(checks, verdict, last_error)

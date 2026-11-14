"""Evidence record, "evidence needed to reconsider", limitations and human review.

Display and record-keeping only: nothing here changes a check, a status or the
verdict. The verdict comes from rules.assign_verdict; a human review is recorded
alongside it and never replaces it.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import checks as C

DECISIONS = ["Approve for this question", "Approve with conditions", "Reject"]
REVIEW_PENDING = "Human review required before this dataset is used"
SCOPE = ("Data readiness assessment of a synthetic dataset for one stated question. "
         "It is not a clinical decision, and it does not validate any model.")

DECISION_RULES = {  # mirrors rules.py, in the order the rules are applied
    "Blocked": "Any hard fail -> Blocked",
    "Insufficient Evidence": "No hard fail, and any unknown -> Insufficient Evidence",
    "Conditional": "No hard fail or unknown, and any soft fail -> Conditional",
    "Ready": "No hard fail, unknown or soft fail -> Ready",
}


def _list(items) -> str:
    return ", ".join(map(str, items))


# ---------------------------------------------------------------- per check

def fields_used(check: dict, req: dict | None, columns: list[str]) -> list[str]:
    """Dataset columns the check read (or would have read)."""
    name = check["name"]
    if name == "identifier_columns":
        return list(columns)  # screens every column of the full file
    if name in {"question_terms_mappable", "question_parsed"} or req is None:
        return []
    mortality = C._is_mortality(req)
    outcome = C.OUTCOME_SOURCES if mortality else [(req.get("outcome") or {}).get("column")]
    outcome = [c for c in outcome if c]
    filters = [f["column"] for f in req.get("target_population") or []]
    if name in {"required_fields_present", "required_field_missingness"}:
        return C._required_columns(req)
    if name in {"outcome_derivable", "early_censoring"}:
        return list(dict.fromkeys(filters + outcome))
    if name == "subgroup_counts":
        groups = [g for g in ("sex", "age", "stage") if g not in filters]
        return list(dict.fromkeys(filters + groups))
    if name == "label_leakage":
        return list(columns)  # screens every column name
    if name == "sample_size":
        return list(dict.fromkeys(filters + outcome + list(req.get("predictor_columns") or [])))
    return []


def evidence_needed(check: dict, req: dict | None) -> str | None:
    """What would have to be shown for this check to be reconsidered. None if it
    passed or was not applicable."""
    name, status, v = check["name"], check["status"], check.get("value")
    if status not in {"fail", "unknown"}:
        return None
    h = (req or {}).get("time_horizon_months")
    if name == "sample_size":
        if status == "fail":
            return (f"A cohort of at least {v['n_required']:,} patients with a known outcome "
                    f"({v['events_required']:,} events), or a pre-specified model with fewer "
                    f"than {v['n_params']} parameters.")
        return "A derivable outcome and a stated number of candidate predictor parameters."
    if name == "required_field_missingness" and status == "fail":
        over = [f"{c} (now {f:.0%})" for c, f in v["by_column"].items() if f > C.MISSING_SOFT]
        return (f"At most 20% missing in {_list(over)}, from source-data review or a "
                "pre-specified, justified imputation plan.")
    if name == "required_fields_present" and status == "fail":
        parts = []
        if v["missing"]:
            parts.append(f"a file that includes {_list(v['missing'])}")
        if v["type_problems"]:
            parts.append(f"corrected values ({'; '.join(v['type_problems'])})")
        return " and ".join(parts).capitalize() + "."
    if name == "outcome_derivable":
        if status == "unknown":
            return "A question with a supported outcome (death) and, for a binary outcome, a time horizon."
        if v is None:
            return "vital_status, diagnosis_date and last_followup_date recorded for each patient."
        return "A cohort with both events and non-events within the time horizon."
    if name == "early_censoring":
        if status == "fail":
            return (f"Follow-up of at least {h} months (or linked vital status) so that at most "
                    f"15% of patients have an unknown outcome (now {v:.0%}).")
        return "A derivable outcome (vital_status, diagnosis_date, last_followup_date)."
    if name == "subgroup_counts":
        if status == "fail":
            small = [f"{k} ({n})" for k, n in v.items() if n < check["threshold"]]
            return (f"At least {check['threshold']} patients in {_list(small)}, or a question "
                    "whose population excludes these subgroups.")
        return "The sex, age and stage columns."
    if name == "question_terms_mappable":
        return f"A column recording {_list(v)}, or a question that does not use it."
    if name == "question_parsed":
        return "A question that names the outcome, time horizon and predictors in the dataset's terms."
    if name == "label_leakage":
        return f"The dataset re-exported without {_list(v)}."
    if name == "identifier_columns":
        cols = [h.split(" (")[0] for h in v]
        return f"A synthetic, de-identified file without {_list(cols)}."
    return f"Evidence that resolves: {check['message']}"


# ---------------------------------------------------------------- record-level

def limitations(result: dict) -> list[str]:
    items = [
        "Synthetic data only. This tool must not be used with real patient data.",
        "Assesses data readiness for one stated question. It does not assess clinical "
        "validity, model performance or fitness for use in care.",
        "Thresholds (20%/40% missingness, 15% early censoring, 30 per subgroup) are "
        "conventions chosen for this tool, not universal standards.",
        "The binary outcome assumes last_followup_date is the date of death for patients "
        "who died; patients alive with follow-up shorter than the horizon are excluded.",
    ]
    ss = next((c for c in result.get("checks") or [] if c["name"] == "sample_size"), None)
    if ss and ss.get("value") and "r2cs_note" in ss["value"]:
        items.append("Sample size (Riley et al.) depends on the anticipated R²cs: "
                     + ss["value"]["r2cs_note"]
                     + " A different R²cs changes the required n.")
    if result.get("parse_source") == "claude":
        items.append("The question was parsed by Claude; check the parsed requirements "
                     "match what was intended.")
    elif result.get("parse_source"):
        items.append(f"Requirements came from a {result['parse_source']}: hand-written for "
                     "this demo case.")
    return items


def validate_review(reviewer: str, reviewed: bool, decision: str | None) -> list[str]:
    errors = []
    if not (reviewer or "").strip():
        errors.append("Enter the reviewer's name.")
    if not reviewed:
        errors.append("Tick \"I have reviewed the evidence\".")
    if decision not in DECISIONS:
        errors.append("Choose a decision.")
    return errors


def make_review(reviewer: str, reviewed: bool, decision: str, notes: str,
                verdict: str, now: datetime | None = None) -> dict:
    errors = validate_review(reviewer, reviewed, decision)
    if errors:
        raise ValueError(" ".join(errors))
    now = now or datetime.now(timezone.utc)
    return {"status": "submitted", "reviewer": reviewer.strip(), "reviewed_evidence": True,
            "decision": decision, "notes": (notes or "").strip(),
            "submitted_at": now.isoformat(timespec="seconds"),
            "tool_verdict_at_review": verdict}


def build_record(result: dict, review: dict | None, dataset: dict | None = None,
                 now: datetime | None = None) -> dict:
    """Everything needed to audit one assessment. Only for status "ok" results."""
    req, cols = result["requirements"], (dataset or {}).get("columns", [])
    checks = [{
        "name": c["name"], "metric_cluster": c["metric_cluster"], "severity": c["severity"],
        "status": c["status"], "fields_used": fields_used(c, req, cols),
        "threshold": c["threshold"], "value": c["value"], "message": c["message"],
        "evidence_needed_to_reconsider": evidence_needed(c, req),
    } for c in result["checks"]]
    return {
        "tool": "Cohort Firewall",
        "generated_at": (now or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
        "scope": SCOPE,
        "dataset": {**(dataset or {}), "columns": cols},
        "question": result["question"],
        "description": result["description"],
        "parse_source": result["parse_source"],
        "parsed_requirements": req,
        "verdict": result["verdict"],
        "verdict_assigned_by": "rules engine (rules.py); a human review never changes it",
        "decision_rule": {"rule": DECISION_RULES[result["verdict"]],
                          "driving_checks": [c["name"] for c in result["driving_checks"]]},
        "checks": checks,
        "limitations": limitations(result),
        "human_review": review or {"status": "pending", "note": REVIEW_PENDING},
    }


def record_json(record: dict) -> str:
    return json.dumps(record, indent=2, default=str, ensure_ascii=False)


def record_markdown(record: dict) -> str:
    r, rev = record, record["human_review"]
    ds = r["dataset"]
    lines = [
        "# Cohort Firewall evidence record", "",
        f"_{r['scope']}_", "",
        f"- Generated: {r['generated_at']}",
        f"- Dataset: {ds.get('name', 'uploaded CSV')}"
        + (f" ({ds['rows']:,} rows x {len(ds['columns'])} columns)" if "rows" in ds else ""),
        f"- Question: {r['question']}",
        f"- Description: {r['description'] or '(none)'}",
        f"- Parse source: {r['parse_source']}", "",
        f"## Verdict: {r['verdict']}", "",
        f"Assigned by the {r['verdict_assigned_by']}.", "",
        f"Decision rule that fired: **{r['decision_rule']['rule']}**. Driving checks: "
        f"{_list(r['decision_rule']['driving_checks']) or 'none'}.", "",
        "## Human review", "",
    ]
    if rev["status"] == "submitted":
        lines += [f"- Reviewer: {rev['reviewer']}",
                  "- I have reviewed the evidence: yes",
                  f"- Decision: {rev['decision']}",
                  f"- Notes: {rev['notes'] or '(none)'}",
                  f"- Submitted: {rev['submitted_at']}",
                  f"- Tool verdict at review: {rev['tool_verdict_at_review']} (unchanged)"]
    else:
        lines.append(f"Pending. {REVIEW_PENDING}.")
    lines += ["", "## Checks", "",
              "| Check | Cluster | Severity | Status | Fields used | Threshold | Message | "
              "Evidence needed to reconsider |",
              "|---|---|---|---|---|---|---|---|"]
    for c in r["checks"]:
        cells = [c["name"], c["metric_cluster"], c["severity"], c["status"],
                 _list(c["fields_used"]) or "-", json.dumps(c["threshold"], default=str),
                 c["message"], c["evidence_needed_to_reconsider"] or "-"]
        lines.append("| " + " | ".join(str(x).replace("|", "/") for x in cells) + " |")
    lines += ["", "## Parsed requirements", "", "```json",
              json.dumps(r["parsed_requirements"], indent=2, default=str), "```", "",
              "## Limitations", ""] + [f"- {x}" for x in r["limitations"]]
    return "\n".join(lines) + "\n"

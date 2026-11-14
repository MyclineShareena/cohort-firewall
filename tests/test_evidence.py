"""Evidence record, evidence-needed text, limitations and review validation (offline)."""
import json
from datetime import datetime, timezone

import pandas as pd
import pytest

import evidence as E
import parser as P
from data.demo_cases import DEMO_CASES
from pipeline import DATA_DIR, assess

NOW = datetime(2026, 11, 14, 12, 0, tzinfo=timezone.utc)
CASES = {c["name"]: c for c in DEMO_CASES}
BANNED = ["clinically valid", "clinical validity has", "production ready", "production-ready",
          "safe for clinical", "validated for"]


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(P, "load_env", lambda: None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)


def run(name):
    case = CASES[name]
    df = pd.read_csv(DATA_DIR / case["file"])
    res = assess(df, case["question"], case["description"])
    ds = {"name": case["file"], "rows": len(df), "columns": list(df.columns)}
    return res, ds


def check(res, name):
    return next(c for c in res["checks"] if c["name"] == name)


def test_sample_size_evidence_needed():
    res, _ = run("epv_vs_riley")
    need = E.evidence_needed(check(res, "sample_size"), res["requirements"])
    assert need.startswith("A cohort of at least 1,205 patients")
    assert "a pre-specified model with fewer than 10 parameters" in need


@pytest.mark.parametrize("name,check_name,expected", [
    ("messy", "required_field_missingness", "At most 20% missing in stage (now 30%)"),
    ("broken", "required_fields_present", "A file that includes vital_status"),
    ("broken", "outcome_derivable", "vital_status, diagnosis_date and last_followup_date"),
    ("leakage", "label_leakage", "re-exported without date_of_death"),
    ("pdl1_unmappable", "question_terms_mappable", "A column recording PD-L1 status"),
])
def test_evidence_needed_per_case(name, check_name, expected):
    res, _ = run(name)
    assert expected in E.evidence_needed(check(res, check_name), res["requirements"])


@pytest.mark.parametrize("case", DEMO_CASES, ids=lambda c: c["name"])
def test_every_failing_check_has_evidence_needed_and_passing_none(case):
    res, _ = run(case["name"])
    for c in res["checks"]:
        need = E.evidence_needed(c, res["requirements"])
        if c["status"] in {"fail", "unknown"}:
            assert need and not need.startswith("Evidence that resolves"), c["name"]
        else:
            assert need is None, c["name"]


@pytest.mark.parametrize("case", DEMO_CASES, ids=lambda c: c["name"])
def test_record_contents(case):
    res, ds = run(case["name"])
    rec = E.build_record(res, None, ds, now=NOW)
    assert rec["question"] == case["question"]
    assert rec["parsed_requirements"] == case["requirements"]
    assert rec["verdict"] == case["verdict"]
    assert rec["decision_rule"]["rule"] == E.DECISION_RULES[case["verdict"]]
    assert set(rec["decision_rule"]["driving_checks"]) == case["driving"]
    assert [c["name"] for c in rec["checks"]] == [c["name"] for c in res["checks"]]
    for c in rec["checks"]:
        assert {"status", "fields_used", "threshold", "message",
                "evidence_needed_to_reconsider"} <= c.keys()
    assert rec["human_review"]["status"] == "pending"
    assert any("Synthetic data only" in x for x in rec["limitations"])
    json.loads(E.record_json(rec))  # serializable


def test_fields_used():
    res, ds = run("clean")
    rec = E.build_record(res, None, ds, now=NOW)
    used = {c["name"]: c["fields_used"] for c in rec["checks"]}
    assert "ldh" in used["required_fields_present"] and "albumin" in used["sample_size"]
    assert used["outcome_derivable"] == ["vital_status", "diagnosis_date", "last_followup_date"]
    assert used["subgroup_counts"] == ["sex", "age", "stage"]
    assert used["identifier_columns"] == ds["columns"]
    assert used["question_terms_mappable"] == []


def test_r2cs_assumption_in_limitations():
    res, _ = run("epv_vs_riley")
    lim = " ".join(E.limitations(res))
    assert "anticipated R²cs" in lim and "15% of max R²cs" in lim
    res_b, _ = run("broken")  # sample size not computed -> no R²cs line
    assert not any("R²cs" in x for x in E.limitations(res_b))


def test_review_validation_and_verdict_unchanged():
    assert len(E.validate_review("", False, None)) == 3
    with pytest.raises(ValueError):
        E.make_review("Dr A", False, "Reject", "", "Blocked")
    res, ds = run("leakage")
    review = E.make_review(" Dr A ", True, "Approve for this question", "ok", res["verdict"], now=NOW)
    rec = E.build_record(res, review, ds, now=NOW)
    assert rec["verdict"] == "Blocked"  # review never changes the verdict
    assert rec["human_review"]["reviewer"] == "Dr A"
    assert rec["human_review"]["decision"] == "Approve for this question"
    assert rec["human_review"]["tool_verdict_at_review"] == "Blocked"


def test_markdown_record():
    res, ds = run("epv_vs_riley")
    review = E.make_review("Dr A", True, "Approve with conditions", "Add patients", "Conditional",
                           now=NOW)
    md = E.record_markdown(E.build_record(res, review, ds, now=NOW))
    for s in ["## Verdict: Conditional", "No hard fail or unknown, and any soft fail -> Conditional",
              "Decision: Approve with conditions", "| sample_size |",
              "A cohort of at least 1,205 patients", "## Limitations", "## Parsed requirements",
              "Synthetic data only"]:
        assert s in md, s
    pending = E.record_markdown(E.build_record(res, None, ds, now=NOW))
    assert "Pending. Human review required before this dataset is used." in pending


@pytest.mark.parametrize("case", DEMO_CASES, ids=lambda c: c["name"])
def test_no_clinical_validity_or_production_claims(case):
    res, ds = run(case["name"])
    md = E.record_markdown(E.build_record(res, None, ds, now=NOW)).lower()
    assert not any(b in md for b in BANNED)

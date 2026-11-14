import numpy as np
import pandas as pd
import pytest

import checks as C
from data.demo_cases import DEMO_CASES, FIELDS_10
from data.generate import build_all, make_cohort
from rules import assign_verdict

KEYS = {"name", "metric_cluster", "severity", "status", "value", "threshold", "message"}


def req(**kw):
    r = {"target_population": [], "outcome": {"column": "vital_status", "type": "binary"},
         "required_fields": list(FIELDS_10), "time_horizon_months": 24,
         "task_type": "prediction", "n_predictor_parameters": 10, "unmappable_terms": []}
    r.update(kw)
    return r


@pytest.fixture
def base():
    # 1000 rows, 300 events, 50 early-censored: passes everything with P=10.
    return make_cohort(n=1000, events_24m=300, censored_early=50, seed=7)


@pytest.fixture(scope="module")
def files():
    return build_all()


def by_name(results):
    return {r["name"]: r for r in results}


# ---------- result shape ----------

def test_all_checks_return_spec_shape(base):
    results = C.run_checks(base, req())
    assert len(results) == 9
    for r in results:
        assert set(r) == KEYS
        assert r["status"] in {"pass", "fail", "unknown", "not_applicable"}
        assert r["severity"] in {"hard", "soft"}
        assert r["metric_cluster"] in C.CLUSTERS
    assert assign_verdict(results)["verdict"] == "Ready"


# ---------- population filter ----------

def test_apply_population(base):
    sub = C.apply_population(base, [{"column": "stage", "operator": "in", "value": ["III", "IV"]},
                                    {"column": "age", "operator": ">=", "value": 65}])
    assert set(sub["stage"]) <= {"III", "IV"} and sub["age"].min() >= 65
    assert len(C.apply_population(base, [])) == len(base)


# ---------- 1. consistency: required fields ----------

def test_required_field_missing_is_hard_fail(base):
    r = C.check_required_fields(base.drop(columns=["ecog"]), req())
    assert (r["status"], r["severity"]) == ("fail", "hard")
    assert "ecog" in r["message"]


def test_outcome_source_columns_are_required(base):
    r = C.check_required_fields(base.drop(columns=["vital_status"]), req())
    assert r["status"] == "fail" and "vital_status" in r["message"]


def test_wrong_type_is_hard_fail(base):
    df = base.astype({"age": object})
    df.loc[0, "age"] = "sixty"
    assert C.check_required_fields(df, req())["status"] == "fail"
    df = base.copy()
    df.loc[0, "stage"] = "V"
    assert C.check_required_fields(df, req())["status"] == "fail"


def test_types_ignore_missing_values(base):
    df = base.copy()
    df.loc[:100, "stage"] = np.nan
    assert C.check_required_fields(df, req())["status"] == "pass"


# ---------- 9. consistency: question terms mappable ----------

def test_unmappable_terms_unknown():
    r = C.check_terms_mappable(req(unmappable_terms=["PD-L1 status"]))
    assert r["status"] == "unknown" and "PD-L1" in r["message"]
    assert r["metric_cluster"] == "consistency"
    assert C.check_terms_mappable(req())["status"] == "pass"


def test_unmappable_terms_not_treated_as_required(base):
    r = C.check_required_fields(base, req(unmappable_terms=["PD-L1 status"]))
    assert r["status"] == "pass"


# ---------- 2. informativeness: missingness ----------

@pytest.mark.parametrize("frac,status,severity", [
    (0.20, "pass", "soft"), (0.25, "fail", "soft"), (0.40, "fail", "soft"), (0.45, "fail", "hard")])
def test_missingness_thresholds(base, frac, status, severity):
    df = base.copy()
    df.loc[: int(frac * len(df)) - 1, "albumin"] = np.nan
    r = C.check_missingness(df, req())
    assert (r["status"], r["severity"]) == (status, severity)


def test_missingness_ignores_non_required_fields(base):
    df = base.copy()
    df["wbc"] = np.nan  # not required
    assert C.check_missingness(df, req())["status"] == "pass"


# ---------- 3. informativeness: outcome ----------

def test_outcome_derivable(base):
    r = C.check_outcome(base, req())
    assert r["status"] == "pass"
    assert r["value"]["events"] == 300 and r["value"]["excluded_unknown"] == 50
    assert "50" in r["message"]


def test_outcome_not_derivable_without_status(base):
    r = C.check_outcome(base.drop(columns=["vital_status"]), req())
    assert (r["status"], r["severity"]) == ("fail", "hard")


def test_outcome_zero_events_fails(base):
    df = base.assign(vital_status="alive")
    assert C.check_outcome(df, req())["status"] == "fail"


def test_outcome_binary_without_horizon_unknown(base):
    assert C.check_outcome(base, req(time_horizon_months=None))["status"] == "unknown"


# ---------- 4. timeliness: early censoring ----------

def test_censoring_not_applicable_without_horizon(base):
    r = C.check_censoring(base, req(time_horizon_months=None,
                                    outcome={"column": "vital_status", "type": "time_to_event"}))
    assert r["status"] == "not_applicable" and "Not applicable" in r["message"]


def test_censoring_threshold():
    ok = make_cohort(n=1000, events_24m=300, censored_early=150, seed=3)   # 15%
    bad = make_cohort(n=1000, events_24m=300, censored_early=160, seed=3)  # 16%
    assert C.check_censoring(ok, req())["status"] == "pass"
    r = C.check_censoring(bad, req())
    assert (r["status"], r["severity"]) == ("fail", "soft")
    assert r["value"] == pytest.approx(0.16)


# ---------- 5. representativeness: subgroups ----------

def test_subgroups_pass_and_fail(base):
    assert C.check_subgroups(base, req())["status"] == "pass"
    df = base[~((base["sex"] == "F") & (base.index % 20 != 0))]  # ~25 women left
    r = C.check_subgroups(df, req())
    assert (r["status"], r["severity"]) == ("fail", "soft")
    assert "sex=F" in r["message"]


def test_subgroups_skip_filtered_column(base):
    r = C.check_subgroups(base, req(target_population=[
        {"column": "stage", "operator": "in", "value": ["III", "IV"]}]))
    assert r["status"] == "pass"  # stage I/II absent by design, not flagged


def test_subgroups_age_80_plus(base):
    df = base[base["age"] < 80]
    r = C.check_subgroups(df, req())
    assert r["status"] == "fail" and "80" in r["message"]


# ---------- 6. measurement process: leakage ----------

def test_leakage(base):
    df = base.assign(date_of_death=base["last_followup_date"].where(base["vital_status"] == "dead"))
    r = C.check_leakage(df, req())
    assert (r["status"], r["severity"]) == ("fail", "hard")
    assert "date_of_death" in r["message"]
    assert C.check_leakage(base, req())["status"] == "pass"


def test_leakage_not_applicable_for_description(base):
    df = base.assign(cause_of_death="x")
    r = C.check_leakage(df, req(task_type="description"))
    assert r["status"] == "not_applicable" and "Not applicable" in r["message"]


# ---------- 7. sample size ----------

def test_sample_size_values(base):
    r = C.check_sample_size(base, req())
    v = r["value"]
    assert r["status"] == "pass" and r["severity"] == "soft"
    assert v["n_available"] == 950 and v["events"] == 300
    assert v["epv"] == pytest.approx(30) and v["epv_pass"]
    assert {"criterion_i", "criterion_ii", "criterion_iii", "n_required", "r2cs_note"} <= set(v)


def test_sample_size_overrides(base):
    assert C.check_sample_size(base, req(), n_params_override=40)["status"] == "fail"
    r = C.check_sample_size(base, req(), anticipated_r2cs=0.3)
    assert r["value"]["r2cs_source"] == "user"


def test_sample_size_not_applicable(base):
    for kw in ({"task_type": "description"},
               {"outcome": {"column": "vital_status", "type": "time_to_event"}}):
        r = C.check_sample_size(base, req(**kw))
        assert r["status"] == "not_applicable" and "Not applicable" in r["message"]


def test_sample_size_unknown_without_params(base):
    assert C.check_sample_size(base, req(n_predictor_parameters=None))["status"] == "unknown"


def test_sample_size_invalid_r2_is_unknown(base):
    assert C.check_sample_size(base, req(), anticipated_r2cs=0.99)["status"] == "unknown"


# ---------- 8. safety: identifiers ----------

@pytest.mark.parametrize("col,val", [
    ("mrn", "123456"), ("ssn", "x"), ("first_name", "Ann"), ("home_address", "1 Main St"),
    ("phone", 5551234567), ("email", "a@b.com"), ("date_of_birth", "1950-01-01"),
    ("dob", "1950-01-01"), ("zip_code", 2115), ("notes", "123-45-6789"),
    ("contact", "ann@example.org")])
def test_identifier_flagged(base, col, val):
    r = C.check_identifiers(base.assign(**{col: val}))
    assert (r["status"], r["severity"]) == ("fail", "hard"), col


def test_non_synthetic_patient_id_flagged(base):
    df = base.copy()
    df.loc[0, "patient_id"] = "A1234567"
    assert C.check_identifiers(df)["status"] == "fail"


def test_identifier_no_false_positive_on_numbers(base):
    # numeric lab values must not trip value patterns
    df = base.assign(lab_code=1234567890, ratio=123.456)
    assert C.check_identifiers(df)["status"] == "pass"


# ---------- demo files: verdict AND exact driving checks ----------

@pytest.mark.parametrize("case", DEMO_CASES, ids=lambda c: c["name"])
def test_demo_verdict_and_driving_checks(files, case):
    df = files[case["file"].removesuffix(".csv")]
    results = C.run_checks(df, case["requirements"])
    v = assign_verdict(results)
    assert v["verdict"] == case["verdict"]
    assert {c["name"] for c in v["driving_checks"]} == case["driving"]


def test_clean_all_pass(files):
    results = C.run_checks(files["clean"], DEMO_CASES[0]["requirements"])
    assert all(r["status"] == "pass" for r in results)


def test_messy_details(files):
    r = by_name(C.run_checks(files["messy"], DEMO_CASES[1]["requirements"]))
    m = r["required_field_missingness"]
    assert m["severity"] == "soft" and "stage" in m["message"] and "ldh" not in m["message"]
    assert r["early_censoring"]["status"] == "pass"
    assert r["outcome_derivable"]["value"]["excluded_unknown"] == 50


def test_epv_vs_riley_details(files):
    r = by_name(C.run_checks(files["epv_vs_riley"], DEMO_CASES[4]["requirements"]))
    v = r["sample_size"]["value"]
    assert v["n_available"] == 1000 and v["events"] == 100
    assert v["epv"] == pytest.approx(10) and v["epv_pass"] is True
    assert v["n_required"] > 1000
    assert r["sample_size"]["severity"] == "soft"


def test_leakage_blocked_only_by_leakage(files):
    r = by_name(C.run_checks(files["leakage"], DEMO_CASES[3]["requirements"]))
    hard_fails = [n for n, x in r.items() if x["status"] == "fail" and x["severity"] == "hard"]
    assert hard_fails == ["label_leakage"]


def test_not_applicable_is_neutral_but_not_pass(files):
    # Description task on clean.csv: leakage and Riley do not apply.
    r = {**DEMO_CASES[0]["requirements"], "task_type": "description"}
    results = by_name(C.run_checks(files["leakage"], r))
    assert results["label_leakage"]["status"] == "not_applicable"
    assert results["sample_size"]["status"] == "not_applicable"
    v = assign_verdict(list(results.values()))
    assert v["verdict"] == "Ready" and v["driving_checks"] == []

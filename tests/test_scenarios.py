import pytest

import checks as C
from data.generate import build_all
from experiments.scenarios import SCENARIOS
from rules import assign_verdict

ALL_CHECK_NAMES = {
    "required_fields_present", "record_plausibility", "question_terms_mappable",
    "required_field_missingness", "outcome_derivable", "early_censoring", "subgroup_counts",
    "label_leakage", "sample_size", "identifier_columns",
}


@pytest.fixture(scope="module")
def files():
    return build_all()


@pytest.mark.parametrize("sc", SCENARIOS, ids=lambda s: s["name"])
def test_scenario_ground_truth(files, sc):
    df = files[sc["file"].removesuffix(".csv")]
    result = C.run_checks(df, sc["requirements"])
    v = assign_verdict(result)
    assert v["verdict"] == sc["verdict"]
    assert {c["name"] for c in v["driving_checks"]} == sc["driving"]


def test_all_four_verdicts_covered():
    assert {sc["verdict"] for sc in SCENARIOS} == {
        "Ready", "Conditional", "Blocked", "Insufficient Evidence"}


def test_all_checks_drive_at_least_one_scenario():
    driving = set().union(*(sc["driving"] for sc in SCENARIOS))
    assert driving == ALL_CHECK_NAMES


def test_at_least_ten_scenarios():
    assert len(SCENARIOS) >= 10

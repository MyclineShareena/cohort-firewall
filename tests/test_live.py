"""Live Claude tests. Excluded by default (pytest.ini: -m "not live").

Run explicitly:  .venv/bin/pytest -m live
Needs ANTHROPIC_API_KEY and ANTHROPIC_MODEL (environment or .env).
"""
import os

import pytest

import parser as P
from data.demo_cases import DEMO_CASES
from data.generate import build_all
from pipeline import assess

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def files():
    P.load_env()  # only when a live test actually runs
    if not (os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("ANTHROPIC_MODEL")):
        pytest.skip("ANTHROPIC_API_KEY / ANTHROPIC_MODEL not set")
    return build_all()


@pytest.mark.parametrize("case", DEMO_CASES, ids=lambda c: c["name"])
def test_live_demo_case(files, case):
    df = files[case["file"].removesuffix(".csv")]
    res = assess(df, case["question"], case["description"])
    assert res["parse"]["ok"], res["parse"]["error"]
    got, want = res["requirements"], case["requirements"]
    assert set(got["required_fields"]) == set(want["required_fields"])
    assert set(got["predictor_columns"]) == set(want["predictor_columns"])
    assert got["outcome"] == want["outcome"]
    assert got["time_horizon_months"] == want["time_horizon_months"]
    assert got["task_type"] == want["task_type"]
    assert got["target_population"] == want["target_population"]
    assert got["n_predictor_parameters"] == want["n_predictor_parameters"]
    assert bool(got["unmappable_terms"]) == bool(want["unmappable_terms"])
    if want["unmappable_terms"]:
        assert any("pd-l1" in t.lower() for t in got["unmappable_terms"])
    assert bool(got["population_from_description"]) == bool(want["population_from_description"])
    assert res["verdict"] == case["verdict"]
    assert {c["name"] for c in res["driving_checks"]} == case["driving"]
    assert res["explanation"]["source"] == "claude", res["explanation"]["error"]

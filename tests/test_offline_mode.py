"""API unavailable: demo cases use cached parses; everything else gets no verdict."""
import anthropic
import httpx2
import pandas as pd
import pytest

import parser as P
from data.demo_cases import DEMO_CASES, DESCRIPTION
from fakes import FakeClient
from pipeline import (CACHED_LABEL, DATA_DIR, OFFLINE_MESSAGE, SETUP_DETAIL, assess,
                      match_demo_case)


@pytest.fixture
def no_setup(monkeypatch):
    monkeypatch.setattr(P, "load_env", lambda: None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)


def read(name):
    return pd.read_csv(DATA_DIR / name)  # as the UI will load it


def api_down(n=4):
    err = anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.example"))
    return FakeClient([err] * n)


@pytest.mark.parametrize("case", DEMO_CASES, ids=lambda c: c["name"])
def test_demo_case_offline_gets_expected_verdict(no_setup, case):
    res = assess(read(case["file"]), case["question"], case["description"])
    assert res["status"] == "ok"
    assert res["parse_source"] == CACHED_LABEL
    assert any(CACHED_LABEL in n for n in res["notes"])
    assert res["verdict"] == case["verdict"]
    assert {c["name"] for c in res["driving_checks"]} == case["driving"]
    assert res["explanation"]["source"] == "fallback"


def test_demo_case_with_api_errors_uses_cache():
    case = DEMO_CASES[1]
    res = assess(read(case["file"]), case["question"], case["description"],
                 client=api_down(), model="test-model")
    assert res["status"] == "ok" and res["parse_source"] == CACHED_LABEL
    assert res["verdict"] == case["verdict"]


def test_non_demo_question_returns_setup_required(no_setup):
    res = assess(read("clean.csv"), "Does albumin predict 12-month death?", DESCRIPTION)
    assert res["status"] == "setup_required"
    assert res["message"] == OFFLINE_MESSAGE
    assert res["detail"] == SETUP_DETAIL
    assert res["verdict"] is None and res["checks"] is None and res["explanation"] is None


def test_demo_question_on_modified_file_returns_setup_required(no_setup):
    df = read("clean.csv").iloc[:-1]
    res = assess(df, DEMO_CASES[0]["question"], DESCRIPTION)
    assert res["status"] == "setup_required" and res["verdict"] is None


def test_demo_question_on_other_demo_file_not_cached(no_setup):
    # epv_vs_riley's question asked of messy.csv is not a demo case.
    case = next(c for c in DEMO_CASES if c["name"] == "epv_vs_riley")
    res = assess(read("messy.csv"), case["question"], DESCRIPTION)
    assert res["status"] == "setup_required"


def test_non_demo_with_api_errors_is_api_unavailable():
    res = assess(read("clean.csv"), "Does albumin predict 12-month death?", DESCRIPTION,
                 client=api_down(), model="test-model")
    assert res["status"] == "api_unavailable" and res["verdict"] is None
    assert res["message"] == OFFLINE_MESSAGE
    assert "APIConnectionError" in res["detail"] and "APIConnectionError" not in res["message"]


def test_question_match_ignores_case_and_whitespace():
    case = DEMO_CASES[0]
    q = "  " + case["question"].upper().replace(" ", "   ") + " "
    assert match_demo_case(read(case["file"]), q) is case


def test_cached_parse_not_used_when_api_works():
    from fakes import claude_parse_output
    case = DEMO_CASES[0]
    err = anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.example"))
    client = FakeClient([claude_parse_output(case["requirements"]), err, err])
    res = assess(read(case["file"]), case["question"], case["description"],
                 client=client, model="test-model")
    assert res["parse_source"] == "claude"


def test_demo_csvs_match_generator():
    # Fingerprints only work if the committed CSVs equal what the generator builds.
    from data.generate import build_all
    from pipeline import _df_fingerprint, _file_fingerprint
    for name, df in build_all().items():
        assert _df_fingerprint(df) == _file_fingerprint(f"{name}.csv"), name

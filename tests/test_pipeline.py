import anthropic
import httpx2
import pytest

from data.demo_cases import DEMO_CASES, DESCRIPTION
from data.generate import build_all
from fakes import FakeClient, claude_parse_output
from pipeline import assess


@pytest.fixture(scope="module")
def files():
    return build_all()


def assess_with(df, case, parse_outputs, **kw):
    # Explainer call fails -> deterministic fallback; keeps the focus on the pipeline.
    err = anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.example"))
    client = FakeClient(parse_outputs + [err, err])
    return assess(df, case["question"], case.get("description"), client=client,
                  model="test-model", **kw)


@pytest.mark.parametrize("case", DEMO_CASES, ids=lambda c: c["name"])
def test_demo_verdicts_end_to_end(files, case):
    df = files[case["file"].removesuffix(".csv")]
    res = assess_with(df, case, [claude_parse_output(case["requirements"])])
    assert res["verdict"] == case["verdict"]
    assert {c["name"] for c in res["driving_checks"]} == case["driving"]
    assert res["requirements"]["n_predictor_parameters"] == case["requirements"]["n_predictor_parameters"]
    assert res["explanation"]["source"] == "fallback"


def test_clean_main_question_with_description_is_ready(files):
    case = DEMO_CASES[0]
    assert "lung cancer patients" in case["question"] and case["description"] == DESCRIPTION
    res = assess_with(files["clean"], case, [claude_parse_output(case["requirements"])])
    assert res["verdict"] == "Ready"
    assert res["requirements"]["population_from_description"] == ["lung cancer patients"]
    assert res["requirements"]["unmappable_terms"] == []


def test_parse_failure_is_insufficient_evidence(files):
    case = DEMO_CASES[0]
    bad = {**claude_parse_output(case["requirements"]), "predictor_columns": ["nope"]}
    res = assess_with(files["clean"], case, [bad, bad])
    assert res["verdict"] == "Insufficient Evidence"
    assert [c["name"] for c in res["driving_checks"]] == ["question_parsed"]
    assert res["requirements"] is None


def test_parse_failure_still_blocks_identifiers(files):
    case = DEMO_CASES[0]
    df = files["clean"].assign(ssn="123-45-6789")
    bad = {**claude_parse_output(case["requirements"]), "predictor_columns": ["nope"]}
    res = assess_with(df, case, [bad, bad])
    assert res["verdict"] == "Blocked"
    assert [c["name"] for c in res["driving_checks"]] == ["identifier_columns"]


def test_p_override_wins(files):
    case = next(c for c in DEMO_CASES if c["name"] == "epv_vs_riley")
    out = claude_parse_output(case["requirements"])
    res = assess_with(files["epv_vs_riley"], case, [out], p_override=5)
    assert res["verdict"] == "Ready"
    ss = next(c for c in res["checks"] if c["name"] == "sample_size")
    assert ss["value"]["n_params"] == 5
    assert any("override" in n.lower() for n in res["notes"])


def test_r2_passed_through(files):
    case = DEMO_CASES[0]
    res = assess_with(files["clean"], case, [claude_parse_output(case["requirements"])], r2=0.2)
    ss = next(c for c in res["checks"] if c["name"] == "sample_size")
    assert ss["value"]["r2cs_source"] == "user"



def test_missing_setup_returns_no_verdict(files, monkeypatch):
    import parser as P
    from pipeline import SETUP_MESSAGE
    monkeypatch.setattr(P, "load_env", lambda: None)
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    res = assess(files["clean"], "Does albumin predict 12-month death?", DESCRIPTION)
    assert res["status"] == "setup_required" and res["message"] == SETUP_MESSAGE
    assert res["verdict"] is None and res["checks"] is None


def test_api_errors_return_no_verdict(files):
    err = anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.example"))
    res = assess(files["clean"], "Does albumin predict 12-month death?", DESCRIPTION,
                 client=FakeClient([err, err]), model="test-model")
    assert res["status"] == "api_unavailable" and res["verdict"] is None

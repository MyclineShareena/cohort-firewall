import inspect
import json

import anthropic
import httpx2
import pytest

import checks as C
import explainer as E
from data.demo_cases import DEMO_CASES
from data.generate import build_all
from fakes import FakeClient, good_explanation
from rules import assign_verdict


@pytest.fixture(scope="module")
def files():
    return build_all()


def scenario(files, name, **req_changes):
    case = next(c for c in DEMO_CASES if c["name"] == name)
    req = {**case["requirements"], **req_changes}
    checks = C.run_checks(files[case["file"].removesuffix(".csv")], req)
    return case["question"], req, checks, assign_verdict(checks)


@pytest.fixture
def messy(files):
    return scenario(files, "messy")


@pytest.fixture
def described(files):
    # Description task on leakage.csv: leakage + sample size are not applicable.
    return scenario(files, "leakage", task_type="description")


def run(outputs, sc):
    client = FakeClient(outputs)
    q, req, checks, v = sc
    return E.explain(q, req, checks, v, client=client, model="test-model"), client


# ---------- happy path ----------

def test_valid_explanation_from_claude(messy):
    out = good_explanation(messy[2], messy[3])
    res, client = run([out], messy)
    assert res["source"] == "claude" and res["error"] is None
    assert {d["name"] for d in res["driving_checks"]} == {"required_field_missingness", "sample_size"}


def test_explainer_sees_only_check_results(messy):
    assert "df" not in inspect.signature(E.explain).parameters
    _, client = run([good_explanation(messy[2], messy[3])], messy)
    text = client.messages.prompt_text(0)
    assert "Missing above 20%: stage 30%" in text and "Conditional" in text
    assert "SYN-0" not in text


# ---------- validators ----------

def errors(sc, **changes):
    out = {**good_explanation(sc[2], sc[3]), **changes}
    return E.validate_explanation(out, sc[2], sc[3])


def test_na_checks_must_all_be_listed(described):
    good = good_explanation(described[2], described[3])
    assert {n["name"] for n in good["not_applicable"]} == {"label_leakage", "sample_size"}
    assert errors(described) == []
    assert errors(described, not_applicable=good["not_applicable"][:1])
    assert errors(described, not_applicable=[])


def test_na_never_described_as_passed(described):
    na = [{"name": "label_leakage", "reason": "Passed: not relevant here."},
          {"name": "sample_size", "reason": "Not evaluated for description tasks."}]
    assert errors(described, not_applicable=na)
    assert errors(described, summary="Ready: all checks passed.")


def test_summary_must_not_name_other_verdicts(messy):
    assert errors(messy, summary="Conditional now, but it would be Ready with more data.")
    assert errors(messy, summary="This is close to Blocked.")
    # next actions may mention other verdicts conditionally
    actions = [{"check_name": "sample_size",
                "action": "Reaching n >= 1570 would move this toward Ready."},
               {"check_name": "required_field_missingness", "action": "Reduce stage missingness."}]
    assert errors(messy, next_actions=actions) == []


def test_driving_checks_must_match(messy):
    assert errors(messy, driving_checks=[{"name": "sample_size", "explanation": "x"}])
    assert errors(messy, driving_checks=[
        {"name": "sample_size", "explanation": "x"},
        {"name": "required_field_missingness", "explanation": "x"},
        {"name": "made_up_check", "explanation": "x"}])


def test_next_actions_target_failing_checks(messy):
    assert errors(messy, next_actions=[{"check_name": "label_leakage", "action": "x"}])
    assert errors(messy, next_actions=[{"check_name": "nope", "action": "x"}])
    assert errors(messy, next_actions=[])  # failures exist -> need actions


# ---------- retry + fallback ----------

def test_retry_with_feedback_then_success(messy):
    bad = {**good_explanation(messy[2], messy[3]), "summary": "Blocked, really."}
    res, client = run([bad, good_explanation(messy[2], messy[3])], messy)
    assert res["source"] == "claude"
    assert "Blocked" in client.messages.prompt_text(1)


def test_fallback_after_two_failures(described):
    bad = {**good_explanation(described[2], described[3]), "not_applicable": []}
    res, _ = run([bad, bad], described)
    assert res["source"] == "fallback" and res["error"]
    assert {n["name"] for n in res["not_applicable"]} == {"label_leakage", "sample_size"}
    assert E.validate_explanation(res, described[2], described[3]) == []


def test_fallback_on_api_error(messy):
    err = anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.example"))
    res, _ = run([err, err], messy)
    assert res["source"] == "fallback"
    assert "Conditional" in res["summary"]
    assert {a["check_name"] for a in res["next_actions"]} == {"required_field_missingness", "sample_size"}
    assert E.validate_explanation(res, messy[2], messy[3]) == []


def test_fallback_without_config(messy, monkeypatch):
    import parser as P
    monkeypatch.setattr(P, "load_env", lambda: None)
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-secret-value")
    q, req, checks, v = messy
    res = E.explain(q, req, checks, v)
    assert res["source"] == "fallback" and "ANTHROPIC_MODEL" in res["error"]
    assert "sk-test-secret-value" not in json.dumps(res)


@pytest.mark.parametrize("name", [c["name"] for c in DEMO_CASES])
def test_fallback_valid_for_every_demo(files, name):
    sc = scenario(files, name)
    res = E.fallback_explanation(sc[2], sc[3], "test")
    assert E.validate_explanation(res, sc[2], sc[3]) == []

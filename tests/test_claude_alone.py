import anthropic
import httpx2
import pytest

import parser as P
from data.generate import build_all
from experiments.claude_alone import assess_alone, claude_alone_user_message
from fakes import FakeClient

QUESTION = "Predict death within 24 months using age, sex, stage, ECOG and albumin."


@pytest.fixture(scope="module")
def profile():
    return P.profile_dataset(build_all()["clean"])


def test_ok_response_parsed(profile):
    out = {"verdict": "Conditional", "concerns": ["small sample"], "reasoning": "n is low"}
    res = assess_alone(QUESTION, profile, "desc", client=FakeClient([out]), model="test-model")
    assert res == {"status": "ok", "verdict": "Conditional", "concerns": ["small sample"],
                   "reasoning": "n is low", "error": None}


def test_invalid_verdict_label_rejected(profile):
    bad = {"verdict": "Fine", "concerns": [], "reasoning": "ok"}
    res = assess_alone(QUESTION, profile, "desc", client=FakeClient([bad]), model="test-model")
    assert res["status"] == "invalid_output"
    assert res["verdict"] is None


def test_api_failure_is_api_unavailable(profile):
    err = anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.example"))
    res = assess_alone(QUESTION, profile, "desc", client=FakeClient([err]), model="test-model")
    assert res["status"] == "api_unavailable"


def test_missing_model_fails_cleanly(profile, monkeypatch):
    monkeypatch.setattr(P, "load_env", lambda: None)
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    res = assess_alone(QUESTION, profile, "desc")
    assert res["status"] == "setup_required"
    assert "ANTHROPIC_MODEL" in res["error"]


def test_prompt_has_profile_and_description_not_rows(profile):
    out = {"verdict": "Ready", "concerns": [], "reasoning": "fine"}
    client = FakeClient([out])
    assess_alone(QUESTION, profile, "Synthetic lung cancer cohort", client=client, model="test-model")
    text = client.messages.prompt_text(0)
    assert "Synthetic lung cancer cohort" in text and "histology" in text and QUESTION in text
    assert "SYN-0" not in text
    assert client.messages.calls[0].get("system")


def test_user_message_has_no_description_fallback(profile):
    text = claude_alone_user_message(QUESTION, profile, None)
    assert "none given" in text

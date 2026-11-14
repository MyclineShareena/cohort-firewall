import anthropic
import httpx2
import pytest

import parser as P
from data.demo_cases import DEMO_CASES, DESCRIPTION
from data.generate import build_all
from fakes import REFUSAL, FakeClient, claude_parse_output


@pytest.fixture(scope="module")
def files():
    return build_all()


@pytest.fixture(scope="module")
def profile(files):
    return P.profile_dataset(files["clean"])


CLEAN = DEMO_CASES[0]


def good_output(**changes):
    out = claude_parse_output(CLEAN["requirements"])
    out.update(changes)
    return out


def run(outputs, profile, description=DESCRIPTION, question=CLEAN["question"]):
    client = FakeClient(outputs)
    res = P.parse_question(question, profile, description, client=client, model="test-model")
    return res, client


# ---------- profile + parameter counting ----------

def test_profile_kinds_and_no_row_values(profile):
    cols = {c["name"]: c for c in profile["columns"]}
    assert cols["age"]["kind"] == "numeric" and "min" in cols["age"]
    assert cols["ecog"]["kind"] == "numeric"
    assert cols["stage"]["kind"] == "categorical" and cols["stage"]["levels"] == ["I", "II", "III", "IV"]
    assert cols["diagnosis_date"]["kind"] == "date"
    assert cols["patient_id"]["kind"] == "identifier" and "levels" not in cols["patient_id"]
    assert "SYN-00001" not in str(profile)


@pytest.mark.parametrize("case", DEMO_CASES, ids=lambda c: c["name"])
def test_parameter_count_matches_demo_cases(files, case):
    prof = P.profile_dataset(files[case["file"].removesuffix(".csv")])
    req = case["requirements"]
    assert P.count_predictor_parameters(prof, req["predictor_columns"]) == req["n_predictor_parameters"]


def test_epv_vs_riley_predictors_give_exactly_10(files):
    case = next(c for c in DEMO_CASES if c["name"] == "epv_vs_riley")
    prof = P.profile_dataset(files["epv_vs_riley"])
    assert P.count_predictor_parameters(prof, case["requirements"]["predictor_columns"]) == 10


# ---------- success, normalization, disagreement note ----------

def test_parse_success_computes_parameters(profile):
    res, client = run([good_output(claimed_n_predictor_parameters=None)], profile)
    assert res["ok"] and res["attempts"] == 1
    req = res["requirements"]
    assert req["n_predictor_parameters"] == 11
    assert req["population_from_description"] == ["lung cancer patients"]
    assert res["notes"] == []
    assert client.messages.calls[0]["model"] == "test-model"


def test_disagreement_note(profile):
    res, _ = run([good_output(claimed_n_predictor_parameters=14)], profile)
    assert res["requirements"]["n_predictor_parameters"] == 11
    assert any("14" in n and "11" in n for n in res["notes"])


def test_normalization(profile):
    out = good_output(required_fields=["age", "vital_status", "diagnosis_date"])
    res, _ = run([out], profile)
    req = res["requirements"]
    assert "vital_status" not in req["required_fields"]
    assert set(CLEAN["requirements"]["predictor_columns"]) <= set(req["required_fields"])


# ---------- retry and failure ----------

def test_retry_once_with_feedback(profile):
    bad = good_output(required_fields=["age", "pdl1_status"])
    res, client = run([bad, good_output()], profile)
    assert res["ok"] and res["attempts"] == 2
    assert "pdl1_status" in client.messages.prompt_text(1)


def test_two_failures_return_error(profile):
    bad = good_output(predictor_columns=["pdl1_status"])
    res, _ = run([bad, bad], profile)
    assert not res["ok"] and res["attempts"] == 2 and res["requirements"] is None
    assert res["status"] == "parse_failed" and "pdl1_status" in res["error"]


def test_refusal_counts_as_attempt(profile):
    res, _ = run([REFUSAL, good_output()], profile)
    assert res["ok"] and res["attempts"] == 2


def test_api_error_counts_as_attempt(profile):
    err = anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.example"))
    res, _ = run([err, err], profile)
    assert not res["ok"] and res["attempts"] == 2 and res["status"] == "api_unavailable"


def test_api_error_then_refusal_is_parse_failure(profile):
    err = anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.example"))
    res, _ = run([err, REFUSAL], profile)
    assert res["status"] == "parse_failed"


# ---------- validators ----------

def errors_for(profile, description=DESCRIPTION, **changes):
    return P.validate_parsed(good_output(**changes), profile, description)


def test_valid_output_has_no_errors(profile):
    assert errors_for(profile) == []


def test_unknown_columns_rejected(profile):
    assert errors_for(profile, outcome={"column": "os_status", "type": "binary"})
    assert errors_for(profile, target_population=[
        {"column": "pdl1", "operator": "==", "value": "high"}])


def test_unmappable_overlap_rejected(profile):
    assert errors_for(profile, unmappable_terms=["stage"])


def test_population_from_description_rules(profile):
    assert errors_for(profile, description=None)  # no description -> must be empty
    assert errors_for(profile, population_from_description=["PD-L1 status"],
                      unmappable_terms=["PD-L1 status"])
    assert errors_for(profile, population_from_description=["ecog"])


def test_bad_predictors_rejected(profile):
    assert errors_for(profile, predictor_columns=["age", "patient_id"])
    assert errors_for(profile, predictor_columns=["age", "vital_status"])


def test_filter_values_checked(profile):
    assert errors_for(profile, target_population=[
        {"column": "stage", "operator": "in", "value": ["III", "V"]}])
    assert errors_for(profile, target_population=[
        {"column": "age", "operator": ">=", "value": "old"}])
    assert errors_for(profile, target_population=[
        {"column": "stage", "operator": "in", "value": "III"}]) == []  # scalar ok for in


# ---------- prompt content + config ----------

def test_prompt_has_profile_and_description_not_rows(profile):
    _, client = run([good_output()], profile)
    text = client.messages.prompt_text(0)
    assert DESCRIPTION in text and "histology" in text and CLEAN["question"] in text
    assert "SYN-0" not in text
    assert client.messages.calls[0].get("system")


def test_missing_model_fails_cleanly(profile, monkeypatch):
    monkeypatch.setattr(P, "load_env", lambda: None)
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-secret-value")
    res = P.parse_question(CLEAN["question"], profile, DESCRIPTION)
    assert not res["ok"] and res["attempts"] == 0 and res["status"] == "setup_required"
    assert "ANTHROPIC_MODEL" in res["error"]
    assert "sk-test-secret-value" not in str(res)


def test_missing_key_fails_cleanly(monkeypatch):
    monkeypatch.setattr(P, "load_env", lambda: None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(P.ConfigError, match="ANTHROPIC_API_KEY"):
        P.get_client()


def test_model_read_from_env(profile, monkeypatch):
    monkeypatch.setattr(P, "load_env", lambda: None)
    monkeypatch.setenv("ANTHROPIC_MODEL", "model-from-env")
    client = FakeClient([good_output()])
    P.parse_question(CLEAN["question"], profile, DESCRIPTION, client=client)
    assert client.messages.calls[0]["model"] == "model-from-env"


def test_schemas_transform_for_structured_outputs():
    from anthropic.lib._parse._transform import transform_schema
    import explainer

    def objects(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                yield node
            for v in node.values():
                yield from objects(v)
        elif isinstance(node, list):
            for v in node:
                yield from objects(v)

    for model in (P.ParsedQuestion, explainer.ExplanationOut):
        schema = transform_schema(model)
        objs = list(objects(schema))
        assert objs and all(o.get("additionalProperties") is False for o in objs)


def test_gitignore_has_env():
    from pathlib import Path
    lines = (Path(P.__file__).parent / ".gitignore").read_text().splitlines()
    assert ".env" in lines


def test_vital_status_outcome_allowed_when_column_missing(files):
    # broken.csv lacks vital_status: the parse must succeed so the checks can Block it.
    prof = P.profile_dataset(files["broken"])
    assert P.validate_parsed(good_output(), prof, DESCRIPTION) == []

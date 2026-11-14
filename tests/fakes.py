"""Offline stand-in for anthropic.Anthropic: replays canned outputs, records calls."""
from __future__ import annotations

import copy
from types import SimpleNamespace


class FakeMessages:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        item = self.outputs.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, SimpleNamespace):  # raw response, e.g. a refusal
            return item
        # Validate through the real Pydantic model, as the SDK would.
        parsed = kwargs["output_format"].model_validate(item)
        return SimpleNamespace(parsed_output=parsed, stop_reason="end_turn")

    def prompt_text(self, i=-1) -> str:
        return "\n".join(m["content"] for m in self.calls[i]["messages"])


class FakeClient:
    def __init__(self, outputs):
        self.messages = FakeMessages(outputs)


REFUSAL = SimpleNamespace(parsed_output=None, stop_reason="refusal")


def claude_parse_output(requirements: dict, claimed: int | None = None) -> dict:
    """What Claude returns for a ground-truth requirements dict (no computed count)."""
    out = copy.deepcopy(requirements)
    out.pop("n_predictor_parameters", None)
    out["claimed_n_predictor_parameters"] = (
        requirements.get("n_predictor_parameters") if claimed is None else claimed)
    return out


def good_explanation(checks: list[dict], verdict: dict) -> dict:
    """A valid explainer output for these checks."""
    driving = [c["name"] for c in verdict["driving_checks"]]
    bad = [c["name"] for c in checks if c["status"] in {"fail", "unknown"}]
    return {
        "summary": f"The rules engine assigned {verdict['verdict']} based on the check results.",
        "driving_checks": [{"name": n, "explanation": "See the check result."} for n in driving],
        "not_applicable": [{"name": c["name"], "reason": "It does not apply to this task."}
                           for c in checks if c["status"] == "not_applicable"],
        "next_actions": [{"check_name": n, "action": "Address this finding."} for n in bad],
    }

"""End to end: question -> requirements (Claude) -> checks -> verdict (rules) -> explanation (Claude).

Result "status":
  "ok"              -> a verdict describing the data
  "setup_required"  -> ANTHROPIC_API_KEY / ANTHROPIC_MODEL missing; no verdict
  "api_unavailable" -> the API could not be reached/used; no verdict
A verdict only ever describes the data, never the tool's own setup.
"""
from __future__ import annotations

import pandas as pd

import checks as C
from explainer import explain
from parser import parse_question, profile_dataset
from rules import assign_verdict

SETUP_MESSAGE = "Set ANTHROPIC_API_KEY and ANTHROPIC_MODEL"


def _no_verdict(status: str, message: str, parse: dict, question, description) -> dict:
    return {"status": status, "message": message, "question": question,
            "description": description, "parse": parse, "parse_source": None,
            "requirements": None, "checks": None, "verdict": None, "driving_checks": None,
            "explanation": None, "notes": []}


def assess(df: pd.DataFrame, question: str, description: str | None = None,
           r2: float | None = None, p_override: int | None = None,
           client=None, model: str | None = None) -> dict:
    parse = parse_question(question, profile_dataset(df), description, client=client, model=model)
    notes = list(parse["notes"])
    parse_source = "claude"

    if parse["status"] in {"setup_required", "api_unavailable"}:
        msg = (SETUP_MESSAGE if parse["status"] == "setup_required"
               else f"Claude API unavailable ({parse['error']}). Check network, key and "
                    "credits, then re-run.")
        return _no_verdict(parse["status"], msg, parse, question, description)

    req = parse["requirements"]
    if req is not None:
        checks = C.run_checks(df, req, anticipated_r2cs=r2, n_params_override=p_override)
        if p_override:
            notes.append(f"Predictor-parameter override in use: {p_override} "
                         f"(computed from predictor columns: {req['n_predictor_parameters']}).")
    else:
        # Claude answered but the question couldn't be mapped; identifiers still screened.
        checks = [C.check_question_parsed(parse["error"]), C.check_identifiers(df)]
    verdict = assign_verdict(checks)
    explanation = explain(question, req, checks, verdict, client=client, model=model)
    return {"status": "ok", "message": None, "question": question, "description": description,
            "parse": parse, "parse_source": parse_source, "requirements": req,
            "checks": checks, "verdict": verdict["verdict"],
            "driving_checks": verdict["driving_checks"], "explanation": explanation,
            "notes": notes}

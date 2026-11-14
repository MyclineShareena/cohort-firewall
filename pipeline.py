"""End to end: question -> requirements (Claude) -> checks -> verdict (rules) -> explanation (Claude).

Result "status":
  "ok"              -> a verdict describing the data
  "setup_required"  -> ANTHROPIC_API_KEY / ANTHROPIC_MODEL missing; no verdict
  "api_unavailable" -> the API could not be reached/used; no verdict
For both, "message" is the user-facing text (no technical detail) and "detail"
carries the technical reason (env var names, API error type).
A verdict only ever describes the data, never the tool's own setup.

Offline demo mode: when the API is unavailable and the input is exactly a demo
case (same file content and same question), that case's hand-written
requirements are used, labeled "cached parse (offline)". Never for other inputs.
"""
from __future__ import annotations

import copy
import hashlib
import re
from pathlib import Path

import pandas as pd

import checks as C
from data.demo_cases import DEMO_CASES
from explainer import explain
from parser import parse_question, profile_dataset
from rules import assign_verdict

DATA_DIR = Path(__file__).resolve().parent / "data"
CACHED_LABEL = "cached parse (offline)"
OFFLINE_MESSAGE = ("Reading a new question needs live Claude. "
                   "The six demo questions work offline.")
SETUP_DETAIL = "Set ANTHROPIC_API_KEY and ANTHROPIC_MODEL"


def _df_fingerprint(df: pd.DataFrame) -> str:
    return hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()


def _file_fingerprint(name: str) -> str | None:
    path = DATA_DIR / name
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _norm_question(q: str) -> str:
    return re.sub(r"\s+", " ", q).strip().casefold()


def match_demo_case(df: pd.DataFrame, question: str) -> dict | None:
    """The demo case whose file content AND question match this input exactly."""
    fp, q = _df_fingerprint(df), _norm_question(question)
    for case in DEMO_CASES:
        if _norm_question(case["question"]) == q and _file_fingerprint(case["file"]) == fp:
            return case
    return None


def _no_verdict(status: str, detail: str, parse: dict, question, description) -> dict:
    return {"status": status, "message": OFFLINE_MESSAGE, "detail": detail, "question": question,
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
        case = match_demo_case(df, question)
        if case is None:
            detail = (SETUP_DETAIL if parse["status"] == "setup_required"
                      else f"Claude API unavailable ({parse['error']}). Check network, key "
                           "and credits, then re-run.")
            return _no_verdict(parse["status"], detail, parse, question, description)
        parse = {**parse, "requirements": copy.deepcopy(case["requirements"])}
        parse_source = CACHED_LABEL
        notes.append(f"Using {CACHED_LABEL}: the hand-written requirements for demo case "
                     f"'{case['name']}', because the Claude API is unavailable.")

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
    return {"status": "ok", "message": None, "detail": None, "question": question, "description": description,
            "parse": parse, "parse_source": parse_source, "requirements": req,
            "checks": checks, "verdict": verdict["verdict"],
            "driving_checks": verdict["driving_checks"], "explanation": explanation,
            "notes": notes}

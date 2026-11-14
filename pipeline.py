"""End to end: question -> requirements (Claude) -> checks -> verdict (rules) -> explanation (Claude)."""
from __future__ import annotations

import pandas as pd

import checks as C
from explainer import explain
from parser import parse_question, profile_dataset
from rules import assign_verdict


def assess(df: pd.DataFrame, question: str, description: str | None = None,
           r2: float | None = None, p_override: int | None = None,
           client=None, model: str | None = None) -> dict:
    parse = parse_question(question, profile_dataset(df), description, client=client, model=model)
    req = parse["requirements"]
    notes = list(parse["notes"])
    if parse["ok"]:
        checks = C.run_checks(df, req, anticipated_r2cs=r2, n_params_override=p_override)
        if p_override:
            notes.append(f"Predictor-parameter override in use: {p_override} "
                         f"(computed from predictor columns: {req['n_predictor_parameters']}).")
    else:
        # The question can't be evaluated, but identifier screening still applies.
        checks = [C.check_question_parsed(parse["error"], config_error=parse["attempts"] == 0),
                  C.check_identifiers(df)]
    verdict = assign_verdict(checks)
    explanation = explain(question, req, checks, verdict, client=client, model=model)
    return {"question": question, "description": description, "parse": parse,
            "requirements": req, "checks": checks, "verdict": verdict["verdict"],
            "driving_checks": verdict["driving_checks"], "explanation": explanation,
            "notes": notes}

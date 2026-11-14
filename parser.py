"""Claude call: plain-English question -> requirements JSON.

Claude maps the question onto the dataset's columns. Python validates the result,
computes n_predictor_parameters from predictor_columns, and retries once with the
validation errors before giving up (the pipeline then returns Insufficient Evidence).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional, Union

import anthropic
import pandas as pd
import pydantic
from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict

import prompts
from checks import DATES, ID_NAME, OUTCOME_SOURCES

ROOT = Path(__file__).resolve().parent
MAX_LEVELS_SHOWN = 12
MAX_TOKENS = 16000


class ConfigError(RuntimeError):
    pass


def load_env() -> None:
    """Load .env from the project root. Called before every API use, so a key added
    to .env while the app is running takes effect on the next run. Non-empty
    environment variables win over .env; empty ones are filled from .env."""
    for key, value in dotenv_values(ROOT / ".env").items():
        if value and not os.environ.get(key):
            os.environ[key] = value


def get_model() -> str:
    load_env()
    model = os.environ.get("ANTHROPIC_MODEL")
    if not model:
        raise ConfigError("ANTHROPIC_MODEL is not set (environment or .env).")
    return model


def get_client() -> anthropic.Anthropic:
    load_env()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise ConfigError("ANTHROPIC_API_KEY is not set (environment or .env).")
    return anthropic.Anthropic()


def call_structured(client, model: str, system: str, user: str, output_format):
    """One structured-output call.

    Returns (parsed dict or None, error or None, api_failed). api_failed is True
    when the API itself could not be used (network, auth, credits, server error).
    """
    try:
        resp = client.messages.parse(
            model=model, max_tokens=MAX_TOKENS, system=system,
            messages=[{"role": "user", "content": user}], output_format=output_format)
    except anthropic.APIError as e:
        return None, f"API call failed: {type(e).__name__}", True
    except (pydantic.ValidationError, ValueError) as e:
        return None, f"invalid structured output: {type(e).__name__}", False
    if resp.stop_reason == "refusal":
        return None, "the model declined the request", False
    if resp.stop_reason == "max_tokens":
        return None, "the response was cut off (max_tokens)", False
    if resp.parsed_output is None:
        return None, "no structured output returned", False
    return resp.parsed_output.model_dump(), None, False


# ------------------------------------------------------------------ schema

class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Filter(_Strict):
    column: str
    operator: Literal["==", "!=", "in", "not_in", "<", "<=", ">", ">="]
    value: Union[float, str, list[Union[float, str]]]


class Outcome(_Strict):
    column: str
    type: Literal["binary", "time_to_event"]


class ParsedQuestion(_Strict):
    target_population: list[Filter]
    outcome: Optional[Outcome]
    required_fields: list[str]
    predictor_columns: list[str]
    time_horizon_months: Optional[float]
    task_type: Literal["prediction", "description", "comparison"]
    claimed_n_predictor_parameters: Optional[int]
    unmappable_terms: list[str]
    population_from_description: list[str]


# ------------------------------------------------------------------ profile

def _kind(name: str, s: pd.Series) -> str:
    if ID_NAME.search(name.lower()):
        return "identifier"
    if pd.api.types.is_bool_dtype(s):
        return "categorical"
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    if name in DATES or name.lower().endswith("_date"):
        return "date"
    return "categorical"


def profile_dataset(df: pd.DataFrame) -> dict:
    """Column-level summary sent to Claude. No row values except category levels."""
    cols = []
    for name in df.columns:
        s = df[name]
        info = {"name": name, "kind": _kind(name, s),
                "missing_fraction": round(float(s.isna().mean()), 3)}
        if info["kind"] == "numeric":
            info.update(min=float(s.min()), max=float(s.max()))
        elif info["kind"] == "categorical":
            levels = sorted(s.dropna().astype(str).unique())
            info["n_levels"] = len(levels)
            if len(levels) <= MAX_LEVELS_SHOWN:
                info["levels"] = levels
        cols.append(info)
    return {"n_rows": len(df), "columns": cols}


def count_predictor_parameters(profile: dict, columns: list[str]) -> int:
    """numeric/date = 1 parameter; categorical with k levels = k - 1."""
    by = {c["name"]: c for c in profile["columns"]}
    total = 0
    for col in dict.fromkeys(columns):
        info = by[col]
        total += max(info["n_levels"] - 1, 0) if info["kind"] == "categorical" else 1
    return total


# ------------------------------------------------------------------ validation

def validate_parsed(parsed: dict, profile: dict, description: str | None) -> list[str]:
    by = {c["name"]: c for c in profile["columns"]}
    names = list(by)
    errs = []

    def unknown(cols, where):
        for c in cols:
            if c not in by:
                errs.append(f"{where}: '{c}' is not a dataset column (columns: {', '.join(names)}). "
                            "If no column represents it, put the term in unmappable_terms.")

    oc = parsed.get("outcome")
    # Death outcomes always map to vital_status, even when the file lacks it:
    # whether the outcome is derivable is decided by the checks (-> Blocked).
    if oc and oc["column"] != "vital_status":
        unknown([oc["column"]], "outcome.column")
    unknown(parsed["required_fields"], "required_fields")
    unknown(parsed["predictor_columns"], "predictor_columns")
    for c in parsed["predictor_columns"]:
        if c in {"vital_status", "last_followup_date"}:
            errs.append(f"predictor_columns: '{c}' is an outcome source column, not a predictor.")
        elif c in by and by[c]["kind"] == "identifier":
            errs.append(f"predictor_columns: '{c}' is an identifier column, not a predictor.")

    for f in parsed["target_population"]:
        col, op, val = f["column"], f["operator"], f["value"]
        if col not in by:
            unknown([col], "target_population")
            continue
        vals = val if isinstance(val, list) else [val]
        info = by[col]
        if op in {"<", "<=", ">", ">="} and not all(isinstance(v, (int, float)) for v in vals):
            errs.append(f"target_population: operator {op} on '{col}' needs a number, got {val!r}.")
        if info.get("levels") and op in {"==", "!=", "in", "not_in"}:
            bad = [v for v in vals if str(v) not in info["levels"]]
            if bad:
                errs.append(f"target_population: {bad} not in levels of '{col}' "
                            f"({', '.join(info['levels'])}).")

    if parsed.get("time_horizon_months") is not None and parsed["time_horizon_months"] <= 0:
        errs.append("time_horizon_months must be positive.")

    cols_lower = {c.lower() for c in parsed["required_fields"] + parsed["predictor_columns"]}
    unm = {t.lower() for t in parsed["unmappable_terms"]}
    pop = {t.lower() for t in parsed["population_from_description"]}
    if unm & cols_lower:
        errs.append(f"unmappable_terms overlap required_fields/predictor_columns: {sorted(unm & cols_lower)}.")
    if pop and not description:
        errs.append("population_from_description must be empty: no dataset description was given.")
    if pop & (unm | cols_lower):
        errs.append("population_from_description must not repeat unmappable_terms or columns: "
                    f"{sorted(pop & (unm | cols_lower))}.")
    return errs


def normalize(parsed: dict, profile: dict) -> tuple[dict, list[str]]:
    """Deterministic cleanup + parameter count. Returns (requirements, notes)."""
    req = dict(parsed)
    claimed = req.pop("claimed_n_predictor_parameters", None)
    predictors = list(dict.fromkeys(req["predictor_columns"]))
    req["predictor_columns"] = predictors
    req["required_fields"] = [c for c in dict.fromkeys(req["required_fields"] + predictors)
                              if c not in OUTCOME_SOURCES]
    req["n_predictor_parameters"] = count_predictor_parameters(profile, predictors) or None
    notes = []
    if claimed is not None and claimed != req["n_predictor_parameters"]:
        notes.append(f"Claude counted {claimed} predictor parameters; the checker computed "
                     f"{req['n_predictor_parameters']} from the predictor columns and uses that.")
    return req, notes


# ------------------------------------------------------------------ entry point

def parse_question(question: str, profile: dict, description: str | None = None,
                   client=None, model: str | None = None) -> dict:
    """Returns {status, ok, requirements, error, attempts, notes}.

    status: "ok" | "parse_failed" (Claude answered, but no valid requirements)
            | "setup_required" (key/model not configured)
            | "api_unavailable" (every attempt failed at the API level)
    """
    def fail(status, error, attempts):
        return {"status": status, "ok": False, "requirements": None, "error": error,
                "attempts": attempts, "notes": []}

    try:
        model = model or get_model()
        client = client or get_client()
    except ConfigError as e:
        return fail("setup_required", str(e), 0)

    feedback, previous, last_error, all_api_failed = None, None, None, True
    for attempt in (1, 2):
        user = prompts.parser_user_message(question, profile, description, feedback, previous)
        parsed, err, api_failed = call_structured(client, model, prompts.PARSER_SYSTEM, user,
                                                  ParsedQuestion)
        all_api_failed &= api_failed
        if parsed is None:
            last_error, feedback, previous = err, None, None
            continue
        errs = validate_parsed(parsed, profile, description)
        if not errs:
            req, notes = normalize(parsed, profile)
            return {"status": "ok", "ok": True, "requirements": req, "error": None,
                    "attempts": attempt, "notes": notes}
        last_error, feedback, previous = "; ".join(errs), errs, parsed
    if all_api_failed:
        return fail("api_unavailable", last_error, 2)
    return fail("parse_failed",
                f"Could not parse the question into valid requirements: {last_error}", 2)

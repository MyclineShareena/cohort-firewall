"""Deterministic dataset checks, labeled by METRIC cluster (Schwabe et al. 2024).

Each check returns {name, metric_cluster, severity, status, value, threshold,
message}. status is pass | fail | unknown | not_applicable; not_applicable
is neutral for the verdict but is never a pass, and its message says why. Checks never assign the verdict; rules.assign_verdict does.
"""
from __future__ import annotations

import re

import pandas as pd

import sample_size as ss

DAYS_PER_MONTH = 30.4375  # 365.25 / 12

# "sample size" and "safety" are labeled here because the spec leaves them open:
# METRIC groups dataset size under representativeness; identifier screening is
# outside METRIC.
CLUSTERS = {"consistency", "informativeness", "timeliness", "representativeness",
            "measurement process", "safety"}

OUTCOME_SOURCES = ["vital_status", "diagnosis_date", "last_followup_date"]

NUMERIC = {"age", "ecog", "hemoglobin", "albumin", "ldh", "wbc"}
DATES = {"diagnosis_date", "last_followup_date", "date_of_death"}
CATEGORIES = {"vital_status": {"alive", "dead"}, "sex": {"m", "f"},
              "stage": {"i", "ii", "iii", "iv"}}

MISSING_SOFT, MISSING_HARD = 0.20, 0.40
CENSORING_SOFT = 0.15
MIN_SUBGROUP = 30
AGE_BINS, AGE_LABELS = [-1, 59, 69, 79, 200], ["<60", "60-69", "70-79", ">=80"]
SUBGROUP_LEVELS = {"sex": ["M", "F"], "age_band": AGE_LABELS, "stage": ["I", "II", "III", "IV"]}
EPV_MIN = 10
# Physically possible ranges only (not clinical normal ranges): flag impossible values,
# not unusual ones. None = no bound.
PLAUSIBLE_RANGES = {"age": (0, 120), "ecog": (0, 5), "hemoglobin": (0, None),
                    "albumin": (0, None), "ldh": (0, None), "wbc": (0, None)}

LEAKAGE_NAME = re.compile(r"death|deceased|died|^dod$|^(os|survival)_(time|days|months)$")
ID_NAME = re.compile(r"(^|_)id$")
SYNTHETIC_ID = r"SYN-\d+"
IDENTIFIER_NAMES = {
    "SSN": re.compile(r"ssn|social_security"),
    "MRN": re.compile(r"mrn|medical_record"),
    "name": re.compile(r"^((first|last|full|patient|given|family)_)?name$|surname"),
    "address": re.compile(r"address|street"),
    "phone": re.compile(r"phone|mobile|fax"),
    "email": re.compile(r"e_?mail"),
    "date of birth": re.compile(r"^dob$|date_of_birth|birth_?date"),
    "ZIP code": re.compile(r"zip|postcode|postal"),
}
IDENTIFIER_VALUES = {  # applied to text columns only
    "SSN": r"\d{3}-\d{2}-\d{4}",
    "email": r"[^@\s]+@[^@\s]+\.[A-Za-z]{2,}",
}


def _result(name, cluster, severity, status, value=None, threshold=None, message=""):
    return {"name": name, "metric_cluster": cluster, "severity": severity, "status": status,
            "value": value, "threshold": threshold, "message": message}


# ---------------------------------------------------------------- helpers

def derive_binary_outcome(
    df: pd.DataFrame,
    horizon_months: float,
    status_col: str = "vital_status",
    start_col: str = "diagnosis_date",
    end_col: str = "last_followup_date",
) -> pd.Series:
    """Derive "died within horizon" from raw columns.

    Assumes last_followup_date is the date of death for patients who died.

    Returns a nullable Int64 series:
      1    -> dead, with follow-up <= horizon (event)
      0    -> follow-up > horizon, alive or died later (known non-event)
      <NA> -> alive with follow-up < horizon (censored early: outcome unknown),
              or the dates/status are missing
    """
    start = pd.to_datetime(df[start_col], errors="coerce")
    end = pd.to_datetime(df[end_col], errors="coerce")
    fu_days = (end - start).dt.days
    horizon_days = horizon_months * DAYS_PER_MONTH
    dead = df[status_col].astype("string").str.lower().eq("dead")

    out = pd.Series(pd.NA, index=df.index, dtype="Int64")
    known = fu_days.notna() & df[status_col].notna()
    out[known & dead & (fu_days <= horizon_days)] = 1
    out[known & (fu_days > horizon_days)] = 0
    return out


def outcome_summary(df: pd.DataFrame, horizon_months: float) -> dict:
    """Event count, known non-events and early-censored exclusions."""
    y = derive_binary_outcome(df, horizon_months)
    return {
        "events": int((y == 1).sum()),
        "non_events": int((y == 0).sum()),
        "excluded_unknown": int(y.isna().sum()),
    }


def apply_population(df: pd.DataFrame, filters: list[dict] | None) -> pd.DataFrame:
    """Apply target_population filters. Filters on absent columns are skipped
    (check_required_fields reports them)."""
    mask = pd.Series(True, index=df.index)
    for f in filters or []:
        col, op, val = f["column"], f["operator"], f["value"]
        if col not in df.columns:
            continue
        s = df[col]
        if op in {"<", "<=", ">", ">="}:
            s = pd.to_numeric(s, errors="coerce")
        if op == "==":
            m = s == val
        elif op == "!=":
            m = s != val
        elif op == "in":
            m = s.isin(val if isinstance(val, list) else [val])
        elif op == "not_in":
            m = ~s.isin(val if isinstance(val, list) else [val])
        elif op == "<":
            m = s < val
        elif op == "<=":
            m = s <= val
        elif op == ">":
            m = s > val
        elif op == ">=":
            m = s >= val
        else:
            raise ValueError(f"unsupported operator: {op}")
        mask &= m.fillna(False).astype(bool)
    return df[mask]


def _is_mortality(req: dict) -> bool:
    return (req.get("outcome") or {}).get("column") == "vital_status"


def _required_columns(req: dict) -> list[str]:
    """Required fields + filter columns + outcome sources. unmappable_terms are
    deliberately NOT included (they go to check_terms_mappable)."""
    cols = list(req.get("required_fields") or [])
    cols += list(req.get("predictor_columns") or [])
    cols += [f["column"] for f in req.get("target_population") or []]
    oc = (req.get("outcome") or {}).get("column")
    if _is_mortality(req):
        cols += OUTCOME_SOURCES
    elif oc:
        cols.append(oc)
    return list(dict.fromkeys(cols))


def _type_problem(s: pd.Series, col: str) -> str | None:
    x = s.dropna()
    if x.empty:
        return None
    if col in NUMERIC:
        bad = pd.to_numeric(x, errors="coerce").isna().sum()
        return f"{col}: {bad} non-numeric value(s)" if bad else None
    if col in DATES:
        bad = pd.to_datetime(x, errors="coerce", format="%Y-%m-%d").isna().sum()
        return f"{col}: {bad} unparseable date(s)" if bad else None
    if col in CATEGORIES:
        bad = sorted(set(x.astype(str).str.lower()) - CATEGORIES[col])
        return f"{col}: unexpected value(s) {bad[:5]}" if bad else None
    return None


# ---------------------------------------------------------------- checks

def check_required_fields(df: pd.DataFrame, req: dict) -> dict:
    """1. Consistency: required fields exist with expected types (hard)."""
    cols = _required_columns(req)
    missing = [c for c in cols if c not in df.columns]
    problems = [p for c in cols if c in df.columns if (p := _type_problem(df[c], c))]
    if missing or problems:
        parts = []
        if missing:
            parts.append(f"missing column(s): {', '.join(missing)}")
        parts += problems
        return _result("required_fields_present", "consistency", "hard", "fail",
                       {"missing": missing, "type_problems": problems}, "all present, valid types",
                       "; ".join(parts) + ".")
    return _result("required_fields_present", "consistency", "hard", "pass",
                   {"missing": [], "type_problems": []}, "all present, valid types",
                   f"All {len(cols)} required columns present with expected types.")


def check_plausibility(df: pd.DataFrame, req: dict) -> dict:
    """10. Consistency: duplicate patients and follow-up before diagnosis (hard);
    physically impossible values in required numeric fields (soft)."""
    name, cl = "record_plausibility", "consistency"
    thr = {"duplicate_patient_ids": 0, "followup_before_diagnosis": 0,
           "ranges": {c: list(r) for c, r in PLAUSIBLE_RANGES.items()}}
    dup = int(df["patient_id"].dropna().duplicated().sum()) if "patient_id" in df else 0
    order = 0
    if _is_mortality(req) and {"diagnosis_date", "last_followup_date"} <= set(df.columns):
        start = pd.to_datetime(df["diagnosis_date"], errors="coerce")
        end = pd.to_datetime(df["last_followup_date"], errors="coerce")
        order = int((end < start).sum())
    out = {}
    for c in _required_columns(req):
        if c in PLAUSIBLE_RANGES and c in df.columns:
            lo, hi = PLAUSIBLE_RANGES[c]
            x = pd.to_numeric(df[c], errors="coerce")
            n = int(((x < lo) | (x > hi if hi is not None else False)).sum())
            if n:
                out[c] = n
    value = {"duplicate_patient_ids": dup, "followup_before_diagnosis": order, "out_of_range": out}
    parts = []
    if dup:
        parts.append(f"{dup} duplicate patient_id row(s)")
    if order:
        parts.append(f"{order} patient(s) with last_followup_date before diagnosis_date")
    parts += [f"{c}: {n} impossible value(s)" for c, n in out.items()]
    if not parts:
        return _result(name, cl, "soft", "pass", value, thr,
                       "No duplicate patients, date-order errors or impossible values found.")
    sev = "hard" if dup or order else "soft"
    return _result(name, cl, sev, "fail", value, thr, "; ".join(parts).capitalize() + ".")


def check_terms_mappable(req: dict) -> dict:
    """9. Consistency: every term in the question maps to a dataset column."""
    terms = list(req.get("unmappable_terms") or [])
    if terms:
        return _result("question_terms_mappable", "consistency", "hard", "unknown", terms, "none",
                       f"Question mentions {', '.join(terms)}, which no column in the dataset "
                       "represents; readiness for that part cannot be assessed.")
    return _result("question_terms_mappable", "consistency", "hard", "pass", [], "none",
                   "All terms in the question map to dataset columns.")


def check_question_parsed(error: str) -> dict:
    """Consistency: Claude answered, but the question could not be turned into valid
    requirements. (API/setup problems never reach the checks: see pipeline.py.)"""
    return _result("question_parsed", "consistency", "hard", "unknown", None, None,
                   f"The question could not be evaluated: {error}")


def check_missingness(df: pd.DataFrame, req: dict) -> dict:
    """2. Informativeness: missingness in required fields only."""
    cols = [c for c in _required_columns(req) if c in df.columns]
    thr = {"soft": MISSING_SOFT, "hard": MISSING_HARD}
    if not cols or df.empty:
        return _result("required_field_missingness", "informativeness", "soft", "unknown",
                       None, thr, "No required columns available to assess missingness.")
    frac = df[cols].isna().mean()
    worst = float(frac.max())
    over = frac[frac > MISSING_SOFT].sort_values(ascending=False)
    detail = {c: round(float(v), 4) for c, v in frac[frac > 0].items()}
    if worst > MISSING_HARD:
        sev, status = "hard", "fail"
    elif worst > MISSING_SOFT:
        sev, status = "soft", "fail"
    else:
        sev, status = "soft", "pass"
    if status == "fail":
        msg = ("Missing above 20%: " + ", ".join(f"{c} {v:.0%}" for c, v in over.items())
               + (" (above 40%: hard fail)." if sev == "hard" else "."))
    else:
        msg = f"Worst required-field missingness {worst:.0%} (limit 20%)."
    return _result("required_field_missingness", "informativeness", sev, status,
                   {"max": worst, "by_column": detail}, thr, msg)


def check_outcome(df: pd.DataFrame, req: dict) -> dict:
    """3. Informativeness: outcome present and derivable (hard)."""
    name, cl = "outcome_derivable", "informativeness"
    oc = req.get("outcome") or {}
    if req.get("task_type") == "description" and not req.get("outcome"):
        return _result(name, cl, "hard", "not_applicable", None, None,
                       "Not applicable: a descriptive question with no outcome to derive.")
    if not _is_mortality(req):
        return _result(name, cl, "hard", "unknown", None, None,
                       f"Outcome '{oc.get('column')}' is not a supported outcome; only death "
                       "derived from vital_status and dates is supported.")
    absent = [c for c in OUTCOME_SOURCES if c not in df.columns]
    if len(absent) == len(OUTCOME_SOURCES):
        return _result(name, cl, "hard", "fail", None, None,
                       "Unsupported schema: none of vital_status, diagnosis_date or "
                       "last_followup_date is present. This tool derives death outcomes only "
                       "from those columns; rename or map the file's columns to them.")
    if absent:
        return _result(name, cl, "hard", "fail", None, None,
                       f"Cannot derive the outcome: missing {', '.join(absent)}.")
    h = req.get("time_horizon_months")
    if oc.get("type") == "binary" and not h:
        return _result(name, cl, "hard", "unknown", None, None,
                       "Binary outcome needs a time horizon, and none was given.")
    if oc.get("type") == "time_to_event" and not h:
        deaths = int(df["vital_status"].astype("string").str.lower().eq("dead").sum())
        status = "pass" if deaths else "fail"
        return _result(name, cl, "hard", status, {"deaths": deaths}, ">= 1 event",
                       f"Time-to-event outcome derivable: {deaths} deaths.")
    o = outcome_summary(df, h)
    status = "pass" if o["events"] > 0 and o["non_events"] > 0 else "fail"
    msg = (f"Death within {h} months: {o['events']} events, {o['non_events']} non-events; "
           f"{o['excluded_unknown']} patients excluded (outcome unknown: alive with follow-up "
           f"under {h} months, or missing status/dates).")
    if status == "fail":
        msg = "Outcome has no variation. " + msg
    return _result(name, cl, "hard", status, o, ">= 1 event and >= 1 non-event", msg)


def check_censoring(df: pd.DataFrame, req: dict) -> dict:
    """4. Timeliness: share excluded for early censoring (soft fail > 15%)."""
    name, cl = "early_censoring", "timeliness"
    h = req.get("time_horizon_months")
    if not h:
        return _result(name, cl, "soft", "not_applicable", None, CENSORING_SOFT,
                       "Not applicable: no time horizon in the question.")
    if not _is_mortality(req) or any(c not in df.columns for c in OUTCOME_SOURCES) or df.empty:
        return _result(name, cl, "soft", "unknown", None, CENSORING_SOFT,
                       "Cannot assess censoring: outcome not derivable.")
    excl = int(derive_binary_outcome(df, h).isna().sum())
    frac = excl / len(df)
    status = "fail" if frac > CENSORING_SOFT else "pass"
    return _result(name, cl, "soft", status, frac, CENSORING_SOFT,
                   f"{excl} of {len(df)} patients ({frac:.1%}) have unknown {h}-month outcome "
                   f"(limit 15%).")


def check_subgroups(df: pd.DataFrame, req: dict) -> dict:
    """5. Representativeness: sex, age band and stage levels >= MIN_SUBGROUP."""
    name, cl = "subgroup_counts", "representativeness"
    filtered = {f["column"] for f in req.get("target_population") or []}
    counts, small, absent = {}, [], []
    for group, levels in SUBGROUP_LEVELS.items():
        src = "age" if group == "age_band" else group
        if src in filtered:
            continue
        if src not in df.columns:
            absent.append(src)
            continue
        s = df[src]
        if group == "age_band":
            s = pd.cut(pd.to_numeric(s, errors="coerce"), AGE_BINS, labels=AGE_LABELS)
        vc = s.astype("string").value_counts()
        for lvl in levels:
            n = int(vc.get(lvl, 0))
            counts[f"{src}={lvl}"] = n
            if n < MIN_SUBGROUP:
                small.append(f"{src}={lvl} ({n})")
    if small:
        return _result(name, cl, "soft", "fail", counts, MIN_SUBGROUP,
                       f"Subgroups under {MIN_SUBGROUP} patients: {', '.join(small)}.")
    if absent:
        return _result(name, cl, "soft", "unknown", counts, MIN_SUBGROUP,
                       f"Cannot assess representativeness: missing {', '.join(absent)}.")
    return _result(name, cl, "soft", "pass", counts, MIN_SUBGROUP,
                   f"All sex, age-band and stage subgroups have >= {MIN_SUBGROUP} patients.")


def check_leakage(df: pd.DataFrame, req: dict) -> dict:
    """6. Measurement process: columns that encode the outcome (hard)."""
    name, cl = "label_leakage", "measurement process"
    if req.get("task_type") != "prediction" or not _is_mortality(req):
        return _result(name, cl, "hard", "not_applicable", None, "none",
                       "Not applicable: leakage is checked for death-prediction tasks.")
    leaks = [c for c in df.columns
             if c not in OUTCOME_SOURCES and LEAKAGE_NAME.search(c.lower())]
    if leaks:
        return _result(name, cl, "hard", "fail", leaks, "none",
                       f"{', '.join(leaks)} directly encodes the outcome (death) and would "
                       "leak the label into a survival prediction model.")
    return _result(name, cl, "hard", "pass", [], "none", "No outcome-encoding columns found.")


def check_sample_size(df: pd.DataFrame, req: dict, anticipated_r2cs: float | None = None,
                      n_params_override: int | None = None) -> dict:
    """7. Sample size: Riley criteria (soft) with EPV >= 10 for comparison."""
    name, cl = "sample_size", "representativeness"
    oc = req.get("outcome") or {}
    if req.get("task_type") != "prediction" or oc.get("type") != "binary":
        return _result(name, cl, "soft", "not_applicable", None, None,
                       "Not applicable: Riley criteria are computed for binary-outcome "
                       "prediction tasks only.")
    p = n_params_override or req.get("n_predictor_parameters")
    if not p:
        return _result(name, cl, "soft", "unknown", None, None,
                       "Number of candidate predictor parameters unknown.")
    h = req.get("time_horizon_months")
    if not _is_mortality(req) or not h or any(c not in df.columns for c in OUTCOME_SOURCES):
        return _result(name, cl, "soft", "unknown", None, None,
                       "Cannot compute sample size: outcome not derivable.")
    o = outcome_summary(df, h)
    n = o["events"] + o["non_events"]
    if n == 0:
        return _result(name, cl, "soft", "unknown", None, None, "No patients with known outcome.")
    phi = o["events"] / n
    try:
        r = ss.riley_binary(int(p), phi, anticipated_r2cs)
    except ValueError as e:
        return _result(name, cl, "soft", "unknown", None, None, f"Cannot compute Riley criteria: {e}")
    e_pp = ss.epv(o["events"], int(p))
    value = {**r, "n_available": n, "events": o["events"],
             "epv": e_pp, "epv_pass": e_pp >= EPV_MIN, "epv_threshold": EPV_MIN}
    status = "pass" if n >= r["n_required"] else "fail"
    msg = (f"Riley: needs n >= {r['n_required']} ({r['events_required']} events; binding "
           f"criterion {r['binding_criterion']}) for {p} parameters; available n = {n} "
           f"({o['events']} events). EPV = {e_pp:.1f} "
           f"({'meets' if value['epv_pass'] else 'below'} the EPV >= 10 rule). {r['r2cs_note']}")
    return _result(name, cl, "soft", status, value, r["n_required"], msg)


def check_identifiers(df: pd.DataFrame) -> dict:
    """8. Safety: columns resembling real patient identifiers (hard)."""
    name, cl = "identifier_columns", "safety"
    hits = []
    for c in df.columns:
        lc = c.lower()
        for label, pat in IDENTIFIER_NAMES.items():
            if pat.search(lc):
                hits.append(f"{c} (name looks like {label})")
                break
        else:
            s = df[c]
            if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
                continue
            x = s.dropna().astype(str).str.strip()
            if ID_NAME.search(lc):
                if not x.str.fullmatch(SYNTHETIC_ID).all():
                    hits.append(f"{c} (ID values not in synthetic SYN-##### format)")
                continue
            for label, pat in IDENTIFIER_VALUES.items():
                if x.str.fullmatch(pat).any():
                    hits.append(f"{c} (values look like {label})")
                    break
    if hits:
        return _result(name, cl, "hard", "fail", hits, "none",
                       "Possible real identifiers: " + "; ".join(hits)
                       + ". Only synthetic, de-identified data may be used.")
    return _result(name, cl, "hard", "pass", [], "none", "No identifier-like columns found.")


def run_checks(df: pd.DataFrame, req: dict, anticipated_r2cs: float | None = None,
               n_params_override: int | None = None) -> list[dict]:
    """Run all checks on the target population. Identifier screening uses the
    full file."""
    cohort = apply_population(df, req.get("target_population"))
    return [
        check_required_fields(cohort, req),
        check_plausibility(cohort, req),
        check_terms_mappable(req),
        check_missingness(cohort, req),
        check_outcome(cohort, req),
        check_censoring(cohort, req),
        check_subgroups(cohort, req),
        check_leakage(cohort, req),
        check_sample_size(cohort, req, anticipated_r2cs, n_params_override),
        check_identifiers(df),
    ]

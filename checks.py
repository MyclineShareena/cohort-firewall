"""Deterministic dataset checks. (Full check suite: Milestone 2.)

Currently contains only the outcome derivation, which the synthetic data
generator also uses for its summary.
"""
from __future__ import annotations

import pandas as pd

DAYS_PER_MONTH = 30.4375  # 365.25 / 12


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

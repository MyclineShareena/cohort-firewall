"""Synthetic lung cancer cohort generator. SYNTHETIC DATA ONLY.

Run from the project root:  .venv/bin/python -m data.generate

Event counts are set exactly instead of left to chance. Each file states
how many patients died within 24 months of diagnosis and how many were
censored early (alive, follow-up < 24 months). Event patients are drawn
with probability proportional to a covariate risk score, so the outcome
still carries signal.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from checks import outcome_summary  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent
SEED = 42
HORIZON_MONTHS = 24
HORIZON_DAYS = 730          # events: follow-up <= 730 days
SAFE_NON_EVENT_DAYS = 732   # non-events: follow-up >= 732 days (> 24 * 30.4375)
DX_START, DX_END = pd.Timestamp("2015-01-01"), pd.Timestamp("2021-12-31")
DATA_CUTOFF = pd.Timestamp("2024-12-31")  # at least 3 years of potential follow-up
# 24-month death rates for early stages in clean.csv (and leakage.csv, derived from it).
STAGE_I_II_RATES = {"I": 0.07, "II": 0.15}


def make_cohort(n: int, events_24m: int, censored_early: int, seed: int,
                stage_event_rates: dict[str, float] | None = None) -> pd.DataFrame:
    """Build a synthetic cohort with exact 24-month event and censoring counts.

    stage_event_rates (optional), e.g. {"I": 0.07, "II": 0.15}: fixes the
    24-month death rate among outcome-known patients in those stages; the
    remaining events are drawn from the other stages by risk score.
    """
    assert events_24m + censored_early <= n
    rng = np.random.default_rng(seed)

    age = np.clip(rng.normal(68, 9, n), 35, 90).round().astype(int)
    sex = rng.choice(["M", "F"], n, p=[0.55, 0.45])
    stage = rng.choice(["I", "II", "III", "IV"], n, p=[0.20, 0.12, 0.25, 0.43])
    histology = rng.choice(
        ["adenocarcinoma", "squamous", "small_cell", "other"], n, p=[0.50, 0.25, 0.13, 0.12]
    )
    ecog = rng.choice([0, 1, 2, 3], n, p=[0.30, 0.40, 0.20, 0.10])

    treatment = np.empty(n, dtype=object)
    early = np.isin(stage, ["I", "II"])
    treatment[early] = rng.choice(["surgery", "surgery_adjuvant_chemo"], early.sum(), p=[0.6, 0.4])
    s3 = stage == "III"
    treatment[s3] = rng.choice(["chemoradiation", "chemoradiation_immunotherapy"], s3.sum())
    s4 = stage == "IV"
    treatment[s4] = rng.choice(
        ["chemotherapy", "immunotherapy", "targeted_therapy", "best_supportive_care"],
        s4.sum(), p=[0.35, 0.35, 0.18, 0.12],
    )

    hemoglobin = rng.normal(12.8, 1.6, n).round(1)
    albumin = np.clip(rng.normal(3.8, 0.5, n), 1.8, 5.2).round(1)
    ldh = rng.lognormal(np.log(220), 0.35, n).round().astype(int)
    wbc = np.clip(rng.normal(8.0, 2.5, n), 2.0, 25.0).round(1)

    # Risk score -> which patients become 24-month events.
    stage_effect = pd.Series(stage).map({"I": 0.0, "II": 0.5, "III": 1.1, "IV": 1.9}).to_numpy()
    lp = (stage_effect + 0.03 * (age - 68) + 0.45 * ecog
          - 0.8 * (albumin - 3.8) + 0.5 * np.log(ldh / 220) + rng.normal(0, 0.3, n))
    p = np.exp(lp) / np.exp(lp).sum()
    idx = np.arange(n)
    if stage_event_rates is None:
        event_idx = rng.choice(idx, events_24m, replace=False, p=p)
        rest = np.setdiff1d(idx, event_idx)
        cens_idx = rng.choice(rest, censored_early, replace=False)
    else:
        # Censor first, so per-stage rates apply to outcome-known patients.
        cens_idx = rng.choice(idx, censored_early, replace=False)
        known = np.setdiff1d(idx, cens_idx)
        picked = []
        for stg, rate in stage_event_rates.items():
            pool = known[stage[known] == stg]
            k = int(round(rate * len(pool)))
            picked.append(rng.choice(pool, k, replace=False, p=p[pool] / p[pool].sum()))
        pool = known[~np.isin(stage[known], list(stage_event_rates))]
        k = events_24m - sum(len(x) for x in picked)
        assert 0 <= k <= len(pool)
        picked.append(rng.choice(pool, k, replace=False, p=p[pool] / p[pool].sum()))
        event_idx = np.concatenate(picked)
    rest = np.setdiff1d(idx, event_idx)
    long_idx = np.setdiff1d(rest, cens_idx)

    dx_span = (DX_END - DX_START).days
    dx = DX_START + pd.to_timedelta(rng.integers(0, dx_span + 1, n), unit="D")
    max_fu = (DATA_CUTOFF - dx).days.to_numpy()  # >= 1096 days

    fu = np.zeros(n, dtype=int)
    status = np.array(["alive"] * n, dtype=object)
    # Events: died within 24 months.
    fu[event_idx] = rng.integers(15, HORIZON_DAYS + 1, len(event_idx))
    status[event_idx] = "dead"
    # Censored early: alive, lost to follow-up before 24 months.
    fu[cens_idx] = rng.integers(30, 700, len(cens_idx))
    # Long follow-up: alive at last contact, or died after 24 months.
    fu[long_idx] = [rng.integers(SAFE_NON_EVENT_DAYS, m + 1) for m in max_fu[long_idx]]
    late_death = rng.random(len(long_idx)) < 0.25
    status[long_idx[late_death]] = "dead"

    df = pd.DataFrame({
        "patient_id": [f"SYN-{i:05d}" for i in range(1, n + 1)],
        "age": age, "sex": sex, "stage": stage, "histology": histology,
        "treatment": treatment, "ecog": ecog,
        "diagnosis_date": dx.strftime("%Y-%m-%d"),
        "last_followup_date": (dx + pd.to_timedelta(fu, unit="D")).strftime("%Y-%m-%d"),
        "vital_status": status,
        "hemoglobin": hemoglobin, "albumin": albumin, "ldh": ldh, "wbc": wbc,
    })
    return df


def build_all() -> dict[str, pd.DataFrame]:
    clean = make_cohort(n=2000, events_24m=560, censored_early=60, seed=SEED,
                        stage_event_rates=STAGE_I_II_RATES)

    messy = make_cohort(n=600, events_24m=40, censored_early=50, seed=SEED + 1)
    rng = np.random.default_rng(SEED + 101)
    messy.loc[rng.random(len(messy)) < 0.30, "stage"] = np.nan
    messy.loc[rng.random(len(messy)) < 0.10, "ldh"] = np.nan

    broken = clean.drop(columns=["vital_status"])

    leakage = clean.copy()
    dead = leakage["vital_status"] == "dead"
    leakage["date_of_death"] = leakage["last_followup_date"].where(dead)

    # n=1000, 100 events, 0 early censoring -> EPV 10 (P=10) passes; Riley (i) needs ~1,200.
    epv_vs_riley = make_cohort(n=1000, events_24m=100, censored_early=0, seed=SEED + 2)

    # clean + a fake SSN-like column, for the identifier_columns check (Milestone 5 scenarios).
    identifiers = clean.copy()
    rng = np.random.default_rng(SEED + 3)
    identifiers["ssn"] = [f"{a:03d}-{b:02d}-{c:04d}" for a, b, c in zip(
        rng.integers(100, 999, len(identifiers)), rng.integers(10, 99, len(identifiers)),
        rng.integers(1000, 9999, len(identifiers)))]

    # clean + one duplicated patient row, for the record_plausibility check (Milestone 5).
    duplicate_patient = pd.concat([clean, clean.iloc[[0]]], ignore_index=True)

    return {"clean": clean, "messy": messy, "broken": broken,
            "leakage": leakage, "epv_vs_riley": epv_vs_riley, "identifiers": identifiers,
            "duplicate_patient": duplicate_patient}


EXPECTED = {"clean": "Ready", "messy": "Conditional", "broken": "Blocked",
            "leakage": "Blocked", "epv_vs_riley": "Conditional (EPV passes, Riley fails)",
            "identifiers": "Blocked (identifier_columns)",
            "duplicate_patient": "Blocked (record_plausibility)"}


def main() -> None:
    files = build_all()
    rows = []
    for name, df in files.items():
        df.to_csv(DATA_DIR / f"{name}.csv", index=False)
        if "vital_status" in df.columns:
            o = outcome_summary(df, HORIZON_MONTHS)
            ev, exc = o["events"], o["excluded_unknown"]
            rate = f"{ev / (o['events'] + o['non_events']):.1%}"
        else:
            ev = exc = rate = "n/a (no vital_status)"
        miss = df.isna().mean()
        miss = ", ".join(f"{c} {v:.0%}" for c, v in miss[miss > 0].items()) or "none"
        rows.append({"file": f"{name}.csv", "rows": len(df), "cols": df.shape[1],
                     "events_24m": ev, "excluded_censored": exc, "event_rate": rate,
                     "missingness": miss, "expected": EXPECTED[name]})
    with pd.option_context("display.width", 200, "display.max_columns", None,
                           "display.max_colwidth", 45):
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()

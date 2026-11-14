import pandas as pd
import pytest

from checks import derive_binary_outcome, outcome_summary
from data.generate import build_all


@pytest.fixture(scope="module")
def files():
    return build_all()


def test_derive_binary_outcome_rules():
    df = pd.DataFrame({
        "diagnosis_date": ["2020-01-01"] * 4,
        "last_followup_date": ["2020-06-01", "2023-01-01", "2020-06-01", "2023-01-01"],
        "vital_status": ["dead", "dead", "alive", "alive"],
    })
    y = derive_binary_outcome(df, 24)
    assert y.tolist()[:2] == [1, 0]
    assert pd.isna(y[2])          # alive, short follow-up -> unknown
    assert y[3] == 0


def test_reproducible():
    a, b = build_all(), build_all()
    for k in a:
        pd.testing.assert_frame_equal(a[k], b[k])


def test_epv_vs_riley_exact_counts(files):
    o = outcome_summary(files["epv_vs_riley"], 24)
    assert len(files["epv_vs_riley"]) == 1000
    assert o["events"] == 100
    assert o["excluded_unknown"] == 0


def test_clean_counts(files):
    o = outcome_summary(files["clean"], 24)
    assert (o["events"], o["excluded_unknown"]) == (560, 60)
    assert files["clean"].isna().sum().sum() == 0


def test_messy_stage_missing(files):
    assert 0.25 < files["messy"]["stage"].isna().mean() < 0.35


def test_broken_and_leakage(files):
    assert "vital_status" not in files["broken"].columns
    lk = files["leakage"]
    assert lk["date_of_death"].notna().eq(lk["vital_status"].eq("dead")).all()


def test_ids_are_synthetic(files):
    for df in files.values():
        assert df["patient_id"].str.fullmatch(r"SYN-\d{5}").all()


def test_clean_stage_i_ii_rates_differ(files):
    df = files["clean"]
    y = derive_binary_outcome(df, 24)
    rate = df.assign(y=y).dropna(subset=["y"]).groupby("stage")["y"].mean()
    assert abs(rate["I"] - 0.07) < 0.01
    assert abs(rate["II"] - 0.15) < 0.01
    assert rate["I"] < rate["II"] < rate["III"] < rate["IV"]

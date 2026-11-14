import math

import pytest

from sample_size import epv, max_r2cs, riley_binary


def test_worked_example_riley():
    # Prevalence 0.174, 24 candidate parameters, anticipated R2cs 0.288.
    # Expected values to be verified against Riley et al. (2018 Part II / BMJ 2020).
    r = riley_binary(n_params=24, phi=0.174, r2cs=0.288)
    assert r["criterion_i"] == 623
    assert r["criterion_ii"] == 662
    assert r["criterion_iii"] == 221
    assert r["n_required"] == 662
    assert r["events_required"] == math.ceil(662 * 0.174)
    assert r["r2cs_source"] == "user"


def test_max_r2cs():
    # 1 - (phi^phi (1-phi)^(1-phi))^2; for phi = 0.5 this is 0.75.
    assert max_r2cs(0.5) == pytest.approx(0.75)
    assert max_r2cs(0.174) == pytest.approx(0.6032, abs=1e-3)


def test_default_r2cs_is_15pct_of_max():
    r = riley_binary(n_params=10, phi=0.10)
    assert r["r2cs"] == pytest.approx(0.15 * max_r2cs(0.10))
    assert r["r2cs_source"] == "default"
    assert "15%" in r["r2cs_note"]
    # epv_vs_riley design: about 1,200 needed for 10 parameters at 10% prevalence.
    assert 1150 < r["n_required"] < 1250
    assert r["n_required"] == r["criterion_i"]


def test_invalid_r2cs():
    with pytest.raises(ValueError):
        riley_binary(n_params=10, phi=0.1, r2cs=0.9)  # above max R2cs
    with pytest.raises(ValueError):
        riley_binary(n_params=10, phi=0.0)


def test_epv():
    assert epv(100, 10) == 10
    assert epv(40, 11) == pytest.approx(40 / 11)

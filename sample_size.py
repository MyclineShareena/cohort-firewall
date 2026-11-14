"""Riley et al. sample size criteria for a binary-outcome prediction model.

Riley et al., BMJ 2020 (doi 10.1136/bmj.m441); binary worked examples in
Riley et al., Stat Med 2018 Part II (doi 10.1002/sim.7992).

  (i)   shrinkage: expected uniform shrinkage factor S >= 0.9
        n = P / ((S - 1) * ln(1 - R2cs / S))
  (ii)  small optimism: apparent minus adjusted Nagelkerke R2 <= delta (0.05)
        S_VH = R2cs / (R2cs + delta * maxR2cs); same formula with S = S_VH
  (iii) precise overall risk: margin of error 0.05 around the outcome proportion
        n = (1.96 / 0.05)^2 * phi * (1 - phi)
"""
from __future__ import annotations

import math

SHRINKAGE = 0.9
DELTA = 0.05
MARGIN = 0.05
Z = 1.96
DEFAULT_R2_FRACTION = 0.15


def max_r2cs(phi: float) -> float:
    """Maximum Cox-Snell R2 for outcome proportion phi: 1 - (phi^phi (1-phi)^(1-phi))^2."""
    if not 0 < phi < 1:
        raise ValueError(f"outcome proportion must be in (0, 1), got {phi}")
    ln_l0_per_n = phi * math.log(phi) + (1 - phi) * math.log(1 - phi)
    return 1 - math.exp(2 * ln_l0_per_n)


def ceil_count(x: float) -> int:
    """Round a required count up, ignoring float noise (0.1 * 1200 -> 120, not 121)."""
    return math.ceil(round(x, 9))


def _n_for_shrinkage(n_params: int, r2cs: float, s: float) -> float:
    return n_params / ((s - 1) * math.log(1 - r2cs / s))


def riley_binary(n_params: int, phi: float, r2cs: float | None = None) -> dict:
    """Minimum sample size (criteria i-iii) for a binary-outcome model."""
    if n_params < 1:
        raise ValueError("need at least one predictor parameter")
    mx = max_r2cs(phi)
    if r2cs is None:
        r2cs, source = DEFAULT_R2_FRACTION * mx, "default"
        note = (f"Anticipated R²cs not given; using 15% of max R²cs "
                f"({mx:.3f}) = {r2cs:.4f}.")
    else:
        source = "user"
        note = f"Anticipated R²cs = {r2cs:.4f} (user input); max R²cs = {mx:.3f}."
    if not 0 < r2cs < mx:
        raise ValueError(f"anticipated R²cs must be in (0, {mx:.3f}), got {r2cs}")

    s_vh = r2cs / (r2cs + DELTA * mx)
    n_i = math.ceil(_n_for_shrinkage(n_params, r2cs, SHRINKAGE))
    n_ii = math.ceil(_n_for_shrinkage(n_params, r2cs, s_vh))
    n_iii = math.ceil((Z / MARGIN) ** 2 * phi * (1 - phi))
    n = max(n_i, n_ii, n_iii)
    return {
        "n_params": n_params, "phi": phi, "r2cs": r2cs, "r2cs_source": source,
        "r2cs_note": note, "max_r2cs": mx, "s_vh": s_vh,
        "criterion_i": n_i, "criterion_ii": n_ii, "criterion_iii": n_iii,
        "n_required": n, "events_required": ceil_count(n * phi),
        "binding_criterion": {n_i: "i", n_ii: "ii", n_iii: "iii"}[n],
    }


def epv(events: int, n_params: int) -> float:
    """Events per candidate predictor parameter (classic rule of thumb: >= 10)."""
    return events / n_params

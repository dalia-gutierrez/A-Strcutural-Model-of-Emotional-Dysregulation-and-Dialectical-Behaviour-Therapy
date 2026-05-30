"""SMM / indirect inference estimation.

First milestone: calibrate omega so that the simulated lifetime suicide
rate matches the empirical 6% (Lak et al. 2025). All other parameters
held at their default (or supplied) values.

Once EMA microdata is available, expand the moment vector to include:
    - mean and variance of x_t conditional on Z bins
    - autocorrelation of x_t conditional on (Z, xi) bins
    - mean NSSI episodes per year
    - mean skills practice per year
    - income elasticity to Z
and jointly estimate the full parameter vector.
"""
from __future__ import annotations

from dataclasses import replace
import numpy as np
from scipy.optimize import brentq

from .model import Params
from .solver import solve_hjb
from .solver_implicit import solve_hjb_implicit
from .simulator import simulate


# Empirical moments (from moments.csv)
EMPIRICAL_MOMENTS = {
    "lifetime_suicide_rate_bpd": 0.06,
    "unemployment_rate_bpd_followup": 0.45,
    "ssdi_rate_bpd_followup": 0.44,
}


def simulated_moments(p: Params, n_sim: int = 3000, seed: int = 0,
                      constrained: bool = True,
                      x0: float = -1.0, Z0: float = 0.8,
                      x0_sd: float = 0.8, Z0_sd: float = 0.7) -> dict:
    """Solve and simulate, return the model's moment vector.

    Default = untreated BPD: constrained agent (no skills) starting from a
    dysregulated initial state (x0 = -1.0, Z0 = 0.5), which is what the
    empirical 6% lifetime suicide rate refers to.
    """
    sol = solve_hjb_implicit(p, constrained=constrained)
    sim = simulate(p, sol, n_sim=n_sim, seed=seed,
                   x0=x0, Z0=Z0, x0_sd=x0_sd, Z0_sd=Z0_sd)
    return {
        "lifetime_suicide_rate_bpd": sim["lifetime_suicide_rate"],
        "mean_NSSI_per_year": sim["mean_NSSI_per_year"],
        "mean_skills_per_year": sim["mean_skills_per_year"],
    }


def smm_objective(theta: np.ndarray, p_base: Params, free_names: list[str],
                  target: dict, weight: dict | None = None, n_sim: int = 2000,
                  seed: int = 0) -> float:
    """Generic SMM criterion.

    theta : free parameter vector (in the order of free_names).
    p_base : a Params with fixed values for the non-free params.
    free_names : names of fields in Params being estimated.
    target : dict of empirical moments to match.
    weight : optional dict of moment weights (defaults to 1/value^2).
    """
    p = replace(p_base, **{name: float(val) for name, val in zip(free_names, theta)})
    m_sim = simulated_moments(p, n_sim=n_sim, seed=seed)
    loss = 0.0
    for k, v_emp in target.items():
        if k not in m_sim:
            continue
        w = (weight or {}).get(k, 1.0 / max(abs(v_emp), 1e-3) ** 2)
        loss += w * (m_sim[k] - v_emp) ** 2
    return loss


def calibrate_omega(p_base: Params, target_rate: float = 0.06,
                    n_sim: int = 3000, omega_bounds: tuple = (-20.0, -0.5),
                    seed: int = 0, tol: float = 5e-3) -> dict:
    """Find omega such that simulated lifetime suicide rate equals target_rate.

    Uses Brent's method on the monotone map omega -> simulated rate
    (more negative omega => lower simulated suicide rate, holding other params).
    """

    def gap(omega_val: float) -> float:
        p = replace(p_base, omega=float(omega_val))
        m = simulated_moments(p, n_sim=n_sim, seed=seed)
        return m["lifetime_suicide_rate_bpd"] - target_rate

    # Ensure bracket has opposite signs (otherwise widen)
    f_lo = gap(omega_bounds[0])
    f_hi = gap(omega_bounds[1])
    if f_lo * f_hi > 0:
        return {"status": "no_bracket", "f_lo": f_lo, "f_hi": f_hi,
                "omega_bounds": omega_bounds}
    omega_hat = brentq(gap, omega_bounds[0], omega_bounds[1], xtol=tol)
    p_hat = replace(p_base, omega=float(omega_hat))
    m = simulated_moments(p_hat, n_sim=n_sim, seed=seed)
    return {
        "status": "ok",
        "omega_hat": float(omega_hat),
        "simulated_rate": m["lifetime_suicide_rate_bpd"],
        "target_rate": target_rate,
    }

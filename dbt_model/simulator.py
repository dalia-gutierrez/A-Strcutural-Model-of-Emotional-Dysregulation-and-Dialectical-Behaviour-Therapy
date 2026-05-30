"""Monte Carlo simulator for the DBT model.

Given a converged solution (V, z_policy, xi_policy) on a grid, simulate
N trajectories of (x_t, Z_t, z_t, xi_t) over [0, Tbar] using Euler-Maruyama.

The agent commits suicide the first time the state enters the stopping
region {V = omega}.
"""
from __future__ import annotations

import numpy as np

from .model import Params, phi, mu, sigma


def _interp2d(x_grid: np.ndarray, Z_grid: np.ndarray,
              field: np.ndarray, x_vals: np.ndarray, Z_vals: np.ndarray) -> np.ndarray:
    """Bilinear interpolation of a 2D field at (x_vals, Z_vals)."""
    Nx, NZ = field.shape
    x_lo = x_grid[0]; x_hi = x_grid[-1]
    Z_lo = Z_grid[0]; Z_hi = Z_grid[-1]
    dx = (x_hi - x_lo) / (Nx - 1)
    dZ = (Z_hi - Z_lo) / (NZ - 1)

    xi = np.clip((x_vals - x_lo) / dx, 0, Nx - 1 - 1e-9)
    zi = np.clip((Z_vals - Z_lo) / dZ, 0, NZ - 1 - 1e-9)
    i = xi.astype(int); j = zi.astype(int)
    fx = xi - i; fz = zi - j

    v00 = field[i,     j    ]
    v10 = field[i + 1, j    ]
    v01 = field[i,     j + 1]
    v11 = field[i + 1, j + 1]
    return (v00 * (1 - fx) * (1 - fz) +
            v10 * fx * (1 - fz) +
            v01 * (1 - fx) * fz +
            v11 * fx * fz)


def simulate(
    p: Params,
    solution: dict,
    n_sim: int = 5000,
    dt: float = 0.05,
    x0: float = 0.0,
    Z0: float = 0.0,
    x0_sd: float = 0.0,
    Z0_sd: float = 0.0,
    seed: int = 0,
) -> dict:
    """Simulate `n_sim` trajectories under the optimal policy.

    Returns
    -------
    dict with:
        suicide_times : np.ndarray (n_sim,)  -- NaN if no suicide
        lifetime_suicide_rate : float
        mean_NSSI_per_year : float
        mean_skills_per_year : float
        x_paths, Z_paths, z_paths, xi_paths : (n_sim, n_steps+1)
                                              (trimmed to before stop)
    """
    rng = np.random.default_rng(seed)
    n_steps = int(np.ceil(p.Tbar / dt))
    sqrt_dt = np.sqrt(dt)

    x_grid = solution["x_grid"]
    Z_grid = solution["Z_grid"]
    z_pol  = solution["z_policy"]
    xi_pol = solution["xi_policy"]
    stop_field = solution["stop"].astype(float)

    x_path  = np.zeros((n_sim, n_steps + 1))
    Z_path  = np.zeros((n_sim, n_steps + 1))
    z_path  = np.zeros((n_sim, n_steps + 1))
    xi_path = np.zeros((n_sim, n_steps + 1))
    # Stochastic initial conditions when x0_sd or Z0_sd > 0.
    if x0_sd > 0:
        x_path[:, 0] = x0 + x0_sd * rng.standard_normal(n_sim)
    else:
        x_path[:, 0] = x0
    if Z0_sd > 0:
        Z_path[:, 0] = np.maximum(Z0 + Z0_sd * rng.standard_normal(n_sim), 0.0)
    else:
        Z_path[:, 0] = Z0

    alive = np.ones(n_sim, dtype=bool)
    suicide_time = np.full(n_sim, np.nan)

    for k in range(n_steps):
        xk = x_path[:, k]
        Zk = Z_path[:, k]

        # Read optimal controls off the grid
        zk  = _interp2d(x_grid, Z_grid, z_pol,  xk, Zk)
        xik = _interp2d(x_grid, Z_grid, xi_pol, xk, Zk)
        zk  = np.maximum(zk, 0.0)
        xik = np.maximum(xik, 0.0)
        z_path[:, k]  = zk
        xi_path[:, k] = xik

        # Stopping check: nearest-neighbour read of stop_field
        in_stop = _interp2d(x_grid, Z_grid, stop_field, xk, Zk) > 0.5
        newly_dead = alive & in_stop
        if newly_dead.any():
            suicide_time[newly_dead] = k * dt
            alive = alive & (~newly_dead)

        # SDE step
        phi_v = phi(Zk, xik, p)
        mu_v  = mu(Zk, xik, p) - p.delta_mu * zk
        sig_v = sigma(Zk, xik, p)
        eps = rng.standard_normal(n_sim)
        dx = phi_v * (mu_v - xk) * dt + sig_v * sqrt_dt * eps
        dZ = (p.kappa * zk - p.rho * Zk) * dt

        x_next = xk + dx
        Z_next = np.maximum(Zk + dZ, 0.0)            # Z >= 0

        # Freeze dead agents
        x_next = np.where(alive, x_next, xk)
        Z_next = np.where(alive, Z_next, Zk)
        x_path[:, k + 1] = x_next
        Z_path[:, k + 1] = Z_next

    lifetime_suicide = ~np.isnan(suicide_time)
    lifetime_suicide_rate = lifetime_suicide.mean()

    # NSSI and skills usage: averaged over alive periods
    # treat z > threshold as an "episode" at that step
    nssi_episodes = (z_path > 0.05) & (np.cumsum(z_path, axis=1) >= 0)  # crude
    mean_NSSI_per_year = nssi_episodes.sum(axis=1).mean() / p.Tbar
    mean_skills_per_year = (xi_path > 0.05).sum(axis=1).mean() / p.Tbar

    return {
        "suicide_times": suicide_time,
        "lifetime_suicide_rate": float(lifetime_suicide_rate),
        "mean_NSSI_per_year": float(mean_NSSI_per_year),
        "mean_skills_per_year": float(mean_skills_per_year),
        "x_paths": x_path,
        "Z_paths": Z_path,
        "z_paths": z_path,
        "xi_paths": xi_path,
        "dt": dt,
    }

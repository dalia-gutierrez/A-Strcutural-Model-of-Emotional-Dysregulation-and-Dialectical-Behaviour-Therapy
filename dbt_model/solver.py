"""HJB solver for the DBT model.

Solves the time-homogeneous (infinite-horizon) approximation:

    rV = sup_{z>=0, xi>=0} { u(c + x + z) + (z - rho*Z) V_Z
                             + phi(Z, xi)(mu(Z, xi) - x) V_x
                             + 0.5 sigma^2(Z, xi) V_xx }   on continuation region
    V >= omega                                              everywhere

where c = cbar - chi*Z. The agent commits suicide on {V = omega}.

We use:
    - Closed-form FOC for z (log utility): z* = max(0, -1/V_Z - (c + x))
    - Grid search over xi in [0, xi_max]
    - Central differences for V_x, V_xx, V_Z (interior); one-sided at the boundary
    - Pseudo-time-stepping (value function iteration) with obstacle projection

This is a first-pass solver: clear, dependency-light, accurate enough for SMM.
A faster solver (implicit time-stepping, semi-Lagrangian schemes) can replace
this later without changing the interface.
"""
from __future__ import annotations

import numpy as np

from .model import Params, phi, mu, sigma


def _derivatives(V: np.ndarray, dx: float, dZ: float):
    """Compute V_x, V_xx, V_Z on a 2D grid with one-sided differences at boundaries."""
    Vx = np.zeros_like(V)
    Vxx = np.zeros_like(V)
    VZ = np.zeros_like(V)

    Vx[1:-1, :] = (V[2:, :] - V[:-2, :]) / (2 * dx)
    Vx[0, :]    = (V[1, :]  - V[0, :])  / dx
    Vx[-1, :]   = (V[-1, :] - V[-2, :]) / dx

    Vxx[1:-1, :] = (V[2:, :] - 2 * V[1:-1, :] + V[:-2, :]) / dx**2
    Vxx[0, :]    = Vxx[1, :]
    Vxx[-1, :]   = Vxx[-2, :]

    VZ[:, 1:-1] = (V[:, 2:] - V[:, :-2]) / (2 * dZ)
    VZ[:, 0]    = (V[:, 1]  - V[:, 0])  / dZ
    VZ[:, -1]   = (V[:, -1] - V[:, -2]) / dZ

    return Vx, Vxx, VZ


def solve_hjb(
    p: Params,
    constrained: bool = False,
    xi_max: float = 3.0,
    n_xi: int = 7,
    max_iter: int = 5000,
    tol: float = 1e-5,
    verbose: bool = False,
) -> dict:
    """Solve the time-homogeneous HJB on the grid in `p`.

    Parameters
    ----------
    p : Params
    constrained : if True, force xi = 0 (no DBT skills available).
    xi_max, n_xi : grid for xi search.
    max_iter, tol : value-iteration controls.

    Returns
    -------
    dict with:
        V          : value function, shape (Nx, NZ)
        z_policy   : optimal z* on the grid
        xi_policy  : optimal xi* on the grid
        stop       : boolean mask of the suicide region
        x_grid, Z_grid
        iterations, residual
    """
    x = np.linspace(p.x_min, p.x_max, p.Nx)
    Z = np.linspace(0.0, p.Z_max, p.NZ)
    dx = x[1] - x[0]
    dZ = Z[1] - Z[0]
    X, ZZ = np.meshgrid(x, Z, indexing="ij")  # (Nx, NZ)
    c_grid = p.cbar - p.chi * ZZ

    xi_grid = np.array([0.0]) if constrained else np.linspace(0.0, xi_max, n_xi)

    # Initialize V slightly above the suicide payoff
    V = np.full((p.Nx, p.NZ), max(p.omega, -1.0) + 1.0, dtype=float)

    # Pseudo-time step: respect CFL on diffusion term
    sigma_max = float(sigma(p.Z_max, 0.0, p))
    dt = 0.5 * dx**2 / (sigma_max**2 + 1e-9)
    dt = min(dt, 0.2 / max(p.r, 1e-3))

    for it in range(max_iter):
        Vx, Vxx, VZ = _derivatives(V, dx, dZ)

        # Closed-form optimal z* given V_Z (log utility)
        VZ_neg = np.minimum(VZ, -1e-8)              # ensure negative for division
        z_star = -1.0 / VZ_neg - (c_grid + X)
        z_star = np.clip(z_star, 0.0, 1.0)          # NSSI episodes per "year"; bounded for realism

        # Maximize over xi on the grid
        best_H = np.full_like(V, -np.inf)
        best_xi = np.zeros_like(V)
        flow_z = np.log(np.maximum(c_grid + X + z_star, 1e-9))
        Z_drift_term = (z_star - p.rho * ZZ) * VZ

        for xi_val in xi_grid:
            phi_v = phi(ZZ, xi_val, p)
            mu_v  = mu(ZZ, xi_val, p)
            sig_v = sigma(ZZ, xi_val, p)
            drift_x = phi_v * (mu_v - X)
            H = flow_z + Z_drift_term + drift_x * Vx + 0.5 * sig_v**2 * Vxx
            mask = H > best_H
            best_H = np.where(mask, H, best_H)
            best_xi = np.where(mask, xi_val, best_xi)

        # Value iteration step: V_new = V + dt*(best_H - r*V), then project onto V >= omega
        V_new = V + dt * (best_H - p.r * V)
        V_new = np.maximum(V_new, p.omega)

        residual = float(np.max(np.abs(V_new - V)))
        V = V_new

        if verbose and it % 200 == 0:
            print(f"iter {it:5d}  residual {residual:.6e}")

        if residual < tol and it > 50:
            break

    # Final policy on the converged V
    Vx, Vxx, VZ = _derivatives(V, dx, dZ)
    VZ_neg = np.minimum(VZ, -1e-8)
    z_policy = np.clip(-1.0 / VZ_neg - (c_grid + X), 0.0, 1.0)
    best_H = np.full_like(V, -np.inf)
    xi_policy = np.zeros_like(V)
    flow_z = np.log(np.maximum(c_grid + X + z_policy, 1e-9))
    Z_drift_term = (z_policy - p.rho * ZZ) * VZ
    for xi_val in xi_grid:
        phi_v = phi(ZZ, xi_val, p)
        mu_v  = mu(ZZ, xi_val, p)
        sig_v = sigma(ZZ, xi_val, p)
        drift_x = phi_v * (mu_v - X)
        H = flow_z + Z_drift_term + drift_x * Vx + 0.5 * sig_v**2 * Vxx
        mask = H > best_H
        best_H = np.where(mask, H, best_H)
        xi_policy = np.where(mask, xi_val, xi_policy)

    stop = (V <= p.omega + 1e-6)

    return {
        "V": V,
        "z_policy": z_policy,
        "xi_policy": xi_policy,
        "stop": stop,
        "x_grid": x,
        "Z_grid": Z,
        "iterations": it + 1,
        "residual": residual,
    }

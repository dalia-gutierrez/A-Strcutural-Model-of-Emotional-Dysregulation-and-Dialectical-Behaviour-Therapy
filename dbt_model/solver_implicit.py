"""Implicit HJB solver, stable under empirical-magnitude phi and sigma.

Policy iteration:
    1. Given V, compute optimal (z*, xi*) at each grid point under the
       regulation-effort budget z + xi <= B.
    2. Given policy (fixing drift and diffusion coefficients), solve the
       linear PDE  r V = flow + drift_x V_x + drift_Z V_Z + 0.5 sigma^2 V_xx
       as a sparse linear system using upwind differences.
    3. Apply the obstacle V <- max(V, omega).
    4. Iterate until V converges.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

from .model import Params, phi, mu, sigma


def _relief(z, p):
    if getattr(p, 'relief_lambda', 0.0) and p.relief_lambda > 0:
        return p.relief_lambda * (1.0 - np.exp(-z))
    return z


def _build_linear_system(p: Params, X, ZZ, c_grid, z_star, xi_star, dx, dZ):
    """Build the sparse operator A and RHS b for r V - L V = flow."""
    Nx, NZ = X.shape
    N = Nx * NZ

    phi_v   = phi(ZZ, xi_star, p)
    mu_v    = mu(ZZ, xi_star, p)
    sig_v   = sigma(ZZ, xi_star, p)
    drift_x = phi_v * (mu_v - p.delta_mu * z_star - X)
    drift_Z = p.kappa * z_star - p.rho * ZZ
    sig2    = sig_v ** 2

    if getattr(p, "utility_type", "additive") == "separable":
        flow = (np.log(np.maximum(c_grid, 1e-9))
                + p.gamma * np.log(np.maximum(1.0 + X + _relief(z_star, p), 1e-9)))
    else:
        flow = np.log(np.maximum(c_grid + X + _relief(z_star, p), 1e-9))

    A = lil_matrix((N, N))
    b = flow.flatten()

    def idx(i, j):
        return i * NZ + j

    inv_dx  = 1.0 / dx
    inv_dx2 = 1.0 / dx ** 2
    inv_dZ  = 1.0 / dZ

    for i in range(Nx):
        for j in range(NZ):
            k  = idx(i, j)
            dx_pos = max(drift_x[i, j], 0.0)
            dx_neg = min(drift_x[i, j], 0.0)
            dZ_pos = max(drift_Z[i, j], 0.0)
            dZ_neg = min(drift_Z[i, j], 0.0)

            diag = p.r
            if 0 < i < Nx - 1:
                A[k, idx(i + 1, j)] += -0.5 * sig2[i, j] * inv_dx2 - dx_pos * inv_dx
                A[k, idx(i - 1, j)] += -0.5 * sig2[i, j] * inv_dx2 + dx_neg * inv_dx
                diag += sig2[i, j] * inv_dx2 + dx_pos * inv_dx - dx_neg * inv_dx
            elif i == 0:
                A[k, idx(i + 1, j)] += -dx_pos * inv_dx
                diag += dx_pos * inv_dx
            else:
                A[k, idx(i - 1, j)] += dx_neg * inv_dx
                diag += -dx_neg * inv_dx

            if 0 < j < NZ - 1:
                A[k, idx(i, j + 1)] += -dZ_pos * inv_dZ
                A[k, idx(i, j - 1)] += dZ_neg * inv_dZ
                diag += dZ_pos * inv_dZ - dZ_neg * inv_dZ
            elif j == 0:
                A[k, idx(i, j + 1)] += -dZ_pos * inv_dZ
                diag += dZ_pos * inv_dZ
            else:
                A[k, idx(i, j - 1)] += dZ_neg * inv_dZ
                diag += -dZ_neg * inv_dZ

            A[k, k] += diag

    return csr_matrix(A), b


def _optimal_policy(p: Params, X, ZZ, c_grid, Vx, Vxx, VZ, xi_grid):
    """Joint (z*, xi*) under the budget z + xi <= B.

    z has THREE effects: bounded immediate relief rho_relief(z)=lambda(1-e^-z)
    in utility (+), accumulation into the harm stock (kappa, via VZ), and direct
    degradation of the affect attractor (delta_mu, effective mu = mu - delta_mu*z).
    When relief saturates the z-FOC is not closed-form, so z is grid-searched
    jointly with xi subject to z + xi <= B. (With relief_lambda<=0 the relief is
    linear and the closed-form z is used for speed.)
    """
    separable = getattr(p, "utility_type", "additive") == "separable"
    saturating = bool(getattr(p, "relief_lambda", 0.0) and p.relief_lambda > 0)
    VZ_neg = np.minimum(VZ, -1e-8)

    best_H  = np.full_like(X, -np.inf)
    xi_star = np.zeros_like(X)
    z_star  = np.zeros_like(X)

    def eval_H(z_try, xi_val, phi_v, mu_v, sig_v):
        rel = _relief(z_try, p)
        if separable:
            flow = (np.log(np.maximum(c_grid, 1e-9))
                    + p.gamma * np.log(np.maximum(1.0 + X + rel, 1e-9)))
        else:
            flow = np.log(np.maximum(c_grid + X + rel, 1e-9))
        mu_eff = mu_v - p.delta_mu * z_try
        return (flow + (p.kappa * z_try - p.rho * ZZ) * VZ
                + phi_v * (mu_eff - X) * Vx + 0.5 * sig_v ** 2 * Vxx)

    for xi_val in xi_grid:
        phi_v = phi(ZZ, xi_val, p); mu_v = mu(ZZ, xi_val, p); sig_v = sigma(ZZ, xi_val, p)
        z_cap = max(p.B - xi_val, 0.0)
        if saturating:
            z_candidates = np.linspace(0.0, z_cap, 17) if z_cap > 0 else np.array([0.0])
        else:
            if separable:
                z0 = p.gamma / np.maximum(phi_v * p.delta_mu * Vx - p.kappa * VZ_neg, 1e-8) - 1.0 - X
            else:
                z0 = 1.0 / np.maximum(phi_v * p.delta_mu * Vx - p.kappa * VZ_neg, 1e-8) - (c_grid + X)
            z_candidates = [np.clip(z0, 0.0, z_cap)]
        for z_try in z_candidates:
            z_try = z_try if np.ndim(z_try) else np.full_like(X, z_try)
            H = eval_H(z_try, xi_val, phi_v, mu_v, sig_v)
            mask = H > best_H
            best_H  = np.where(mask, H, best_H)
            xi_star = np.where(mask, xi_val, xi_star)
            z_star  = np.where(mask, z_try, z_star)
    return z_star, xi_star


def _derivatives(V, dx, dZ):
    Vx  = np.zeros_like(V)
    Vxx = np.zeros_like(V)
    VZ  = np.zeros_like(V)
    Vx [1:-1, :] = (V[2:, :] - V[:-2, :]) / (2 * dx)
    Vx [0, :]    = (V[1, :] - V[0, :]) / dx
    Vx [-1, :]   = (V[-1, :] - V[-2, :]) / dx
    Vxx[1:-1, :] = (V[2:, :] - 2 * V[1:-1, :] + V[:-2, :]) / dx ** 2
    Vxx[0, :]    = Vxx[1, :]
    Vxx[-1, :]   = Vxx[-2, :]
    VZ [:, 1:-1] = (V[:, 2:] - V[:, :-2]) / (2 * dZ)
    VZ [:, 0]    = (V[:, 1] - V[:, 0]) / dZ
    VZ [:, -1]   = (V[:, -1] - V[:, -2]) / dZ
    return Vx, Vxx, VZ


def solve_hjb_implicit(p: Params, constrained: bool = False,
                       n_xi: int = 21,
                       max_iter: int = 60, tol: float = 1e-4,
                       verbose: bool = False) -> dict:
    """Implicit policy-iteration solver. Stable for any phi, sigma.

    Skillful coping xi is grid-searched over [0, B] (the estimated budget),
    not an arbitrary cap. The constrained (no-DBT) agent has xi == 0 and the
    full budget B available for ineffective coping z.
    """
    x = np.linspace(p.x_min, p.x_max, p.Nx)
    Z = np.linspace(0.0, p.Z_max, p.NZ)
    dx = x[1] - x[0]
    dZ = Z[1] - Z[0]
    X, ZZ = np.meshgrid(x, Z, indexing="ij")
    c_grid = np.maximum(p.cbar - p.chi * ZZ, p.c_min)
    xi_grid = np.array([0.0]) if constrained else np.linspace(0.0, p.B, n_xi)

    z_init = 0.5 * np.ones_like(X)
    if getattr(p, "utility_type", "additive") == "separable":
        flow_init = (np.log(np.maximum(c_grid, 1e-9))
                     + p.gamma * np.log(np.maximum(1.0 + X + z_init, 1e-9)))
    else:
        flow_init = np.log(np.maximum(c_grid + X + z_init, 1e-9))
    V = flow_init / p.r
    V = np.maximum(V, p.omega)

    it = 0
    residual = np.inf
    for it in range(max_iter):
        Vx, Vxx, VZ = _derivatives(V, dx, dZ)
        z_star, xi_star = _optimal_policy(p, X, ZZ, c_grid, Vx, Vxx, VZ, xi_grid)

        A, b = _build_linear_system(p, X, ZZ, c_grid, z_star, xi_star, dx, dZ)
        V_new = spsolve(A, b).reshape(p.Nx, p.NZ)
        V_new = np.maximum(V_new, p.omega)

        residual = float(np.max(np.abs(V_new - V)))
        V = V_new
        if verbose:
            print(f"iter {it:3d}  residual {residual:.3e}")
        if residual < tol and it > 2:
            break

    Vx, Vxx, VZ = _derivatives(V, dx, dZ)
    z_policy, xi_policy = _optimal_policy(p, X, ZZ, c_grid, Vx, Vxx, VZ, xi_grid)
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

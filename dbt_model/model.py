"""Primitives of the DBT structural model."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class Params:
    # Time and preferences
    r: float = 0.04
    Tbar: float = 52.0
    omega: float = -5.0

    # Endowment
    cbar: float = 1.0
    chi: float = 0.20
    c_min: float = 0.0           # income floor (safety-net counterfactual)

    # Utility specification: "additive" = log(c+x+z); "separable" = log(c) + gamma*log(1+x+z)
    utility_type: str = "additive"
    gamma: float = 0.5

    # Bounded relief: effective relief from ineffective coping is
    #   rho_relief(z) = lambda*(1 - exp(-z))  when relief_lambda > 0,
    # else linear z (original). The ceiling lambda is the most momentary
    # affect relief coping can buy (affect-SD units); estimated ~0.75
    # from the within-person affect headroom in the Fisher2017 EMA.
    relief_lambda: float = 0.0

    # Regulation-effort budget: z_t + xi_t <= B. Ineffective coping z and
    # skillful coping xi compete for one finite per-period regulatory capacity.
    # Estimated from EMA coping intensity (see estimate_B.py); baseline 3.85
    # affect-SD units (95th pct of observed coping intensity, Fisher2017).
    B: float = 3.85

    # Z dynamics: dZ = (kappa*z - rho*Z) dt. kappa<1 separates the
    # immediate-relief intensity of ineffective coping from its (slower)
    # accumulation into the lasting harm stock. Identified from the
    # remission/income/suicide moments.
    kappa: float = 1.0
    rho: float = 0.20

    # Mean reversion: phi(Z, xi) = phi_0 + alpha_phi*(1 - exp(-xi)) - beta_phi*Z
    phi_0: float = 1.0
    alpha_phi: float = 0.5
    beta_phi: float = 0.30

    # Ineffective coping z directly worsens the affect attractor:
    #   effective mu = mu(Z, xi) - delta_mu * z
    # This is what makes z 'ineffective': relief now (in utility) but the
    # mood attractor is pulled down, a rumination/avoidance spiral.
    delta_mu: float = 0.0

    # Long-run mean: mu(Z, xi) = mu_0 + alpha_mu*(1 - exp(-xi)) - beta_mu*Z
    mu_0: float = 0.0
    alpha_mu: float = 0.8
    beta_mu: float = 0.40

    # Volatility: sigma(Z, xi) = sigma_0 * exp(beta_sigma*Z - alpha_sigma*xi)
    sigma_0: float = 0.6
    alpha_sigma: float = 0.3
    beta_sigma: float = 0.30

    # Numerical grid
    Nx: int = 81
    NZ: int = 41
    Nt: int = 200
    x_min: float = -3.0
    x_max: float = 3.0
    Z_max: float = 3.0


def phi(Z, xi, p: Params):
    val = p.phi_0 + p.alpha_phi * (1.0 - np.exp(-xi)) - p.beta_phi * Z
    return np.maximum(val, 1e-6)


def mu(Z, xi, p: Params):
    return p.mu_0 + p.alpha_mu * (1.0 - np.exp(-xi)) - p.beta_mu * Z


def sigma(Z, xi, p: Params):
    return p.sigma_0 * np.exp(p.beta_sigma * Z - p.alpha_sigma * xi)


def u(y):
    return np.log(np.maximum(y, 1e-12))


def consumption(Z, p: Params):
    return np.maximum(p.cbar - p.chi * Z, p.c_min)


def default_params() -> Params:
    return Params()

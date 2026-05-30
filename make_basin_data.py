"""Regenerate the lifetime-suicide basin data for Figure 'fig:basin'.

Writes figures/basin_untreated.dat and figures/basin_dbt.dat: lifetime suicide
probability across initial states (x0, Z0) under the constrained (no-DBT) and
unconstrained (DBT) policies, at the calibrated exit value omega = 0.25. The
pgfplots surf plot reads these three-column (x0  Z0  prob) tables.

Run:  PYTHONPATH=. python3 make_basin_data.py
"""
from __future__ import annotations
import os
import numpy as np
from dbt_model import Params
from dbt_model.solver_implicit import solve_hjb_implicit
from dbt_model.simulator import simulate

THETA = dict(
    utility_type="separable", gamma=0.586,
    r=0.04, Tbar=52.0, cbar=1.0, chi=0.024,
    phi_0=4.0, alpha_phi=0.28, beta_phi=0.30,
    mu_0=-0.002, alpha_mu=0.90, beta_mu=0.40,
    sigma_0=0.60, alpha_sigma=0.51, beta_sigma=0.30,
    rho=0.20, kappa=0.05, delta_mu=0.08, relief_lambda=0.75, B=3.85,
    Nx=81, NZ=41, Z_max=3.0, x_min=-3.0, x_max=3.0,
)
OMEGA = 0.25
SEED = 12345
N_SIM = 1500
NX0, NZ0 = 10, 10            # figure grid (mesh/rows=10 in the .tex)

def main():
    os.makedirs("figures", exist_ok=True)
    p = Params(omega=OMEGA, **THETA)
    sol_c = solve_hjb_implicit(p, constrained=True, max_iter=15)
    sol_u = solve_hjb_implicit(p, constrained=False, max_iter=15)
    x0s = np.linspace(-2.0, 1.0, NX0)
    Z0s = np.linspace(0.0, 2.4, NZ0)
    for sol, name in [(sol_c, "figures/basin_untreated.dat"),
                      (sol_u, "figures/basin_dbt.dat")]:
        with open(name, "w") as f:
            f.write("x0 Z0 prob\n")
            for x0 in x0s:
                for Z0 in Z0s:
                    sim = simulate(p, sol, n_sim=N_SIM, dt=0.05, seed=SEED,
                                   x0=float(x0), Z0=float(Z0))
                    f.write(f"{x0:.4f} {Z0:.4f} {sim['lifetime_suicide_rate']:.4f}\n")
                f.write("\n")   # blank line between rows for pgfplots surf
        print(f"wrote {name}", flush=True)
    print("BASIN_DATA_DONE", flush=True)

if __name__ == "__main__":
    main()

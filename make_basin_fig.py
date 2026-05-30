"""Single combined figure replacing the old Figures 1 and 3.

Three panels: baseline constrained, halved chi, unconstrained.
Each panel shows lifetime suicide probability (red shading) with the
instantaneous stopping set boundary overlaid in black.
"""
import json
from dataclasses import replace
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dbt_model import Params
from dbt_model.solver_implicit import solve_hjb_implicit
from dbt_model.simulator import simulate

with open("first_estimation.json") as f: est = json.load(f)
P0 = Params(**est["estimated_params"])
P_sep = replace(P0, utility_type="separable", gamma=0.586, omega=-10.53)

sol_c  = solve_hjb_implicit(P_sep, constrained=True)
sol_u  = solve_hjb_implicit(P_sep, constrained=False)
P_lo   = replace(P_sep, chi=P_sep.chi * 0.5)
sol_lo = solve_hjb_implicit(P_lo, constrained=True)

x0_grid = np.linspace(-2.0, 1.5, 10)
Z0_grid = np.linspace(0.0, 2.5, 8)

def basin(P, sol):
    B = np.zeros((len(x0_grid), len(Z0_grid)))
    for i, x0 in enumerate(x0_grid):
        for j, Z0 in enumerate(Z0_grid):
            s = simulate(P, sol, n_sim=60, seed=0,
                         x0=x0, Z0=Z0, x0_sd=0.0, Z0_sd=0.0)
            B[i, j] = s["lifetime_suicide_rate"]
    return B

print("computing baseline basin...")
B_base = basin(P_sep, sol_c)
print("computing halved-chi basin...")
B_lo   = basin(P_lo,  sol_lo)
print("computing unconstrained basin...")
B_u    = basin(P_sep, sol_u)

fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), sharey=True)
panels = [
    (axes[0], B_base, sol_c,  "Constrained, baseline"),
    (axes[1], B_lo,   sol_lo, r"Constrained, $\chi$ halved"),
    (axes[2], B_u,    sol_u,  "Unconstrained (with DBT)"),
]
ext = [x0_grid[0], x0_grid[-1], Z0_grid[0], Z0_grid[-1]]
for ax, B, sol_here, title in panels:
    im = ax.imshow(B.T, origin="lower", aspect="auto", extent=ext,
                   cmap="Reds", vmin=0, vmax=1)
    X, Z = np.meshgrid(sol_here["x_grid"], sol_here["Z_grid"], indexing="ij")
    ax.contour(X, Z, sol_here["stop"].astype(float),
               levels=[0.5], colors="black", linewidths=1.4)
    ax.set_title(title)
    ax.set_xlabel(r"initial state $x_0$")
axes[0].set_ylabel(r"initial stock $Z_0$")
fig.colorbar(im, ax=axes, label="lifetime suicide probability",
             location="right", shrink=0.85)
fig.suptitle(
    "Lifetime suicide basin (red shading) and instantaneous stopping set "
    "boundary (black)", fontsize=10)
fig.savefig("figures/basin_combined.png", dpi=140, bbox_inches="tight")
print("saved figures/basin_combined.png")

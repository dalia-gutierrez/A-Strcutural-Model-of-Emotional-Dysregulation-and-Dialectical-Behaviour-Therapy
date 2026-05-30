"""Canonical reproduction script for all numerical results in DBT.tex.

Regenerates every number in the paper from the estimated / calibrated
parameter vector (Table 1, "Theta-hat"). Supersedes the older ad-hoc drivers
(results.py, rerun_results.py), which loaded a stale parameter file and did not
reflect the revised model (regulation budget B, accumulation rate kappa,
coping-harm delta, bounded relief lambda).

Sections, keyed to the manuscript:
  A. Welfare table by initial condition .......... Table 4  (omega = deep floor)
  B. Which-coefficients-matter sensitivity ....... Table 3
  C. Utility-weight (gamma) robustness ........... Table 5
  D. r / rho / chi robustness .................... Section 5.5 text
  E. Suicide margin .............................. Section 4.3 (omega = 0.25)

Outputs: results/repro_all.json, results/table_welfare.csv, and a live
progress log at results/_progress.txt.

Run:  PYTHONPATH=. python3 reproduce_all.py
"""
from __future__ import annotations
import json, os, csv, time
from dataclasses import replace
import numpy as np

from dbt_model import Params
from dbt_model.solver_implicit import solve_hjb_implicit
from dbt_model.simulator import simulate

MIDUS = 46072.0
OMEGA_FLOOR = -12.0          # deep floor for welfare (welfare is invariant to it)
OMEGA_SUICIDE = 0.25         # calibrated exit value for the suicide section
SEED = 12345
# Welfare gain is identical at iterations 5/10/20/40 (verified), so 15 is a safe
# margin. The residual chatters at one high-affect grid point that enters no
# reported quantity.
MAXIT = 15
N_SIM = 8000

THETA = dict(
    utility_type="separable", gamma=0.586,
    r=0.04, Tbar=52.0, cbar=1.0, chi=0.024,
    phi_0=4.0, alpha_phi=0.28, beta_phi=0.30,
    mu_0=-0.002, alpha_mu=0.90, beta_mu=0.40,
    sigma_0=0.60, alpha_sigma=0.51, beta_sigma=0.30,
    rho=0.20, kappa=0.05, delta_mu=0.08, relief_lambda=0.75, B=3.85,
    Nx=81, NZ=41, Z_max=3.0, x_min=-3.0, x_max=3.0,
)
ICs = [(-1.5,0.5),(-1.0,0.5),(-0.5,0.5),(-1.5,1.0),
       (-1.0,1.0),(-0.5,1.0),(-1.0,0.2),(0.0,0.5)]


def progress(msg):
    with open("results/_progress.txt", "a") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
    print(msg, flush=True)


def base(omega=OMEGA_FLOOR, **kw):
    d = dict(THETA); d.update(kw); d["omega"] = omega
    return Params(**d)


def solve(p, constrained):
    return solve_hjb_implicit(p, constrained=constrained, max_iter=MAXIT, tol=1e-4)


def val(sol, x0, Z0):
    xg, Zg, V = sol["x_grid"], sol["Z_grid"], sol["V"]
    i = int(np.clip(np.searchsorted(xg, x0)-1, 0, len(xg)-2))
    j = int(np.clip(np.searchsorted(Zg, Z0)-1, 0, len(Zg)-2))
    fx = (x0-xg[i])/(xg[i+1]-xg[i]); fz = (Z0-Zg[j])/(Zg[j+1]-Zg[j])
    return float((1-fx)*(1-fz)*V[i,j] + fx*(1-fz)*V[i+1,j]
                 + (1-fx)*fz*V[i,j+1] + fx*fz*V[i+1,j+1])


def gain_dollars(su, sc, x0, Z0, r):
    g = val(su, x0, Z0) - val(sc, x0, Z0)
    pct = np.exp(r*g) - 1.0
    return g, float(pct), float(pct*MIDUS)


def avg_dollars(su, sc, r):
    return float(np.mean([gain_dollars(su, sc, x, z, r)[2] for x, z in ICs]))


def main():
    os.makedirs("results", exist_ok=True)
    open("results/_progress.txt", "w").close()
    out = {"params": THETA, "seed": SEED, "max_iter": MAXIT}
    t0 = time.time()

    # ---- A. Welfare table (Table 4) --------------------------------------
    p = base()
    su = solve(p, False); sc = solve(p, True)
    rows = []
    for x0, Z0 in ICs:
        Vc = val(sc, x0, Z0); Vu = val(su, x0, Z0)
        g, pct, dol = gain_dollars(su, sc, x0, Z0, p.r)
        rows.append(dict(x0=x0, Z0=Z0, Vc=Vc, Vu=Vu, gain=g, pct=pct, dollars=dol))
    avg = float(np.mean([r["dollars"] for r in rows]))
    avg_pct = float(np.mean([r["pct"] for r in rows]))
    out["welfare_table"] = rows
    out["welfare_avg_dollars"] = avg
    out["welfare_avg_pct"] = avg_pct
    out["welfare_pv"] = avg/p.r
    with open("results/table_welfare.csv","w",newline="") as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    progress(f"A welfare avg ${avg:,.0f}/yr ({avg_pct*100:.1f}%) PV ${avg/p.r:,.0f}")

    # ---- B. Sensitivity (Table 3) ----------------------------------------
    sens = {}
    sc_base = sc
    def span_alpha(name, lo, hi, npts=5):
        ds=[avg_dollars(solve(base(**{name: float(v)}), False), sc_base, THETA["r"])
            for v in np.linspace(lo, hi, npts)]
        progress(f"B {name} [{lo},{hi}] -> {min(ds):,.0f} to {max(ds):,.0f}")
        return [lo,hi],[float(min(ds)),float(max(ds))],[float(x) for x in ds]
    def span_beta(name, lo, hi, npts=5):
        ds=[]
        for v in np.linspace(lo, hi, npts):
            pv=base(**{name: float(v)})
            ds.append(avg_dollars(solve(pv,False), solve(pv,True), THETA["r"]))
        progress(f"B {name} [{lo},{hi}] -> {min(ds):,.0f} to {max(ds):,.0f}")
        return [lo,hi],[float(min(ds)),float(max(ds))],[float(x) for x in ds]
    sens["alpha_mu"]    = span_alpha("alpha_mu",   0.4, 1.4)
    sens["alpha_phi"]   = span_alpha("alpha_phi",  0.0, 1.0)
    sens["alpha_sigma"] = span_alpha("alpha_sigma",0.0, 1.0)
    sens["beta_mu"]     = span_beta("beta_mu",     0.0, 0.8)
    sens["beta_phi"]    = span_beta("beta_phi",    0.0, 0.6)
    sens["beta_sigma"]  = span_beta("beta_sigma",  0.0, 0.6)
    out["sensitivity"] = sens

    # ---- C. gamma robustness (Table 5) at (-1,0.5) -----------------------
    grows = []
    pa = replace(base(), utility_type="additive")
    su_a = solve(pa, False); sc_a = solve(pa, True)
    g,pct,dol = gain_dollars(su_a, sc_a, -1.0, 0.5, pa.r)
    grows.append(dict(spec="additive", gamma=None, gain=g, pct=pct, dollars=dol))
    for gm in (1.0, 0.586, 0.5, 0.25):
        pg = base(gamma=gm)
        g,pct,dol = gain_dollars(solve(pg,False), solve(pg,True), -1.0, 0.5, pg.r)
        grows.append(dict(spec="separable", gamma=gm, gain=g, pct=pct, dollars=dol))
    out["gamma_table"] = grows
    progress("C gamma table done")

    # ---- D. r / rho / chi robustness -------------------------------------
    rob = {}
    for r_ in (0.03, 0.04, 0.05):
        pr=base(r=r_); rob[f"r_{r_}"]=avg_dollars(solve(pr,False), solve(pr,True), r_)
    for rho_ in (0.10, 0.20, 0.30):
        pr=base(rho=rho_); rob[f"rho_{rho_}"]=avg_dollars(solve(pr,False), solve(pr,True), THETA["r"])
    for chi_ in (0.012, 0.024, 0.077):
        pr=base(chi=chi_); rob[f"chi_{chi_}"]=avg_dollars(solve(pr,False), solve(pr,True), THETA["r"])
    out["robustness"] = rob
    progress("D robustness done")

    # ---- E. Suicide margin (omega = 0.25) --------------------------------
    ps = base(omega=OMEGA_SUICIDE)
    su_s = solve(ps, False); sc_s = solve(ps, True)
    sim_c = simulate(ps, sc_s, n_sim=N_SIM, dt=0.05, seed=SEED,
                     x0=-1.0, Z0=0.8, x0_sd=0.8, Z0_sd=0.7)
    sim_u = simulate(ps, su_s, n_sim=N_SIM, dt=0.05, seed=SEED,
                     x0=-1.0, Z0=0.8, x0_sd=0.8, Z0_sd=0.7)
    out["suicide"] = dict(
        omega=OMEGA_SUICIDE,
        untreated_rate=sim_c["lifetime_suicide_rate"],
        treated_rate=sim_u["lifetime_suicide_rate"],
        exit_region_frac_constrained=float(sc_s["stop"].mean()),
        exit_region_frac_unconstrained=float(su_s["stop"].mean()),
    )
    progress(f"E suicide untreated={sim_c['lifetime_suicide_rate']:.4f} "
             f"treated={sim_u['lifetime_suicide_rate']:.4f} "
             f"exit_con={sc_s['stop'].mean()*100:.1f}%")

    out["runtime_sec"] = time.time()-t0
    with open("results/repro_all.json","w") as f:
        json.dump(out, f, indent=2)
    progress(f"DONE in {out['runtime_sec']:.0f}s")


if __name__ == "__main__":
    main()

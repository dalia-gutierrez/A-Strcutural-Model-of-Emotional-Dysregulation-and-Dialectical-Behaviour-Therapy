# A-Strcutural-Model-of-Emotional-Dysregulation-and-Dialectical-Behaviour-Therapy
Replication package for the working paper A Strcutural Model of Emotional Dysregulation and Dialectical Behaviour Therapy

Dalia Gutierrez Valencia

This package reproduces every number, table, and figure in the paper from raw,
publicly available data. A full run takes a few minutes on a laptop.

---

## 1. Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
bash run_all.sh                    # estimate -> solve -> tables -> bootstrap CI
```

Outputs land in `results/` (JSON + CSV) and `figures/` (`.dat` tables for the
pgfplots figure). To rebuild the PDF you also need a LaTeX distribution with the
`elsarticle` and `pgfplots` packages:

```bash
pdflatex DBT && bibtex DBT && pdflatex DBT && pdflatex DBT
```

---

## 2. What produces each result

| Paper object | Script | Output |
|---|---|---|
| Table 1 — $\phi_0,\mu_0,\sigma_0$ | `estimate_dynamics.py` | console / `first_estimation.json` |
| Table 1 — $B,\lambda,\delta$ | `estimate_budget_params.py`, `estimate_B.py` | `budget_params.json`, `B_estimate.json` |
| Table 1 — $\alpha_\phi,\alpha_\mu,\alpha_\sigma$ | `estimate_alphas.py` | `alpha_estimates.json` |
| Table 1 — $\gamma$ | `estimate_gamma.py` | `gamma_estimate.json` |
| Table 1 — $\chi$ (panel FE / FD / pooled) | `estimate_chi_panel.py` | `chi_panel.json` |
| Table 3 — sensitivity | `reproduce_all.py` (section B) | `results/repro_all.json` |
| Table 4 — welfare by initial state | `reproduce_all.py` (section A) | `results/table_welfare.csv` |
| Table 5 — $\gamma$ robustness | `reproduce_all.py` (section C) | `results/repro_all.json` |
| §5.5 — $r,\rho,\chi$ robustness | `reproduce_all.py` (section D) | `results/repro_all.json` |
| §4.3 — suicide margin ($\omega=0.25$) | `reproduce_all.py` (section E) | `results/repro_all.json` |
| Fig. (basin) | `make_basin_data.py` | `figures/basin_untreated.dat`, `figures/basin_dbt.dat` |
| Welfare 90% CI | `bootstrap_meta.py`, `bootstrap_table1.py`, `summarize_ci.py` | `boot_*.jsonl` |

The model lives in `dbt_model/`:

* `model.py` — `Params` dataclass and the functional forms ($\phi,\mu,\sigma$,
  bounded relief, consumption).
* `solver_implicit.py` — HJB solver by implicit Howard policy iteration on a
  sparse upwind discretisation; joint $(z,\xi)$ optimum under the budget
  $z+\xi\le B$; exit option imposed as the obstacle $V\leftarrow\max(V,\omega)$.
* `simulator.py` — Euler–Maruyama Monte Carlo under the optimal policy.
* `estimation.py` — indirect-inference helper that calibrates $\omega$ to a
  target suicide rate.

`solver.py` is the original explicit solver, kept for reference; all paper
results use `solver_implicit.py`.

---

## 3. Reproducibility notes

* **Determinism.** All Monte Carlo uses a fixed seed (`12345`). The HJB solver
  is deterministic.
* **Iteration count.** `reproduce_all.py` runs the policy iteration for 15
  sweeps. The welfare gain is identical (to the dollar) at 5, 10, 20 and 40
  sweeps; 15 is a safe margin. The implicit solver's residual chatters at a
  single high-affect grid point that enters none of the reported quantities.
* **Grids.** Results are computed on the $81\times41$ $(x,Z)$ grid stated in the
  paper. They are unchanged on finer control grids ($n_\xi$ up to 81) and on
  wider state grids ($x\in[-5,5]$, $Z\le5$); simulated state paths never
  accumulate at a grid edge (0.00% of state-time at any boundary). See
  `VERIFICATION_REPORT.md`.
* **Reported bootstrap prior.** The paper's 90% interval (\$9,200–\$27,100)
  comes from `bootstrap_table1.py`, which draws every primitive from its
  Table-1 point estimate and standard error ($\chi=0.024$, $B=3.85$, etc.) —
  fully consistent with the estimated parameter vector. `bootstrap_meta.py`
  (raw cross-section $\chi,B$) is kept for reference and gives the same interval
  to within rounding, because the gain is nearly flat in $\chi$ and $B$.

---

## 4. Data

See `DATA_SOURCES.md` and `ICPSR_DOWNLOAD_GUIDE.md` for full provenance and
download instructions. In brief:

* **Experience-sampling (ESM/EMA).** `data_ESM/` holds the open archives used
  for the affect dynamics and the budget/relief/harm parameters
  (Wright et al. 2017; Fisher et al. 2017; Rowland & Wenzel 2020). These are
  redistributed here under their original open licences.
* **MIDUS 1–3.** The income-penalty and well-being parameters come from the
  Midlife in the United States panel (ICPSR 02760, 04652, 36346). ICPSR terms
  do **not** permit redistribution, so the `.dta` files are not included in a
  public release; `ICPSR_DOWNLOAD_GUIDE.md` gives step-by-step retrieval and the
  expected directory layout (`data_MIDUS/MIDUS{1,2,3}/`).

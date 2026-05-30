"""Estimate the regulation-effort budget B from EMA coping-intensity data.

Economic object
---------------
The model gives the agent a finite per-period affect-regulation capacity B and
lets her split it between ineffective coping z (immediate hedonic relief, enters
utility through 1+x+z, accumulates the harmful stock Z) and skillful coping xi
(no immediate relief, improves the affect process). The constraint is

        z_t + xi_t <= B,   z_t, xi_t >= 0.

B is therefore the *ceiling on total regulatory intensity* a person can mobilise
at one moment. Without DBT (xi=0) the whole budget is available for ineffective
quick-fixes; DBT lets the agent reallocate part of B to skills, paying the
opportunity cost of the forgone immediate relief gamma/(1+x+z).

Identification
--------------
z and x share a scale: they enter utility additively as 1+x+z. The model's x is
a standardised affect index (onset x0 ~ N(-1, 0.8^2), i.e. SD ~ 1 in affect
z-score units). So B must be measured in units of the affect composite's
standard deviation.

Fisher et al. (2017) EMA records momentary *ineffective coping* (rumination,
behavioural avoidance, reassurance-seeking, procrastination) on the SAME 0-100
scale as momentary negative affect. We:
  1. build a negative-affect composite (the x scale) and a coping composite
     (the z construct) per beep,
  2. aggregate to daily means per subject (matching the dynamics estimation),
  3. divide the coping composite by the total SD of the affect composite, so
     coping is expressed in affect-SD units (the model's x/z units), and
  4. take a high percentile of that distribution as B: the regulatory intensity
     that is exceeded only (1-q) of the time -- the empirical ceiling.

We report B at the 90th/95th/99th percentiles for sensitivity; the 95th is the
baseline.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd

NEG_AFFECT = ["irritable", "restless", "worried", "guilty", "afraid",
              "angry", "hopeless", "down", "tension", "threatened"]
COPING = ["ruminate", "avoid_act", "reassure", "procrast"]


def estimate_B(path: str = "data_ESM/data_Fisher2017.csv",
               q_baseline: float = 0.95) -> dict:
    d = pd.read_csv(path)
    d["na"] = d[NEG_AFFECT].mean(axis=1)
    d["cope"] = d[COPING].mean(axis=1)
    d = d.dropna(subset=["na", "cope", "subj_id"]).copy()

    # Reconstruct a beep/day index (Fisher2017 has ~4 beeps/day)
    d["beep"] = d.groupby("subj_id").cumcount()
    d["day"] = d["beep"] // 4
    daily = d.groupby(["subj_id", "day"], as_index=False)[["na", "cope"]].mean()

    sd_na = float(daily["na"].std())          # affect-SD yardstick (model x ~ z-scored)
    z_affect_units = daily["cope"] / sd_na    # coping intensity in affect-SD units

    out = {
        "n_subjects": int(daily["subj_id"].nunique()),
        "n_person_days": int(len(daily)),
        "sd_affect_composite": sd_na,
        "B_q90": float(z_affect_units.quantile(0.90)),
        "B_q95": float(z_affect_units.quantile(0.95)),
        "B_q99": float(z_affect_units.quantile(0.99)),
        "B_max": float(z_affect_units.max()),
        "q_baseline": q_baseline,
        "B": float(z_affect_units.quantile(q_baseline)),
        "source": "Fisher et al. 2017 EMA (rumination/avoidance/reassurance/procrastination)",
    }
    return out


if __name__ == "__main__":
    res = estimate_B()
    with open("B_estimate.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"B (95th pct, baseline) = {res['B']:.2f}  affect-SD units")
    print(f"  sensitivity: B90={res['B_q90']:.2f}  B95={res['B_q95']:.2f}  "
          f"B99={res['B_q99']:.2f}  Bmax={res['B_max']:.2f}")
    print(f"  from {res['n_subjects']} subjects, {res['n_person_days']} person-days")

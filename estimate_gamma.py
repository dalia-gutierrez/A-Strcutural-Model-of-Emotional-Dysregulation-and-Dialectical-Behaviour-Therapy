"""Estimate gamma from MIDUS subjective-wellbeing data.

Model utility:
    u(c, x, z) = log(c) + gamma * log(1 + x + z)

Empirical analogue:
    WB_i = alpha + beta_1 * log(income_i) + beta_2 * affect_i + controls + eps_i

where WB is the average of the six Ryff psychological-wellbeing subscales
(Autonomy, Environmental mastery, personal Growth, positive Relations,
Self-acceptance, purpose in Life), and affect is the standardised positive
minus standardised negative affect.

Identification:
    gamma_observable = beta_2 / beta_1
This is the ratio of "utility units per standard deviation of affect"
to "utility units per log dollar of consumption." We then standardise the
model's x to have unit cross-sectional variance so that one unit of x in
the model corresponds to one cross-sectional SD of affect in MIDUS. With
that calibration, gamma in the model equals gamma_observable.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

PATH = "data_MIDUS/MIDUS1/02760-0001-Data.dta"


def load_with_cleaning():
    df = pd.read_stata(PATH, convert_categoricals=False)
    # Income: 999_998 and 999_999 are DK / refuse
    s = pd.to_numeric(df["A1SHWEARN"], errors="coerce")
    df["income"] = s.where((s >= 1) & (s < 999_000))

    # Affect scales: valid range 1 to 5
    for col in ("A1SPOSAF", "A1SNEGAF"):
        s = pd.to_numeric(df[col], errors="coerce")
        df[col + "_v"] = s.where((s >= 1) & (s <= 5))

    # PWB subscales: valid range typically 1 to 21 (3 items, 7-point scale)
    pwb_cols = ["A1SPWBA", "A1SPWBE", "A1SPWBG", "A1SPWBR", "A1SPWBU", "A1SPWBS"]
    for col in pwb_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        df[col + "_v"] = s.where((s >= 3) & (s <= 21))

    df["PWB"] = df[[c + "_v" for c in pwb_cols]].mean(axis=1)

    # Sex/Age cleanups
    s = pd.to_numeric(df["A1PRSEX"], errors="coerce")
    df["sex"] = s.where(s.isin([1, 2]))
    df["age"] = pd.to_numeric(df["A1PAGE_M2"], errors="coerce")
    return df


def main():
    df = load_with_cleaning()

    # Standardised affect index (positive minus negative, then z-scored)
    pa = df["A1SPOSAF_v"]
    na = df["A1SNEGAF_v"]
    affect = (pa - pa.mean()) / pa.std() - (na - na.mean()) / na.std()
    df["affect_z"] = affect

    # z-scored wellbeing
    df["wb_z"] = (df["PWB"] - df["PWB"].mean()) / df["PWB"].std()
    df["log_inc"] = np.log(df["income"].clip(lower=1))

    use = df.dropna(subset=["wb_z", "log_inc", "affect_z", "age", "sex"])
    print(f"sample size: {len(use):,}")

    m = smf.ols("wb_z ~ log_inc + affect_z + age + C(sex)",
                data=use).fit()
    print(m.summary())

    b1 = float(m.params["log_inc"])
    b2 = float(m.params["affect_z"])
    gamma_hat = b2 / b1                                 # observable units
    print(f"\nbeta_log_inc  = {b1:.4f}")
    print(f"beta_affect   = {b2:.4f}")
    print(f"gamma = beta_affect / beta_log_inc = {gamma_hat:.3f}")

    # Bootstrap SE on gamma
    rng = np.random.default_rng(7)
    n = len(use)
    Xcols = ["log_inc", "affect_z", "age"]
    use = use.copy()
    use["sex_F"] = (use["sex"] == 2).astype(float)
    Xmat = np.column_stack([np.ones(n)] + [use[c].values for c in Xcols] + [use["sex_F"].values])
    y = use["wb_z"].values
    gammas = np.empty(300)
    for b in range(300):
        idx = rng.integers(0, n, n)
        beta, *_ = np.linalg.lstsq(Xmat[idx], y[idx], rcond=None)
        gammas[b] = beta[2] / beta[1]
    gamma_se = float(np.std(gammas, ddof=1))
    print(f"bootstrap SE(gamma) over 300 reps: {gamma_se:.3f}")

    out = {
        "gamma_hat":      float(gamma_hat),
        "gamma_se":       gamma_se,
        "beta_log_inc":   b1,
        "beta_affect":    b2,
        "n":              int(len(use)),
        "interpretation": ("ratio of utility units per SD of affect to "
                           "utility units per log dollar of income; the model's "
                           "x is calibrated to unit affect SD"),
    }
    with open("gamma_estimate.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved gamma_estimate.json")


if __name__ == "__main__":
    main()

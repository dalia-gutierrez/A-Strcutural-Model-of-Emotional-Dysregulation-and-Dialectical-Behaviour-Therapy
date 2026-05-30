"""Estimate baseline OU dynamics (phi_0, mu_0, sigma_0) from Wright2017 ESM data.

Model (held at xi = 0, Z = 0, i.e. baseline):
    dx_t = phi_0 (mu_0 - x_t) dt + sigma_0 dW_t

Discretized at beep interval dt:
    x_{t+1} - x_t = phi_0 (mu_0 - x_t) dt + sigma_0 sqrt(dt) eps_t

This is an AR(1):
    x_{t+1} = (1 - phi_0 dt) x_t + phi_0 mu_0 dt + sigma_0 sqrt(dt) eps_t
            = a + b x_t + u_t,    var(u_t) = sigma_0^2 dt

with mapping:
    phi_0 = (1 - b) / dt
    mu_0  = a / (1 - b)
    sigma_0 = sqrt(var(u) / dt)

We fit within-person, then pool.

Beep interval (Wright2017): ~6 beeps/day, so dt approx 16 waking hours / 6 / (365 * 24)
years. We use dt = 1/(365 * 6) years (six measurements per day, treated as one
per discrete step).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NEG_AFFECT_ITEMS = [
    "Afraid", "Distressed", "Hostile", "Irritable", "Nervous",
    "Scared", "Upset", "Angry", "Sad", "Blue", "Downhearted", "Lonely",
]


def load_wright2017(path: str = "data_ESM/data_Wright2017.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    # Composite negative affect: mean of available items
    df["x"] = df[NEG_AFFECT_ITEMS].mean(axis=1)
    # Drop rows with no affect data
    df = df.dropna(subset=["x", "subj_id"])
    return df


def fit_ar1_within(df: pd.DataFrame, dt: float = 1.0 / (365 * 6)) -> dict:
    """Pool within-person AR(1) estimates with subject fixed effects via
    centering, then return implied OU parameters.
    """
    # Sort and lag within subject
    df = df.sort_values(["subj_id", "day", "beep"]).copy()
    df["x_lag"] = df.groupby("subj_id")["x"].shift(1)
    df = df.dropna(subset=["x", "x_lag"])

    # Within-subject demeaning so subject fixed effects don't contaminate b
    df["x_dm"]     = df["x"]     - df.groupby("subj_id")["x"].transform("mean")
    df["x_lag_dm"] = df["x_lag"] - df.groupby("subj_id")["x_lag"].transform("mean")

    # OLS slope (no intercept after demeaning)
    num = float((df.x_dm * df.x_lag_dm).sum())
    den = float((df.x_lag_dm ** 2).sum())
    b   = num / den

    # Intercept = mean(x) - b * mean(x_lag), using pooled means
    a = float(df["x"].mean() - b * df["x_lag"].mean())

    # Residual variance
    resid = df.x - (a + b * df.x_lag)
    sigma2_innov = float(resid.var(ddof=1))

    # Map back to OU
    phi_0   = (1.0 - b) / dt
    mu_0    = a / (1.0 - b) if abs(1 - b) > 1e-9 else float(df.x.mean())
    sigma_0 = float(np.sqrt(sigma2_innov / dt))

    # Recenter mu_0 to model units: shift so that mean(x) = 0 in our model.
    # In MIDUS units affect runs roughly 1-5; in our model x runs roughly -3 to 3.
    # Subtract the sample mean of x to recenter.
    x_mean_raw = float(df["x"].mean())

    return {
        "b": b,
        "a": a,
        "phi_0_per_year": phi_0,
        "mu_0_raw_scale": mu_0,
        "mu_0_centered": mu_0 - x_mean_raw,
        "sigma_0_per_sqrt_year": sigma_0,
        "innov_var": sigma2_innov,
        "n_subjects": int(df.subj_id.nunique()),
        "n_obs": int(len(df)),
        "dt_years": dt,
        "x_mean_raw_scale": x_mean_raw,
    }


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse beeps to daily mean affect per subject."""
    return (df.groupby(["subj_id", "day"], as_index=False)["x"].mean())


def main():
    print("Loading Wright2017 ESM...")
    df = load_wright2017()
    print(f"  {df.subj_id.nunique()} subjects, {len(df)} beeps")

    print("\n--- Fit at beep resolution (dt = 1/(365*6) yr) ---")
    res_beep = fit_ar1_within(df, dt=1.0 / (365 * 6))
    print(f"  phi_0 (per yr): {res_beep['phi_0_per_year']:.1f}   "
          f"sigma_0 (per sqrt yr): {res_beep['sigma_0_per_sqrt_year']:.1f}")
    print("  (these are technically correct but huge because affect mean-reverts in hours)")

    print("\n--- Fit at daily resolution (dt = 1/365 yr) ---")
    df_daily = aggregate_daily(df)
    df_daily = df_daily.rename(columns={"day": "beep"})
    df_daily["day"] = df_daily["beep"]                # for the sort key below
    res_day = fit_ar1_within(df_daily, dt=1.0 / 365)
    for k, v in res_day.items():
        if isinstance(v, float):
            print(f"  {k:30s} {v:>10.4f}")
        else:
            print(f"  {k:30s} {v}")

    print("\nDaily-scale values to use as DBT-model defaults:")
    print(f"  phi_0   = {res_day['phi_0_per_year']:.2f}    (mean reversion, per year)")
    print(f"  sigma_0 = {res_day['sigma_0_per_sqrt_year']:.2f}    (per sqrt year)")
    print(f"  mu_0    = {res_day['mu_0_centered']:.3f}    (centered)")
    print("\nNote: this is a cluster B (pathological narcissism) sample, the closest")
    print("Cluster B sample; refit on Trull BPD data when available.")


if __name__ == "__main__":
    main()

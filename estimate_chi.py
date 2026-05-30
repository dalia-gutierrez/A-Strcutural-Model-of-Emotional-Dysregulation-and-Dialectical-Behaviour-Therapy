"""Estimate chi from MIDUS once you have downloaded the data.

Run after placing the MIDUS 1 Stata file at:
    data_MIDUS/MIDUS1/02760-0001-Data.dta

This script:
    1. Loads MIDUS 1.
    2. Builds the BPD-features proxy from Neuroticism, low Conscientiousness, depression.
    3. Regresses log(income) on the proxy with age and sex controls.
    4. Maps the coefficient to an implied chi at the sample mean income.

If MIDUS variable names differ in your codebook, edit the NAME_MAP dict below.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import statsmodels.formula.api as smf
except ImportError:
    raise SystemExit("Install statsmodels: pip install statsmodels")

# MIDUS 1 (wave A1) variable names.
NAME_MAP = {
    "hh_income":      "A1SHWEARN",   # SAQ household earnings (dollars)
    "neuroticism":    "A1SNEURO",    # 4-item adjective scale, 1-4
    "conscientious":  "A1SCONS",     # 5-item conscientiousness, 1-4
    "depression":     "A1PDEPDX",    # 12-month MDD diagnosis (0/1)
    "age":            "A1PAGE_M2",
    "sex":            "A1PRSEX",     # 1 = male, 2 = female; 7-9 missing
}


def load_midus1(path: str) -> pd.DataFrame:
    df = pd.read_stata(path, convert_categoricals=False)
    # MIDUS-specific missing-code cleanup (per the ICPSR codebook).
    # Income: 999,999 and 999,998 mean DK / refuse.
    income_col = NAME_MAP["hh_income"]
    if income_col in df.columns:
        s = pd.to_numeric(df[income_col], errors="coerce")
        df[income_col] = s.where((s >= 0) & (s < 999000))
    # Personality scales: 1-4 valid; negatives and 9+ are missing.
    for col in (NAME_MAP["neuroticism"], NAME_MAP["conscientious"]):
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            df[col] = s.where((s >= 1) & (s <= 4))
    # Sex: 1=male, 2=female; 7/8/9 missing.
    sex_col = NAME_MAP["sex"]
    if sex_col in df.columns:
        s = pd.to_numeric(df[sex_col], errors="coerce")
        df[sex_col] = s.where(s.isin([1, 2]))
    return df


def build_proxy(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for short, var in NAME_MAP.items():
        if var not in out.columns:
            raise KeyError(f"Expected MIDUS variable {var} (label '{short}') not found. "
                           f"Check the codebook and edit NAME_MAP in estimate_chi.py.")
        out[short] = pd.to_numeric(out[var], errors="coerce")

    def z(x):
        return (x - x.mean()) / x.std(ddof=0)

    out["bpd_proxy"] = z(out.neuroticism) - z(out.conscientious) + z(out.depression)
    out["log_inc"]   = np.log(out.hh_income.clip(lower=1))
    return out


def fit(df: pd.DataFrame):
    m = smf.ols(
        "log_inc ~ bpd_proxy + age + C(sex)",
        data=df.dropna(subset=["log_inc", "bpd_proxy", "age", "sex"]),
    ).fit()
    return m


def chi_from_coef(coef_bpd_proxy: float, mean_income: float) -> float:
    """Map the OLS coefficient to chi.

    With log-income on the LHS, the marginal effect of one SD increase in
    the BPD-features proxy is mean_income * coef. That SD corresponds to
    roughly one unit of Z in the model. So chi (the income loss per unit Z)
    is approximately -mean_income * coef (positive number).
    """
    return -float(mean_income) * float(coef_bpd_proxy)


def main():
    path = "data_MIDUS/MIDUS1/02760-0001-Data.dta"
    print(f"Loading {path}...")
    df = load_midus1(path)
    print(f"  shape = {df.shape}")

    print("Building BPD-features proxy...")
    df = build_proxy(df)

    print("Fitting OLS: log(income) ~ bpd_proxy + age + sex...")
    m = fit(df)
    print(m.summary())

    coef = m.params["bpd_proxy"]
    mean_inc = df.hh_income.dropna().mean()
    chi_hat = chi_from_coef(coef, mean_inc)
    print(f"\nMean household income (sample): ${mean_inc:,.0f}")
    print(f"OLS coefficient on bpd_proxy:   {coef:.4f} (log income)")
    print(f"Implied chi (income units per unit Z): {chi_hat:,.2f}")

    # Express in cbar = 1 normalization so the structural model can use it directly.
    chi_model = -float(coef)        # log-income slope is the percentage-loss-per-Z;
                                    # cbar normalized to 1 implies chi = % loss per Z.
    import json
    with open("chi_estimate.json", "w") as f:
        json.dump({
            "chi_hat":            chi_hat,
            "chi_model_units":    chi_model,
            "ols_coef_bpd_proxy": coef,
            "mean_income":        float(mean_inc),
            "n_obs":              int(m.nobs),
        }, f, indent=2)
    print(f"Saved chi_estimate.json (chi in model units, cbar=1: {chi_model:.4f})")


if __name__ == "__main__":
    main()

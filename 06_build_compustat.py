#!/usr/bin/env python3
"""
06_build_compustat.py - construct firm-level Compustat controls

Input:
    compustat_russell3000_2024.csv (raw, one row per firm, fyear 2024)

Output:
    compustat_summary.csv - one row per firm with:
        - log_at              : log(total assets) - SIZE control (replaces
                                log market cap, which isn't in this data)
        - leverage_w          : leverage winsorized at 1%/99%
        - roa_w               : ROA winsorized at 1%/99%
        - log_firm_age        : log(2024 - ipo_year + 1)
        - firm_age            : 2024 - ipo_year (raw)
        - firm_age_missing    : dummy = 1 if ipo_year is unknown
        - sic_code            : 4-digit SIC
        - sic_2digit          : 2-digit SIC (for industry FE)
        - sic_1digit          : 1-digit SIC (for coarse industry FE)
        - at, leverage, roa, sale, emp, cash_ratio (raw, unwinsorized)

Notes on choices:
    - log(at) used as size control because market cap is not in the file
    - leverage and roa winsorized at 1%/99% to limit influence of extreme
      outliers (a few tiny firms have leverage > 4 or roa < -10).
    - For firm age: where ipo_year is missing we set firm_age_missing=1
      and impute log_firm_age = log(median firm_age). This lets you keep
      those firms in the regression while accounting for the missingness.

Usage:
    python 06_build_compustat.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

INPUT_CSV = Path("compustat_russell3000_2024.csv")
OUTPUT_CSV = Path("compustat_summary.csv")


def winsorize(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Clip a series at the lower and upper quantiles."""
    lo = s.quantile(lower)
    hi = s.quantile(upper)
    return s.clip(lower=lo, upper=hi)


def main() -> None:
    if not INPUT_CSV.exists():
        print(f"ERROR: {INPUT_CSV} not found in current folder.")
        return

    cp = pd.read_csv(INPUT_CSV, low_memory=False)
    print(f"Loaded {len(cp)} rows from {INPUT_CSV}")
    print(f"  Unique tickers: {cp['tic'].nunique()}")

    # ------------------------------------------------------------------
    # Rename for consistency with other files (which use 'ticker')
    # ------------------------------------------------------------------
    cp = cp.rename(columns={'tic': 'ticker'})

    # ------------------------------------------------------------------
    # SIZE: log(total assets). Replaces log(market_cap) which isn't here.
    # ------------------------------------------------------------------
    cp['log_at'] = np.log(cp['at'].clip(lower=1))  # clip avoids log(0)

    # ------------------------------------------------------------------
    # LEVERAGE: pre-computed by your professor, but winsorize for outliers
    # ------------------------------------------------------------------
    cp['leverage_w'] = winsorize(cp['leverage'])

    # ------------------------------------------------------------------
    # ROA: pre-computed, but has extreme outliers (min -12.49 in raw).
    # Winsorize at 1%/99% to limit their influence.
    # ------------------------------------------------------------------
    cp['roa_w'] = winsorize(cp['roa'])

    # ------------------------------------------------------------------
    # FIRM AGE: only 60% have ipo_year. Handle missingness explicitly.
    # ------------------------------------------------------------------
    cp['firm_age'] = 2024 - cp['ipo_year']
    cp['firm_age_missing'] = cp['ipo_year'].isna().astype(int)
    # Where firm_age known, log(age + 1). Where missing, impute median.
    known_age = cp.loc[cp['firm_age'].notna(), 'firm_age']
    median_age = known_age.median()
    cp['firm_age_imputed'] = cp['firm_age'].fillna(median_age)
    cp['log_firm_age'] = np.log(cp['firm_age_imputed'] + 1)

    # ------------------------------------------------------------------
    # INDUSTRY: 1- and 2-digit SIC
    # ------------------------------------------------------------------
    cp['sic_2digit'] = (cp['sic_code'] // 100).astype('Int64')
    cp['sic_1digit'] = (cp['sic_code'] // 1000).astype('Int64')

    # ------------------------------------------------------------------
    # Select output columns (one row per firm)
    # ------------------------------------------------------------------
    out_cols = [
        'ticker',
        'gvkey',
        'conm',                 # company name (for sanity-checking)
        'log_at',               # SIZE control
        'leverage_w',           # winsorized leverage
        'roa_w',                # winsorized ROA
        'log_firm_age',         # firm-age control (with imputation)
        'firm_age',             # raw firm age (for descriptive stats)
        'firm_age_missing',     # dummy: 1 if ipo_year is unknown
        'sic_code',             # 4-digit (for reference)
        'sic_2digit',           # for industry FE (default)
        'sic_1digit',           # for industry FE (coarser alternative)
        'at',                   # raw assets (for descriptive stats)
        'leverage',             # raw leverage (descriptive)
        'roa',                  # raw ROA (descriptive)
        'sale',                 # sales
        'emp',                  # employees
        'cash_ratio',           # extra (could be used as alt size proxy)
    ]
    out = cp[out_cols].copy()

    # Report
    print(f"\nFinal Compustat dataset: {len(out)} firms")
    print(f"\nKey variable distributions:")
    for c in ['log_at', 'leverage_w', 'roa_w', 'log_firm_age']:
        s = out[c].dropna()
        print(f"  {c:22s} n={len(s):>5d}  mean={s.mean():>7.3f}  "
              f"median={s.median():>7.3f}  std={s.std():>7.3f}")
    print(f"\n  firm_age_missing dummy: {out['firm_age_missing'].sum()} firms "
          f"({100*out['firm_age_missing'].mean():.1f}%)")
    print(f"\n  Distinct 2-digit SIC industries: {out['sic_2digit'].nunique()}")
    print(f"  Distinct 1-digit SIC industries: {out['sic_1digit'].nunique()}")

    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(out)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

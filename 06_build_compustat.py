#!/usr/bin/env python3
# Builds the firm-level Compustat controls from the raw fyear-2024 pull and
# writes compustat_summary.csv, one row per firm.
#
# Choices worth remembering:
#   - log(at) is the size control because market cap isn't in this extract.
#   - leverage and roa are winsorized at 1%/99%; a handful of tiny firms have
#     leverage above 4 or roa below -10 and would otherwise dominate.
#   - Only ~60% of firms have an ipo_year. Those without get
#     firm_age_missing=1 and log_firm_age imputed from the median age, so
#     they stay in the regression with the missingness accounted for.
#
#   python 06_build_compustat.py

import numpy as np
import pandas as pd
from pathlib import Path

INPUT_CSV = Path("compustat_russell3000_2024.csv")
OUTPUT_CSV = Path("compustat_summary.csv")

OUT_COLS = [
    'ticker',
    'gvkey',
    'conm',              # company name, for eyeballing
    'log_at',            # size
    'leverage_w',
    'roa_w',
    'log_firm_age',      # imputed where ipo_year is missing
    'firm_age',
    'firm_age_missing',
    'sic_code',
    'sic_2digit',        # industry FE default
    'sic_1digit',        # coarser alternative
    'at',
    'leverage',
    'roa',
    'sale',
    'emp',
    'cash_ratio',
]

SUMMARY_COLS = ['log_at', 'leverage_w', 'roa_w', 'log_firm_age']


def winsorize(s, lower=0.01, upper=0.99):
    return s.clip(lower=s.quantile(lower), upper=s.quantile(upper))


def main():
    if not INPUT_CSV.exists():
        print(f"ERROR: {INPUT_CSV} not found in current folder.")
        return

    cp = pd.read_csv(INPUT_CSV, low_memory=False)
    print(f"Loaded {len(cp)} rows from {INPUT_CSV}")
    print(f"  Unique tickers: {cp['tic'].nunique()}")

    # everything else in the pipeline calls it 'ticker'
    cp = cp.rename(columns={'tic': 'ticker'})

    cp['log_at'] = np.log(cp['at'].clip(lower=1))  # clip avoids log(0)
    cp['leverage_w'] = winsorize(cp['leverage'])
    cp['roa_w'] = winsorize(cp['roa'])

    cp['firm_age'] = 2024 - cp['ipo_year']
    cp['firm_age_missing'] = cp['ipo_year'].isna().astype(int)
    age_imputed = cp['firm_age'].fillna(cp['firm_age'].median())
    cp['log_firm_age'] = np.log(age_imputed + 1)

    cp['sic_2digit'] = (cp['sic_code'] // 100).astype('Int64')
    cp['sic_1digit'] = (cp['sic_code'] // 1000).astype('Int64')

    out = cp[OUT_COLS].copy()

    print(f"\nFinal Compustat dataset: {len(out)} firms")
    print(f"\nKey variable distributions:")
    for c in SUMMARY_COLS:
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

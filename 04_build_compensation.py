#!/usr/bin/env python3
# Builds firm-level CEO compensation variables from the raw WRDS ExecComp
# pull (FY2024 only — the whole file is 2024) and writes comp_summary.csv,
# one row per ticker.
#
# Decisions worth remembering:
#   - equity_share = (stock_awards + option_awards) / total_sec for the CEO.
#     That's the main independent variable.
#   - Firms with a CEO transition have two CEO rows; keep the one with the
#     higher total_sec, i.e. the year's full-year CEO rather than the partial.
#   - equity_share > 1.0 is mathematically impossible, so those are data
#     errors and get dropped (one firm, KNX).
#   - The pooled robustness measure is sum(equity)/sum(total_sec) across all
#     named executives, not the average of per-exec ratios — averaging blows
#     up when one exec has near-zero total comp.
#
#   python 04_build_compensation.py

import numpy as np
import pandas as pd
from pathlib import Path

INPUT_CSV = Path("execcomp_russell3000_2024.csv")
OUTPUT_CSV = Path("comp_summary.csv")

# source column -> output name (identity where nothing changes)
CEO_COLUMNS = {
    'ticker': 'ticker',
    'exec_fullname': 'ceo_name',
    'equity_share_exec': 'equity_share',
    'log_total_sec': 'log_total_sec',
    'total_sec': 'ceo_total_sec',
    'salary': 'ceo_salary',
    'stock_awards': 'ceo_stock_awards',
    'option_awards': 'ceo_option_awards',
    'shrown_tot_pct': 'ceo_share_ownership',
    'age': 'ceo_age',
    'ceo_tenure_years': 'ceo_tenure_years',
}

FINAL_COLS = [
    'ticker',
    'ceo_name',
    'equity_share',         # main variable
    'equity_share_pooled',  # robustness
    'log_total_sec',
    'ceo_total_sec',
    'ceo_salary',
    'ceo_stock_awards',
    'ceo_option_awards',
    'ceo_share_ownership',
    'ceo_age',
    'ceo_tenure_years',
    'n_executives',
]

SUMMARY_COLS = ['equity_share', 'equity_share_pooled', 'log_total_sec',
                'ceo_age', 'ceo_tenure_years', 'ceo_share_ownership']


def main():
    if not INPUT_CSV.exists():
        print(f"ERROR: {INPUT_CSV} not found in current folder.")
        return

    ec = pd.read_csv(INPUT_CSV, low_memory=False)
    print(f"Loaded {len(ec)} executive rows from {INPUT_CSV}")
    print(f"  Unique firms (ticker): {ec['ticker'].nunique()}")

    # missing component = the exec simply got none of that type
    ec['equity_comp'] = ec['stock_awards'].fillna(0) + ec['option_awards'].fillna(0)
    ec['equity_share_exec'] = ec['equity_comp'] / ec['total_sec']

    ceos = ec[ec['ceoann'] == 'CEO'].copy()
    print(f"\nCEO rows: {len(ceos)}")

    ceos = (ceos.sort_values('total_sec', ascending=False)
                .drop_duplicates(subset='ticker', keep='first'))
    print(f"After dedup (one CEO per ticker): {len(ceos)}")

    n_before = len(ceos)
    ceos = ceos[
        (ceos['equity_share_exec'].notna())
        & (ceos['equity_share_exec'] >= 0)
        & (ceos['equity_share_exec'] <= 1.0)
    ].copy()
    print(f"After dropping equity_share > 1.0: {len(ceos)} (dropped {n_before - len(ceos)})")

    ceos['becameceo_dt'] = pd.to_datetime(ceos['becameceo'], errors='coerce')
    ceos['ceo_tenure_years'] = 2024 - ceos['becameceo_dt'].dt.year
    ceos['log_total_sec'] = np.log(ceos['total_sec'].clip(lower=1))

    ceo_cols = ceos[list(CEO_COLUMNS)].rename(columns=CEO_COLUMNS)

    # pooled measure runs over every exec, not just the CEO
    pooled = ec.groupby('ticker', as_index=False).agg(
        equity_comp_sum=('equity_comp', 'sum'),
        total_sec_sum=('total_sec', 'sum'),
        n_executives=('exec_fullname', 'count'),
    )
    pooled['equity_share_pooled'] = pooled['equity_comp_sum'] / pooled['total_sec_sum']
    # same >1 / inf problem as above
    pooled.loc[~pooled['equity_share_pooled'].between(0, 1.0), 'equity_share_pooled'] = np.nan
    pooled = pooled[['ticker', 'equity_share_pooled', 'n_executives']]

    out = ceo_cols.merge(pooled, on='ticker', how='left')[FINAL_COLS]

    print(f"\nFinal compensation dataset: {len(out)} firms")
    print(f"\nKey variable distributions:")
    for c in SUMMARY_COLS:
        s = out[c].dropna()
        if len(s) > 0:
            print(f"  {c:25s}  n={len(s):>4d}  mean={s.mean():>7.2f}  "
                  f"median={s.median():>7.2f}  std={s.std():>7.2f}")

    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(out)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

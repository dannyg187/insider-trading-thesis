#!/usr/bin/env python3
"""
04_build_compensation.py - construct firm-level CEO compensation variables

Input:
    execcomp_russell3000_2024.csv (raw ExecComp from WRDS)

Output:
    comp_summary.csv - one row per firm (ticker) with:
        - equity_share         : (stock_awards + option_awards) / total_sec
                                 for the CEO. MAIN INDEPENDENT VARIABLE.
        - equity_share_pooled  : same ratio but summed across all named
                                 executives. ROBUSTNESS CHECK.
        - log_total_sec        : log of CEO total compensation (level control)
        - ceo_total_sec        : CEO total compensation (USD thousands)
        - ceo_salary, ceo_stock_awards, ceo_option_awards (raw components)
        - ceo_share_ownership  : CEO total share ownership as % of company
        - ceo_age              : CEO age in years
        - ceo_tenure_years     : years since person became CEO
        - n_executives         : number of named executives in ExecComp
        - ceo_name             : for sanity-checking

Decisions / data cleaning:
    - FY2024 only (the entire file is already 2024)
    - For firms with multiple CEO rows (transitions), keeps the row with
      highest total_sec (the year's "full" CEO, not the partial-year one)
    - Drops cases where equity_share > 1.0 (only 1 firm: KNX, data error)
    - For pooled measure, uses sum(equity)/sum(total_sec) across all execs
      (more robust than averaging per-exec ratios, which blow up when an
      executive has near-zero total comp)

Usage:
    python 04_build_compensation.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

INPUT_CSV = Path("execcomp_russell3000_2024.csv")
OUTPUT_CSV = Path("comp_summary.csv")


def main() -> None:
    if not INPUT_CSV.exists():
        print(f"ERROR: {INPUT_CSV} not found in current folder.")
        return

    ec = pd.read_csv(INPUT_CSV, low_memory=False)
    print(f"Loaded {len(ec)} executive rows from {INPUT_CSV}")
    print(f"  Unique firms (ticker): {ec['ticker'].nunique()}")

    # ------------------------------------------------------------------
    # Step 1: Compute executive-level equity_comp and equity_share
    # ------------------------------------------------------------------
    # Fill missing components with 0 (some execs simply have 0 of a type)
    ec['equity_comp'] = (
        ec['stock_awards'].fillna(0) + ec['option_awards'].fillna(0)
    )
    # equity_share at executive level (used for CEO measure)
    ec['equity_share_exec'] = ec['equity_comp'] / ec['total_sec']

    # ------------------------------------------------------------------
    # Step 2: Build CEO-level dataset (MAIN measure)
    # ------------------------------------------------------------------
    ceos = ec[ec['ceoann'] == 'CEO'].copy()
    print(f"\nCEO rows: {len(ceos)}")

    # Some firms have multiple CEOs in a year (transitions).
    # Keep the one with highest total_sec (the year's "main" CEO).
    ceos = ceos.sort_values('total_sec', ascending=False)
    ceos = ceos.drop_duplicates(subset='ticker', keep='first')
    print(f"After dedup (one CEO per ticker): {len(ceos)}")

    # Drop outliers: equity_share > 1.0 means equity grants exceed total
    # comp, which is mathematically impossible -> data error.
    n_before = len(ceos)
    ceos = ceos[
        (ceos['equity_share_exec'].notna())
        & (ceos['equity_share_exec'] >= 0)
        & (ceos['equity_share_exec'] <= 1.0)
    ].copy()
    print(f"After dropping equity_share > 1.0: {len(ceos)} (dropped {n_before - len(ceos)})")

    # CEO tenure: years between becameceo date and 2024
    ceos['becameceo_dt'] = pd.to_datetime(ceos['becameceo'], errors='coerce')
    ceos['ceo_tenure_years'] = 2024 - ceos['becameceo_dt'].dt.year

    # Log of total comp (handles right-skew, useful as a control)
    ceos['log_total_sec'] = np.log(ceos['total_sec'].clip(lower=1))

    # Rename for clarity in the output
    ceo_cols = ceos[[
        'ticker',
        'exec_fullname',
        'equity_share_exec',
        'log_total_sec',
        'total_sec',
        'salary',
        'stock_awards',
        'option_awards',
        'shrown_tot_pct',
        'age',
        'ceo_tenure_years',
    ]].rename(columns={
        'exec_fullname': 'ceo_name',
        'equity_share_exec': 'equity_share',
        'total_sec': 'ceo_total_sec',
        'salary': 'ceo_salary',
        'stock_awards': 'ceo_stock_awards',
        'option_awards': 'ceo_option_awards',
        'shrown_tot_pct': 'ceo_share_ownership',
        'age': 'ceo_age',
    })

    # ------------------------------------------------------------------
    # Step 3: Build all-executives POOLED measure (ROBUSTNESS)
    # ------------------------------------------------------------------
    # Sum equity_comp and total_sec across all execs per firm, then divide.
    # This is more robust than averaging individual equity_share values
    # because it doesn't blow up when an exec has near-zero total comp.
    pooled = ec.groupby('ticker', as_index=False).agg(
        equity_comp_sum=('equity_comp', 'sum'),
        total_sec_sum=('total_sec', 'sum'),
        n_executives=('exec_fullname', 'count'),
    )
    pooled['equity_share_pooled'] = (
        pooled['equity_comp_sum'] / pooled['total_sec_sum']
    )
    # Drop infinite values and >1 outliers in the pooled measure too
    pooled.loc[
        ~pooled['equity_share_pooled'].between(0, 1.0), 'equity_share_pooled'
    ] = np.nan
    pooled = pooled[['ticker', 'equity_share_pooled', 'n_executives']]

    # ------------------------------------------------------------------
    # Step 4: Merge CEO + pooled
    # ------------------------------------------------------------------
    out = ceo_cols.merge(pooled, on='ticker', how='left')

    # Reorder for readability
    final_cols = [
        'ticker',
        'ceo_name',
        'equity_share',           # MAIN VARIABLE
        'equity_share_pooled',    # ROBUSTNESS
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
    out = out[final_cols]

    # ------------------------------------------------------------------
    # Step 5: Report and save
    # ------------------------------------------------------------------
    print(f"\nFinal compensation dataset: {len(out)} firms")
    print(f"\nKey variable distributions:")
    for c in ['equity_share', 'equity_share_pooled', 'log_total_sec',
              'ceo_age', 'ceo_tenure_years', 'ceo_share_ownership']:
        s = out[c].dropna()
        if len(s) > 0:
            print(f"  {c:25s}  n={len(s):>4d}  mean={s.mean():>7.2f}  "
                  f"median={s.median():>7.2f}  std={s.std():>7.2f}")

    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(out)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
10_add_factset_ownership.py — merge FactSet Ownership Summary into analysis

Input:
    analysis_v2.csv                       (from step 07)
    factset_ownership_summary_q4_2024.csv (FactSet Ownership Summary from WRDS)
                                          Rename your WRDS download to this
                                          filename before running.

Output:
    analysis_v3.csv                       (with FactSet ownership variables added)

What we add:
    mktcap           Market capitalization in USD millions
    log_mktcap       ln(mktcap)  — REPLACES log_at as primary size control
    io               Institutional ownership as % of market cap (0-1 scale)
    ibh_5pct         Blockholder ownership (institutions with >=5% stake, 0-1)
    top5             Ownership share of top 5 institutions (0-1)
    herf             Herfindahl index of ownership concentration
    nbr_firms        Number of institutional owners
    log_n_inst       ln(nbr_firms + 1)

Merge key:
    ticker (after stripping FactSet's "-US" suffix)

Coverage:
    ~832 firms match. The 106 unmatched are mostly REITs (58) — FactSet
    excludes REITs from their aggregated ownership summary. These firms
    will have null values in the FactSet variables and will drop from
    regressions using them. Main regression uses log_mktcap and thus
    excludes REITs; robustness spec uses log_at and includes REITs.

Usage:
    python 10_add_factset_ownership.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

ANALYSIS_CSV = Path("analysis_v2.csv")
FS_CSV = Path("factset_ownership_summary_q4_2024.csv")
OUTPUT_CSV = Path("analysis_v3.csv")


def main() -> None:
    for f in (ANALYSIS_CSV, FS_CSV):
        if not f.exists():
            print(f"ERROR: {f} not found in current folder.")
            return

    analysis = pd.read_csv(ANALYSIS_CSV)
    fs = pd.read_csv(FS_CSV, low_memory=False)
    print(f"Loaded {len(analysis)} firms from {ANALYSIS_CSV}")
    print(f"Loaded {len(fs)} rows from {FS_CSV}")

    # Strip the "-US" suffix that FactSet appends to tickers
    fs = fs.copy()
    fs['ticker'] = fs['ticker'].str.replace('-US', '', regex=False)

    # Sanity check: one row per ticker per quarter
    if fs['ticker'].duplicated().any():
        n_dup = fs['ticker'].duplicated().sum()
        print(f"  WARNING: {n_dup} duplicate tickers in FactSet file. "
              f"Keeping first occurrence.")
        fs = fs.drop_duplicates(subset='ticker', keep='first')

    # Compute log-transformed variables
    fs['log_mktcap'] = np.log(fs['mktcap'].clip(lower=1))
    fs['log_n_inst'] = np.log(fs['nbr_firms'].fillna(0) + 1)

    # Keep only the columns we need
    fs_out = fs[[
        'ticker',
        'mktcap',
        'log_mktcap',
        'io',
        'ibh_5pct',
        'top5',
        'herf',
        'nbr_firms',
        'log_n_inst',
    ]]

    # Merge into analysis dataset
    merged = analysis.merge(fs_out, on='ticker', how='left', indicator=True)
    n_matched = (merged['_merge'] == 'both').sum()
    n_unmatched = (merged['_merge'] == 'left_only').sum()
    print(f"\nMerge results:")
    print(f"  Total analysis rows:   {len(merged)}")
    print(f"  Matched to FactSet:    {n_matched}  "
          f"({100*n_matched/len(merged):.1f}%)")
    print(f"  Unmatched (no FactSet): {n_unmatched}")

    merged = merged.drop(columns=['_merge'])

    # Report distributions
    print(f"\nFactSet variable distributions (matched firms):")
    for c in ['mktcap', 'log_mktcap', 'io', 'ibh_5pct', 'top5', 'herf',
              'nbr_firms']:
        s = merged[c].dropna()
        if len(s) > 0:
            print(f"  {c:12s}  n={len(s):>4d}  mean={s.mean():>10.3f}  "
                  f"median={s.median():>10.3f}")

    # Preview the final regression samples
    print(f"\nProjected regression samples:")
    days = pd.to_numeric(
        merged['blackout_start_days_before_quarter_end'], errors='coerce')
    main_ok = (days.notna()
               & merged['equity_share'].notna()
               & merged['log_mktcap'].notna()
               & merged['leverage_w'].notna()
               & merged['roa_w'].notna()
               & merged['log_firm_age'].notna())
    robust_ok = (days.notna()
                 & merged['equity_share'].notna()
                 & merged['log_at'].notna()
                 & merged['leverage_w'].notna()
                 & merged['roa_w'].notna()
                 & merged['log_firm_age'].notna())
    print(f"  Main spec (log_mktcap): {main_ok.sum()} firms")
    print(f"  Robustness (log_at):    {robust_ok.sum()} firms")

    merged.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(merged)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

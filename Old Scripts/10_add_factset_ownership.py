#!/usr/bin/env python3
"""
10_add_factset_ownership.py — merge FactSet Ownership Summary AND
                              build CEO stock-holdings variables

Inputs:
    analysis_v2.csv                       (from step 07)
    factset_ownership_summary_q4_2024.csv (FactSet Ownership Summary from WRDS)

Output:
    analysis_v3.csv                       (final analysis dataset)

FactSet variables added (per firm):
    mktcap           Market capitalization, USD millions
    log_mktcap       ln(mktcap)  — main size control
    io               Institutional ownership, share of market cap [0-1]
    ibh_5pct         Blockholder ownership (5%+ stakes)
    top5             Top-5 investor ownership
    herf             Herfindahl ownership concentration
    nbr_firms        Number of institutional owners
    log_n_inst       ln(nbr_firms + 1)

CEO holdings variables added (professor's suggested extension):
    ceo_holdings_musd     Dollar value of CEO holdings in USD millions
                          = (ceo_share_ownership / 100) * mktcap
    log_ceo_holdings      ln(ceo_holdings_musd)
    holdings_to_comp      CEO holdings expressed as multiples of annual comp
                          = ceo_holdings_musd * 1000 / ceo_total_sec
    log_holdings_to_comp  ln(holdings_to_comp)

The "holdings_to_comp" measure follows the professor's suggestion to scale
CEO holdings by average CEO compensation, expressing accumulated equity
exposure in units of annual comp equivalent.

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

    fs = fs.copy()
    fs['ticker'] = fs['ticker'].str.replace('-US', '', regex=False)

    if fs['ticker'].duplicated().any():
        n_dup = fs['ticker'].duplicated().sum()
        print(f"  WARNING: {n_dup} duplicate tickers in FactSet file. "
              f"Keeping first occurrence.")
        fs = fs.drop_duplicates(subset='ticker', keep='first')

    fs['log_mktcap'] = np.log(fs['mktcap'].clip(lower=1))
    fs['log_n_inst'] = np.log(fs['nbr_firms'].fillna(0) + 1)

    fs_out = fs[[
        'ticker', 'mktcap', 'log_mktcap', 'io', 'ibh_5pct', 'top5',
        'herf', 'nbr_firms', 'log_n_inst',
    ]]

    merged = analysis.merge(fs_out, on='ticker', how='left', indicator=True)
    n_matched = (merged['_merge'] == 'both').sum()
    n_unmatched = (merged['_merge'] == 'left_only').sum()
    print(f"\nMerge results:")
    print(f"  Total rows:            {len(merged)}")
    print(f"  Matched to FactSet:    {n_matched}  "
          f"({100*n_matched/len(merged):.1f}%)")
    print(f"  Unmatched (no FactSet): {n_unmatched}")
    merged = merged.drop(columns=['_merge'])

    # -----------------------------------------------------------------
    # CEO holdings variables (professor's extension)
    # -----------------------------------------------------------------
    # ceo_share_ownership is in percentage points (e.g. 3.5 for 3.5%)
    # mktcap is in USD millions
    # ceo_total_sec is in USD thousands (ExecComp convention)
    #
    # Dollar value of holdings = (pct / 100) * mktcap  → USD millions
    # Scaled by comp: multiply by 1000 to convert to thousands then divide by comp
    merged['ceo_holdings_musd'] = (
        (merged['ceo_share_ownership'] / 100) * merged['mktcap']
    )
    merged['log_ceo_holdings'] = np.log(
        merged['ceo_holdings_musd'].clip(lower=0.001)
    )
    merged['holdings_to_comp'] = (
        merged['ceo_holdings_musd'] * 1000 / merged['ceo_total_sec']
    )
    merged['log_holdings_to_comp'] = np.log(
        merged['holdings_to_comp'].clip(lower=0.001)
    )

    print(f"\nCEO holdings variables (matched firms with both mktcap and "
          f"ceo_share_ownership):")
    for c in ['ceo_holdings_musd', 'holdings_to_comp',
              'log_ceo_holdings', 'log_holdings_to_comp']:
        s = merged[c].dropna()
        if len(s) > 0:
            print(f"  {c:22s}  n={len(s):>4d}  mean={s.mean():>10.2f}  "
                  f"median={s.median():>10.2f}")

    # -----------------------------------------------------------------
    # Preview regression samples
    # (main spec no longer requires firm_age)
    # -----------------------------------------------------------------
    days = pd.to_numeric(
        merged['blackout_start_days_before_quarter_end'], errors='coerce')
    main_ok = (days.notna()
               & merged['equity_share'].notna()
               & merged['log_mktcap'].notna()
               & merged['leverage_w'].notna()
               & merged['roa_w'].notna())
    holdings_ok = main_ok & merged['log_ceo_holdings'].notna()
    print(f"\nProjected regression samples:")
    print(f"  Main spec (log_mktcap):        {main_ok.sum()} firms")
    print(f"  CEO holdings extension:        {holdings_ok.sum()} firms")

    merged.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(merged)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

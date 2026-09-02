#!/usr/bin/env python3
"""
07_build_analysis_dataset.py - final merge: policy + comp + Compustat

Inputs:
    analysis_v1.csv         (policy + compensation merged earlier)
    compustat_summary.csv   (firm-level controls)

Output:
    analysis_v2.csv         (one row per firm, all variables for regression)

Usage:
    python 07_build_analysis_dataset.py
"""

import pandas as pd
from pathlib import Path

POLICY_COMP_CSV = Path("analysis_v1.csv")
COMPUSTAT_CSV = Path("compustat_summary.csv")
OUTPUT_CSV = Path("analysis_v2.csv")


def main() -> None:
    for f in (POLICY_COMP_CSV, COMPUSTAT_CSV):
        if not f.exists():
            print(f"ERROR: {f} not found. Run earlier scripts first.")
            return

    base = pd.read_csv(POLICY_COMP_CSV)
    cp = pd.read_csv(COMPUSTAT_CSV)
    print(f"Loaded {len(base)} rows from {POLICY_COMP_CSV}")
    print(f"Loaded {len(cp)} rows from {COMPUSTAT_CSV}")

    # Drop columns from Compustat that duplicate base (e.g. ticker already there)
    # Then left-join
    merged = base.merge(cp, on='ticker', how='left', indicator=True)

    n_matched = (merged['_merge'] == 'both').sum()
    n_unmatched = (merged['_merge'] == 'left_only').sum()
    print(f"\nMerge results:")
    print(f"  Total rows:      {len(merged)}")
    print(f"  Matched:         {n_matched}  ({100*n_matched/len(merged):.1f}%)")
    print(f"  No Compustat:    {n_unmatched}")

    merged = merged.drop(columns=['_merge'])

    # ----------------- FINAL REGRESSION SAMPLE CHECK ------------------
    days = pd.to_numeric(merged['blackout_start_days_before_quarter_end'], errors='coerce')
    required_cols = ['equity_share', 'log_at', 'leverage_w', 'roa_w',
                     'log_firm_age', 'sic_2digit']
    has_all = days.notna()
    for c in required_cols:
        has_all = has_all & merged[c].notna()

    print(f"\n=== FINAL REGRESSION SAMPLE ===")
    print(f"Rows with ALL of: blackout_days, equity_share, and controls")
    print(f"  ({', '.join(required_cols)}):")
    print(f"  Final n = {has_all.sum()} firms")

    print(f"\nSample of merged data (5 firms with complete data):")
    show = merged[has_all][[
        'ticker', 'company_name',
        'blackout_start_days_before_quarter_end',
        'equity_share', 'log_at', 'leverage_w', 'roa_w',
        'log_firm_age', 'sic_2digit',
    ]].head(5)
    print(show.to_string(index=False))

    merged.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(merged)} rows to {OUTPUT_CSV}")
    print(f"This is your final dataset for the regression.")


if __name__ == "__main__":
    main()

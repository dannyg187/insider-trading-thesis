#!/usr/bin/env python3
"""
05_merge_policy_comp.py - merge policy data with compensation data

Inputs:
    blackout_summary_dedup.csv  (one row per firm, policy features)
    comp_summary.csv            (one row per firm, CEO compensation)

Output:
    analysis_v1.csv             (one row per firm with policy + comp variables)

Merge logic:
    Left join on ticker, keeping the policy dataset as the base. This means
    every firm in the policy data appears in the output; firms without a
    match in ExecComp will have null compensation fields and will be
    dropped by the regression automatically (or you can filter them out
    explicitly later).

Usage:
    python 05_merge_policy_comp.py
"""

import pandas as pd
from pathlib import Path

POLICY_CSV = Path("blackout_summary_dedup.csv")
COMP_CSV = Path("comp_summary.csv")
OUTPUT_CSV = Path("analysis_v1.csv")


def main() -> None:
    for f in (POLICY_CSV, COMP_CSV):
        if not f.exists():
            print(f"ERROR: {f} not found. Run earlier scripts first.")
            return

    policy = pd.read_csv(POLICY_CSV)
    comp = pd.read_csv(COMP_CSV)
    print(f"Loaded {len(policy)} policy rows from {POLICY_CSV}")
    print(f"Loaded {len(comp)} compensation rows from {COMP_CSV}")

    # Left-join: policy is the base, attach comp where available
    merged = policy.merge(comp, on='ticker', how='left', indicator=True)

    # Report the merge result
    n_matched = (merged['_merge'] == 'both').sum()
    n_only_policy = (merged['_merge'] == 'left_only').sum()
    print(f"\nMerge results:")
    print(f"  Total rows:       {len(merged)}")
    print(f"  Matched (policy + comp):  {n_matched}  ({100*n_matched/len(merged):.1f}%)")
    print(f"  Policy only (no comp):    {n_only_policy}")

    # Drop the merge indicator before saving
    merged = merged.drop(columns=['_merge'])

    # How many rows have BOTH a numeric blackout-days value AND equity_share?
    days = pd.to_numeric(merged['blackout_start_days_before_quarter_end'], errors='coerce')
    has_both = days.notna() & merged['equity_share'].notna()
    print(f"\nRows usable for main regression "
          f"(both blackout_days and equity_share): {has_both.sum()}")

    # Show a sample
    print(f"\nSample of merged data (first 5 matched firms):")
    sample = merged[merged['equity_share'].notna()][[
        'ticker', 'company_name', 'blackout_start_days_before_quarter_end',
        'requires_preclearance', 'prohibits_hedging',
        'equity_share', 'log_total_sec', 'ceo_age'
    ]].head(5)
    print(sample.to_string(index=False))

    # Save
    merged.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(merged)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

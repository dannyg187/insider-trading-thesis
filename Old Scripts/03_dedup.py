#!/usr/bin/env python3
"""
03_dedup.py - one row per firm

Goal:
    The merged dataset (blackout_summary_final.csv) can have multiple rows
    per firm because (a) some filings contain two EX-19 exhibits and (b)
    some firms filed in two fiscal years. The regression needs one row per
    firm, so we deduplicate by ticker.

Rule:
    For each ticker, prefer:
      1. The most recent filing_date (covers cross-year duplicates)
      2. Among same-date duplicates, the row with a non-null
         blackout-days value (covers EX-19.1 vs EX-19.2 split filings)
      3. As a final tiebreak, has_recurring_blackout=True over False

Output:
    blackout_summary_dedup.csv  -- one row per firm, ready for merging
                                   with ExecComp / Compustat

Usage:
    python 03_dedup.py
"""

import pandas as pd
from pathlib import Path

INPUT_CSV = Path("blackout_summary_final.csv")
OUTPUT_CSV = Path("blackout_summary_dedup.csv")


def main() -> None:
    if not INPUT_CSV.exists():
        print(f"ERROR: {INPUT_CSV} not found. Run 02_merge_recovery.py first.")
        return

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} rows from {INPUT_CSV}")
    print(f"Unique tickers: {df['ticker'].nunique()}")

    # Build sort keys
    # filing_date is in DD-MM-YY format
    df['_filing_dt'] = pd.to_datetime(
        df['filing_date'], format='%d-%m-%y', errors='coerce'
    )
    df['_days_numeric'] = pd.to_numeric(
        df['blackout_start_days_before_quarter_end'], errors='coerce'
    )
    df['_has_days'] = df['_days_numeric'].notna()
    df['_has_recurring'] = df['has_recurring_blackout'].astype(bool)

    # Sort by ticker (asc), filing_date (desc = most recent first),
    # has_days (desc = True first), has_recurring (desc = True first)
    df_sorted = df.sort_values(
        by=['ticker', '_filing_dt', '_has_days', '_has_recurring'],
        ascending=[True, False, False, False],
    )

    # Keep the first row for each ticker
    df_dedup = df_sorted.drop_duplicates(subset='ticker', keep='first').copy()

    # Drop the helper sort columns
    df_dedup = df_dedup.drop(
        columns=['_filing_dt', '_days_numeric', '_has_days', '_has_recurring']
    )

    # Report
    print(f"\nAfter dedup: {len(df_dedup)} rows (dropped {len(df) - len(df_dedup)})")

    days = pd.to_numeric(
        df_dedup['blackout_start_days_before_quarter_end'], errors='coerce'
    )
    print(f"\nBlackout-days coverage:")
    print(f"  Numeric:  {days.notna().sum()} firms ({100*days.notna().sum()/len(df_dedup):.1f}%)")
    print(f"  Null:     {days.isna().sum()} firms ({100*days.isna().sum()/len(df_dedup):.1f}%)")

    print(f"\nBlackout-days distribution (numeric only):")
    print(days.describe().to_string())

    # Save
    df_dedup.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(df_dedup)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

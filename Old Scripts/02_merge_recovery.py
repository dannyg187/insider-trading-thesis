#!/usr/bin/env python3
"""
02_merge_recovery.py - combine original extraction with recovery results

Goal:
    Take blackout_summary.csv (original) and blackout_summary_recovery.csv
    (recovered values for 132 problem files) and produce a clean merged
    blackout_summary_final.csv.

Merge rules:
    For every row in the original CSV:
      - If the filename is in the recovery CSV: replace the policy-level
        fields (blackout days, descriptions, has_recurring_blackout, etc.)
        with the recovered values. Metadata (ticker, CIK, market_value etc.)
        stays the same in either case.
      - Otherwise: keep the original row untouched.

Output:
    blackout_summary_final.csv  -- this is the dataset you use for analysis
    merge_report.txt           -- summary of what changed

Usage:
    python 02_merge_recovery.py
"""

import csv
from pathlib import Path

ORIGINAL_CSV = Path("blackout_summary.csv")
RECOVERY_CSV = Path("blackout_summary_recovery.csv")
OUTPUT_CSV = Path("blackout_summary_final.csv")
REPORT_FILE = Path("merge_report.txt")

# Columns that get OVERWRITTEN with recovery data if a match exists.
# (Metadata columns like ticker/CIK/market_value are NOT in this list -
#  we always keep them from the original.)
OVERWRITE_COLUMNS = [
    "has_recurring_blackout",
    "has_ad_hoc_blackout",
    "requires_preclearance",
    "preclearance_description",
    "prohibits_hedging",
    "hedging_description",
    "general_description",
    "group_name",
    "blackout_description",
    "blackout_start_days_before_quarter_end",
    "groups_raw_json",
]


def main() -> None:
    if not ORIGINAL_CSV.exists():
        print(f"ERROR: {ORIGINAL_CSV} not found in current folder.")
        return
    if not RECOVERY_CSV.exists():
        print(f"ERROR: {RECOVERY_CSV} not found. Did the recovery script finish?")
        return

    # Load both CSVs
    with ORIGINAL_CSV.open("r", encoding="utf-8", newline="") as f:
        original_rows = list(csv.DictReader(f))
        fieldnames = list(original_rows[0].keys()) if original_rows else []

    with RECOVERY_CSV.open("r", encoding="utf-8", newline="") as f:
        recovery_rows = list(csv.DictReader(f))

    print(f"Loaded {len(original_rows)} original rows, {len(recovery_rows)} recovery rows")

    # Index recovery by filename
    recovery_by_fn = {r["filename"]: r for r in recovery_rows}

    # Track changes for reporting
    n_filled = 0          # null → value
    n_overwritten = 0     # value → different value
    n_still_null = 0      # null → still null
    n_no_change = 0       # value unchanged
    n_no_match = 0        # not in recovery file

    changes_log = []  # for the merge report

    merged_rows = []
    for row in original_rows:
        fn = row["filename"]
        if fn not in recovery_by_fn:
            merged_rows.append(row)
            n_no_match += 1
            continue

        rec = recovery_by_fn[fn]
        new_row = dict(row)  # copy original (preserves metadata)

        # Track whether the headline value changed
        old_days = (row.get("blackout_start_days_before_quarter_end") or "").strip()
        new_days = (rec.get("blackout_start_days_before_quarter_end") or "").strip()

        if old_days == "" and new_days != "":
            n_filled += 1
            changes_log.append(
                f"FILLED   {row.get('ticker', '?'):8s} {row.get('company_name', '?')[:35]:35s} "
                f"null → {new_days}"
            )
        elif old_days != "" and new_days == "":
            n_still_null += 1
            changes_log.append(
                f"NULLED   {row.get('ticker', '?'):8s} {row.get('company_name', '?')[:35]:35s} "
                f"{old_days} → null"
            )
        elif old_days != "" and new_days != "" and old_days != new_days:
            n_overwritten += 1
            changes_log.append(
                f"CHANGED  {row.get('ticker', '?'):8s} {row.get('company_name', '?')[:35]:35s} "
                f"{old_days} → {new_days}"
            )
        elif old_days == "" and new_days == "":
            n_still_null += 1
        else:
            n_no_change += 1

        # Overwrite the policy-level columns
        for col in OVERWRITE_COLUMNS:
            if col in rec:
                new_row[col] = rec[col]

        merged_rows.append(new_row)

    # Write final CSV
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged_rows)

    # Write report
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("MERGE REPORT")
    report_lines.append("=" * 60)
    report_lines.append(f"Total rows in final CSV:           {len(merged_rows)}")
    report_lines.append(f"Rows untouched (not in recovery):  {n_no_match}")
    report_lines.append(f"Rows matched with recovery:        {len(merged_rows) - n_no_match}")
    report_lines.append("")
    report_lines.append("Change summary for blackout_start_days_before_quarter_end:")
    report_lines.append(f"  Filled  (null → value):          {n_filled}")
    report_lines.append(f"  Changed (value → diff value):    {n_overwritten}")
    report_lines.append(f"  Still null after recovery:       {n_still_null}")
    report_lines.append(f"  Unchanged:                       {n_no_change}")
    report_lines.append("")
    report_lines.append("-" * 60)
    report_lines.append("DETAILED CHANGES")
    report_lines.append("-" * 60)
    for line in changes_log:
        report_lines.append(line)

    REPORT_FILE.write_text("\n".join(report_lines), encoding="utf-8")

    # Also print the summary part to console
    print("\n".join(report_lines[:14]))
    print(f"\nWrote {len(merged_rows)} rows to {OUTPUT_CSV}")
    print(f"Wrote detailed merge report to {REPORT_FILE}")


if __name__ == "__main__":
    main()

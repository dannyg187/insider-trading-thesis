#!/usr/bin/env python3
# Merges blackout_summary.csv with blackout_summary_recovery.csv into
# blackout_summary_final.csv — the file the analysis actually reads.
# Recovery values win for the policy columns; metadata always comes from the
# original. Also drops a merge_report.txt saying what moved.
#
#   python 02_merge_recovery.py

import csv
from pathlib import Path

ORIGINAL_CSV = Path("blackout_summary.csv")
RECOVERY_CSV = Path("blackout_summary_recovery.csv")
OUTPUT_CSV = Path("blackout_summary_final.csv")
REPORT_FILE = Path("merge_report.txt")

BAR = "=" * 60
DASH = "-" * 60

# only these get taken from the recovery run; ticker/cik/market_value etc.
# stay as they were
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


def main():
    if not ORIGINAL_CSV.exists():
        print(f"ERROR: {ORIGINAL_CSV} not found in current folder.")
        return
    if not RECOVERY_CSV.exists():
        print(f"ERROR: {RECOVERY_CSV} not found. Did the recovery script finish?")
        return

    with ORIGINAL_CSV.open("r", encoding="utf-8", newline="") as f:
        original_rows = list(csv.DictReader(f))
        fieldnames = list(original_rows[0].keys()) if original_rows else []

    with RECOVERY_CSV.open("r", encoding="utf-8", newline="") as f:
        recovery_rows = list(csv.DictReader(f))

    print(f"Loaded {len(original_rows)} original rows, {len(recovery_rows)} recovery rows")

    recovery_by_fn = {r["filename"]: r for r in recovery_rows}

    n_filled = 0        # null → value
    n_overwritten = 0   # value → different value
    n_still_null = 0    # null → still null
    n_no_change = 0     # value unchanged
    n_no_match = 0      # not in recovery file
    changes_log = []

    merged_rows = []
    for row in original_rows:
        fn = row["filename"]
        if fn not in recovery_by_fn:
            merged_rows.append(row)
            n_no_match += 1
            continue

        rec = recovery_by_fn[fn]
        new_row = dict(row)

        old_days = (row.get("blackout_start_days_before_quarter_end") or "").strip()
        new_days = (rec.get("blackout_start_days_before_quarter_end") or "").strip()

        if old_days == new_days:
            tag = None
            if old_days:
                n_no_change += 1
            else:
                n_still_null += 1
        elif not old_days:
            tag = "FILLED"
            n_filled += 1
        elif not new_days:
            tag = "NULLED"
            n_still_null += 1
        else:
            tag = "CHANGED"
            n_overwritten += 1

        if tag:
            changes_log.append(
                f"{tag:<8} {row.get('ticker', '?'):8s} {row.get('company_name', '?')[:35]:35s} "
                f"{old_days or 'null'} → {new_days or 'null'}"
            )

        for col in OVERWRITE_COLUMNS:
            if col in rec:
                new_row[col] = rec[col]

        merged_rows.append(new_row)

    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged_rows)

    summary = f"""{BAR}
MERGE REPORT
{BAR}
Total rows in final CSV:           {len(merged_rows)}
Rows untouched (not in recovery):  {n_no_match}
Rows matched with recovery:        {len(merged_rows) - n_no_match}

Change summary for blackout_start_days_before_quarter_end:
  Filled  (null → value):          {n_filled}
  Changed (value → diff value):    {n_overwritten}
  Still null after recovery:       {n_still_null}
  Unchanged:                       {n_no_change}

{DASH}"""

    report = "\n".join([summary, "DETAILED CHANGES", DASH] + changes_log)
    REPORT_FILE.write_text(report, encoding="utf-8")

    print(summary)
    print(f"\nWrote {len(merged_rows)} rows to {OUTPUT_CSV}")
    print(f"Wrote detailed merge report to {REPORT_FILE}")


if __name__ == "__main__":
    main()

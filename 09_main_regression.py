#!/usr/bin/env python3
"""
09_main_regression.py — three regression tables (main, robustness, alternative outcomes)

Input:
    analysis_v3.csv (produced by 10_add_factset_ownership.py)

Outputs (./output/):
    table3_main_regression.txt      — 3-column main spec on blackout days
    table4_robustness.txt           — Governance + IV + REIT-inclusion robustness
    table5_alternative_outcomes.txt — Composite score, hedging, pre-clearance
    Also .csv versions of each for pasting into Word

Table structure (following professor's guidance: 3-4 tables answering
specific research questions):

  TABLE 3 — Main: does CEO equity share predict blackout timing?
    (1) Baseline
    (2) + firm controls
    (3) + industry FE  <- preferred spec

  TABLE 4 — Robustness: is the effect robust to alternative specifications?
    (1) Baseline preferred spec (for reference)
    (2) + institutional ownership (io) — addresses governance alternative
    (3) + blockholder ownership (ibh_5pct) — alternative governance measure
    (4) All-executives pooled equity share as IV — better conceptual match
    (5) With log(total_assets) instead of log(market_cap) — includes REITs

  TABLE 5 — Alternative outcomes: does the effect appear in other policy dims?
    (1) Composite restrictiveness (0-3), CEO equity share
    (2) Composite restrictiveness (0-3), all-execs pooled
    (3) Prohibits hedging (LPM), CEO
    (4) Requires pre-clearance (LPM), CEO

Usage:
    python 09_main_regression.py
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings('ignore')

INPUT_CSV = Path("analysis_v3.csv")
OUT_DIR = Path("output")
OUT_DIR.mkdir(exist_ok=True)


def stars(p: float) -> str:
    if p < 0.01:
        return '***'
    if p < 0.05:
        return '**'
    if p < 0.10:
        return '*'
    return ''


def coef_pair(model, var):
    """Return (coef_string_with_stars, se_string) or ('', '') if var missing."""
    if var not in model.params.index:
        return ('', '')
    b = model.params[var]
    se = model.bse[var]
    p = model.pvalues[var]
    return (f"{b:+.3f}{stars(p)}", f"({se:.3f})")


def run_ols(y, X):
    return sm.OLS(y, X).fit(cov_type='HC3')


def make_sic_dummies(series):
    return pd.get_dummies(
        series.astype(int), prefix='sic', drop_first=True
    ).astype(float)


def build_table(rows, models, ind_fe_notes, sample_notes=None):
    """Build a formatted regression table with rows for each variable."""
    ncols = len(models)
    lines = []
    # Header
    col_labels = [f"({i+1})" for i in range(ncols)]
    header = f"{'Variable':<32s}" + ''.join(f"{c:>14s}" for c in col_labels)
    lines.append(header)
    lines.append('-' * len(header))
    # Coef rows
    for var, label in rows:
        pairs = [coef_pair(m, var) for m in models]
        line1 = f"{label:<32s}" + ''.join(f"{p[0]:>14s}" for p in pairs)
        line2 = f"{'':<32s}" + ''.join(f"{p[1]:>14s}" for p in pairs)
        lines.append(line1)
        lines.append(line2)
    lines.append('-' * len(header))
    # Footer
    lines.append(f"{'Industry FE':<32s}"
                 + ''.join(f"{n:>14s}" for n in ind_fe_notes))
    lines.append(f"{'N':<32s}"
                 + ''.join(f"{int(m.nobs):>14d}" for m in models))
    lines.append(f"{'R-squared':<32s}"
                 + ''.join(f"{m.rsquared:>14.3f}" for m in models))
    if sample_notes:
        lines.append(f"{'Sample':<32s}"
                     + ''.join(f"{n:>14s}" for n in sample_notes))
    lines.append('-' * len(header))
    lines.append("HC3-robust SE in parentheses. *** p<0.01, ** p<0.05, * p<0.10.")
    return '\n'.join(lines)


def main() -> None:
    if not INPUT_CSV.exists():
        print(f"ERROR: {INPUT_CSV} not found. Run 10_add_factset_ownership.py first.")
        return

    df = pd.read_csv(INPUT_CSV)
    df['days'] = pd.to_numeric(
        df['blackout_start_days_before_quarter_end'], errors='coerce')
    # Convert binaries to numeric
    for c in ['has_recurring_blackout', 'requires_preclearance',
              'prohibits_hedging']:
        df[c + '_i'] = df[c].astype(str).str.lower().eq('true').astype(int)
    df['restrict_score'] = (df['has_recurring_blackout_i']
                            + df['requires_preclearance_i']
                            + df['prohibits_hedging_i'])

    # =================================================================
    # TABLE 3 — Main regression on blackout days
    # =================================================================
    print("="*80)
    print("TABLE 3 — MAIN REGRESSION")
    print("Dep. var: blackout_start_days_before_quarter_end")
    print("Main spec: log(market cap) as size control")
    print("="*80)

    # Get consistent sample across all three columns of Table 3
    controls_main = ['log_mktcap', 'leverage_w', 'roa_w',
                     'log_firm_age', 'firm_age_missing']
    required_main = ['days', 'equity_share'] + controls_main + ['sic_1digit']
    m3 = df.dropna(subset=required_main).copy()
    y = m3['days']
    sic_d3 = make_sic_dummies(m3['sic_1digit'])

    X_c1 = sm.add_constant(m3[['equity_share']])
    X_c2 = sm.add_constant(m3[['equity_share'] + controls_main])
    X_c3 = sm.add_constant(
        pd.concat([m3[['equity_share'] + controls_main], sic_d3], axis=1))

    r3_1 = run_ols(y, X_c1)
    r3_2 = run_ols(y, X_c2)
    r3_3 = run_ols(y, X_c3)

    table3_rows = [
        ('equity_share', 'Equity pay share (CEO)'),
        ('log_mktcap', 'Log(market cap)'),
        ('leverage_w', 'Leverage'),
        ('roa_w', 'ROA'),
        ('log_firm_age', 'Log(firm age)'),
        ('firm_age_missing', 'Firm age missing'),
    ]
    table3 = build_table(table3_rows, [r3_1, r3_2, r3_3],
                         ind_fe_notes=['No', 'No', 'Yes'])
    print(table3)
    (OUT_DIR / 'table3_main_regression.txt').write_text(table3)
    _save_table_csv(table3_rows, [r3_1, r3_2, r3_3],
                    OUT_DIR / 'table3_main_regression.csv')

    # =================================================================
    # TABLE 4 — Robustness
    # =================================================================
    print("\n\n" + "="*80)
    print("TABLE 4 — ROBUSTNESS")
    print("Dep. var: blackout_days across all columns; specification varies")
    print("="*80)

    # Column 1: preferred spec (same as Table 3 col 3, shown for comparison)
    r4_1 = r3_3

    # Column 2: + io
    req_c2 = required_main + ['io']
    m4_2 = df.dropna(subset=req_c2).copy()
    sic_d4_2 = make_sic_dummies(m4_2['sic_1digit'])
    X = sm.add_constant(
        pd.concat([m4_2[['equity_share'] + controls_main + ['io']],
                   sic_d4_2], axis=1))
    r4_2 = run_ols(m4_2['days'], X)

    # Column 3: + ibh_5pct
    req_c3 = required_main + ['ibh_5pct']
    m4_3 = df.dropna(subset=req_c3).copy()
    sic_d4_3 = make_sic_dummies(m4_3['sic_1digit'])
    X = sm.add_constant(
        pd.concat([m4_3[['equity_share'] + controls_main + ['ibh_5pct']],
                   sic_d4_3], axis=1))
    r4_3 = run_ols(m4_3['days'], X)

    # Column 4: all-executives pooled equity share
    req_c4 = ['days', 'equity_share_pooled'] + controls_main + ['sic_1digit']
    m4_4 = df.dropna(subset=req_c4).copy()
    sic_d4_4 = make_sic_dummies(m4_4['sic_1digit'])
    X = sm.add_constant(
        pd.concat([m4_4[['equity_share_pooled'] + controls_main],
                   sic_d4_4], axis=1))
    r4_4 = run_ols(m4_4['days'], X)

    # Column 5: log(total_assets) instead of log(market_cap)
    # (adds REITs back to the sample)
    controls_at = ['log_at', 'leverage_w', 'roa_w',
                   'log_firm_age', 'firm_age_missing']
    req_c5 = ['days', 'equity_share'] + controls_at + ['sic_1digit']
    m4_5 = df.dropna(subset=req_c5).copy()
    sic_d4_5 = make_sic_dummies(m4_5['sic_1digit'])
    X = sm.add_constant(
        pd.concat([m4_5[['equity_share'] + controls_at], sic_d4_5], axis=1))
    r4_5 = run_ols(m4_5['days'], X)

    table4_rows = [
        ('equity_share', 'Equity pay share (CEO)'),
        ('equity_share_pooled', 'Equity pay share (all execs)'),
        ('io', 'Institutional ownership'),
        ('ibh_5pct', 'Blockholder ownership'),
        ('log_mktcap', 'Log(market cap)'),
        ('log_at', 'Log(total assets)'),
        ('leverage_w', 'Leverage'),
        ('roa_w', 'ROA'),
        ('log_firm_age', 'Log(firm age)'),
        ('firm_age_missing', 'Firm age missing'),
    ]
    table4 = build_table(
        table4_rows,
        [r4_1, r4_2, r4_3, r4_4, r4_5],
        ind_fe_notes=['Yes'] * 5,
        sample_notes=['no REIT', 'no REIT', 'no REIT', 'no REIT', '+REIT'],
    )
    print(table4)
    (OUT_DIR / 'table4_robustness.txt').write_text(table4)
    _save_table_csv(table4_rows, [r4_1, r4_2, r4_3, r4_4, r4_5],
                    OUT_DIR / 'table4_robustness.csv')

    # =================================================================
    # TABLE 5 — Alternative dependent variables
    # =================================================================
    print("\n\n" + "="*80)
    print("TABLE 5 — ALTERNATIVE OUTCOMES")
    print("="*80)

    # Column 1: composite score, CEO equity share
    req = ['equity_share'] + controls_main + ['sic_1digit', 'restrict_score']
    m5_1 = df.dropna(subset=req).copy()
    sic_d5_1 = make_sic_dummies(m5_1['sic_1digit'])
    X = sm.add_constant(
        pd.concat([m5_1[['equity_share'] + controls_main], sic_d5_1], axis=1))
    r5_1 = run_ols(m5_1['restrict_score'], X)

    # Column 2: composite score, all-execs pooled
    req = ['equity_share_pooled'] + controls_main + ['sic_1digit',
                                                    'restrict_score']
    m5_2 = df.dropna(subset=req).copy()
    sic_d5_2 = make_sic_dummies(m5_2['sic_1digit'])
    X = sm.add_constant(
        pd.concat([m5_2[['equity_share_pooled'] + controls_main],
                   sic_d5_2], axis=1))
    r5_2 = run_ols(m5_2['restrict_score'], X)

    # Column 3: prohibits hedging (LPM), CEO
    req = ['equity_share'] + controls_main + ['sic_1digit',
                                              'prohibits_hedging_i']
    m5_3 = df.dropna(subset=req).copy()
    sic_d5_3 = make_sic_dummies(m5_3['sic_1digit'])
    X = sm.add_constant(
        pd.concat([m5_3[['equity_share'] + controls_main], sic_d5_3], axis=1))
    r5_3 = run_ols(m5_3['prohibits_hedging_i'], X)

    # Column 4: requires pre-clearance (LPM), CEO
    req = ['equity_share'] + controls_main + ['sic_1digit',
                                              'requires_preclearance_i']
    m5_4 = df.dropna(subset=req).copy()
    sic_d5_4 = make_sic_dummies(m5_4['sic_1digit'])
    X = sm.add_constant(
        pd.concat([m5_4[['equity_share'] + controls_main], sic_d5_4], axis=1))
    r5_4 = run_ols(m5_4['requires_preclearance_i'], X)

    table5_rows = [
        ('equity_share', 'Equity pay share (CEO)'),
        ('equity_share_pooled', 'Equity pay share (all execs)'),
        ('log_mktcap', 'Log(market cap)'),
        ('leverage_w', 'Leverage'),
        ('roa_w', 'ROA'),
        ('log_firm_age', 'Log(firm age)'),
        ('firm_age_missing', 'Firm age missing'),
    ]
    dv_notes = ['Composite', 'Composite', 'Hedging', 'Pre-clear']
    table5 = build_table(
        table5_rows,
        [r5_1, r5_2, r5_3, r5_4],
        ind_fe_notes=['Yes'] * 4,
        sample_notes=dv_notes,
    )
    print(table5)
    (OUT_DIR / 'table5_alternative_outcomes.txt').write_text(table5)
    _save_table_csv(table5_rows, [r5_1, r5_2, r5_3, r5_4],
                    OUT_DIR / 'table5_alternative_outcomes.csv')

    print(f"\nAll tables saved to {OUT_DIR}/")


def _save_table_csv(rows, models, path):
    """Save a table in CSV form for easy pasting into Word/LaTeX."""
    csv_rows = []
    for var, label in rows:
        row = {'Variable': label}
        for i, model in enumerate(models, 1):
            pair = coef_pair(model, var)
            row[f'({i}) coef'] = pair[0]
            row[f'({i}) se'] = pair[1]
        csv_rows.append(row)
    # Footer rows
    footer_n = {'Variable': 'N'}
    footer_r2 = {'Variable': 'R-squared'}
    for i, model in enumerate(models, 1):
        footer_n[f'({i}) coef'] = int(model.nobs)
        footer_n[f'({i}) se'] = ''
        footer_r2[f'({i}) coef'] = f"{model.rsquared:.3f}"
        footer_r2[f'({i}) se'] = ''
    csv_rows.append(footer_n)
    csv_rows.append(footer_r2)
    pd.DataFrame(csv_rows).to_csv(path, index=False)


if __name__ == "__main__":
    main()

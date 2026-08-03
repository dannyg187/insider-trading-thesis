#!/usr/bin/env python3
"""
09_main_regression.py — four regression tables

Input:
    analysis_v3.csv (from step 10)

Outputs (./output/):
    table3_main_regression.txt / .csv   — 3-column main spec
    table4_robustness.txt / .csv        — governance controls, all-execs, +REIT
    table5_alternative_outcomes.txt/csv — composite, hedging, pre-clearance
    table6_stock_vs_flow.txt / .csv     — CEO holdings extension (NEW)

Changes from previous version:
    - Dropped log(firm_age) and firm_age_missing from all specifications.
      Firm age had 46% imputed values in the regression sample, so per the
      professor's guidance ("wenn es gut populated ist, kannst du log(age)
      als control behalten sonst lass es lieber weg") we drop it.
    - Added Table 6: CEO stock-vs-flow decomposition. Extension proposed by
      the professor — measures CEO accumulated holdings and their scaled
      version (holdings / annual comp), alongside the flow (equity_share).

Table structure:
  TABLE 3 — Main: does CEO equity share predict blackout timing?
  TABLE 4 — Robustness: alternative specifications
  TABLE 5 — Alternative outcomes: does the effect show up in other policy dims?
  TABLE 6 — Stock vs. flow: does CEO accumulated holdings matter alongside
            annual equity grants? [NEW]

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


def build_table(rows, models, ind_fe_notes, sample_notes=None,
                col_labels=None):
    ncols = len(models)
    lines = []
    if col_labels is None:
        col_labels = [f"({i+1})" for i in range(ncols)]
    header = f"{'Variable':<34s}" + ''.join(f"{c:>14s}" for c in col_labels)
    lines.append(header)
    lines.append('-' * len(header))
    for var, label in rows:
        pairs = [coef_pair(m, var) for m in models]
        line1 = f"{label:<34s}" + ''.join(f"{p[0]:>14s}" for p in pairs)
        line2 = f"{'':<34s}" + ''.join(f"{p[1]:>14s}" for p in pairs)
        lines.append(line1)
        lines.append(line2)
    lines.append('-' * len(header))
    lines.append(f"{'Industry FE':<34s}"
                 + ''.join(f"{n:>14s}" for n in ind_fe_notes))
    lines.append(f"{'N':<34s}"
                 + ''.join(f"{int(m.nobs):>14d}" for m in models))
    lines.append(f"{'R-squared':<34s}"
                 + ''.join(f"{m.rsquared:>14.3f}" for m in models))
    if sample_notes:
        lines.append(f"{'Sample':<34s}"
                     + ''.join(f"{n:>14s}" for n in sample_notes))
    lines.append('-' * len(header))
    lines.append("HC3-robust SE in parentheses. *** p<0.01, ** p<0.05, * p<0.10.")
    return '\n'.join(lines)


def save_table_csv(rows, models, path, col_labels=None):
    csv_rows = []
    if col_labels is None:
        col_labels = [f"({i+1})" for i in range(len(models))]
    for var, label in rows:
        row = {'Variable': label}
        for i, model in enumerate(models):
            pair = coef_pair(model, var)
            row[f'{col_labels[i]} coef'] = pair[0]
            row[f'{col_labels[i]} se'] = pair[1]
        csv_rows.append(row)
    for name, values in [('N', [int(m.nobs) for m in models]),
                         ('R-squared',
                          [f"{m.rsquared:.3f}" for m in models])]:
        r = {'Variable': name}
        for i, v in enumerate(values):
            r[f'{col_labels[i]} coef'] = v
            r[f'{col_labels[i]} se'] = ''
        csv_rows.append(r)
    pd.DataFrame(csv_rows).to_csv(path, index=False)


def main() -> None:
    if not INPUT_CSV.exists():
        print(f"ERROR: {INPUT_CSV} not found.")
        return

    df = pd.read_csv(INPUT_CSV)
    df['days'] = pd.to_numeric(
        df['blackout_start_days_before_quarter_end'], errors='coerce')
    for c in ['has_recurring_blackout', 'requires_preclearance',
              'prohibits_hedging']:
        df[c + '_i'] = df[c].astype(str).str.lower().eq('true').astype(int)
    df['restrict_score'] = (df['has_recurring_blackout_i']
                            + df['requires_preclearance_i']
                            + df['prohibits_hedging_i'])

    # Main controls: NO firm_age (per professor's feedback)
    controls_main = ['log_mktcap', 'leverage_w', 'roa_w']

    # =================================================================
    # TABLE 3 — Main
    # =================================================================
    print("="*82)
    print("TABLE 3 — MAIN REGRESSION (blackout days)")
    print("="*82)

    req = ['days', 'equity_share'] + controls_main + ['sic_1digit']
    m3 = df.dropna(subset=req).copy()
    y = m3['days']
    sic_d3 = make_sic_dummies(m3['sic_1digit'])

    X1 = sm.add_constant(m3[['equity_share']])
    X2 = sm.add_constant(m3[['equity_share'] + controls_main])
    X3 = sm.add_constant(
        pd.concat([m3[['equity_share'] + controls_main], sic_d3], axis=1))

    r3_1 = run_ols(y, X1)
    r3_2 = run_ols(y, X2)
    r3_3 = run_ols(y, X3)

    table3_rows = [
        ('equity_share', 'Equity pay share (CEO)'),
        ('log_mktcap', 'Log(market cap)'),
        ('leverage_w', 'Leverage'),
        ('roa_w', 'ROA'),
    ]
    table3 = build_table(table3_rows, [r3_1, r3_2, r3_3],
                         ind_fe_notes=['No', 'No', 'Yes'])
    print(table3)
    (OUT_DIR / 'table3_main_regression.txt').write_text(table3)
    save_table_csv(table3_rows, [r3_1, r3_2, r3_3],
                   OUT_DIR / 'table3_main_regression.csv')

    # =================================================================
    # TABLE 4 — Robustness
    # =================================================================
    print("\n\n" + "="*82)
    print("TABLE 4 — ROBUSTNESS")
    print("="*82)

    r4_1 = r3_3

    req = ['days', 'equity_share'] + controls_main + ['sic_1digit', 'io']
    m = df.dropna(subset=req).copy()
    sd = make_sic_dummies(m['sic_1digit'])
    X = sm.add_constant(
        pd.concat([m[['equity_share'] + controls_main + ['io']], sd], axis=1))
    r4_2 = run_ols(m['days'], X)

    req = ['days', 'equity_share'] + controls_main + ['sic_1digit',
                                                     'ibh_5pct']
    m = df.dropna(subset=req).copy()
    sd = make_sic_dummies(m['sic_1digit'])
    X = sm.add_constant(
        pd.concat([m[['equity_share'] + controls_main + ['ibh_5pct']], sd],
                  axis=1))
    r4_3 = run_ols(m['days'], X)

    req = ['days', 'equity_share_pooled'] + controls_main + ['sic_1digit']
    m = df.dropna(subset=req).copy()
    sd = make_sic_dummies(m['sic_1digit'])
    X = sm.add_constant(
        pd.concat([m[['equity_share_pooled'] + controls_main], sd], axis=1))
    r4_4 = run_ols(m['days'], X)

    controls_at = ['log_at', 'leverage_w', 'roa_w']
    req = ['days', 'equity_share'] + controls_at + ['sic_1digit']
    m = df.dropna(subset=req).copy()
    sd = make_sic_dummies(m['sic_1digit'])
    X = sm.add_constant(
        pd.concat([m[['equity_share'] + controls_at], sd], axis=1))
    r4_5 = run_ols(m['days'], X)

    table4_rows = [
        ('equity_share', 'Equity pay share (CEO)'),
        ('equity_share_pooled', 'Equity pay share (all execs)'),
        ('io', 'Institutional ownership'),
        ('ibh_5pct', 'Blockholder ownership'),
        ('log_mktcap', 'Log(market cap)'),
        ('log_at', 'Log(total assets)'),
        ('leverage_w', 'Leverage'),
        ('roa_w', 'ROA'),
    ]
    table4 = build_table(
        table4_rows, [r4_1, r4_2, r4_3, r4_4, r4_5],
        ind_fe_notes=['Yes'] * 5,
        sample_notes=['no REIT', 'no REIT', 'no REIT', 'no REIT', '+REIT'],
    )
    print(table4)
    (OUT_DIR / 'table4_robustness.txt').write_text(table4)
    save_table_csv(table4_rows, [r4_1, r4_2, r4_3, r4_4, r4_5],
                   OUT_DIR / 'table4_robustness.csv')

    # =================================================================
    # TABLE 5 — Alternative outcomes
    # =================================================================
    print("\n\n" + "="*82)
    print("TABLE 5 — ALTERNATIVE OUTCOMES")
    print("="*82)

    req = ['equity_share'] + controls_main + ['sic_1digit', 'restrict_score']
    m = df.dropna(subset=req).copy()
    sd = make_sic_dummies(m['sic_1digit'])
    X = sm.add_constant(
        pd.concat([m[['equity_share'] + controls_main], sd], axis=1))
    r5_1 = run_ols(m['restrict_score'], X)

    req = ['equity_share_pooled'] + controls_main + ['sic_1digit',
                                                    'restrict_score']
    m = df.dropna(subset=req).copy()
    sd = make_sic_dummies(m['sic_1digit'])
    X = sm.add_constant(
        pd.concat([m[['equity_share_pooled'] + controls_main], sd], axis=1))
    r5_2 = run_ols(m['restrict_score'], X)

    req = ['equity_share'] + controls_main + ['sic_1digit',
                                              'prohibits_hedging_i']
    m = df.dropna(subset=req).copy()
    sd = make_sic_dummies(m['sic_1digit'])
    X = sm.add_constant(
        pd.concat([m[['equity_share'] + controls_main], sd], axis=1))
    r5_3 = run_ols(m['prohibits_hedging_i'], X)

    req = ['equity_share'] + controls_main + ['sic_1digit',
                                              'requires_preclearance_i']
    m = df.dropna(subset=req).copy()
    sd = make_sic_dummies(m['sic_1digit'])
    X = sm.add_constant(
        pd.concat([m[['equity_share'] + controls_main], sd], axis=1))
    r5_4 = run_ols(m['requires_preclearance_i'], X)

    table5_rows = [
        ('equity_share', 'Equity pay share (CEO)'),
        ('equity_share_pooled', 'Equity pay share (all execs)'),
        ('log_mktcap', 'Log(market cap)'),
        ('leverage_w', 'Leverage'),
        ('roa_w', 'ROA'),
    ]
    dv_notes = ['Composite', 'Composite', 'Hedging', 'Pre-clear']
    table5 = build_table(
        table5_rows, [r5_1, r5_2, r5_3, r5_4],
        ind_fe_notes=['Yes'] * 4,
        sample_notes=dv_notes,
    )
    print(table5)
    (OUT_DIR / 'table5_alternative_outcomes.txt').write_text(table5)
    save_table_csv(table5_rows, [r5_1, r5_2, r5_3, r5_4],
                   OUT_DIR / 'table5_alternative_outcomes.csv')

    # =================================================================
    # TABLE 6 — Stock vs. flow decomposition (NEW, professor's suggestion)
    # =================================================================
    print("\n\n" + "="*82)
    print("TABLE 6 — CEO HOLDINGS: STOCK vs FLOW")
    print("Dep. var: blackout_days. All columns use log(mktcap) as size,")
    print("industry FE, and same firm controls as main spec.")
    print("="*82)

    # Column 1: CEO ownership (%) — raw
    req = ['days', 'ceo_share_ownership'] + controls_main + ['sic_1digit']
    m = df.dropna(subset=req).copy()
    sd = make_sic_dummies(m['sic_1digit'])
    X = sm.add_constant(
        pd.concat([m[['ceo_share_ownership'] + controls_main], sd], axis=1))
    r6_1 = run_ols(m['days'], X)

    # Column 2: log(CEO holdings dollar value)
    req = ['days', 'log_ceo_holdings'] + controls_main + ['sic_1digit']
    m = df.dropna(subset=req).copy()
    sd = make_sic_dummies(m['sic_1digit'])
    X = sm.add_constant(
        pd.concat([m[['log_ceo_holdings'] + controls_main], sd], axis=1))
    r6_2 = run_ols(m['days'], X)

    # Column 3: log(holdings / annual comp) — professor's scaled version
    req = ['days', 'log_holdings_to_comp'] + controls_main + ['sic_1digit']
    m = df.dropna(subset=req).copy()
    sd = make_sic_dummies(m['sic_1digit'])
    X = sm.add_constant(
        pd.concat([m[['log_holdings_to_comp'] + controls_main], sd], axis=1))
    r6_3 = run_ols(m['days'], X)

    # Column 4: horse race — flow (equity_share) + stock (log_ceo_holdings)
    req = ['days', 'equity_share', 'log_ceo_holdings'] + controls_main + ['sic_1digit']
    m = df.dropna(subset=req).copy()
    sd = make_sic_dummies(m['sic_1digit'])
    X = sm.add_constant(
        pd.concat([m[['equity_share', 'log_ceo_holdings'] + controls_main],
                   sd], axis=1))
    r6_4 = run_ols(m['days'], X)

    table6_rows = [
        ('equity_share', 'Equity pay share (CEO)'),
        ('ceo_share_ownership', 'CEO share ownership (%)'),
        ('log_ceo_holdings', 'Log(CEO holdings, USD)'),
        ('log_holdings_to_comp', 'Log(holdings / annual comp)'),
        ('log_mktcap', 'Log(market cap)'),
        ('leverage_w', 'Leverage'),
        ('roa_w', 'ROA'),
    ]
    table6 = build_table(
        table6_rows,
        [r6_1, r6_2, r6_3, r6_4],
        ind_fe_notes=['Yes'] * 4,
        col_labels=['(1) Raw %', '(2) log($)', '(3) log(/comp)',
                    '(4) Horse race'],
    )
    print(table6)
    (OUT_DIR / 'table6_stock_vs_flow.txt').write_text(table6)
    save_table_csv(
        table6_rows,
        [r6_1, r6_2, r6_3, r6_4],
        OUT_DIR / 'table6_stock_vs_flow.csv',
        col_labels=['(1)', '(2)', '(3)', '(4)'],
    )

    print(f"\nAll tables saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()

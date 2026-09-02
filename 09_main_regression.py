#!/usr/bin/env python3
# Four regression tables, all from analysis_v3.csv (step 10), written to
# ./output/ as both .txt and .csv:
#
#   Table 3 — main: does CEO equity share predict blackout timing?
#   Table 4 — robustness: governance controls, all-execs measure, +REIT
#   Table 5 — alternative outcomes: composite, hedging, pre-clearance
#   Table 6 — stock vs flow: accumulated CEO holdings alongside annual grants
#
# Changes from the previous version:
#   - log(firm_age) and firm_age_missing dropped from every specification.
#     Firm age was 46% imputed in the regression sample, and per the
#     professor: "wenn es gut populated ist, kannst du log(age) als control
#     behalten sonst lass es lieber weg".
#   - Table 6 is new, also the professor's suggestion: decompose the CEO's
#     accumulated holdings (stock) from the annual equity grant (flow), with
#     a scaled version (holdings / annual comp).
#
#   python 09_main_regression.py

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings('ignore')

INPUT_CSV = Path("analysis_v3.csv")
OUT_DIR = Path("output")
OUT_DIR.mkdir(exist_ok=True)

# no firm_age — see note at the top
CONTROLS_MAIN = ['log_mktcap', 'leverage_w', 'roa_w']
CONTROLS_AT = ['log_at', 'leverage_w', 'roa_w']

BINARY_COLS = ['has_recurring_blackout', 'requires_preclearance',
               'prohibits_hedging']

TABLE3_ROWS = [
    ('equity_share', 'Equity pay share (CEO)'),
    ('log_mktcap', 'Log(market cap)'),
    ('leverage_w', 'Leverage'),
    ('roa_w', 'ROA'),
]

TABLE4_ROWS = [
    ('equity_share', 'Equity pay share (CEO)'),
    ('equity_share_pooled', 'Equity pay share (all execs)'),
    ('io', 'Institutional ownership'),
    ('ibh_5pct', 'Blockholder ownership'),
    ('log_mktcap', 'Log(market cap)'),
    ('log_at', 'Log(total assets)'),
    ('leverage_w', 'Leverage'),
    ('roa_w', 'ROA'),
]

TABLE5_ROWS = [
    ('equity_share', 'Equity pay share (CEO)'),
    ('equity_share_pooled', 'Equity pay share (all execs)'),
    ('log_mktcap', 'Log(market cap)'),
    ('leverage_w', 'Leverage'),
    ('roa_w', 'ROA'),
]

TABLE6_ROWS = [
    ('equity_share', 'Equity pay share (CEO)'),
    ('ceo_share_ownership', 'CEO share ownership (%)'),
    ('log_ceo_holdings', 'Log(CEO holdings, USD)'),
    ('log_holdings_to_comp', 'Log(holdings / annual comp)'),
    ('log_mktcap', 'Log(market cap)'),
    ('leverage_w', 'Leverage'),
    ('roa_w', 'ROA'),
]


def stars(p):
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


def drop_singleton_industries(m, min_firms=2):
    """
    Drop observations belonging to industries with fewer than `min_firms`
    firms in the estimation sample.

    A singleton industry dummy perfectly predicts its one observation,
    giving that observation a leverage (hat) value of exactly 1.0. HC3
    weights each observation's squared residual by 1/(1 - h_ii)^2, so a
    leverage of 1.0 produces a division by zero and propagates infinite
    standard errors through the entire covariance matrix.

    Because a singleton contributes no within-group variation, dropping it
    leaves all coefficient estimates unchanged; only the variance
    calculation is affected. Dropping singleton fixed-effect groups is
    standard practice in applied work for this reason.
    """
    counts = m['sic_1digit'].value_counts()
    keep = counts[counts >= min_firms].index
    return m[m['sic_1digit'].isin(keep)].copy()


def estimation_sample(df, columns):
    return drop_singleton_industries(
        df.dropna(subset=list(columns) + ['sic_1digit']))


def fit(m, dv, regressors, industry_fe=True):
    X = m[list(regressors)]
    if industry_fe:
        X = pd.concat([X, make_sic_dummies(m['sic_1digit'])], axis=1)
    return run_ols(m[dv], sm.add_constant(X))


def fit_spec(df, dv, regressors):
    """Estimate on the sample where dv, the regressors and sic_1digit are all present."""
    return fit(estimation_sample(df, [dv] + list(regressors)), dv, regressors)


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


def emit(name, table, rows, models, csv_col_labels=None):
    print(table)
    (OUT_DIR / f'{name}.txt').write_text(table)
    save_table_csv(rows, models, OUT_DIR / f'{name}.csv',
                   col_labels=csv_col_labels)


def main():
    if not INPUT_CSV.exists():
        print(f"ERROR: {INPUT_CSV} not found.")
        return

    df = pd.read_csv(INPUT_CSV)
    df['days'] = pd.to_numeric(
        df['blackout_start_days_before_quarter_end'], errors='coerce')
    for c in BINARY_COLS:
        df[c + '_i'] = df[c].astype(str).str.lower().eq('true').astype(int)
    df['restrict_score'] = (df['has_recurring_blackout_i']
                            + df['requires_preclearance_i']
                            + df['prohibits_hedging_i'])

    print("="*82)
    print("TABLE 3 — MAIN REGRESSION (blackout days)")
    print("="*82)

    # all three columns share one sample so the N is comparable across them
    m3 = estimation_sample(df, ['days', 'equity_share'] + CONTROLS_MAIN)
    r3_1 = fit(m3, 'days', ['equity_share'], industry_fe=False)
    r3_2 = fit(m3, 'days', ['equity_share'] + CONTROLS_MAIN, industry_fe=False)
    r3_3 = fit(m3, 'days', ['equity_share'] + CONTROLS_MAIN)

    emit('table3_main_regression',
         build_table(TABLE3_ROWS, [r3_1, r3_2, r3_3],
                     ind_fe_notes=['No', 'No', 'Yes']),
         TABLE3_ROWS, [r3_1, r3_2, r3_3])

    print("\n\n" + "="*82)
    print("TABLE 4 — ROBUSTNESS")
    print("="*82)

    r4_1 = r3_3
    r4_2 = fit_spec(df, 'days', ['equity_share'] + CONTROLS_MAIN + ['io'])
    r4_3 = fit_spec(df, 'days', ['equity_share'] + CONTROLS_MAIN + ['ibh_5pct'])
    r4_4 = fit_spec(df, 'days', ['equity_share_pooled'] + CONTROLS_MAIN)
    r4_5 = fit_spec(df, 'days', ['equity_share'] + CONTROLS_AT)
    models4 = [r4_1, r4_2, r4_3, r4_4, r4_5]

    emit('table4_robustness',
         build_table(TABLE4_ROWS, models4,
                     ind_fe_notes=['Yes'] * 5,
                     sample_notes=['no REIT', 'no REIT', 'no REIT',
                                   'no REIT', '+REIT']),
         TABLE4_ROWS, models4)

    print("\n\n" + "="*82)
    print("TABLE 5 — ALTERNATIVE OUTCOMES")
    print("="*82)

    r5_1 = fit_spec(df, 'restrict_score', ['equity_share'] + CONTROLS_MAIN)
    r5_2 = fit_spec(df, 'restrict_score', ['equity_share_pooled'] + CONTROLS_MAIN)
    r5_3 = fit_spec(df, 'prohibits_hedging_i', ['equity_share'] + CONTROLS_MAIN)
    r5_4 = fit_spec(df, 'requires_preclearance_i', ['equity_share'] + CONTROLS_MAIN)
    models5 = [r5_1, r5_2, r5_3, r5_4]

    emit('table5_alternative_outcomes',
         build_table(TABLE5_ROWS, models5,
                     ind_fe_notes=['Yes'] * 4,
                     sample_notes=['Composite', 'Composite', 'Hedging',
                                   'Pre-clear']),
         TABLE5_ROWS, models5)

    print("\n\n" + "="*82)
    print("TABLE 6 — CEO HOLDINGS: STOCK vs FLOW")
    print("Dep. var: blackout_days. All columns use log(mktcap) as size,")
    print("industry FE, and same firm controls as main spec.")
    print("="*82)

    r6_1 = fit_spec(df, 'days', ['ceo_share_ownership'] + CONTROLS_MAIN)
    r6_2 = fit_spec(df, 'days', ['log_ceo_holdings'] + CONTROLS_MAIN)
    r6_3 = fit_spec(df, 'days', ['log_holdings_to_comp'] + CONTROLS_MAIN)
    r6_4 = fit_spec(df, 'days',
                    ['equity_share', 'log_ceo_holdings'] + CONTROLS_MAIN)
    models6 = [r6_1, r6_2, r6_3, r6_4]

    emit('table6_stock_vs_flow',
         build_table(TABLE6_ROWS, models6,
                     ind_fe_notes=['Yes'] * 4,
                     col_labels=['(1) Raw %', '(2) log($)', '(3) log(/comp)',
                                 '(4) Horse race']),
         TABLE6_ROWS, models6,
         csv_col_labels=['(1)', '(2)', '(3)', '(4)'])

    print(f"\nAll tables saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()

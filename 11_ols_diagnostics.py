#!/usr/bin/env python3
"""
11_ols_diagnostics.py — test the OLS assumptions for the main specification

Input:
    analysis_v3.csv

Outputs (./output/):
    table7_ols_diagnostics.txt / .csv  — diagnostic test results
    figure5_residual_diagnostics.png   — 4-panel residual plots

Tests performed on the main specification (Table 3, column 3):

  1. No perfect multicollinearity     — variance inflation factors
  2. Homoskedasticity                 — Breusch-Pagan and White tests
  3. Normality of the error term      — Jarque-Bera test, skewness, kurtosis
  4. Correct functional form          — Ramsey RESET test
  5. No single influential observation— Cook's distance

Assumptions that cannot be tested with cross-sectional data are noted in
the output: exogeneity of the regressors (zero conditional mean) is not
directly testable and is discussed as a limitation in the thesis.

Usage:
    python 11_ols_diagnostics.py
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_white, linear_reset
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import jarque_bera

warnings.filterwarnings('ignore')

INPUT_CSV = Path("analysis_v3.csv")
OUT_DIR = Path("output")
OUT_DIR.mkdir(exist_ok=True)

CONTROLS = ['log_mktcap', 'leverage_w', 'roa_w']


def main() -> None:
    if not INPUT_CSV.exists():
        print(f"ERROR: {INPUT_CSV} not found.")
        return

    df = pd.read_csv(INPUT_CSV)
    df['days'] = pd.to_numeric(
        df['blackout_start_days_before_quarter_end'], errors='coerce')

    req = ['days', 'equity_share'] + CONTROLS + ['sic_1digit']
    m = df.dropna(subset=req).copy()
    counts = m['sic_1digit'].value_counts()
    m = m[m['sic_1digit'].isin(counts[counts >= 2].index)].copy()

    sic_d = pd.get_dummies(
        m['sic_1digit'].astype(int), prefix='sic', drop_first=True).astype(float)
    X = sm.add_constant(pd.concat([m[['equity_share'] + CONTROLS], sic_d], axis=1))
    y = m['days']

    # Diagnostics are computed on the non-robust fit; HC3 affects only the
    # standard errors, not the residuals or the fitted values.
    r = sm.OLS(y, X).fit()

    lines = []
    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 68)
    out("OLS DIAGNOSTICS — MAIN SPECIFICATION (Table 3, column 3)")
    out("=" * 68)
    out(f"Observations: {int(r.nobs)}")
    out(f"Regressors (incl. constant and industry dummies): {X.shape[1]}")
    out(f"R-squared: {r.rsquared:.4f}")
    out()

    # --- 1. Multicollinearity -------------------------------------------
    out("1. MULTICOLLINEARITY — variance inflation factors")
    out("-" * 68)
    Xv = X.astype(float).values
    vifs = [(X.columns[i], variance_inflation_factor(Xv, i))
            for i in range(X.shape[1])]
    for name, v in vifs:
        if name == 'const':
            continue
        flag = "   <-- above 10" if v > 10 else ""
        out(f"   {name:22s} {v:8.3f}{flag}")
    main_max = max(v for n, v in vifs if n in ['equity_share'] + CONTROLS)
    out(f"   Maximum VIF among the main regressors: {main_max:.3f}")
    out()

    # --- 2. Homoskedasticity --------------------------------------------
    out("2. HOMOSKEDASTICITY")
    out("-" * 68)
    bp_lm, bp_p, bp_f, bp_fp = het_breuschpagan(r.resid, X.astype(float))
    out(f"   Breusch-Pagan:  LM = {bp_lm:.3f},  p = {bp_p:.4f}")
    try:
        wh_lm, wh_p, wh_f, wh_fp = het_white(r.resid, X.astype(float))
        out(f"   White:          LM = {wh_lm:.3f},  p = {wh_p:.4f}")
    except Exception:
        out("   White test: not computable for this design matrix")
    out()

    # --- 3. Normality ----------------------------------------------------
    out("3. NORMALITY OF THE ERROR TERM")
    out("-" * 68)
    jb, jb_p, skew, kurt = jarque_bera(r.resid)
    out(f"   Jarque-Bera:  JB = {jb:.3f},  p = {jb_p:.4f}")
    out(f"   Skewness = {skew:.3f}   Kurtosis = {kurt:.3f}")
    out()

    # --- 4. Functional form ----------------------------------------------
    out("4. FUNCTIONAL FORM")
    out("-" * 68)
    reset = linear_reset(r, power=2, use_f=True)
    out(f"   Ramsey RESET (squared fitted values): "
        f"F = {reset.fvalue:.3f},  p = {reset.pvalue:.4f}")
    out()

    # --- 5. Influential observations -------------------------------------
    out("5. INFLUENTIAL OBSERVATIONS")
    out("-" * 68)
    cooks = r.get_influence().cooks_distance[0]
    thresh = 4 / len(m)
    out(f"   Cook's distance threshold (4/n): {thresh:.5f}")
    out(f"   Observations above threshold: {(cooks > thresh).sum()} "
        f"({100 * (cooks > thresh).sum() / len(m):.1f}%)")
    out(f"   Maximum Cook's distance: {cooks.max():.4f}")
    out(f"   Observations with Cook's distance above 1: {(cooks > 1).sum()}")
    out()

    out("-" * 68)
    out("Exogeneity of the regressors cannot be tested directly with")
    out("cross-sectional data and is addressed as a limitation.")
    out("-" * 68)

    (OUT_DIR / 'table7_ols_diagnostics.txt').write_text("\n".join(lines))

    # CSV summary for pasting into the thesis
    summary = pd.DataFrame([
        {'Assumption': 'No multicollinearity', 'Test': 'Max VIF (main regressors)',
         'Statistic': round(main_max, 3), 'p-value': ''},
        {'Assumption': 'Homoskedasticity', 'Test': 'Breusch-Pagan',
         'Statistic': round(bp_lm, 3), 'p-value': round(bp_p, 4)},
        {'Assumption': 'Homoskedasticity', 'Test': 'White',
         'Statistic': round(wh_lm, 3), 'p-value': round(wh_p, 4)},
        {'Assumption': 'Normality of errors', 'Test': 'Jarque-Bera',
         'Statistic': round(jb, 3), 'p-value': round(jb_p, 4)},
        {'Assumption': 'Correct functional form', 'Test': 'Ramsey RESET',
         'Statistic': round(reset.fvalue, 3), 'p-value': round(reset.pvalue, 4)},
        {'Assumption': 'No influential outliers', 'Test': "Max Cook's distance",
         'Statistic': round(cooks.max(), 4), 'p-value': ''},
    ])
    summary.to_csv(OUT_DIR / 'table7_ols_diagnostics.csv', index=False)

    # --- Residual plots ---------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    color = '#2c3e50'

    axes[0, 0].scatter(r.fittedvalues, r.resid, s=12, alpha=0.4,
                       color=color, edgecolors='none')
    axes[0, 0].axhline(0, color='crimson', linewidth=1, linestyle='--')
    axes[0, 0].set_xlabel('Fitted values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs fitted')

    stats.probplot(r.resid, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Normal Q-Q plot')
    axes[0, 1].get_lines()[0].set_markerfacecolor(color)
    axes[0, 1].get_lines()[0].set_markeredgecolor('none')
    axes[0, 1].get_lines()[0].set_markersize(4)
    axes[0, 1].get_lines()[1].set_color('crimson')

    axes[1, 0].hist(r.resid, bins=30, color=color, edgecolor='white')
    axes[1, 0].set_xlabel('Residual')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of residuals')

    markerline, stemlines, baseline = axes[1, 1].stem(
        np.arange(len(cooks)), cooks, markerfmt=',', linefmt='-', basefmt=' ')
    plt.setp(stemlines, color=color, linewidth=0.6)
    plt.setp(markerline, color=color)
    axes[1, 1].axhline(thresh, color='crimson', linewidth=1, linestyle='--',
                       label=f'4/n = {thresh:.4f}')
    axes[1, 1].set_xlabel('Observation')
    axes[1, 1].set_ylabel("Cook's distance")
    axes[1, 1].set_title("Cook's distance")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'figure5_residual_diagnostics.png',
                dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved diagnostics to {OUT_DIR}/")


if __name__ == "__main__":
    main()

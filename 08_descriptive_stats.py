#!/usr/bin/env python3
# Tables and figures for the Data section, built from analysis_v3.csv (step 10).
#
# Everything is computed on the main regression sample: firms with non-missing
# blackout_days, equity_share and the main controls (log_mktcap, leverage_w,
# roa_w), i.e. the Table 3 sample.
#
# Changed from the previous version:
#   - log(firm_age) dropped from the required-variable list; per the
#     professor's feedback it's no longer used as a control.
#   - CEO holdings variables added to Table 1 (they feed the Table 6 extension).
#
# Writes table1/2/3 csvs and figure1-4 pngs into ./output/.
#
#   python 08_descriptive_stats.py

import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', category=FutureWarning)

INPUT_CSV = Path("analysis_v3.csv")
OUT_DIR = Path("output")
OUT_DIR.mkdir(exist_ok=True)

COLOR = '#2c3e50'

BINARY_COLS = ['has_recurring_blackout', 'has_ad_hoc_blackout',
               'requires_preclearance', 'prohibits_hedging']

# no firm_age here — see note at the top
MAIN_REQUIRED = ['equity_share', 'log_mktcap', 'leverage_w', 'roa_w',
                 'sic_1digit']

SUMMARY_VARS = [
    # dependent variables
    ('days', 'Blackout days before quarter end'),
    ('has_recurring_blackout', 'Has recurring blackout (0/1)'),
    ('requires_preclearance', 'Requires pre-clearance (0/1)'),
    ('prohibits_hedging', 'Prohibits hedging (0/1)'),
    # compensation (main IVs)
    ('equity_share', 'CEO equity pay share'),
    ('equity_share_pooled', 'All-execs equity pay share (pooled)'),
    # CEO holdings (Table 6 extension)
    ('ceo_share_ownership', 'CEO share ownership (%)'),
    ('ceo_holdings_musd', 'CEO holdings value (USD millions)'),
    ('holdings_to_comp', 'CEO holdings / annual comp'),
    # firm characteristics
    ('log_mktcap', 'Log(market cap, USD millions)'),
    ('mktcap', 'Market cap (USD millions)'),
    ('log_at', 'Log(total assets) [robustness]'),
    ('leverage_w', 'Leverage (winsorized)'),
    ('roa_w', 'ROA (winsorized)'),
    # governance / institutional ownership
    ('io', 'Institutional ownership (%)'),
    ('ibh_5pct', 'Blockholder ownership (5%+)'),
    ('top5', 'Top-5 investor ownership'),
    ('herf', 'Herfindahl concentration index'),
    ('nbr_firms', 'Number of institutional owners'),
    # CEO demographics
    ('ceo_age', 'CEO age'),
    ('ceo_tenure_years', 'CEO tenure (years)'),
]

CORR_VARS = ['days', 'equity_share', 'equity_share_pooled',
             'ceo_share_ownership', 'log_ceo_holdings', 'log_mktcap',
             'log_at', 'leverage_w', 'roa_w', 'io', 'ibh_5pct',
             'has_recurring_blackout', 'requires_preclearance',
             'prohibits_hedging']

SIC1_LABELS = {
    0: '0 — Agriculture/Forestry/Fishing',
    1: '1 — Mining & Construction',
    2: '2 — Manufacturing (low SIC)',
    3: '3 — Manufacturing (high SIC)',
    4: '4 — Transport/Communication/Utilities',
    5: '5 — Wholesale & Retail Trade',
    6: '6 — Finance/Insurance/Real Estate',
    7: '7 — Services (business)',
    8: '8 — Services (health, legal, ed.)',
    9: '9 — Public Administration',
}

BIN_VARS = [
    ('has_recurring_blackout', 'Has recurring\nblackout'),
    ('has_ad_hoc_blackout', 'Has ad-hoc\nblackout'),
    ('requires_preclearance', 'Requires\npre-clearance'),
    ('prohibits_hedging', 'Prohibits\nhedging'),
]


def hist(ax, s, xlabel, title, median_line=False):
    ax.hist(s.dropna(), bins=20, color=COLOR, edgecolor='white')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Firms')
    ax.set_title(title)
    if median_line:
        med = s.median()
        ax.axvline(med, color='crimson', linestyle='--', linewidth=1,
                   label=f"Median = {med:.2f}")
        ax.legend()


def main():
    if not INPUT_CSV.exists():
        print(f"ERROR: {INPUT_CSV} not found.")
        return

    df = pd.read_csv(INPUT_CSV)
    df['days'] = pd.to_numeric(
        df['blackout_start_days_before_quarter_end'], errors='coerce'
    )
    for c in BINARY_COLS:
        df[c] = df[c].astype(str).str.lower().eq('true').astype(int)

    reg = df.dropna(subset=['days'] + MAIN_REQUIRED).copy()

    # A singleton industry dummy perfectly predicts its one observation and
    # breaks HC3 standard errors, so 09_main_regression.py drops those firms.
    # Do the same here or Table 1 describes a different sample than Table 3.
    counts = reg['sic_1digit'].value_counts()
    keep = counts[counts >= 2].index
    n_before = len(reg)
    reg = reg[reg['sic_1digit'].isin(keep)].copy()
    if n_before != len(reg):
        print(f"Dropped {n_before - len(reg)} firm(s) in singleton industries")

    print(f"Main regression sample: {len(reg)} firms")

    rows = []
    for v, label in SUMMARY_VARS:
        if v not in reg.columns:
            continue
        s = pd.to_numeric(reg[v], errors='coerce').dropna()
        if len(s) == 0:
            continue
        rows.append({
            'Variable': label,
            'N': len(s),
            'Mean': round(s.mean(), 3),
            'SD': round(s.std(), 3),
            'p25': round(s.quantile(0.25), 3),
            'Median': round(s.median(), 3),
            'p75': round(s.quantile(0.75), 3),
            'Min': round(s.min(), 3),
            'Max': round(s.max(), 3),
        })
    table1 = pd.DataFrame(rows)
    table1.to_csv(OUT_DIR / "table1_summary_stats.csv", index=False)
    print(f"\nTable 1 — Summary statistics (n={len(reg)}):")
    print(table1.to_string(index=False))

    corr_vars = [v for v in CORR_VARS if v in reg.columns]
    corr = reg[corr_vars].corr().round(3)
    corr.to_csv(OUT_DIR / "table2_correlation_matrix.csv")
    print(f"\nTable 2 — Correlation matrix saved")

    ind = reg['sic_1digit'].dropna().astype(int).value_counts().sort_index()
    table3 = pd.DataFrame({
        'SIC 1-digit': [SIC1_LABELS.get(i, f"{i}") for i in ind.index],
        'N firms': ind.values,
        'Share (%)': (100 * ind.values / ind.sum()).round(1),
    })
    table3.to_csv(OUT_DIR / "table3_industry_breakdown.csv", index=False)
    print(f"\nTable 3 — Industry breakdown:")
    print(table3.to_string(index=False))

    # figure 1 — six distribution panels
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    hist(axes[0, 0], reg['days'],
         'Days before quarter-end', 'Blackout timing', median_line=True)
    hist(axes[0, 1], reg['equity_share'],
         'CEO equity share', 'CEO equity pay share')
    hist(axes[0, 2], reg['log_mktcap'],
         'Log(market cap)', 'Firm size')
    hist(axes[1, 0], reg['io'],
         'Institutional ownership (%)', 'Institutional ownership')
    hist(axes[1, 1], reg['roa_w'],
         'ROA (winsorized)', 'Profitability')
    hist(axes[1, 2], reg['leverage_w'],
         'Leverage (winsorized)', 'Leverage')

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'figure1_distributions.png',
                dpi=150, bbox_inches='tight')
    plt.close()

    # figure 2 — industry breakdown
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(ind)), ind.values, color=COLOR, edgecolor='white')
    ax.set_xticks(range(len(ind)))
    ax.set_xticklabels(ind.index, rotation=0)
    ax.set_xlabel('SIC 1-digit')
    ax.set_ylabel('Number of firms')
    ax.set_title(f'Sample composition by 1-digit SIC (n={len(reg)})')
    for i, v in enumerate(ind.values):
        ax.text(i, v + 2, str(v), ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'figure2_industry_breakdown.png',
                dpi=150, bbox_inches='tight')
    plt.close()

    # figure 3 — equity share vs blackout timing
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(reg['equity_share'], reg['days'], s=15, alpha=0.4,
               color=COLOR, edgecolors='none')
    x = reg['equity_share'].dropna()
    y = reg.loc[x.index, 'days']
    valid = y.notna()
    if valid.sum() > 2:
        coef = np.polyfit(x[valid], y[valid], 1)
        xx = np.linspace(x.min(), x.max(), 100)
        ax.plot(xx, np.polyval(coef, xx), color='crimson', linewidth=2,
                label=f'Linear fit (slope={coef[0]:+.2f})')
    ax.set_xlabel('CEO equity pay share')
    ax.set_ylabel('Days before quarter-end (blackout starts)')
    ax.set_title('Equity pay share vs blackout timing (raw)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'figure3_scatter_main.png',
                dpi=150, bbox_inches='tight')
    plt.close()

    # figure 4 — prevalence of the binary policy features
    fig, ax = plt.subplots(figsize=(8, 4))
    names = [lbl for _, lbl in BIN_VARS]
    vals = [100 * reg[v].mean() for v, _ in BIN_VARS]
    bars = ax.bar(names, vals, color=COLOR, edgecolor='white')
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 1, f'{v:.0f}%',
                ha='center', fontsize=10)
    ax.set_ylabel('% of firms (main regression sample)')
    ax.set_ylim(0, 110)
    ax.set_title('Prevalence of policy features')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'figure4_binary_features.png',
                dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nAll outputs saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()

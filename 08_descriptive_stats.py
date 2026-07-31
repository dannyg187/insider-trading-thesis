#!/usr/bin/env python3
"""
08_descriptive_stats.py — tables and figures for the Data section

Uses analysis_v3.csv (with FactSet Ownership Summary variables merged in).
Descriptive statistics are computed on the MAIN regression sample:
firms with non-missing blackout_days, equity_share, and all main controls
(including log_mktcap), which excludes REITs.

Input:
    analysis_v3.csv (from step 10)

Outputs (./output/):
    table1_summary_stats.csv         — Summary stats for the main sample
    table2_correlation_matrix.csv    — Pearson correlations
    table3_industry_breakdown.csv    — Firms per 1-digit SIC industry
    figure1_distributions.png        — 6-panel histogram of key variables
    figure2_industry_breakdown.png   — Bar chart of industries
    figure3_scatter_main.png         — Scatter: equity_share vs blackout_days
    figure4_binary_features.png      — Bar chart: % firms with each feature

Usage:
    python 08_descriptive_stats.py
"""

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


def main() -> None:
    if not INPUT_CSV.exists():
        print(f"ERROR: {INPUT_CSV} not found.")
        return

    df = pd.read_csv(INPUT_CSV)
    df['days'] = pd.to_numeric(
        df['blackout_start_days_before_quarter_end'], errors='coerce'
    )
    for c in ['has_recurring_blackout', 'has_ad_hoc_blackout',
              'requires_preclearance', 'prohibits_hedging']:
        df[c] = df[c].astype(str).str.lower().eq('true').astype(int)

    # Main regression sample: firms with all variables for the main spec
    # (log_mktcap-based, i.e. FactSet available -> excludes REITs)
    main_required = ['equity_share', 'log_mktcap', 'leverage_w', 'roa_w',
                     'log_firm_age', 'sic_1digit']
    reg = df.dropna(subset=['days'] + main_required).copy()
    print(f"Main regression sample: {len(reg)} firms")
    print(f"(Firms with FactSet ownership summary AND all controls)")

    # ===================================================================
    # TABLE 1 — Summary statistics on the main sample
    # ===================================================================
    summary_vars = [
        # Dependent variables
        ('days', 'Blackout days before quarter end'),
        ('has_recurring_blackout', 'Has recurring blackout (0/1)'),
        ('requires_preclearance', 'Requires pre-clearance (0/1)'),
        ('prohibits_hedging', 'Prohibits hedging (0/1)'),
        # Compensation (main IVs)
        ('equity_share', 'CEO equity pay share'),
        ('equity_share_pooled', 'All-execs equity pay share (pooled)'),
        # Firm characteristics
        ('log_mktcap', 'Log(market cap, USD millions)'),
        ('mktcap', 'Market cap (USD millions)'),
        ('log_at', 'Log(total assets)'),
        ('leverage_w', 'Leverage (winsorized)'),
        ('roa_w', 'ROA (winsorized)'),
        ('log_firm_age', 'Log(firm age + 1)'),
        # Governance / institutional ownership
        ('io', 'Institutional ownership (%)'),
        ('ibh_5pct', 'Blockholder ownership (5%+)'),
        ('top5', 'Top-5 investor ownership'),
        ('herf', 'Herfindahl concentration index'),
        ('nbr_firms', 'Number of institutional owners'),
        # CEO characteristics
        ('ceo_age', 'CEO age'),
        ('ceo_tenure_years', 'CEO tenure (years)'),
        ('ceo_share_ownership', 'CEO share ownership (%)'),
    ]
    rows = []
    for v, label in summary_vars:
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

    # ===================================================================
    # TABLE 2 — Correlation matrix
    # ===================================================================
    corr_vars = ['days', 'equity_share', 'equity_share_pooled', 'log_mktcap',
                 'log_at', 'leverage_w', 'roa_w', 'log_firm_age', 'io',
                 'ibh_5pct', 'has_recurring_blackout', 'requires_preclearance',
                 'prohibits_hedging']
    corr_vars = [v for v in corr_vars if v in reg.columns]
    corr = reg[corr_vars].corr().round(3)
    corr.to_csv(OUT_DIR / "table2_correlation_matrix.csv")
    print(f"\nTable 2 — Correlation matrix saved")

    # ===================================================================
    # TABLE 3 — Industry breakdown
    # ===================================================================
    sic1_labels = {
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
    ind = reg['sic_1digit'].dropna().astype(int).value_counts().sort_index()
    table3 = pd.DataFrame({
        'SIC 1-digit': [sic1_labels.get(i, f"{i}") for i in ind.index],
        'N firms': ind.values,
        'Share (%)': (100 * ind.values / ind.sum()).round(1),
    })
    table3.to_csv(OUT_DIR / "table3_industry_breakdown.csv", index=False)
    print(f"\nTable 3 — Industry breakdown:")
    print(table3.to_string(index=False))

    # ===================================================================
    # FIGURE 1 — 6-panel distributions
    # ===================================================================
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    color = '#2c3e50'

    def hist(ax, s, xlabel, title, median_line=False):
        ax.hist(s.dropna(), bins=20, color=color, edgecolor='white')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Firms')
        ax.set_title(title)
        if median_line:
            med = s.median()
            ax.axvline(med, color='crimson', linestyle='--', linewidth=1,
                       label=f"Median = {med:.2f}")
            ax.legend()

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

    # ===================================================================
    # FIGURE 2 — Industry breakdown
    # ===================================================================
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(ind)), ind.values, color=color, edgecolor='white')
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

    # ===================================================================
    # FIGURE 3 — Scatter of equity_share vs blackout_days
    # ===================================================================
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(reg['equity_share'], reg['days'], s=15, alpha=0.4,
               color=color, edgecolors='none')
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
    ax.set_title('Equity pay share vs blackout timing (raw, no controls)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'figure3_scatter_main.png',
                dpi=150, bbox_inches='tight')
    plt.close()

    # ===================================================================
    # FIGURE 4 — Binary policy features
    # ===================================================================
    fig, ax = plt.subplots(figsize=(8, 4))
    bin_vars = [
        ('has_recurring_blackout', 'Has recurring\nblackout'),
        ('has_ad_hoc_blackout', 'Has ad-hoc\nblackout'),
        ('requires_preclearance', 'Requires\npre-clearance'),
        ('prohibits_hedging', 'Prohibits\nhedging'),
    ]
    names = [lbl for _, lbl in bin_vars]
    vals = [100 * reg[v].mean() for v, _ in bin_vars]
    bars = ax.bar(names, vals, color=color, edgecolor='white')
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

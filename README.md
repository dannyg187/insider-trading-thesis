# Insider Trading Policies and Executive Compensation

## Overview

The empirical analysis tests whether firms with higher equity-based executive compensation impose more restrictive insider trading policies. Policy features are extracted from SEC EX-19 filings using an LLM-based text extraction pipeline. Compensation and firm-level controls come from WRDS (ExecComp, Compustat) and FactSet (Ownership Summary).

## Main findings

- On blackout timing, the CEO-only equity-share coefficient is positive across all specifications but not statistically distinguishable from zero after industry fixed effects are added.
- When the equity-pay measure is broadened to all named executives — matching the scope of the insider trading policy itself — the coefficient reaches marginal significance (β ≈ +4.2, p ≈ 0.10 in the main specification).
- The result is robust to controlling for institutional ownership and blockholder ownership, addressing the concern that a general governance-quality factor drives both compensation structure and policy strictness.
- Decomposing CEO equity exposure into flows (annual grants) and stocks (accumulated holdings) shows that the flow measure retains its coefficient while the stock measure is close to zero. This suggests firm policy design responds to the incentives being actively created rather than to accumulated CEO wealth.

Overall the evidence provides qualified support for the "complements" view: equity pay and policy restrictiveness move together, with the effect concentrated in the broader executive-team measure that matches the scope of firm-wide policies.

## Data sources (not included in this repository)

1. **EX-19 policy filings** — 1,073 HTML documents from SEC EDGAR, provided by the supervisor. Each firm's insider trading policy is filed as Exhibit 19 to its annual 10-K under Item 408 of Regulation S-K.
2. **ExecComp** — CEO and named-executive compensation data, WRDS.
3. **Compustat Annual Fundamentals** — firm-level accounting variables, WRDS.
4. **FactSet Ownership Summary** — market capitalization and institutional ownership data, WRDS.

## Pipeline structure

Scripts are numbered by dependency order rather than strictly by execution order (see the run sequence below):

```
01_read_files.py               Extract policy features from EX-19 files via GPT-5.4 nano
                               (based on the supervisor's script, with the JSON schema
                               extended to also capture pre-clearance and hedging rules)
01b_recover_missing.py         Re-run on flagged files with a sharpened prompt
02_merge_recovery.py           Merge recovery values into the main extraction
03_dedup.py                    Deduplicate to one row per firm
04_build_compensation.py       Build CEO equity share and all-executives pooled measure
05_merge_policy_comp.py        Merge policy + compensation
06_build_compustat.py          Build firm-level controls (size, leverage, ROA, industry)
07_build_analysis_dataset.py   Produce analysis_v2.csv
10_add_factset_ownership.py    Merge FactSet Ownership Summary and construct CEO
                               holdings variables → analysis_v3.csv
08_descriptive_stats.py        Produce Table 1, correlation matrix, figures
09_main_regression.py          Produce Tables 3, 4, 5, and 6
10_add_factset_ownership.py    Merge FactSet Ownership Summary and holdings variables
11_ols_diagnostics.py          OLS assumption tests and inference detail
```

Note that 08 and 09 must be run *after* 10, because they rely on FactSet-derived variables (market capitalization, institutional ownership, and CEO holdings).

## Outputs

The `output/` folder contains all tables and figures for the thesis:

- `table1_summary_stats.csv` — summary statistics for the main regression sample
- `table2_correlation_matrix.csv` — Pearson correlations across key variables
- `table3_industry_breakdown.csv` — sample composition by 1-digit SIC
- `table3_main_regression.txt` / `.csv` — main 3-column regression (blackout days on CEO equity share)
- `table4_robustness.txt` / `.csv` — governance controls, all-executives measure, log(assets) alternative including REITs
- `table5_alternative_outcomes.txt` / `.csv` — composite restrictiveness score, hedging LPM, pre-clearance LPM
- `table6_stock_vs_flow.txt` / `.csv` — CEO holdings extension: raw ownership, log dollar value, holdings scaled by annual compensation, and a horse race against the flow measure
- `figure1_distributions.png` — 6-panel histograms of key variables
- `figure2_industry_breakdown.png` — sample composition bar chart
- `figure3_scatter_main.png` — equity share vs blackout days
- `figure4_binary_features.png` — prevalence of policy features
- `table7_ols_diagnostics.txt` / `.csv` — VIF, Breusch-Pagan, White, Jarque-Bera, Ramsey RESET, Cook's distance, plus confidence intervals, joint significance tests, minimum detectable effect and IQR-scaled effect sizes
- `figure5_residual_diagnostics.png` — 4-panel residual plots

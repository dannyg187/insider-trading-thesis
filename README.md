# Insider Trading Policies and Executive Compensation

Bachelor thesis empirical pipeline. This repository contains the full Python code used to construct the analysis dataset and produce the regression tables and figures. Data files are excluded due to WRDS and SEC licensing restrictions; see the Data sources section below for how to obtain them.

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

Reproducing this analysis requires a valid WRDS subscription and access to the specific tables above.

## Pipeline structure

Scripts are numbered by dependency order rather than strictly by execution order (see the run sequence below):

```
01_read_files.py               Extract policy features from EX-19 files via GPT-4o-mini
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
```

Note that 08 and 09 must be run *after* 10, because they rely on FactSet-derived variables (market capitalization, institutional ownership, and CEO holdings).

## Setup

Python 3.10 or later. Install dependencies:

```
pip install -r requirements.txt
```

Create a `.env` file in the project root with your OpenAI API key (needed only for scripts 01 and 01b):

```
OPENAI_API_KEY=sk-...
```

Place data files in the project root:

```
ex19_policies/*.htm
ex19_policies/ex19_metadata.csv
execcomp_russell3000_2024.csv
compustat_russell3000_2024.csv
factset_ownership_summary_q4_2024.csv
```

Run the pipeline in the following order:

```
python 01_read_files.py
python 01b_recover_missing.py
python 02_merge_recovery.py
python 03_dedup.py
python 04_build_compensation.py
python 05_merge_policy_comp.py
python 06_build_compustat.py
python 07_build_analysis_dataset.py
python 10_add_factset_ownership.py
python 08_descriptive_stats.py
python 09_main_regression.py
```

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

## Notes on the LLM extraction

The extraction pipeline uses OpenAI's GPT-4o-mini model via the Chat Completions API. Results may differ slightly across runs due to model stochasticity, but the integer-valued and boolean fields that enter the regressions are stable in repeated runs. A subsample of extractions was verified against the underlying EX-19 texts; error categorization and recovery procedures are documented in the accompanying methodology document.

## Attribution

The base extraction script (`01_read_files.py`) is adapted from a template provided by the thesis supervisor. All other scripts, and the extensions to `01_read_files.py` (pre-clearance and hedging feature extraction), were written by the author for this thesis. All methodological decisions — hypothesis, model specification, robustness checks, and interpretation of results — are the author's own.

Coding assistance (Anthropic Claude) was used for debugging and drafting scripts. All code was reviewed and tested by the author before use.

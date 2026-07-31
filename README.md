# Insider Trading Policies and Executive Compensation

Bachelor thesis empirical pipeline. This repository contains the full Python code used to construct the analysis dataset and produce the regression tables and figures. Data files are excluded due to WRDS licensing restrictions; see the Data section below for how to obtain them.

## Overview

The empirical analysis tests whether firms with higher equity-based executive compensation impose more restrictive insider trading policies. Policy features are extracted from SEC EX-19 filings using an LLM-based text extraction pipeline. Compensation and firm-level controls come from WRDS (ExecComp, Compustat) and FactSet (Ownership Summary).

Main finding: firms with higher all-executives equity intensity have blackout periods that start approximately five days earlier before quarter-end, with statistical significance at the 10% level after controlling for firm size, leverage, profitability, firm age, industry fixed effects, and institutional ownership. The CEO-only measure shows the same direction but is not statistically distinguishable from zero after industry controls.

## Data sources (not included in this repository)

1. **EX-19 policy filings** — 1,073 HTML documents from SEC EDGAR, provided by the supervisor. Each firm's insider trading policy is filed as Exhibit 19 to its annual 10-K.
2. **ExecComp** — CEO and named-executive compensation data, WRDS.
3. **Compustat Annual Fundamentals** — firm-level accounting variables, WRDS.
4. **FactSet Ownership Summary** — market capitalization and institutional ownership data, WRDS.

Reproducing this analysis requires a valid WRDS subscription and access to the specific tables above.

## Pipeline structure

Scripts are numbered in the order they should be run:

```
01_read_files.py               Extract policy features from EX-19 files via GPT-4o-mini
01b_recover_missing.py         Re-run on flagged files with a sharpened prompt
02_merge_recovery.py           Merge recovery values into the main extraction
03_dedup.py                    Deduplicate to one row per firm
04_build_compensation.py       Build CEO equity share and pooled measure
05_merge_policy_comp.py        Merge policy + compensation
06_build_compustat.py          Build firm-level controls
07_build_analysis_dataset.py   Produce analysis_v2.csv
10_add_factset_ownership.py    Merge FactSet Ownership Summary → analysis_v3.csv
08_descriptive_stats.py        Produce Table 1, correlation matrix, figures
09_main_regression.py          Produce Tables 3, 4, 5
```

Note that 08 and 09 should be run after 10 (dependency order does not match numbering, for historical reasons).

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
ex19_policies/*.htm          (folder with EX-19 HTML files)
ex19_policies/ex19_metadata.csv
execcomp_russell3000_2024.csv
compustat_russell3000_2024.csv
factset_ownership_summary_q4_2024.csv
```

Run the pipeline:

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

- `table1_summary_stats.csv` — summary statistics for the regression sample
- `table2_correlation_matrix.csv` — Pearson correlations
- `table3_industry_breakdown.csv` — sample composition by 1-digit SIC
- `table3_main_regression.txt` / `.csv` — main 3-column regression (blackout days on CEO equity share)
- `table4_robustness.txt` / `.csv` — governance controls, all-executives measure, log(assets) alternative
- `table5_alternative_outcomes.txt` / `.csv` — composite restrictiveness, hedging LPM, pre-clearance LPM
- `figure1_distributions.png` — 6-panel histograms of key variables
- `figure2_industry_breakdown.png` — sample composition bar chart
- `figure3_scatter_main.png` — equity share vs blackout days
- `figure4_binary_features.png` — prevalence of policy features

## Notes on the LLM extraction

The extraction script uses OpenAI's GPT-4o-mini model. Results may differ slightly on repeated runs due to model stochasticity, but the integer-valued and boolean fields that enter the regressions are stable across runs. Manual verification of a subsample confirmed extraction quality; error categorization and recovery procedures are documented in `methodology_notes.docx`.

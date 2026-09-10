# xG Forecast

Code for **From Shots to Forecasts: One-Step-Ahead Prediction of Team-Level Expected Goals in Football**.

## Data and environment

Run commands from the project root. The licensed 23 July 2026 snapshot is `df.csv`;
`df_model.csv` contains the derived team-match records and original within-panel
splits. Data are not redistributed. `reproducibility/input_checksums.json` identifies
the inputs used for the reported results. A fresh API download is not a replacement
for the frozen snapshot. `1_get_data.R` requires the user's authorized provider access.

The verified local environments use R 4.5.1 and Python 3.12.13. R dependencies are
recorded in `renv.lock`; `requirements.txt` contains portable Python package pins.
The previous environment export, including nonportable build paths, is preserved
as `reproducibility/requirements_original.txt`. A clean installation on a new
machine has not been tested. Device and library changes may alter numerical results,
especially neural training; fixed seeds alone do not guarantee bitwise GPU replay.

```sh
Rscript --vanilla -e 'renv::restore(prompt = FALSE)'
python -m pip install -r requirements.txt
Rscript --vanilla 2_prepare_data.R
```

Preparation excludes whole matches with extra time or shoot-outs and uses the
published 70/15/15 within-competition-season split, with at least three prior
matches. Engineered histories and TCN inputs reset within team-seasons. The
primary rolling benchmark instead carries previous eligible team records across
seasons; its LOCO variant carries them within competition and team. All histories
exclude the target match. Calendar evaluation uses separate date-based assignments.

## Main analysis

With the original prediction and configuration files in `results/`, regenerate
summaries, bootstrap comparisons and the original tables/figures without refitting:

```sh
Rscript --vanilla 11_main.R summary
```

The driver defaults to `summary`. To rerun all original model searches and summaries:

```sh
XG_DEVICE=cpu Rscript --vanilla 11_main.R all /path/to/python
```

`models` runs model scripts only. Full execution includes both response scales,
100-trial XGBoost/TCN searches, six XGBoost ablations and 30 LOCO folds. Run it in a
copy when preserving archived outputs. It writes `results/` and `figs/`; it does
not download data or rerun preparation. `XG_DEVICE=cuda` selects CUDA when available.
Without an override, XGBoost retains its CUDA setting and TCN selects an available
CUDA/MPS/CPU device. `XG_PYTHON` can specify Python instead of the second argument.

Recorded run seeds are the defaults:

| Analysis | Seed |
|---|---:|
| Main XGBoost, log | 210787338 |
| Main XGBoost, raw | 2357141922 (scale seed 2357141923) |
| XGBoost ablations | 350162372, with recorded variant/trial offsets |
| Direct xGD XGBoost | 2174043077 |
| TCN, log then raw | 3990080874 |
| Match bootstrap versus rolling | 739899332 |
| Top-three paired bootstrap | 26629490 |
| Competition-season bootstrap | 2100877207 |

`XG_SEED` overrides the run seed for an individual script. Keep TCN's original
log-then-raw execution order when replaying the recorded run because the random
stream continues between scales. `XG_RESPONSE_SCALE=raw` selects the response
for individual R model/figure scripts. `XG_RESPONSE_SCALES=raw` restricts the main
XGBoost script; its scale offset still matches the archived raw configuration.

Archived configurations are in `reproducibility/`. LOCO defaults to
`reproducibility/xgb_loco_log_best_config.csv`, the full-panel configuration used
for the archived LOCO run, rather than a subsequently replaced main-run file.
`XG_LOCO_CONFIG=/path/to/config.csv` overrides it. LOCO fits each excluded-competition
fold with that fixed configuration; it is not nested selection. Earlier local
matches remain available for lag construction. Ablation variants are independently
tuned on the original validation split; comparisons reuse the primary test period.

## History ablation, smearing and residual/coverage diagnostics

```sh
Rscript --vanilla 12_revision_R1_lmm.R . results/reviewer1
python 13_revision_R1_xgb_smearing.py . results/reviewer1
python 14_revision_R1_comparisons.py . results/reviewer1
python 15_revision_R1_tables.py results/reviewer1 results/reviewer1/latex_tables
Rscript --vanilla 16_model_diagnostics_stability.R . results/reviewer1/lmm_full_model.rds results/reviewer3
```

The two selected-model refits support validation smearing and LMM history ablation.
The comparison script uses base seed 20260910 and records contrast-specific offsets.
The residual/coverage script reads saved fits and predictions; it calls
`17_plot_diagnostics_stability.R`. Details are in `README_revision_R1.md` and
`README_model_diagnostics.md`. Smearing and ablations are exploratory shared-test analyses.

## Calendar and point-forecast analyses

```sh
Rscript --vanilla 18_calendar_forecasting.R . results/calendar
python 19_forecast_diagnostics.py . results/diagnostics results/calendar
Rscript --vanilla 20_plot_forecast_calibration.R results/diagnostics
python 21_forecast_tables.py results/calendar results/diagnostics results/latex_tables
```

Calendar training ends on 30 June 2024, validation on 30 June 2025, and testing
uses later matches through the snapshot. Candidate windows 3/5/8/10 share eligible
complete pairs. Windows are selected using validation xGF MAE; final LMM scaling
and REML fits use training plus validation only. Model parameters remain fixed,
and histories update one match at a time. Unseen seasonal random intercepts
contribute zero, while earlier local history remains available. New competition-
seasons are defined relative to the final fitting sample. The snapshot was already
used in the main study, so this is exploratory calendar sensitivity, not an
untouched prospective holdout. Pointwise panel-bootstrap intervals use 5,000
replicates, base seed 20260911 and fixed contrast offsets.

Point diagnostics use saved forecasts, complete common pairs, a high-xG threshold
fixed from primary training data, and magnitude weights `abs(observed xGD)`.
Forecast-binned means assess mean alignment, not predictive-interval coverage.
The exclusion audit counts unique matches and the overlap of extra time/shoot-outs;
the source extract has no competition-stage variable.

## Output mapping

`reproducibility/table_map.csv` lists all 21 tables individually. Most original
table layouts are authored in `main.tex`; the commands regenerate their numerical
inputs, while scripts 15, 17 and 21 also export the corresponding LaTeX tables.

| Manuscript output | Scripts / result files |
|---|---|
| Data, main errors, model parameters, LOCO, scale/ablation/direct comparisons (Tables 1-15, 18; Figures 1-8) | `2_prepare_data.R`, model scripts 3-7, `6_compare_xgdiff_approaches.R`, `8_summary.R`, `9_bootCI*.R`, `10_tables_figures.R`; manuscript also retains authored tables and a TikZ protocol figure |
| Matched-information/history contrasts (Table 16) | `12_revision_R1_lmm.R`, `14_revision_R1_comparisons.py`, `15_revision_R1_tables.py` |
| Smearing (Table 17) | Scripts 12-15; `results/reviewer1/smearing_scale_comparison.csv` |
| Coverage (Table 19, Figure 9), residuals (Figure A1) | Scripts 16-17; `results/reviewer3/` |
| Calendar forecasts (Table 20) | Scripts 18-19 and 21; `results/calendar/` |
| Point diagnostics (Table 21, Figure 10) | Scripts 19-21; `results/diagnostics/` |

`21_forecast_tables.py` writes the two purple-marked LaTeX tables; the manuscript
defines `revFour` and `reviewerFour`. Figures are rendered in final scientific
colors. Diagram sources remain in the separate manuscript LaTeX package.

## Verification

The calendar analysis checks chronological separation, shared pairs and grouping
levels and saves convergence information, selected windows, predictions and R
session information. The diagnostic script validates keys, paired responses,
finite available forecasts and target consistency and writes input SHA-256 hashes.
With recorded seeds, the three bootstrap scripts reproduced all 22 archived
comparison rows within `1e-12`; the primary rolling forecasts agreed within
`1.4e-15`. Full original neural tuning searches were not rerun during this audit.
Configurations and original predictions therefore remain necessary to distinguish
recomputing reported summaries from fitting new models.

## License

The source code is available under the MIT License. Data and third-party materials
remain subject to their respective terms; the code license does not authorize
redistribution of licensed data.

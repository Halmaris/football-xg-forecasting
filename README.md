# xG Forecast

Code for *From Shots to Forecasts: One-Step-Ahead Prediction of Team-Level Expected Goals in Football*.

## Data and dependencies

Run scripts from this directory. `df.csv` is the licensed event-data snapshot
from 23 July 2026; `2_prepare_data.R` produces `df_model.csv`. Data are not
redistributed. Downloading a new snapshot can change the results.

R dependencies are recorded in `renv.lock`. Python dependencies for the original
XGBoost and TCN models are in `requirements.txt`. The analyses use R 4.5.1 and
Python 3.12.13. Restore the environments with:

```sh
Rscript -e 'renv::restore(prompt = FALSE)'
python3 -m pip install -r requirements.txt
```

## Original analyses

`11_main.R` is a sequential list of model and summary scripts. Its `python`
variable points to the local analysis environment; adjust it on another machine.
Then run:

```r
source('2_prepare_data.R')  # Only when df_model.csv needs to be created.
source('11_main.R')
```

This runs both response scales, XGBoost/TCN tuning, ablations, LOCO, summaries,
bootstrap comparisons, figures, and `revision.R`. Individual scripts can also
be run separately. Model results are written to `results/`; original figures
are written to `figs/`. To use CPU for the Python models, set `XG_DEVICE=cpu`.

With saved forecasts, `source('6_compare_xgdiff_approaches.R')` regenerates
the direct-versus-derived xGD comparison (Table 18), and
`source('10_tables_figures.R')` regenerates the original figures and descriptive
summaries. Tables 2 and 6 are exported as `results/data_coverage.csv` and
`results/descriptive_team_match.csv`, retaining teams with no shots.
Table 7 uses each main model's validation-selected configuration and metrics;
the XGBoost file is `results/xgb_log_best_config.csv`.

The primary split is chronological within each competition-season (70/15/15).
Engineered rolling means use up to K available previous matches and reset within
team-seasons; TCN sequences require a complete lookback within the team-season. The rolling
benchmark carries earlier eligible team records across seasons. Seeds are set
in the scripts. `XG_SEED` can override the seed of an individual original script.
LOCO uses the fixed `reproducibility/xgb_loco_log_best_config.csv` configuration;
selection is not nested within folds. The recorded configurations are retained
in `reproducibility/`.

## Analyses following peer review

All of these calculations are in **`revision.R`**, in seven numbered sections:

1. LMM history ablation and validation-based smearing for LMM and XGBoost.
2. Paired bootstrap comparisons, including matched-information XGBoost versus TCN.
3. Calendar-time validation and testing, including previously unseen seasons.
4. LMM residuals, competition coverage, effect sizes, and TCN capacity.
5. High-xG errors, directional accuracy, binned calibration, and match exclusions.
6. Publication figures.
7. LaTeX tables.

With the original predictions and configurations already in `results/`, run:

```r
source('revision.R')
```

No Python process is called by `revision.R`. It refits the LMMs in R and reads
the original XGBoost/TCN forecasts. Smearing uses validation predictions from
a model fitted only on training data. `6_model_xgb.py` exports the selected
XGBoost validation forecasts to `results/xgb_log_validation_predictions.csv`;
the existing analysis already supplies this file, so no new XGBoost fit is needed.

Outputs are in `results/revision/`, with five LaTeX tables in its `tables/`
subdirectory, three PDF figures, prediction and summary CSVs, fitted LMMs,
bootstrap draws, and `sessionInfo.txt`. Figure colors are final publication
colors; red revision markings apply only to table text and captions.
Earlier result folders are retained unchanged.

Calendar validation starts on 1 July 2024 and testing on 1 July 2025. Fits are
frozen before testing; histories use earlier observed matches. Bootstrap uses
5,000 resamples of whole matches or competition-seasons, seed 20260910 and
contrast-specific offsets. Holm adjustment covers four contrasts per resampling
scheme in the primary paired comparison. The R bootstrap uses different random
draws from the earlier NumPy implementation, so confidence limits and p-values
can differ slightly through Monte Carlo error; point estimates are unchanged.
The manuscript uses the R bootstrap results in `results/revision/`; earlier
NumPy outputs are historical. Tables 16, 17, 19, 20 and 21 are generated in the
`tables/` subdirectory. These analyses reuse the existing data snapshot and are exploratory.

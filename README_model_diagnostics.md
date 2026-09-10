# Model diagnostics and temporal coverage

`16_model_diagnostics_stability.R` summarizes saved fits and predictions and calls `17_plot_diagnostics_stability.R` to render the figures and coverage table. It does not fit, tune or modify a model.

Run from the football-xg-forecasting project directory:

```sh
Rscript --vanilla 16_model_diagnostics_stability.R . results/reviewer1/lmm_full_model.rds results/reviewer3
```

The second argument may instead point to the saved `lmm_full_model.rds` in the supplied code/results package. It is the final full REML fit produced by `12_revision_R1_lmm.R`. Recreating that fit is only necessary if the saved object is unavailable.

To regenerate only the graphics and table from the saved CSV results:

```sh
Rscript --vanilla 17_plot_diagnostics_stability.R results/reviewer3
```

Inputs also include `df_model.csv` and the saved log-response rolling predictions, TCN tuning/configuration files, and paired-bootstrap comparison results in `results/`. All inputs are read only. Match/team keys, paired records, response values, sample sizes, the selected TCN parameter count and the selection criterion are checked before reporting results.

The analysis used R 4.5.1, lme4 1.1-37, dplyr 1.1.4, readr 2.1.5 and jsonlite 2.0.0. Exact session information is written to `session_info.txt`. Plotting also uses ggplot2 4.0.3 and ggrepel 0.9.6. Numerical summaries involve no random sampling; label placement uses the fixed seed 20260910.

`lmm_residual_diagnostics.pdf` uses conditional response residuals from the 48,671 fitting observations. The Q-Q panel displays 3,000 evenly spaced order statistics, including the extremes; the residual/fitted panel contains all observations. The black solid curve shows a Gaussian-kernel local mean; dashed curves show the local mean plus/minus one local SD. At each of 401 equally spaced fitted values, weights are proportional to `exp(-0.5 * ((fitted - x) / 0.10)^2)`, the mean is the weighted residual mean, and the SD is the square root of the weighted squared deviation from that mean. Curves cover the full observed fitted-value range and are descriptive, with less local information near the extremes. The unchanged decile summaries remain in `lmm_residual_fitted_deciles.csv`. These diagnostics do not validate the coverage of the Wald intervals or independence of errors.

`rolling_error_stability.pdf` and `rolling_coverage.tex` use the unchanged 9,686 primary rolling forecasts (4,843 complete matches, 30 competitions, 114 competition-season panels). Metrics are observation-weighted; coverage is the number of observed seasons, while rolling histories reset within each season. The analysis is descriptive, uses the shared primary test period and cannot identify provider recalibration.

The two figure PDFs belong in the LaTeX `figs/` directory and `rolling_coverage.tex` in `tables/`. Revision colors apply to manuscript text, table entries and captions; plots have neutral labels and parenthesized panel letters. All 30 competitions are labelled. Ekstraklasa and La Liga 2 use matching blue/orange symbols in both panels, with other competition points filled black. These two long-coverage examples were named in the review; they are not the only competitions with long coverage. Ekstraklasa has eight seasons; La Liga 2, Primeira Liga, Allsvenskan and Swiss Super League each have six. The CSV outputs retain the underlying numerical values and all competition/season summaries.

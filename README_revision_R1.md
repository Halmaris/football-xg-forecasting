# Reviewer 1 revision: reproducible analyses

The revision adds mixed-effects history ablation, a paired comparison of existing
information-restricted XGBoost and Temporal ConvNet predictions, and validation-based
smearing for the selected LMM and XGBoost configurations. It does not rerun tuning,
change the original split, or overwrite the original scripts and results.

## Run from the existing project directory

```bash
Rscript --vanilla 12_revision_R1_lmm.R
python 13_revision_R1_xgb_smearing.py
python 14_revision_R1_comparisons.py
python 15_revision_R1_tables.py
```

The first three scripts accept optional `project_dir` and `output_dir` arguments.
Their defaults are the current working directory and `results/reviewer1`.
The fourth accepts a results directory and a destination for the LaTeX tables;
its defaults are `results/reviewer1` and `results/reviewer1/latex_tables`.
The generated table fragments are included by the revised manuscript, whose
`revOne` macro marks Reviewer 1 changes in blue.

The inputs are the existing `df_model.csv`, the function definitions in
`5_model_lmm.R` and `6_model_xgb.py`, and these files in `results/`:

- `lmm_log_best_config.csv`, `xgb_log_best_config.csv`;
- `lmm_log_test_predictions.csv`, `xgb_log_test_predictions.csv`;
- `lmm_raw_test_predictions.csv`, `xgb_raw_test_predictions.csv`;
- `tcn_log_test_predictions.csv`, `xgb_ablation_log_test_predictions.csv`.

The refit scripts load function definitions without executing the original
scripts' analysis blocks. They therefore do not trigger the original tuning or
permutation-importance runs. Licensed input data are not included in the revision
code bundle.

## Fitting and smearing

The full LMM retains the saved three-match window. Calibration uses the original
training-only ML specification; the final full and context-only fits use REML on
the identical 48,671 training-plus-validation observations. Context-only retains
home/away and the competition, team-season and opponent-season random intercepts,
but removes historical fixed effects. Both variants have 9,694 test observations.

XGBoost uses the archived configuration, including learning rate
0.037281314977209365 and 1,147 trees. The selected validation seed is 210787421;
the final refit seed is 210787338. CPU histogram fitting uses four threads.
No configuration is chosen using the new test results.

For each model, smearing is the average of `exp(log1p(actual) - predicted_log)`
over 9,618 validation predictions from a training-only model. It is applied as
`max(S * exp(predicted_log) - 1, 0)` to final predictions. In particular, it is
not multiplication of an already back-transformed xG value by S. Calibration
factors are saved before final test evaluation. The validation sample had also
served model selection; it is not a newly reserved calibration set. Transferring
a global factor to the final refit assumes a sufficiently stable exponential
residual moment. This sensitivity does not solve the common-calendar limitation.

## Paired inference

The comparisons are context-only minus full LMM (4,847 matches) and matched-information
XGBoost minus Temporal ConvNet (4,795 common matches). Both use xGF and derived xGD.
There are 5,000 bootstrap samples per comparison, target and resampling unit.
The base seed is 20260910; exact offsets are saved for every contrast. Match
sampling retains both team perspectives. Panel sampling retains each complete
competition-season and estimates observation-weighted MAE, not an equally weighted
average of panel MAEs. There are 114 LMM panels and 112 common TCN/XGBoost panels.

Intervals are pointwise 95% percentile intervals. The two-sided bootstrap
tail-probability calculation follows the existing project convention and uses a
plus-one correction. Holm adjustment covers all four new comparisons-by-target
contrasts, separately for match and panel resampling. A pointwise interval may
exclude zero while its multiplicity-adjusted p-value exceeds 0.05. No equivalence
margin is selected. These revision-stage analyses reuse the original test period
and are exploratory follow-ups.

## Recorded run (2026-09-10)

- R 4.5.1; lme4 1.1-37; complete R session in `lmm_session_info.txt`.
- Python 3.12.13; package versions in `requirements_revision_R1.txt` and metadata.
- Refit agreement with archived predictions: LMM maximum difference 4.44e-16;
  XGBoost maximum difference 1.18e-7 (CSV storage precision).
- Smearing factors: LMM 1.0449875645; XGBoost 1.0420099854.
- LMM xGF MAE without/with history: 0.546103/0.545166. The small gain does not
  remain significant after Holm adjustment or panel resampling.
- On common matches, TCN xGD MAE is 0.829502 versus 0.832163 for matched-information
  XGBoost. This small difference remains significant after Holm adjustment under
  both resampling schemes; xGF intervals include zero.
- Smearing reduces xGF bias to approximately -0.025 in both models, lowers RMSE,
  and slightly increases MAE. Directional accuracy is unchanged by the global
  factor in these complete-pair predictions.

The saved files include calibration predictions, fitted model objects, complete
test predictions, metrics, paired errors, bootstrap replicates, model diagnostics,
runtime versions and input SHA-256 checksums. `upper_decile_bias_descriptive.csv`
is an exploratory description of the top decile of observed test xGF; its threshold
is not used in fitting, calibration or model selection. The manuscript does not
interpret global bias reduction as full conditional calibration.

## Manuscript consistency

The original Table 4 had several XGBoost values inconsistent with the saved
configuration. The configuration reproduced the archived forecasts, so its values
were used to correct Table 4 and the learning-rate statement in Section 3.2.
Original main-result metrics were preserved. New manuscript Tables 16 and 17 are
generated directly from the revision result CSVs by the fourth script.

All results remain local. No remote repository update or publication is performed
by these scripts.

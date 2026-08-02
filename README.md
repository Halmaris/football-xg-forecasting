# Football xG Forecasting

Code accompanying the study:

**From Shots to Forecasts: xG-Based Time Series for Football Team Performance Prediction**

The repository contains R and Python scripts for forecasting team-level expected goals using rolling means, ARIMA, linear mixed-effects models, XGBoost and Temporal Convolutional Networks.

## Files

All scripts and input data are expected to be located in the main project directory.

The analysis is organised as follows:

* `1_get_data.R` – downloads shot-level StatsBomb data,
* `2_prepare_data.R` – creates the team–match dataset and chronological train, validation and test splits,
* `3_model_rolling.R` – rolling-mean benchmark,
* `3_model_rolling_loco.R` – leave-one-competition-out evaluation of the rolling mean,
* `4_model_arima.R` – ARIMA models,
* `5_model_lmm.R` – linear mixed-effects models,
* `6_model_xgb.py` – XGBoost models,
* `6_model_xgb_direct_xgdiff.py` – direct modelling of xG difference,
* `6_model_xgb_ablation.py` – XGBoost ablation analysis,
* `6_model_xgb_loco.py` – leave-one-competition-out XGBoost evaluation,
* `6_compare_xgdiff_approaches.R` – comparison of xG-difference modelling approaches,
* `7_model_tcn.py` – Temporal Convolutional Network,
* `8_summary.R` – summary of model performance,
* `9_bootCI.R` – bootstrap comparisons with the rolling-mean benchmark,
* `10_bootCI_pairwise.R` – pairwise bootstrap comparisons,
* `11_tables_figures.R` – manuscript tables and figures,
* `12_main.R` – runs the main analysis scripts.

## Data

The raw and processed data are not included in the repository.

The scripts use:

* `df.csv` – shot-level data,
* `df_model.csv` – processed team–match data.

`1_get_data.R` requires valid StatsBomb credentials.

## Running the analysis

Run the scripts in numerical order, starting with:

```text
1_get_data.R
2_prepare_data.R
```

The model scripts can then be executed separately.

`12_main.R` can be used to run the complete analysis, but the path to the local Python interpreter must first be updated.

The scripts create two output directories:

```text
results/
figs/
```

These directories contain generated results and figures and are not included in the repository.

XGBoost scripts use a CUDA GPU by default. Set:

```python
use_gpu = False
```

to run them on a CPU.

## Software

The analysis requires R and Python.

Main R packages include:

```text
dplyr, tidyr, purrr, readr, slider, zoo, forecast, lme4,
StatsBombR, ggplot2, patchwork, scales and tidytext
```

Main Python packages include:

```text
numpy, pandas, scikit-learn, optuna, xgboost, torch and matplotlib
```

## Licence

The source code is available under the MIT License.

The licence applies only to the source code. Data and third-party materials are subject to their respective terms and licences.

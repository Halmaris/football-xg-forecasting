library(dplyr)
library(tidyr)
library(purrr)
library(readr)

# Summarise test results from all forecasting models.
#
# Input
# -----
# Model-specific CSV files stored in results/.
#
# Output
# ------
# Summary tables comparing errors, directional accuracy, response scales,
# improvement over the rolling mean and the best-performing models.
# All files are saved in results/.

results_dir <- 'results'

response_scales <- c('log', 'raw')
model_order <- c(
  'Mixed-effects model',
  'XGBoost',
  'Temporal ConvNet',
  'ARIMA',
  'Rolling mean'
)
target_order <- c('xG_for', 'xG_against', 'xG_diff')
report_targets <- c('xG_for', 'xG_diff')

dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

read_result <- function(prefix, scale, suffix) {
  file <- file.path(results_dir, paste0(prefix, '_', scale, '_', suffix))
  if (!file.exists(file)) return(tibble())
  read_csv(file, show_col_types = FALSE) %>%
    mutate(response_scale = scale)
}

read_models <- function(suffix, arima_suffix = suffix) {
  sources <- tribble(
    ~prefix,    ~model,                   ~file_suffix,
    'rolling',  'Rolling mean',           suffix,
    'arima',    'ARIMA',                  arima_suffix,
    'lmm',      'Mixed-effects model',    suffix,
    'xgb',      'XGBoost',                suffix,
    'tcn',      'Temporal ConvNet',       suffix
  )

  pmap_dfr(sources, function(prefix, model, file_suffix) {
    map_dfr(response_scales, ~ read_result(prefix, .x, file_suffix)) %>%
      mutate(model = model)
  })
}

write_result <- function(df, file) {
  write_csv(df, file.path(results_dir, file))
}

# Test-set error metrics
df_all_test_errors <- read_models('test_metrics.csv') %>%
  mutate(
    model = factor(model, levels = model_order),
    response_scale = factor(response_scale, levels = response_scales),
    target = factor(target, levels = target_order),
    n = as.integer(n),
    across(c(mae, rmse, bias), as.numeric)
  ) %>%
  select(
    model, response_scale, target,
    any_of(c(
      'rolling_window', 'lookback', 'nrounds', 'model_type',
      'random_effects', 'singular', 'best_epoch', 'filters',
      'kernel_size', 'n_blocks', 'dropout', 'dropout_rate',
      'dense_units', 'learning_rate', 'weight_decay',
      'batch_size', 'epochs', 'n_numeric_features',
      'n_categorical_features'
    )),
    n, mae, rmse, bias
  ) %>%
  arrange(response_scale, target, mae, model)

df_report_test_errors <- df_all_test_errors %>%
  filter(target %in% report_targets) %>%
  mutate(target = factor(as.character(target), levels = report_targets)) %>%
  arrange(response_scale, target, mae, model)

df_test_errors_log <- df_all_test_errors %>%
  filter(response_scale == 'log')

df_test_errors_raw <- df_all_test_errors %>%
  filter(response_scale == 'raw')

df_test_performance_short <- df_all_test_errors %>%
  select(model, response_scale, target, n, mae, rmse, bias)

df_report_test_performance_short <- df_report_test_errors %>%
  select(model, response_scale, target, n, mae, rmse, bias)

# Directional accuracy
df_direction_accuracy <- read_models(
  suffix = 'direction_accuracy.csv',
  arima_suffix = 'test_direction_accuracy.csv'
) %>%
  mutate(
    target = coalesce(target, 'xG_diff'),
    model = factor(model, levels = model_order),
    response_scale = factor(response_scale, levels = response_scales),
    target = factor(target, levels = target_order),
    n = as.integer(n),
    direction_accuracy = as.numeric(direction_accuracy)
  ) %>%
  select(model, response_scale, target, n, direction_accuracy) %>%
  arrange(response_scale, desc(direction_accuracy), model)

# Improvement relative to the rolling mean
rolling_baseline <- df_all_test_errors %>%
  filter(model == 'Rolling mean') %>%
  select(
    response_scale,
    target,
    baseline_mae = mae,
    baseline_rmse = rmse
  )

df_improvement_all <- df_all_test_errors %>%
  left_join(rolling_baseline, by = c('response_scale', 'target')) %>%
  mutate(
    mae_improvement_pct = 100 * (baseline_mae - mae) / baseline_mae,
    rmse_improvement_pct = 100 * (baseline_rmse - rmse) / baseline_rmse
  ) %>%
  select(
    model, response_scale, target, n, mae, rmse, bias,
    baseline_mae, baseline_rmse,
    mae_improvement_pct, rmse_improvement_pct
  ) %>%
  arrange(response_scale, target, desc(mae_improvement_pct))

df_improvement_report <- df_improvement_all %>%
  filter(target %in% report_targets)

# Log-versus-raw comparison
df_scale_comparison_all <- df_all_test_errors %>%
  select(model, target, response_scale, n, mae, rmse, bias) %>%
  mutate(response_scale = as.character(response_scale)) %>%
  pivot_wider(
    names_from = response_scale,
    values_from = c(n, mae, rmse, bias),
    names_glue = '{.value}_{response_scale}'
  ) %>%
  mutate(
    delta_mae_log_minus_raw = mae_log - mae_raw,
    delta_rmse_log_minus_raw = rmse_log - rmse_raw,
    delta_bias_log_minus_raw = bias_log - bias_raw,
    log_mae_improvement_pct = 100 * (mae_raw - mae_log) / mae_raw,
    log_rmse_improvement_pct = 100 * (rmse_raw - rmse_log) / rmse_raw
  ) %>%
  arrange(target, model)

df_scale_comparison_report <- df_scale_comparison_all %>%
  filter(target %in% report_targets)

df_direction_scale_comparison <- df_direction_accuracy %>%
  mutate(response_scale = as.character(response_scale)) %>%
  pivot_wider(
    names_from = response_scale,
    values_from = c(n, direction_accuracy),
    names_glue = '{.value}_{response_scale}'
  ) %>%
  mutate(
    delta_direction_accuracy_log_minus_raw =
      direction_accuracy_log - direction_accuracy_raw
  ) %>%
  arrange(desc(direction_accuracy_log), model)

# Best-performing models
df_best_by_target_and_scale <- df_all_test_errors %>%
  group_by(response_scale, target) %>%
  arrange(mae, rmse, abs(bias), .by_group = TRUE) %>%
  slice(1) %>%
  ungroup()

df_best_by_report_target_and_scale <- df_report_test_errors %>%
  group_by(response_scale, target) %>%
  arrange(mae, rmse, abs(bias), .by_group = TRUE) %>%
  slice(1) %>%
  ungroup()

df_best_model_and_scale_by_target <- df_all_test_errors %>%
  group_by(target) %>%
  arrange(mae, rmse, abs(bias), .by_group = TRUE) %>%
  slice(1) %>%
  ungroup()

df_best_model_and_scale_by_report_target <- df_report_test_errors %>%
  group_by(target) %>%
  arrange(mae, rmse, abs(bias), .by_group = TRUE) %>%
  slice(1) %>%
  ungroup()

# Export
write_result(df_all_test_errors, 'summary_all_test_errors.csv')
write_result(df_report_test_errors, 'summary_report_test_errors.csv')
write_result(
  df_test_performance_short,
  'summary_test_performance_short_all_targets.csv'
)
write_result(
  df_report_test_performance_short,
  'summary_test_performance_short_report_targets.csv'
)
write_result(df_test_errors_log, 'summary_test_performance_log.csv')
write_result(df_test_errors_raw, 'summary_test_performance_raw.csv')
write_result(
  df_improvement_all,
  'summary_improvement_vs_rolling_all_targets.csv'
)
write_result(
  df_improvement_report,
  'summary_improvement_vs_rolling_report_targets.csv'
)
write_result(df_scale_comparison_all, 'summary_raw_vs_log_all_targets.csv')
write_result(
  df_scale_comparison_report,
  'summary_raw_vs_log_report_targets.csv'
)
write_result(df_direction_accuracy, 'summary_direction_accuracy.csv')
write_result(
  df_direction_scale_comparison,
  'summary_raw_vs_log_direction_accuracy.csv'
)
write_result(
  df_best_by_target_and_scale,
  'summary_best_by_target_and_scale.csv'
)
write_result(
  df_best_by_report_target_and_scale,
  'summary_best_by_report_target_and_scale.csv'
)
write_result(
  df_best_model_and_scale_by_target,
  'summary_best_model_and_scale_by_target.csv'
)
write_result(
  df_best_model_and_scale_by_report_target,
  'summary_best_model_and_scale_by_report_target.csv'
)

df_report_test_performance_short
df_scale_comparison_report
df_improvement_report
df_direction_accuracy
df_direction_scale_comparison
df_best_by_report_target_and_scale
df_best_model_and_scale_by_report_target

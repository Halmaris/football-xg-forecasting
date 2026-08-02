# Paired match-level bootstrap comparisons of MAE against the rolling-mean benchmark.
#
# Input:
#   results/*_log_test_predictions_long.csv
#
# Output:
#   results/bootstrap_log_mae_differences_vs_rolling.csv
#   results/bootstrap_log_mae_replicates_vs_rolling.csv
#   results/bootstrap_log_paired_errors_vs_rolling.csv
#   results/bootstrap_log_settings.csv

library(dplyr)
library(tidyr)
library(purrr)
library(readr)

results_dir <- 'results'
baseline_model <- 'Rolling mean'
targets <- c('xG_for', 'xG_diff')
n_boot <- 5000

run_seed <- sample.int(.Machine$integer.max, 1)
set.seed(run_seed)

model_files <- tibble::tribble(
  ~model,                ~file,
  'Rolling mean',        'rolling_log_test_predictions_long.csv',
  'ARIMA',               'arima_log_test_predictions_long.csv',
  'Mixed-effects model', 'lmm_log_test_predictions_long.csv',
  'XGBoost',             'xgb_log_test_predictions_long.csv',
  'Temporal ConvNet',    'tcn_log_test_predictions_long.csv'
)

models_to_compare <- c(
  'Mixed-effects model',
  'XGBoost',
  'Temporal ConvNet',
  'ARIMA'
)

predictions <- map2_dfr(
  model_files$model,
  model_files$file,
  function(model_name, file_name) {
    read_csv(file.path(results_dir, file_name), show_col_types = FALSE) %>%
      transmute(
        model = model_name,
        match_id = as.character(match_id),
        team_id = as.character(team_id),
        target,
        actual,
        predicted
      ) %>%
      filter(target %in% targets, !is.na(actual), !is.na(predicted)) %>%
      distinct(model, match_id, team_id, target, .keep_all = TRUE)
  }
)

common_keys <- predictions %>%
  group_by(match_id, team_id, target) %>%
  summarise(n_models = n_distinct(model), .groups = 'drop') %>%
  filter(n_models == nrow(model_files)) %>%
  select(-n_models)

predictions <- predictions %>%
  inner_join(common_keys, by = c('match_id', 'team_id', 'target'))

baseline <- predictions %>%
  filter(model == baseline_model) %>%
  transmute(
    match_id,
    team_id,
    target,
    actual_baseline = actual,
    predicted_baseline = predicted,
    error_baseline = abs(actual - predicted)
  )

paired_errors <- predictions %>%
  filter(model %in% models_to_compare) %>%
  transmute(
    model,
    match_id,
    team_id,
    target,
    actual_model = actual,
    predicted_model = predicted,
    error_model = abs(actual - predicted)
  ) %>%
  inner_join(baseline, by = c('match_id', 'team_id', 'target')) %>%
  mutate(
    baseline_model = baseline_model,
    response_scale = 'log',
    .after = model
  )

match_errors <- paired_errors %>%
  group_by(model, baseline_model, response_scale, target, match_id) %>%
  summarise(
    model_error = mean(error_model),
    baseline_error = mean(error_baseline),
    n_rows = n(),
    .groups = 'drop'
  )

bootstrap_target <- function(df, target_name) {
  model_errors <- df %>%
    select(match_id, model, model_error) %>%
    pivot_wider(names_from = model, values_from = model_error) %>%
    arrange(match_id)

  baseline_errors <- df %>%
    distinct(match_id, baseline_error) %>%
    arrange(match_id)

  wide <- left_join(model_errors, baseline_errors, by = 'match_id')

  model_matrix <- as.matrix(wide[, models_to_compare])
  baseline_vector <- wide$baseline_error
  difference_matrix <- sweep(model_matrix, 1, baseline_vector, '-')

  n_matches <- nrow(wide)
  n_models <- length(models_to_compare)

  model_mae <- colMeans(model_matrix)
  baseline_mae <- mean(baseline_vector)
  mae_difference <- colMeans(difference_matrix)
  standard_error <- apply(difference_matrix, 2, sd) / sqrt(n_matches)
  test_statistic <- mae_difference / standard_error

  boot_model_mae <- matrix(NA_real_, n_boot, n_models)
  boot_baseline_mae <- numeric(n_boot)
  boot_difference <- matrix(NA_real_, n_boot, n_models)
  boot_statistic <- matrix(NA_real_, n_boot, n_models)

  for (b in seq_len(n_boot)) {
    rows <- sample.int(n_matches, n_matches, replace = TRUE)
    sampled_difference <- difference_matrix[rows, , drop = FALSE]

    boot_model_mae[b, ] <- colMeans(model_matrix[rows, , drop = FALSE])
    boot_baseline_mae[b] <- mean(baseline_vector[rows])
    boot_difference[b, ] <- colMeans(sampled_difference)
    boot_se <- apply(sampled_difference, 2, sd) / sqrt(n_matches)
    boot_statistic[b, ] <-
      (boot_difference[b, ] - mae_difference) / boot_se
  }

  p_raw <- (
    colSums(abs(boot_statistic) >= abs(test_statistic), na.rm = TRUE) + 1
  ) / (n_boot + 1)

  order_test <- order(abs(test_statistic), decreasing = TRUE)
  p_stepdown <- numeric(n_models)

  for (j in seq_along(order_test)) {
    remaining <- order_test[j:length(order_test)]
    max_statistic <- apply(
      abs(boot_statistic[, remaining, drop = FALSE]),
      1,
      max,
      na.rm = TRUE
    )

    p_stepdown[j] <- (
      sum(max_statistic >= abs(test_statistic[order_test[j]])) + 1
    ) / (n_boot + 1)
  }

  p_stepdown <- cummax(p_stepdown)
  p_romano_wolf <- numeric(n_models)
  p_romano_wolf[order_test] <- p_stepdown

  max_t <- apply(abs(boot_statistic), 1, max, na.rm = TRUE)
  max_t_critical <- quantile(max_t, 0.95, na.rm = TRUE)

  simultaneous_ci_low <-
    mae_difference - max_t_critical * standard_error
  simultaneous_ci_high <-
    mae_difference + max_t_critical * standard_error

  pointwise_ci_low <- apply(
    boot_difference, 2, quantile, probs = 0.025, na.rm = TRUE
  )
  pointwise_ci_high <- apply(
    boot_difference, 2, quantile, probs = 0.975, na.rm = TRUE
  )

  improvement_pct_boot <- sweep(
    -boot_difference,
    1,
    boot_baseline_mae,
    '/'
  ) * 100

  row_counts <- df %>%
    group_by(model) %>%
    summarise(n_rows = sum(n_rows), .groups = 'drop')

  summary <- tibble(
    model = models_to_compare,
    baseline_model = baseline_model,
    response_scale = 'log',
    target = target_name,
    n_rows = row_counts$n_rows[match(models_to_compare, row_counts$model)],
    n_matches = n_matches,
    model_mae = model_mae,
    baseline_mae = baseline_mae,
    mae_difference = mae_difference,
    standard_error = standard_error,
    max_t_critical = as.numeric(max_t_critical),
    mae_improvement = -mae_difference,
    mae_improvement_pct = 100 * (-mae_difference) / baseline_mae,
    mae_difference_ci_low = simultaneous_ci_low,
    mae_difference_ci_high = simultaneous_ci_high,
    mae_improvement_ci_low = -simultaneous_ci_high,
    mae_improvement_ci_high = -simultaneous_ci_low,
    mae_difference_pointwise_ci_low = pointwise_ci_low,
    mae_difference_pointwise_ci_high = pointwise_ci_high,
    mae_improvement_pointwise_ci_low = -pointwise_ci_high,
    mae_improvement_pointwise_ci_high = -pointwise_ci_low,
    mae_improvement_pct_pointwise_ci_low = apply(
      improvement_pct_boot, 2, quantile, probs = 0.025, na.rm = TRUE
    ),
    mae_improvement_pct_pointwise_ci_high = apply(
      improvement_pct_boot, 2, quantile, probs = 0.975, na.rm = TRUE
    ),
    p_boot_difference_from_zero = p_raw,
    p_romano_wolf = p_romano_wolf
  )

  boot <- map_dfr(seq_along(models_to_compare), function(j) {
    tibble(
      boot_id = seq_len(n_boot),
      model = models_to_compare[j],
      baseline_model = baseline_model,
      response_scale = 'log',
      target = target_name,
      model_mae = boot_model_mae[, j],
      baseline_mae = boot_baseline_mae,
      mae_difference = boot_difference[, j],
      studentized_statistic = boot_statistic[, j],
      mae_improvement = -boot_difference[, j],
      mae_improvement_pct = improvement_pct_boot[, j]
    )
  })

  list(summary = summary, boot = boot)
}

results <- match_errors %>%
  group_by(target) %>%
  nest() %>%
  mutate(result = map2(data, target, bootstrap_target)) %>%
  ungroup()

bootstrap_summary <- results %>%
  transmute(summary = map(result, 'summary')) %>%
  unnest(summary) %>%
  mutate(n_boot = n_boot, run_seed = run_seed) %>%
  arrange(target, mae_difference)

bootstrap_replicates <- results %>%
  transmute(boot = map(result, 'boot')) %>%
  unnest(boot) %>%
  mutate(n_boot = n_boot, run_seed = run_seed)

write_csv(
  bootstrap_summary,
  file.path(results_dir, 'bootstrap_log_mae_differences_vs_rolling.csv')
)
write_csv(
  bootstrap_replicates,
  file.path(results_dir, 'bootstrap_log_mae_replicates_vs_rolling.csv')
)
write_csv(
  paired_errors,
  file.path(results_dir, 'bootstrap_log_paired_errors_vs_rolling.csv')
)
write_csv(
  tibble(
    response_scale = 'log',
    baseline_model = baseline_model,
    targets = paste(targets, collapse = ', '),
    models_compared = paste(models_to_compare, collapse = ', '),
    n_boot = n_boot,
    bootstrap_unit = 'match_id',
    multiplicity_adjustment = paste(
      'step-down Romano-Wolf maxT, separately within each target'
    ),
    confidence_intervals = paste(
      'simultaneous 95% studentized maxT intervals,',
      'separately within each target'
    ),
    pointwise_intervals = paste(
      'pointwise percentile intervals retained in separate columns'
    ),
    run_seed = run_seed
  ),
  file.path(results_dir, 'bootstrap_log_settings.csv')
)

bootstrap_summary

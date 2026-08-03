# Paired match-level bootstrap comparisons of MAE among the three best models.
#
# Input:
#   results/lmm_log_test_predictions_long.csv
#   results/xgb_log_test_predictions_long.csv
#   results/tcn_log_test_predictions_long.csv
#
# Output:
#   results/bootstrap_log_mae_pairwise_top3.csv
#   results/bootstrap_log_mae_replicates_pairwise_top3.csv
#   results/bootstrap_log_paired_errors_pairwise_top3.csv
#   results/bootstrap_log_pairwise_top3_settings.csv

library(dplyr)
library(tidyr)
library(purrr)
library(readr)

results_dir <- 'results'
targets <- c('xG_for', 'xG_diff')
n_boot <- 5000

dir.create(results_dir, showWarnings = FALSE)

run_seed <- sample.int(.Machine$integer.max, 1)
set.seed(run_seed)

model_files <- tibble::tribble(
  ~model,                ~file,
  'Mixed-effects model', 'lmm_log_test_predictions_long.csv',
  'XGBoost',             'xgb_log_test_predictions_long.csv',
  'Temporal ConvNet',    'tcn_log_test_predictions_long.csv'
)

comparisons <- tibble::tribble(
  ~model_1,              ~model_2,
  'Mixed-effects model', 'XGBoost',
  'Mixed-effects model', 'Temporal ConvNet',
  'XGBoost',             'Temporal ConvNet'
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
      filter(target %in% targets, !is.na(actual), !is.na(predicted))
  }
)

make_paired_errors <- function(model_1_name, model_2_name) {
  model_1_data <- predictions %>%
    filter(model == model_1_name) %>%
    transmute(
      match_id,
      team_id,
      target,
      actual_model_1 = actual,
      predicted_model_1 = predicted,
      error_model_1 = abs(actual - predicted)
    )

  model_2_data <- predictions %>%
    filter(model == model_2_name) %>%
    transmute(
      match_id,
      team_id,
      target,
      actual_model_2 = actual,
      predicted_model_2 = predicted,
      error_model_2 = abs(actual - predicted)
    )

  inner_join(
    model_1_data,
    model_2_data,
    by = c('match_id', 'team_id', 'target')
  ) %>%
    mutate(
      model_1 = model_1_name,
      model_2 = model_2_name,
      response_scale = 'log',
      .before = match_id
    )
}

paired_errors <- map2_dfr(
  comparisons$model_1,
  comparisons$model_2,
  make_paired_errors
)

match_errors <- paired_errors %>%
  group_by(model_1, model_2, response_scale, target, match_id) %>%
  summarise(
    error_model_1 = mean(error_model_1),
    error_model_2 = mean(error_model_2),
    n_rows = n(),
    .groups = 'drop'
  )

bootstrap_one <- function(df) {
  n_matches <- nrow(df)
  mae_model_1 <- mean(df$error_model_1)
  mae_model_2 <- mean(df$error_model_2)

  boot_matrix <- replicate(n_boot, {
    rows <- sample.int(n_matches, n_matches, replace = TRUE)
    boot_mae_model_1 <- mean(df$error_model_1[rows])
    boot_mae_model_2 <- mean(df$error_model_2[rows])

    c(
      mae_model_1 = boot_mae_model_1,
      mae_model_2 = boot_mae_model_2,
      mae_difference = boot_mae_model_1 - boot_mae_model_2
    )
  })

  boot <- as_tibble(t(boot_matrix)) %>%
    mutate(boot_id = row_number(), .before = 1)

  p_left <- (sum(boot$mae_difference <= 0) + 1) / (n_boot + 1)
  p_right <- (sum(boot$mae_difference >= 0) + 1) / (n_boot + 1)

  summary <- tibble(
    n_rows = sum(df$n_rows),
    n_matches = n_matches,
    mae_model_1 = mae_model_1,
    mae_model_2 = mae_model_2,
    mae_difference = mae_model_1 - mae_model_2,
    mae_difference_ci_low = quantile(boot$mae_difference, 0.025),
    mae_difference_ci_high = quantile(boot$mae_difference, 0.975),
    p_boot_difference_from_zero = min(1, 2 * min(p_left, p_right))
  )

  list(summary = summary, boot = boot)
}

results <- match_errors %>%
  group_by(model_1, model_2, response_scale, target) %>%
  nest() %>%
  mutate(result = map(data, bootstrap_one))

bootstrap_summary <- results %>%
  transmute(
    model_1,
    model_2,
    response_scale,
    target,
    summary = map(result, 'summary')
  ) %>%
  unnest(summary) %>%
  group_by(target) %>%
  mutate(p_holm = p.adjust(p_boot_difference_from_zero, method = 'holm')) %>%
  ungroup() %>%
  mutate(n_boot = n_boot, run_seed = run_seed) %>%
  arrange(target, mae_difference)

bootstrap_replicates <- results %>%
  transmute(
    model_1,
    model_2,
    response_scale,
    target,
    boot = map(result, 'boot')
  ) %>%
  unnest(boot) %>%
  mutate(n_boot = n_boot, run_seed = run_seed)

write_csv(
  bootstrap_summary,
  file.path(results_dir, 'bootstrap_log_mae_pairwise_top3.csv')
)
write_csv(
  bootstrap_replicates,
  file.path(results_dir, 'bootstrap_log_mae_replicates_pairwise_top3.csv')
)
write_csv(
  paired_errors,
  file.path(results_dir, 'bootstrap_log_paired_errors_pairwise_top3.csv')
)
write_csv(
  tibble(
    response_scale = 'log',
    models = paste(model_files$model, collapse = ', '),
    comparisons = paste(
      paste(comparisons$model_1, comparisons$model_2, sep = ' vs '),
      collapse = '; '
    ),
    targets = paste(targets, collapse = ', '),
    n_boot = n_boot,
    bootstrap_unit = 'match_id',
    mae_difference = 'MAE(model_1) - MAE(model_2)',
    holm_correction = 'separately within each target',
    run_seed = run_seed
  ),
  file.path(results_dir, 'bootstrap_log_pairwise_top3_settings.csv')
)

bootstrap_summary

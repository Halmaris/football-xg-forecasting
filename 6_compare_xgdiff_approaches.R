library(dplyr)
library(readr)

# Compare three XGBoost approaches to team-level xG difference.
# Inputs:
#   results/xgb_log_test_predictions.csv
#   results/xgb_direct_xgdiff_test_predictions.csv
# Outputs:
#   results/xgb_xgdiff_three_approaches_metrics.csv
#   results/xgb_xgdiff_three_approaches_predictions.csv
#   results/xgb_xgdiff_three_approaches_diagnostics.csv

results_dir <- 'results'

paired_file <- file.path(
  results_dir,
  'xgb_log_test_predictions.csv'
)

direct_file <- file.path(
  results_dir,
  'xgb_direct_xgdiff_test_predictions.csv'
)

metrics_file <- file.path(
  results_dir,
  'xgb_xgdiff_three_approaches_metrics.csv'
)

predictions_file <- file.path(
  results_dir,
  'xgb_xgdiff_three_approaches_predictions.csv'
)

diagnostics_file <- file.path(
  results_dir,
  'xgb_xgdiff_three_approaches_diagnostics.csv'
)

required_columns <- c(
  'match_id',
  'team_id',
  'opponent_id',
  'actual_xG_diff',
  'predicted_xG_diff'
)

check_columns <- function(df, required, file_name) {
  missing <- setdiff(required, names(df))

  if (length(missing) > 0) {
    stop(
      'Missing columns in ',
      file_name,
      ': ',
      paste(missing, collapse = ', ')
    )
  }
}

calculate_metrics <- function(actual, predicted) {
  error <- predicted - actual

  tibble(
    n = length(error),
    mae = mean(abs(error)),
    rmse = sqrt(mean(error^2)),
    bias = mean(error),
    direction_accuracy = mean(sign(actual) == sign(predicted)),
    direction_accuracy_percent =
      100 * mean(sign(actual) == sign(predicted))
  )
}

dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

paired <- read_csv(paired_file, show_col_types = FALSE)
direct <- read_csv(direct_file, show_col_types = FALSE)

check_columns(paired, required_columns, paired_file)
check_columns(direct, required_columns, direct_file)

n_paired_original <- nrow(paired)
n_direct_original <- nrow(direct)

if (anyDuplicated(paired[c('match_id', 'team_id')]) > 0) {
  stop('Duplicate match-team rows in ', paired_file)
}

if (anyDuplicated(direct[c('match_id', 'team_id')]) > 0) {
  stop('Duplicate match-team rows in ', direct_file)
}

paired <- paired %>%
  select(
    match_id,
    team_id,
    opponent_id,
    actual_xG_diff,
    predicted_xG_diff
  ) %>%
  rename(
    actual_xG_diff_paired = actual_xG_diff,
    predicted_paired = predicted_xG_diff
  )

direct_opponent <- direct %>%
  select(
    match_id,
    team_id,
    predicted_xG_diff
  ) %>%
  rename(
    opponent_id = team_id,
    predicted_direct_opponent = predicted_xG_diff
  )

direct <- direct %>%
  select(
    match_id,
    team_id,
    opponent_id,
    actual_xG_diff,
    predicted_xG_diff
  ) %>%
  left_join(
    direct_opponent,
    by = c('match_id', 'opponent_id'),
    relationship = 'one-to-one'
  ) %>%
  mutate(
    predicted_direct_unreconciled = predicted_xG_diff,
    predicted_direct_reconciled = (
      predicted_xG_diff - predicted_direct_opponent
    ) / 2
  ) %>%
  select(
    match_id,
    team_id,
    opponent_id,
    actual_xG_diff,
    predicted_direct_unreconciled,
    predicted_direct_reconciled
  )

analysis <- paired %>%
  inner_join(
    direct,
    by = c('match_id', 'team_id', 'opponent_id'),
    relationship = 'one-to-one'
  )

if (nrow(analysis) == 0) {
  stop('No common test observations were found.')
}

max_actual_difference <- max(
  abs(
    analysis$actual_xG_diff_paired -
      analysis$actual_xG_diff
  ),
  na.rm = TRUE
)

if (max_actual_difference > 1e-10) {
  stop(
    'Observed xG differences are inconsistent between prediction files.'
  )
}

analysis <- analysis %>%
  group_by(match_id) %>%
  filter(
    n() == 2,
    setequal(team_id, opponent_id)
  ) %>%
  ungroup() %>%
  transmute(
    match_id,
    team_id,
    opponent_id,
    actual = actual_xG_diff,
    predicted_paired,
    predicted_direct_unreconciled,
    predicted_direct_reconciled
  )

if (nrow(analysis) == 0) {
  stop('No complete match pairs were found in common.')
}

approaches <- c(
  'Derived from paired xG_for predictions' = 'predicted_paired',
  'Direct xG_diff model, unreconciled' =
    'predicted_direct_unreconciled',
  'Direct xG_diff model, reconciled' =
    'predicted_direct_reconciled'
)

metrics <- bind_rows(
  lapply(names(approaches), function(approach) {
    prediction_column <- approaches[[approach]]

    calculate_metrics(
      analysis$actual,
      analysis[[prediction_column]]
    ) %>%
      mutate(
        approach = approach,
        n_matches = n_distinct(analysis$match_id),
        .before = 1
      )
  })
)

paired_sums <- analysis %>%
  group_by(match_id) %>%
  summarise(
    prediction_sum = sum(predicted_paired),
    .groups = 'drop'
  )

reconciled_sums <- analysis %>%
  group_by(match_id) %>%
  summarise(
    prediction_sum = sum(predicted_direct_reconciled),
    .groups = 'drop'
  )

diagnostics <- tibble(
  paired_rows_original = n_paired_original,
  direct_rows_original = n_direct_original,
  common_complete_rows = nrow(analysis),
  common_complete_matches = n_distinct(analysis$match_id),
  max_actual_difference_between_files = max_actual_difference,
  max_abs_paired_prediction_sum_within_match =
    max(abs(paired_sums$prediction_sum)),
  max_abs_reconciled_prediction_sum_within_match =
    max(abs(reconciled_sums$prediction_sum))
)

write_csv(metrics, metrics_file)
write_csv(analysis, predictions_file)
write_csv(diagnostics, diagnostics_file)

print(metrics)
print(diagnostics)

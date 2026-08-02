library(dplyr)
library(purrr)
library(readr)
library(zoo)

# Fits rolling-mean models for team-level xG_for forecasting.
# Input: df_model.csv with team-match observations and train/validation/test splits.
# Outputs: selected window, validation results, test predictions, MAE, RMSE, bias
# and directional accuracy, saved in results/.

input_file <- 'df_model.csv'
output_dir <- 'results'
response_scale <- 'log'  # 'log' or 'raw'
rolling_windows <- c(3, 5, 8, 10)

file_prefix <- paste0('rolling_', response_scale)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

output_file <- function(name) {
  file.path(output_dir, paste0(file_prefix, '_', name, '.csv'))
}

summarise_metrics <- function(actual, predicted) {
  tibble(
    n = sum(is.finite(actual) & is.finite(predicted)),
    mae = mean(abs(actual - predicted), na.rm = TRUE),
    rmse = sqrt(mean((actual - predicted)^2, na.rm = TRUE)),
    bias = mean(predicted - actual, na.rm = TRUE)
  )
}

make_predictions <- function(df, k, evaluation_split) {
  rolling_col <- paste0('rolling', k, '_response')

  df %>%
    filter(split == evaluation_split) %>%
    transmute(
      match_id,
      team_id,
      split,
      target = 'xG_for',
      actual = xG_for,
      predicted = if (response_scale == 'log') {
        expm1(.data[[rolling_col]])
      } else {
        .data[[rolling_col]]
      }
    ) %>%
    filter(is.finite(actual), is.finite(predicted))
}

pair_predictions <- function(predictions) {
  opponent <- predictions %>%
    select(match_id, split, team_id, actual, predicted) %>%
    rename(
      opponent_team_id = team_id,
      actual_xG_against = actual,
      predicted_xG_against = predicted
    )

  predictions %>%
    rename(
      actual_xG_for = actual,
      predicted_xG_for = predicted
    ) %>%
    inner_join(opponent, by = c('match_id', 'split'), relationship = 'many-to-many') %>%
    filter(team_id != opponent_team_id) %>%
    group_by(match_id, split, team_id) %>%
    slice(1) %>%
    ungroup() %>%
    mutate(
      actual_xG_diff = actual_xG_for - actual_xG_against,
      predicted_xG_diff = predicted_xG_for - predicted_xG_against
    )
}

make_long_predictions <- function(paired) {
  bind_rows(
    paired %>% transmute(
      match_id, team_id, opponent_team_id, split, target = 'xG_for',
      actual = actual_xG_for, predicted = predicted_xG_for
    ),
    paired %>% transmute(
      match_id, team_id, opponent_team_id, split, target = 'xG_against',
      actual = actual_xG_against, predicted = predicted_xG_against
    ),
    paired %>% transmute(
      match_id, team_id, opponent_team_id, split, target = 'xG_diff',
      actual = actual_xG_diff, predicted = predicted_xG_diff
    )
  )
}

df_model <- read_csv(input_file, show_col_types = FALSE) %>%
  mutate(
    match_date = as.Date(match_date),
    response = if (response_scale == 'log') log1p(xG_for) else xG_for
  ) %>%
  arrange(team_id, match_date, match_id)

for (k in rolling_windows) {
  rolling_col <- paste0('rolling', k, '_response')
  df_model <- df_model %>%
    group_by(team_id) %>%
    mutate(!!rolling_col := rollmeanr(lag(response), k, fill = NA)) %>%
    ungroup()
}

tuning_results <- map_dfr(rolling_windows, function(k) {
  predictions <- make_predictions(df_model, k, 'validation')
  summarise_metrics(predictions$actual, predictions$predicted) %>%
    mutate(
      response_scale = response_scale,
      target = 'xG_for',
      rolling_window = k,
      split = 'validation',
      .before = 1
    )
}) %>%
  arrange(mae)

best_config <- tuning_results %>%
  slice_min(mae, n = 1, with_ties = FALSE)

best_k <- best_config$rolling_window[[1]]
test_xg_for <- make_predictions(df_model, best_k, 'test')
test_paired <- pair_predictions(test_xg_for)
test_long <- make_long_predictions(test_paired)

test_metrics <- test_long %>%
  group_by(target) %>%
  summarise(
    n = n(),
    mae = mean(abs(actual - predicted)),
    rmse = sqrt(mean((actual - predicted)^2)),
    bias = mean(predicted - actual),
    .groups = 'drop'
  ) %>%
  mutate(
    model = 'Rolling mean',
    response_scale = response_scale,
    rolling_window = best_k,
    .before = 1
  )

direction_accuracy <- test_paired %>%
  summarise(
    n = n(),
    direction_accuracy = mean(
      sign(actual_xG_diff) == sign(predicted_xG_diff)
    )
  ) %>%
  mutate(
    model = 'Rolling mean',
    response_scale = response_scale,
    target = 'xG_diff',
    rolling_window = best_k,
    .before = 1
  )

write_csv(tuning_results, output_file('tuning_results'))
write_csv(best_config, output_file('best_config'))
write_csv(test_metrics, output_file('test_metrics'))
write_csv(direction_accuracy, output_file('direction_accuracy'))
write_csv(test_paired, output_file('test_predictions'))
write_csv(test_long, output_file('test_predictions_long'))

print(tuning_results)
print(test_metrics)

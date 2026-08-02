library(dplyr)
library(purrr)
library(readr)
library(forecast)

# Fits rolling one-step-ahead ARIMA models to team-season xG_for series.
# Input: df_model.csv with team-match observations and train/validation/test splits.
# Outputs: configuration, model counts, predictions, MAE, RMSE, bias and
# directional accuracy for the validation and test sets, saved in results/.

input_file <- 'df_model.csv'
output_dir <- 'results'
response_scale <- 'log'  # 'log' or 'raw'
min_train <- 8

use_log <- response_scale == 'log'
file_prefix <- paste0('arima_', response_scale)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

output_file <- function(name) {
  file.path(output_dir, paste0(file_prefix, '_', name, '.csv'))
}

forecast_series <- function(df) {
  df <- df %>%
    arrange(match_date, match_week, match_id) %>%
    filter(!is.na(xG_for))

  map_dfr(which(df$split %in% c('validation', 'test')), function(i) {
    history <- df$xG_for[seq_len(i - 1)]

    if (length(history) < min_train) {
      predicted <- NA_real_
      arima_model <- NA_character_
    } else {
      y <- if (use_log) log1p(history) else history

      fit <- tryCatch(
        auto.arima(
          y,
          seasonal = FALSE,
          stepwise = TRUE,
          approximation = FALSE,
          allowdrift = TRUE,
          allowmean = TRUE
        ),
        error = function(e) NULL
      )

      if (is.null(fit)) {
        predicted <- mean(y, na.rm = TRUE)
        arima_model <- 'fallback_mean'
      } else {
        predicted <- as.numeric(forecast(fit, h = 1)$mean[1])
        arima_model <- paste0(
          'ARIMA(', paste(arimaorder(fit), collapse = ','), ')'
        )
      }

      if (use_log) predicted <- expm1(predicted)
      predicted <- max(predicted, 0)
    }

    tibble(
      match_id = df$match_id[i],
      team_id = df$team_id[i],
      opponent_id = df$opponent_id[i],
      split = df$split[i],
      target = 'xG_for',
      actual = df$xG_for[i],
      predicted = predicted,
      arima_model = arima_model
    )
  })
}

pair_predictions <- function(predictions) {
  opponent <- predictions %>%
    select(match_id, split, team_id, actual, predicted) %>%
    rename(
      opponent_id = team_id,
      actual_xG_against = actual,
      predicted_xG_against = predicted
    )

  predictions %>%
    select(
      match_id,
      team_id,
      opponent_id,
      split,
      actual_xG_for = actual,
      predicted_xG_for = predicted
    ) %>%
    inner_join(opponent, by = c('match_id', 'split', 'opponent_id')) %>%
    mutate(
      opponent_team_id = opponent_id,
      actual_xG_diff = actual_xG_for - actual_xG_against,
      predicted_xG_diff = predicted_xG_for - predicted_xG_against
    ) %>%
    select(-opponent_id)
}

make_long_predictions <- function(paired) {
  bind_rows(
    paired %>%
      transmute(
        match_id, team_id, opponent_team_id, split,
        target = 'xG_for',
        actual = actual_xG_for,
        predicted = predicted_xG_for
      ),
    paired %>%
      transmute(
        match_id, team_id, opponent_team_id, split,
        target = 'xG_against',
        actual = actual_xG_against,
        predicted = predicted_xG_against
      ),
    paired %>%
      transmute(
        match_id, team_id, opponent_team_id, split,
        target = 'xG_diff',
        actual = actual_xG_diff,
        predicted = predicted_xG_diff
      )
  )
}

summarise_metrics <- function(df) {
  df %>%
    filter(is.finite(actual), is.finite(predicted)) %>%
    group_by(target) %>%
    summarise(
      n = n(),
      mae = mean(abs(actual - predicted)),
      rmse = sqrt(mean((actual - predicted)^2)),
      bias = mean(predicted - actual),
      .groups = 'drop'
    ) %>%
    mutate(
      model = 'ARIMA',
      response_scale = response_scale,
      .before = 1
    )
}

df_model <- read_csv(input_file, show_col_types = FALSE)

xg_for_predictions <- df_model %>%
  arrange(competition_id, season_id, team_id, match_date, match_week, match_id) %>%
  group_by(
    competition_id,
    competition_name,
    season_id,
    season_name,
    team_id,
    team_name
  ) %>%
  group_split() %>%
  map_dfr(forecast_series)

paired_predictions <- pair_predictions(xg_for_predictions)
long_predictions <- make_long_predictions(paired_predictions)

config <- tibble(
  model = 'ARIMA',
  target = 'xG_for',
  response_scale = response_scale,
  min_train = min_train,
  evaluation = 'rolling one-step-ahead'
)

model_counts <- xg_for_predictions %>%
  count(split, target, arima_model, sort = TRUE) %>%
  mutate(response_scale = response_scale, .before = 1)

write_csv(config, output_file('config'))
write_csv(model_counts, output_file('model_counts'))

for (evaluation_split in c('validation', 'test')) {
  paired_split <- paired_predictions %>%
    filter(split == evaluation_split)

  long_split <- long_predictions %>%
    filter(split == evaluation_split)

  metrics <- summarise_metrics(long_split)

  direction <- paired_split %>%
    filter(is.finite(actual_xG_diff), is.finite(predicted_xG_diff)) %>%
    summarise(
      n = n(),
      direction_accuracy = mean(
        sign(actual_xG_diff) == sign(predicted_xG_diff)
      )
    ) %>%
    mutate(
      model = 'ARIMA',
      target = 'xG_diff',
      response_scale = response_scale,
      .before = 1
    )

  write_csv(metrics, output_file(paste0(evaluation_split, '_metrics')))
  write_csv(direction, output_file(paste0(evaluation_split, '_direction_accuracy')))
  write_csv(paired_split, output_file(paste0(evaluation_split, '_predictions')))
  write_csv(long_split, output_file(paste0(evaluation_split, '_predictions_long')))

  cat('\n', tools::toTitleCase(evaluation_split), ' metrics:\n', sep = '')
  print(metrics)
}

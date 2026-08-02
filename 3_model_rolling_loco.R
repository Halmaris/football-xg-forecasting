library(dplyr)
library(purrr)
library(readr)
library(zoo)

# Leave-one-competition-out evaluation for the rolling-mean benchmark.
#
# For each held-out competition:
#   1. select the rolling window using validation observations from all
#      remaining competitions;
#   2. evaluate the selected window on test observations from the held-out
#      competition;
#   3. pool predictions from all held-out competitions for overall metrics.
#
# The rolling mean remains a one-step-ahead benchmark: a test prediction may
# use outcomes from earlier matches in the same competition, including earlier
# test matches, exactly as in the original rolling-mean script.
#
# Input: df_model.csv
# Output: LOCO tuning, selected windows, predictions and metrics in results/.

# Settings ####

script_version <- '2026-07-31-01'
input_file <- 'df_model.csv'
output_dir <- 'results'
response_scale <- 'log'  # 'log' or 'raw'
rolling_windows <- c(3, 5, 8, 10)

file_prefix <- paste0('rolling_loco_', response_scale)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

output_file <- function(name) {
  file.path(output_dir, paste0(file_prefix, '_', name, '.csv'))
}

# Utility functions ####

validate_input <- function(df) {
  required <- c(
    'competition_id', 'competition_name', 'season_id', 'match_id',
    'match_date', 'team_id', 'opponent_id', 'split', 'xG_for'
  )

  missing <- setdiff(required, names(df))

  if (length(missing) > 0) {
    stop(
      'Missing required columns: ',
      paste(missing, collapse = ', '),
      call. = FALSE
    )
  }

  invisible(df)
}

summarise_metrics <- function(actual, predicted) {
  keep <- is.finite(actual) & is.finite(predicted)
  actual <- actual[keep]
  predicted <- predicted[keep]

  if (length(actual) == 0) {
    return(tibble(
      n = 0L,
      mae = NA_real_,
      rmse = NA_real_,
      bias = NA_real_
    ))
  }

  tibble(
    n = length(actual),
    mae = mean(abs(actual - predicted)),
    rmse = sqrt(mean((actual - predicted)^2)),
    bias = mean(predicted - actual)
  )
}

add_rolling_features <- function(df) {
  df <- df %>%
    mutate(
      response = if (response_scale == 'log') {
        log1p(xG_for)
      } else {
        xG_for
      }
    ) %>%
    arrange(
      competition_id, team_id, match_date,
      season_id, match_id
    )

  # Competition is included in the grouping to ensure that a held-out
  # competition uses only its own previous matches for the local benchmark.
  # History is allowed to continue between seasons for the same club.
  for (k in rolling_windows) {
    rolling_col <- paste0('rolling', k, '_response')

    df <- df %>%
      group_by(competition_id, team_id) %>%
      mutate(
        !!rolling_col := rollmeanr(
          lag(response),
          k = k,
          fill = NA_real_
        )
      ) %>%
      ungroup()
  }

  df
}

make_predictions <- function(df, k, evaluation_split, competition_ids) {
  rolling_col <- paste0('rolling', k, '_response')

  df %>%
    filter(
      split == evaluation_split,
      competition_id %in% competition_ids
    ) %>%
    transmute(
      country_name = if ('country_name' %in% names(df)) {
        country_name
      } else {
        NA_character_
      },
      competition_id,
      competition_name,
      season_id,
      season_name = if ('season_name' %in% names(df)) {
        season_name
      } else {
        NA_character_
      },
      match_id,
      match_date,
      team_id,
      team_name = if ('team_name' %in% names(df)) {
        team_name
      } else {
        NA_character_
      },
      opponent_id,
      opponent_name = if ('opponent_name' %in% names(df)) {
        opponent_name
      } else {
        NA_character_
      },
      split = as.character(split),
      target = 'xG_for',
      actual = xG_for,
      predicted = if (response_scale == 'log') {
        pmax(expm1(.data[[rolling_col]]), 0)
      } else {
        pmax(.data[[rolling_col]], 0)
      }
    ) %>%
    filter(is.finite(actual), is.finite(predicted))
}

pair_predictions <- function(predictions) {
  opponent <- predictions %>%
    select(
      competition_id, season_id, match_id, split,
      opponent_id = team_id,
      actual_xG_against = actual,
      predicted_xG_against = predicted
    )

  predictions %>%
    rename(
      actual_xG_for = actual,
      predicted_xG_for = predicted
    ) %>%
    inner_join(
      opponent,
      by = c(
        'competition_id', 'season_id', 'match_id',
        'split', 'opponent_id'
      )
    ) %>%
    mutate(
      opponent_team_id = opponent_id,
      actual_xG_diff = actual_xG_for - actual_xG_against,
      predicted_xG_diff = predicted_xG_for - predicted_xG_against
    )
}

make_long_predictions <- function(paired) {
  metadata <- c(
    'country_name', 'competition_id', 'competition_name',
    'season_id', 'season_name', 'match_id', 'match_date',
    'team_id', 'team_name', 'opponent_team_id', 'opponent_name',
    'split', 'held_out_competition_id',
    'held_out_competition_name', 'rolling_window'
  )

  bind_rows(
    paired %>%
      transmute(
        across(all_of(metadata)),
        target = 'xG_for',
        actual = actual_xG_for,
        predicted = predicted_xG_for
      ),
    paired %>%
      transmute(
        across(all_of(metadata)),
        target = 'xG_against',
        actual = actual_xG_against,
        predicted = predicted_xG_against
      ),
    paired %>%
      transmute(
        across(all_of(metadata)),
        target = 'xG_diff',
        actual = actual_xG_diff,
        predicted = predicted_xG_diff
      )
  )
}

# One LOCO fold ####

run_loco_fold <- function(df, held_out_competition_id) {
  held_out_info <- df %>%
    filter(competition_id == held_out_competition_id) %>%
    distinct(competition_id, competition_name) %>%
    slice(1)

  held_out_competition_name <- held_out_info$competition_name[[1]]
  training_competitions <- setdiff(
    unique(df$competition_id),
    held_out_competition_id
  )

  tuning_results <- map_dfr(rolling_windows, function(k) {
    validation_predictions <- make_predictions(
      df = df,
      k = k,
      evaluation_split = 'validation',
      competition_ids = training_competitions
    )

    summarise_metrics(
      actual = validation_predictions$actual,
      predicted = validation_predictions$predicted
    ) %>%
      mutate(
        script_version = script_version,
        model = 'Rolling mean',
        response_scale = response_scale,
        target = 'xG_for',
        held_out_competition_id = held_out_competition_id,
        held_out_competition_name = held_out_competition_name,
        rolling_window = k,
        split = 'validation',
        .before = 1
      )
  }) %>%
    arrange(mae, rolling_window)

  best_config <- tuning_results %>%
    filter(is.finite(mae)) %>%
    slice_min(mae, n = 1, with_ties = FALSE)

  if (nrow(best_config) == 0) {
    stop(
      'No valid validation predictions when holding out competition ',
      held_out_competition_id,
      call. = FALSE
    )
  }

  best_k <- best_config$rolling_window[[1]]

  test_xg_for <- make_predictions(
    df = df,
    k = best_k,
    evaluation_split = 'test',
    competition_ids = held_out_competition_id
  ) %>%
    mutate(
      held_out_competition_id = held_out_competition_id,
      held_out_competition_name = held_out_competition_name,
      rolling_window = best_k
    )

  test_paired <- pair_predictions(test_xg_for)
  test_long <- make_long_predictions(test_paired)

  test_metrics <- test_long %>%
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
      script_version = script_version,
      model = 'Rolling mean',
      response_scale = response_scale,
      held_out_competition_id = held_out_competition_id,
      held_out_competition_name = held_out_competition_name,
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
      script_version = script_version,
      model = 'Rolling mean',
      response_scale = response_scale,
      target = 'xG_diff',
      held_out_competition_id = held_out_competition_id,
      held_out_competition_name = held_out_competition_name,
      rolling_window = best_k,
      .before = 1
    )

  list(
    tuning_results = tuning_results,
    best_config = best_config,
    test_metrics = test_metrics,
    direction_accuracy = direction_accuracy,
    test_predictions = test_paired,
    test_predictions_long = test_long
  )
}

# Data ####

df_model <- read_csv(input_file, show_col_types = FALSE) %>%
  mutate(
    match_date = as.Date(match_date),
    across(
      c(competition_id, season_id, match_id, team_id, opponent_id),
      as.character
    ),
    split = as.character(split)
  )

validate_input(df_model)
df_model <- add_rolling_features(df_model)

competitions <- df_model %>%
  distinct(competition_id, competition_name) %>%
  arrange(competition_name, competition_id)

# Run LOCO ####

loco_folds <- map(
  competitions$competition_id,
  ~ run_loco_fold(df_model, .x)
)

loco_tuning_results <- map_dfr(loco_folds, 'tuning_results')
loco_best_config <- map_dfr(loco_folds, 'best_config')
loco_test_metrics_by_competition <- map_dfr(loco_folds, 'test_metrics')
loco_direction_by_competition <- map_dfr(loco_folds, 'direction_accuracy')
loco_test_predictions <- map_dfr(loco_folds, 'test_predictions')
loco_test_predictions_long <- map_dfr(loco_folds, 'test_predictions_long')

# Overall metrics pooled across all held-out competitions ####

loco_test_metrics <- loco_test_predictions_long %>%
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
    script_version = script_version,
    model = 'Rolling mean',
    response_scale = response_scale,
    window_selection = 'LOCO-specific validation choice',
    n_competitions = n_distinct(
      loco_test_predictions_long$held_out_competition_id
    ),
    .before = 1
  )

loco_direction_accuracy <- loco_test_predictions %>%
  summarise(
    n = n(),
    direction_accuracy = mean(
      sign(actual_xG_diff) == sign(predicted_xG_diff)
    )
  ) %>%
  mutate(
    script_version = script_version,
    model = 'Rolling mean',
    response_scale = response_scale,
    target = 'xG_diff',
    window_selection = 'LOCO-specific validation choice',
    n_competitions = n_distinct(
      loco_test_predictions$held_out_competition_id
    ),
    .before = 1
  )

# Distribution of competition-specific performance ####

loco_competition_summary <- loco_test_metrics_by_competition %>%
  group_by(target) %>%
  summarise(
    n_competitions = n(),
    mean_mae = mean(mae, na.rm = TRUE),
    median_mae = median(mae, na.rm = TRUE),
    q1_mae = quantile(mae, 0.25, na.rm = TRUE),
    q3_mae = quantile(mae, 0.75, na.rm = TRUE),
    mean_rmse = mean(rmse, na.rm = TRUE),
    median_rmse = median(rmse, na.rm = TRUE),
    mean_bias = mean(bias, na.rm = TRUE),
    median_bias = median(bias, na.rm = TRUE),
    .groups = 'drop'
  ) %>%
  mutate(
    script_version = script_version,
    model = 'Rolling mean',
    response_scale = response_scale,
    .before = 1
  )

run_config <- tibble(
  script_version = script_version,
  model = 'Rolling mean',
  analysis = 'Leave-one-competition-out',
  response_scale = response_scale,
  rolling_windows = paste(rolling_windows, collapse = ', '),
  history_grouping = 'competition_id + team_id',
  tuning_split = 'validation in non-held-out competitions',
  evaluation_split = 'test in held-out competition',
  forecasting_scheme = 'rolling one-step-ahead'
)

# Export ####

write_csv(run_config, output_file('run_config'))
write_csv(loco_tuning_results, output_file('tuning_results'))
write_csv(loco_best_config, output_file('best_config'))
write_csv(loco_test_metrics, output_file('test_metrics'))
write_csv(
  loco_test_metrics_by_competition,
  output_file('test_metrics_by_competition')
)
write_csv(loco_direction_accuracy, output_file('direction_accuracy'))
write_csv(
  loco_direction_by_competition,
  output_file('direction_accuracy_by_competition')
)
write_csv(
  loco_competition_summary,
  output_file('competition_summary')
)
write_csv(loco_test_predictions, output_file('test_predictions'))
write_csv(
  loco_test_predictions_long,
  output_file('test_predictions_long')
)

print(loco_best_config)
print(loco_test_metrics)
print(loco_competition_summary)

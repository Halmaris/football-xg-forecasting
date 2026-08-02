library(dplyr)
library(purrr)
library(readr)
library(lme4)

# Fits mixed-effects models for team-level xG_for forecasting.
# Input: df_model.csv with team-match observations, lagged and rolling predictors,
# grouping identifiers and train/validation/test splits.
# Outputs: selected window, test predictions and metrics, model coefficients,
# variance components and diagnostics, saved in results/.

input_file <- 'df_model.csv'
output_dir <- 'results'
response_scale <- 'log'  # 'log' or 'raw'
rolling_windows <- c(3, 5, 8, 10)

file_prefix <- paste0('lmm_', response_scale)
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

get_predictors <- function(k) {
  c(
    'lag1_xG_for',
    'lag1_xG_against',
    paste0('rolling', k, '_xG_for'),
    paste0('rolling', k, '_xG_against'),
    paste0('opponent_rolling', k, '_xG_for'),
    paste0('opponent_rolling', k, '_xG_against')
  )
}

prepare_data <- function(df, k) {
  predictors <- get_predictors(k)
  required <- c(
    'xG_for', predictors, 'home_away', 'competition_id',
    'team_season_id', 'opponent_season_id', 'match_id', 'team_id', 'split'
  )

  df %>%
    mutate(
      response = if (response_scale == 'log') log1p(xG_for) else xG_for,
      home_away = factor(home_away, levels = c('away', 'home')),
      across(
        c(competition_id, team_season_id, opponent_season_id),
        as.factor
      )
    ) %>%
    filter(if_all(all_of(required), ~ !is.na(.x)))
}

fit_scaler <- function(df, predictors) {
  tibble(
    variable = predictors,
    center = map_dbl(predictors, ~ mean(df[[.x]])),
    scale = map_dbl(predictors, ~ sd(df[[.x]]))
  )
}

apply_scaler <- function(df, scaler) {
  for (i in seq_len(nrow(scaler))) {
    variable <- scaler$variable[i]
    df[[paste0(variable, '_z')]] <-
      (df[[variable]] - scaler$center[i]) / scaler$scale[i]
  }
  df
}

fit_model <- function(df, k, reml) {
  predictors <- get_predictors(k)
  scaler <- fit_scaler(df, predictors)
  df_scaled <- apply_scaler(df, scaler)
  
  contrasts(df_scaled$home_away) <- contr.treatment(
    n = 2,
    base = 1
  )

  formula <- as.formula(paste(
    'response ~',
    paste(c(
      paste0(predictors, '_z'),
      'home_away',
      '(1 | competition_id)',
      '(1 | team_season_id)',
      '(1 | opponent_season_id)'
    ), collapse = ' + ')
  ))

  fit <- lmer(
    formula,
    data = df_scaled,
    REML = reml,
    control = lmerControl(
      optimizer = 'bobyqa',
      optCtrl = list(maxfun = 2e5)
    )
  )

  list(
    fit = fit,
    scaler = scaler,
    formula = formula,
    predictors = predictors,
    reml = reml
  )
}

predict_xg_for <- function(model, newdata) {
  newdata_scaled <- apply_scaler(newdata, model$scaler)
  prediction <- predict(
    model$fit,
    newdata = newdata_scaled,
    allow.new.levels = TRUE
  )

  if (response_scale == 'log') prediction <- expm1(prediction)

  tibble(
    match_id = newdata$match_id,
    team_id = newdata$team_id,
    split = newdata$split,
    target = 'xG_for',
    actual = newdata$xG_for,
    predicted = pmax(prediction, 0)
  )
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

df_model <- read_csv(input_file, show_col_types = FALSE)

tuning_results <- map_dfr(rolling_windows, function(k) {
  data <- prepare_data(df_model, k)
  train <- data %>% filter(split == 'train')
  validation <- data %>% filter(split == 'validation')
  model <- fit_model(train, k, reml = FALSE)
  predictions <- predict_xg_for(model, validation)

  summarise_metrics(predictions$actual, predictions$predicted) %>%
    mutate(
      response_scale = response_scale,
      target = 'xG_for',
      rolling_window = k,
      estimation = 'ML',
      model_type = 'lmer',
      random_effects = paste(
        c(
          '(1 | competition_id)',
          '(1 | team_season_id)',
          '(1 | opponent_season_id)'
        ),
        collapse = ' + '
      ),
      singular = isSingular(model$fit, tol = 1e-4),
      formula = paste(deparse(model$formula), collapse = ' '),
      .before = 1
    )
}) %>%
  arrange(mae)

best_config <- tuning_results %>%
  slice_min(mae, n = 1, with_ties = FALSE)

best_k <- best_config$rolling_window[[1]]
final_data <- prepare_data(df_model, best_k)
train_validation <- final_data %>% filter(split %in% c('train', 'validation'))
test <- final_data %>% filter(split == 'test')
final_model <- fit_model(train_validation, best_k, reml = TRUE)

test_xg_for <- predict_xg_for(final_model, test)
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
    model = 'Mixed-effects model',
    response_scale = response_scale,
    rolling_window = best_k,
    estimation = 'REML',
    model_type = 'lmer',
    random_effects = paste(
      c(
        '(1 | competition_id)',
        '(1 | team_season_id)',
        '(1 | opponent_season_id)'
      ),
      collapse = ' + '
    ),
    singular = isSingular(final_model$fit, tol = 1e-4),
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
    model = 'Mixed-effects model',
    target = 'xG_diff',
    response_scale = response_scale,
    rolling_window = best_k,
    .before = 1
  )

best_config <- best_config %>%
  mutate(
    model = 'Mixed-effects model',
    final_estimation = 'REML',
    .before = 1
  )

model_formula <- tibble(
  model = 'Mixed-effects model',
  response_scale = response_scale,
  estimation = 'REML',
  formula = paste(deparse(final_model$formula), collapse = ' ')
)

coefficient_matrix <- coef(summary(final_model$fit))
fixed_effects <- as.data.frame(coefficient_matrix) %>%
  tibble::rownames_to_column('term') %>%
  as_tibble() %>%
  transmute(
    term,
    estimate = Estimate,
    std_error = `Std. Error`,
    statistic = `t value`,
    conf_low = estimate - 1.96 * std_error,
    conf_high = estimate + 1.96 * std_error,
    multiplicative_factor_1plus_xg = if (response_scale == 'log') {
      exp(estimate)
    } else {
      NA_real_
    }
  )

random_effects <- as.data.frame(VarCorr(final_model$fit)) %>%
  as_tibble() %>%
  mutate(
    variance_share = vcov / sum(vcov),
    variance_share_percent = 100 * variance_share
  )

grouping_levels <- tibble(
  grouping_factor = names(ngrps(final_model$fit)),
  n_levels = as.integer(ngrps(final_model$fit))
)

new_level_summary <- map_dfr(
  c('competition_id', 'team_season_id', 'opponent_season_id'),
  function(group) {
    train_levels <- unique(as.character(train_validation[[group]]))
    test_values <- as.character(test[[group]])
    new_level <- !test_values %in% train_levels

    tibble(
      grouping_factor = group,
      test_observations = length(test_values),
      observations_with_new_level = sum(new_level),
      percent_with_new_level = 100 * mean(new_level),
      unique_test_levels = n_distinct(test_values),
      unique_new_levels = n_distinct(test_values[new_level])
    )
  }
)

convergence_messages <- final_model$fit@optinfo$conv$lme4$messages
dropped_columns <- attr(getME(final_model$fit, 'X'), 'col.dropped')

diagnostics <- tibble(
  model_type = 'lmer',
  estimation = 'REML',
  n_observations = nobs(final_model$fit),
  n_fixed_effects = length(fixef(final_model$fit)),
  n_dropped_fixed_effect_columns = ifelse(
    is.null(dropped_columns), 0L, length(dropped_columns)
  ),
  singular = isSingular(final_model$fit, tol = 1e-4),
  convergence_message = ifelse(
    is.null(convergence_messages),
    '',
    paste(convergence_messages, collapse = ' | ')
  ),
  log_likelihood = as.numeric(logLik(final_model$fit)),
  aic = AIC(final_model$fit),
  bic = BIC(final_model$fit),
  residual_sd = sigma(final_model$fit)
)

write_csv(tuning_results, output_file('tuning_results'))
write_csv(best_config, output_file('best_config'))
write_csv(test_metrics, output_file('test_metrics'))
write_csv(direction_accuracy, output_file('direction_accuracy'))
write_csv(test_paired, output_file('test_predictions'))
write_csv(test_long, output_file('test_predictions_long'))
write_csv(model_formula, output_file('formula'))
write_csv(fixed_effects, output_file('fixed_effects'))
write_csv(random_effects, output_file('random_effects'))
write_csv(grouping_levels, output_file('grouping_levels'))
write_csv(new_level_summary, output_file('new_level_summary'))
write_csv(diagnostics, output_file('diagnostics'))
write_csv(final_model$scaler, output_file('scaler'))

print(tuning_results)
print(test_metrics)
print(summary(final_model$fit))

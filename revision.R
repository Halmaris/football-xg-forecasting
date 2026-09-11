# Run from the project directory: source('revision.R')
library(dplyr)
library(readr)
library(lme4)
library(zoo)
library(ggplot2)
library(ggrepel)

# Settings ----
output_dir <- 'results/revision'
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(output_dir, 'tables'), showWarnings = FALSE)
rolling_windows <- c(3, 5, 8, 10)
validation_start <- as.Date('2024-07-01')
test_start <- as.Date('2025-07-01')
n_boot <- 5000
seed <- 20260910

read_result <- function(name) read_csv(file.path('results', name), show_col_types = FALSE)
save_csv <- function(df, name) write_csv(df, file.path(output_dir, paste0(name, '.csv')))
df <- read_csv('df_model.csv', show_col_types = FALSE)
keys <- c('match_id', 'team_id')
targets <- c('xG_for', 'xG_diff')
metadata <- select(df, all_of(keys), competition_id, season_id)
stopifnot(!anyDuplicated(df[keys]))

# Shared statistical functions ----
lmm_predictors <- function(k) c('lag1_xG_for', 'lag1_xG_against',
  paste0('rolling', k, c('_xG_for', '_xG_against')),
  paste0('opponent_rolling', k, c('_xG_for', '_xG_against')))

prepare_lmm <- function(df, k) df %>%
  mutate(response = log1p(xG_for), home_away = factor(home_away, c('away', 'home')),
    across(c(competition_id, team_season_id, opponent_season_id), as.factor)) %>%
  filter(if_all(all_of(c('xG_for', lmm_predictors(k), 'home_away',
    'competition_id', 'team_season_id', 'opponent_season_id')), ~ !is.na(.x)))

scale_lmm <- function(df, scaler) {
  for (i in seq_len(nrow(scaler))) {
    x <- scaler$variable[i]
    df[[paste0(x, '_z')]] <- (df[[x]] - scaler$center[i]) / scaler$scale[i]
  }
  df
}

fit_lmm <- function(df, k = NULL, reml = TRUE) {
  predictors <- if (is.null(k)) character() else lmm_predictors(k)
  scaler <- tibble(variable = predictors,
    center = vapply(df[predictors], mean, numeric(1)),
    scale = vapply(df[predictors], sd, numeric(1)))
  df <- scale_lmm(df, scaler)
  contrasts(df$home_away) <- contr.treatment(2, base = 1)
  scaled <- if (length(predictors)) paste0(predictors, '_z') else character()
  formula <- as.formula(paste('response ~', paste(c(scaled, 'home_away', '(1 | competition_id)',
    '(1 | team_season_id)', '(1 | opponent_season_id)'), collapse = ' + ')))
  fit <- lmer(formula, data = df, REML = reml,
    control = lmerControl(optimizer = 'bobyqa', optCtrl = list(maxfun = 2e5)))
  list(fit = fit, scaler = scaler)
}

predict_log <- function(model, df) as.numeric(predict(model$fit,
  newdata = scale_lmm(df, model$scaler), allow.new.levels = TRUE))

pair_forecasts <- function(df, predicted) df %>%
  transmute(match_id, team_id, actual_xG_for = xG_for, predicted_xG_for = predicted) %>%
  group_by(match_id) %>% filter(n() == 2L) %>%
  mutate(opponent_team_id = rev(team_id), actual_xG_against = rev(actual_xG_for),
    predicted_xG_against = rev(predicted_xG_for),
    actual_xG_diff = actual_xG_for - actual_xG_against,
    predicted_xG_diff = predicted_xG_for - predicted_xG_against) %>% ungroup()

read_predictions <- function(name, variant = NULL) {
  df <- read_result(name)
  if (!is.null(variant)) df <- filter(df, .data$variant == .env$variant)
  df <- df %>% select(all_of(keys), starts_with('actual_xG'), starts_with('predicted_xG')) %>%
    filter(if_all(-all_of(keys), is.finite)) %>%
    group_by(match_id) %>% filter(n() == 2L) %>% ungroup()
  stopifnot(!anyDuplicated(df[keys]), nrow(df) > 0)
  df
}

forecast_metrics <- function(df) bind_rows(lapply(targets, function(target) {
  error <- df[[paste0('predicted_', target)]] - df[[paste0('actual_', target)]]
  tibble(target, n = length(error), mae = mean(abs(error)),
    rmse = sqrt(mean(error^2)), bias = mean(error))
}))

# Whole matches or competition-seasons are sampled; the estimand is row-weighted MAE.
bootstrap_mae <- function(df, unit, seed) {
  groups <- if (unit == 'match') 'match_id' else c('competition_id', 'season_id')
  units <- df %>% group_by(across(all_of(groups))) %>%
    summarise(total = sum(difference), n = n(), .groups = 'drop')
  set.seed(seed)
  values <- replicate(n_boot, {
    i <- sample.int(nrow(units), nrow(units), replace = TRUE)
    sum(units$total[i]) / sum(units$n[i])
  })
  ci <- quantile(values, c(0.025, 0.975))
  list(summary = tibble(n_units = nrow(units), ci_low = ci[1], ci_high = ci[2],
    p_boot = min(1, 2 * (1 + min(sum(values <= 0), sum(values >= 0))) / (n_boot + 1))),
    values = values)
}

paired_errors <- function(a, b, target) {
  p <- inner_join(a, b, by = keys, suffix = c('_1', '_2'), relationship = 'one-to-one')
  actual <- p[[paste0('actual_', target, '_1')]]
  stopifnot(all(table(p$match_id) == 2L),
    max(abs(actual - p[[paste0('actual_', target, '_2')]])) < 1e-8)
  p %>% transmute(match_id, team_id,
    error_1 = abs(.data[[paste0('predicted_', target, '_1')]] - actual),
    error_2 = abs(.data[[paste0('predicted_', target, '_2')]] - actual),
    difference = error_1 - error_2)
}

# 1. LMM history ablation and validation smearing ----
message('LMM history ablation and smearing')
k <- read_result('lmm_log_best_config.csv')$rolling_window[1]
lmm_data <- prepare_lmm(df, k)
train <- filter(lmm_data, split == 'train')
validation <- filter(lmm_data, split == 'validation')
fit_data <- filter(lmm_data, split != 'test')
test <- filter(lmm_data, split == 'test')
lmm_validation <- fit_lmm(train, k, reml = FALSE)
lmm_full <- fit_lmm(fit_data, k)
lmm_context <- fit_lmm(fit_data)
saveRDS(list(validation = lmm_validation, full = lmm_full, context = lmm_context),
  file.path(output_dir, 'lmm_models.rds'))

calibration <- validation %>% transmute(match_id, team_id, split,
  actual_xG_for = xG_for, predicted_log = predict_log(lmm_validation, validation),
  log_residual = log1p(xG_for) - predicted_log)
lmm_factor <- mean(exp(calibration$log_residual))
save_csv(calibration, 'lmm_validation_calibration')
test_log <- predict_log(lmm_full, test)
lmm_predictions <- bind_rows(
  full = pair_forecasts(test, pmax(expm1(test_log), 0)),
  full_smearing = pair_forecasts(test, pmax(lmm_factor * exp(test_log) - 1, 0)),
  context_only = pair_forecasts(test, pmax(expm1(predict_log(lmm_context, test)), 0)),
  .id = 'variant')
save_csv(lmm_predictions, 'lmm_test_predictions')

# Validation forecasts come from the train-only selected XGBoost model.
# The original 6_model_xgb.py exports these alongside its test forecasts.
calibration <- read_result('xgb_log_validation_predictions.csv')
stopifnot(all(calibration$split == 'validation'),
  !anyDuplicated(calibration[keys]), all(is.finite(calibration$predicted_log)))
xgb_factor <- mean(exp(log1p(calibration$actual_xG_for) - calibration$predicted_log))
xgb_full <- read_predictions('xgb_log_test_predictions.csv')
stopifnot(all(xgb_full$predicted_xG_for > 0))
xgb_predictions <- bind_rows(full = xgb_full,
  full_smearing = pair_forecasts(rename(xgb_full, xG_for = actual_xG_for),
    pmax(xgb_factor * (1 + xgb_full$predicted_xG_for) - 1, 0)), .id = 'variant')
save_csv(xgb_predictions, 'xgb_test_predictions')
save_csv(tibble(model = c('LMM', 'XGBoost'), factor = c(lmm_factor, xgb_factor),
  n_calibration = c(nrow(validation), nrow(calibration)), fit_split = 'train',
  calibration_split = 'validation'), 'smearing_factors')

smearing_metrics <- bind_rows(lapply(c('LMM', 'XGBoost'), function(model) {
  p <- if (model == 'LMM') lmm_predictions else xgb_predictions
  prefix <- if (model == 'LMM') 'lmm' else 'xgb'
  bind_rows(log = forecast_metrics(filter(p, variant == 'full')),
    smearing = forecast_metrics(filter(p, variant == 'full_smearing')),
    raw = forecast_metrics(read_predictions(paste0(prefix, '_raw_test_predictions.csv'))),
    .id = 'variant') %>% mutate(model, .before = 1)
}))
save_csv(smearing_metrics, 'smearing_scale_comparison')

# 2. Paired history and matched-information comparisons ----
message('Paired bootstrap comparisons')
comparisons <- list(
  'LMM history ablation' = list(filter(lmm_predictions, variant == 'context_only'),
    filter(lmm_predictions, variant == 'full')),
  'Matched-information comparison' = list(
    read_predictions('xgb_ablation_log_test_predictions.csv', 'matched_information'),
    read_predictions('tcn_log_test_predictions.csv')))
inference <- draws <- list()
for (i in seq_along(comparisons)) {
  for (j in seq_along(targets)) {
    p <- paired_errors(comparisons[[i]][[1]], comparisons[[i]][[2]], targets[j]) %>%
      left_join(metadata, by = keys, relationship = 'one-to-one')
    for (u in seq_along(c('match', 'competition_season'))) {
      unit <- c('match', 'competition_season')[u]
      contrast_seed <- seed + (i - 1) * 100 + (j - 1) * 10 + u - 1
      boot <- bootstrap_mae(p, unit, contrast_seed)
      label <- tibble(comparison = names(comparisons)[i], target = targets[j], unit)
      inference[[length(inference) + 1]] <- bind_cols(label,
        tibble(n = nrow(p), n_matches = n_distinct(p$match_id),
          mae_1 = mean(p$error_1), mae_2 = mean(p$error_2),
          delta_mae = mean(p$difference), seed = contrast_seed, n_boot), boot$summary)
      draws[[length(draws) + 1]] <- bind_cols(label,
        tibble(replicate = seq_len(n_boot), delta_mae = boot$values))
    }
  }
}
inference <- bind_rows(inference) %>% group_by(unit) %>%
  mutate(p_holm = p.adjust(p_boot, method = 'holm')) %>% ungroup()
save_csv(inference, 'paired_mae_comparisons')
save_csv(bind_rows(draws), 'paired_bootstrap_replicates')

# 3. Calendar-time forecasts and previously unseen seasons ----
message('Calendar-time validation')
calendar <- df %>% mutate(match_date = as.Date(match_date), original_split = split,
  split = case_when(match_date < validation_start ~ 'train',
    match_date < test_start ~ 'validation', TRUE ~ 'test')) %>%
  arrange(team_id, match_date, match_id)
for (k in rolling_windows) {
  calendar <- calendar %>% group_by(team_id) %>%
    mutate(!!paste0('baseline_', k) := expm1(rollmeanr(lag(log1p(xG_for)), k,
      fill = NA_real_)), last_history_date = lag(match_date)) %>% ungroup()
}
stopifnot(all(calendar$last_history_date < calendar$match_date, na.rm = TRUE))
required <- c(unique(unlist(lapply(rolling_windows, lmm_predictors))),
  paste0('baseline_', rolling_windows), 'xG_for')
calendar <- prepare_lmm(calendar, 3) %>% filter(if_all(all_of(required), is.finite)) %>%
  group_by(match_id) %>% filter(n() == 2L) %>% ungroup()
save_csv(calendar %>% group_by(split) %>% summarise(n = n(),
  matches = n_distinct(match_id), competitions = n_distinct(competition_id),
  first_date = min(match_date), last_date = max(match_date), .groups = 'drop'),
  'calendar_split_counts')
train <- filter(calendar, split == 'train')
validation <- filter(calendar, split == 'validation')
fit_data <- filter(calendar, split != 'test')
test <- filter(calendar, split == 'test')
stopifnot(max(train$match_date) < min(validation$match_date),
  max(fit_data$match_date) < min(test$match_date))
tuning <- bind_rows(lapply(rolling_windows, function(k) {
  message('Calendar LMM: K = ', k)
  fitted_model <- fit_lmm(train, k, reml = FALSE)
  lmm_mae <- mean(abs(pmax(expm1(predict_log(fitted_model, validation)), 0) - validation$xG_for))
  rolling_mae <- mean(abs(validation[[paste0('baseline_', k)]] - validation$xG_for))
  tibble(model = c('LMM full', 'Rolling mean'), k, mae = c(lmm_mae, rolling_mae))
}))
selected <- tuning %>% arrange(mae, k) %>% group_by(model) %>% slice(1) %>% ungroup()
save_csv(tuning, 'calendar_validation')
save_csv(selected, 'calendar_selected_windows')
calendar_full <- fit_lmm(fit_data, selected$k[selected$model == 'LMM full'])
calendar_context <- fit_lmm(fit_data)
saveRDS(list(full = calendar_full, context = calendar_context),
  file.path(output_dir, 'calendar_models.rds'))
calendar_metadata <- test %>% transmute(match_id, team_id, match_date,
  competition_id, season_id, original_split,
  new_team_season = !team_season_id %in% fit_data$team_season_id,
  new_opponent_season = !opponent_season_id %in% fit_data$opponent_season_id,
  new_competition_season = !paste(competition_id, season_id) %in%
    paste(fit_data$competition_id, fit_data$season_id))
calendar_predictions <- bind_rows(
  'LMM full' = pair_forecasts(test, pmax(expm1(predict_log(calendar_full, test)), 0)),
  'LMM context only' = pair_forecasts(test, pmax(expm1(predict_log(calendar_context, test)), 0)),
  'Rolling mean' = pair_forecasts(test, test[[paste0('baseline_',
    selected$k[selected$model == 'Rolling mean'])]]), .id = 'model') %>%
  left_join(calendar_metadata, by = keys, relationship = 'many-to-one')
save_csv(calendar_predictions, 'calendar_test_predictions')
save_csv(calendar_metadata %>% count(new_competition_season, new_team_season,
  new_opponent_season, name = 'n'), 'calendar_new_levels')
cohorts <- list(All = calendar_predictions,
  'Observed season' = filter(calendar_predictions, !new_competition_season),
  'New season' = filter(calendar_predictions, new_competition_season))
calendar_metrics <- bind_rows(lapply(cohorts, function(p) {
  bind_rows(lapply(split(p, p$model), forecast_metrics), .id = 'model')
}), .id = 'cohort')
save_csv(calendar_metrics, 'calendar_metrics')
calendar_intervals <- calendar_draws <- list()
for (cohort in names(cohorts)) {
  p <- cohorts[[cohort]]
  for (comparator in c('LMM context only', 'Rolling mean')) {
    for (target in targets) {
      errors <- paired_errors(filter(p, model == comparator),
        filter(p, model == 'LMM full'), target) %>%
        left_join(select(calendar_metadata, all_of(keys), competition_id, season_id),
          by = keys, relationship = 'one-to-one')
      contrast_seed <- seed + 1 + length(calendar_intervals)
      boot <- bootstrap_mae(errors, 'competition_season', contrast_seed)
      label <- tibble(cohort, comparator, target)
      calendar_intervals[[length(calendar_intervals) + 1]] <- bind_cols(label,
        tibble(n = nrow(errors), difference = mean(errors$difference), seed = contrast_seed),
        select(boot$summary, panels = n_units, ci_low, ci_high))
      calendar_draws[[length(calendar_draws) + 1]] <- bind_cols(label,
        tibble(replicate = seq_len(n_boot), delta_mae = boot$values))
    }
  }
}
save_csv(bind_rows(calendar_intervals), 'calendar_paired_differences')
save_csv(bind_rows(calendar_draws), 'calendar_bootstrap_replicates')

# 4. LMM residuals, competition coverage and TCN capacity ----
message('Residual and coverage diagnostics')
model <- lmm_full$fit
residual_data <- tibble(fitted_log = as.numeric(fitted(model)),
  residual_log = as.numeric(residuals(model)),
  standardized_residual = residual_log / sigma(model), fitted_decile = ntile(fitted_log, 10))
residual_bins <- residual_data %>% group_by(fitted_decile) %>% summarise(n = n(),
  fitted_mean = mean(fitted_log), residual_mean = mean(residual_log),
  residual_sd = sd(residual_log), .groups = 'drop')
z <- residual_data$standardized_residual - mean(residual_data$standardized_residual)
save_csv(residual_data, 'lmm_conditional_residuals')
save_csv(residual_bins, 'lmm_residual_fitted_deciles')
save_csv(tibble(n = nobs(model), residual_sd = sigma(model),
  mean_residual = mean(residual_data$residual_log), skewness = mean(z^3) / mean(z^2)^1.5,
  excess_kurtosis = mean(z^4) / mean(z^2)^2 - 3,
  min_decile_sd = min(residual_bins$residual_sd), max_decile_sd = max(residual_bins$residual_sd),
  min_decile_mean = min(residual_bins$residual_mean), max_decile_mean = max(residual_bins$residual_mean)),
  'lmm_residual_diagnostics')
models <- list(lmm_validation = lmm_validation, lmm_full = lmm_full,
  lmm_context = lmm_context, calendar_full = calendar_full, calendar_context = calendar_context)
save_csv(bind_rows(lapply(models, function(m) tibble(n = nobs(m$fit),
  reml = isREML(m$fit), singular = isSingular(m$fit),
  convergence = paste(m$fit@optinfo$conv$lme4$messages, collapse = ' | '))), .id = 'model'),
  'lmm_fit_diagnostics')

panels <- df %>% distinct(competition_id, country_name, competition_name, season_id,
  season_name) %>% mutate(season_end = as.integer(substr(season_name,
    nchar(season_name) - 3, nchar(season_name))))
competitions <- panels %>% group_by(competition_id, country_name, competition_name) %>%
  summarise(n_seasons = n(), .groups = 'drop')
rolling <- read_predictions('rolling_log_test_predictions.csv') %>%
  left_join(metadata, by = keys, relationship = 'one-to-one')
rolling_metrics <- function(df, ...) df %>% group_by(...) %>% summarise(n = n(),
  mae_xgf = mean(abs(predicted_xG_for - actual_xG_for)),
  mae_xgd = mean(abs(predicted_xG_diff - actual_xG_diff)),
  bias_xgf = mean(predicted_xG_for - actual_xG_for), .groups = 'drop')
by_competition <- rolling_metrics(rolling, competition_id) %>%
  left_join(competitions, by = 'competition_id', relationship = 'one-to-one')
by_season <- rolling_metrics(rolling, competition_id, season_id) %>%
  left_join(panels, by = c('competition_id', 'season_id'), relationship = 'one-to-one')
coverage <- rolling %>% left_join(select(competitions, competition_id, n_seasons),
  by = 'competition_id', relationship = 'many-to-one') %>%
  mutate(length_group = case_when(n_seasons <= 3 ~ '2-3 seasons',
    n_seasons == 5 ~ '5 seasons', n_seasons >= 6 ~ '6-8 seasons'))
coverage_metrics <- rolling_metrics(coverage, length_group) %>% left_join(
  coverage %>% group_by(length_group) %>% summarise(n_competitions = n_distinct(competition_id),
    n_panels = n_distinct(competition_id, season_id), .groups = 'drop'), by = 'length_group')
long_panels <- by_season %>% filter(competition_name %in% c('Ekstraklasa', 'La Liga 2')) %>%
  arrange(country_name, season_end)
save_csv(by_competition, 'rolling_errors_by_competition')
save_csv(by_season, 'rolling_errors_by_season')
save_csv(coverage_metrics, 'rolling_errors_by_coverage')
save_csv(long_panels, 'rolling_errors_long_coverage')
tcn_tuning <- read_result('tcn_log_tuning_results.csv') %>% mutate(
  parameter_count = filters * kernel_size + filters +
    (n_blocks - 1) * (filters^2 * kernel_size + filters) +
    (filters + 1) * dense_units + 2 * dense_units + 1)
save_csv(tcn_tuning, 'tcn_tuning_capacity')
save_csv(filter(tcn_tuning, trial == read_result('tcn_log_best_config.csv')$selected_trial[1]),
  'tcn_selected_capacity')
save_csv(read_result('bootstrap_log_mae_differences_vs_rolling.csv') %>%
  filter(model == 'Mixed-effects model'), 'paired_effect_sizes')

# 5. Extreme-match errors, binned calibration and excluded matches ----
predictions <- list('Rolling mean' = rolling,
  ARIMA = read_predictions('arima_log_test_predictions.csv'),
  LMM = read_predictions('lmm_log_test_predictions.csv'), XGBoost = xgb_full,
  TCN = read_predictions('tcn_log_test_predictions.csv'),
  'LMM smearing' = filter(lmm_predictions, variant == 'full_smearing'),
  'XGBoost smearing' = filter(xgb_predictions, variant == 'full_smearing'))
common_keys <- Reduce(function(a, b) inner_join(a, b, by = keys, relationship = 'one-to-one'),
  lapply(predictions, function(p) select(p, all_of(keys))))
threshold <- quantile(df$xG_for[df$split == 'train'], 0.9)
point_diagnostics <- bind_rows(lapply(names(predictions), function(model) {
  p <- semi_join(predictions[[model]], common_keys, by = keys)
  stopifnot(all(table(p$match_id) == 2L))
  extreme <- filter(p, actual_xG_for >= threshold)
  correct <- sign(p$actual_xG_diff) == sign(p$predicted_xG_diff)
  error <- extreme$predicted_xG_for - extreme$actual_xG_for
  tibble(model, common_n = nrow(p), threshold = as.numeric(threshold), n = nrow(extreme),
    mae = mean(abs(error)), rmse = sqrt(mean(error^2)), bias = mean(error),
    direction_accuracy = 100 * mean(correct),
    weighted_direction_accuracy = 100 * weighted.mean(correct, abs(p$actual_xG_diff)))
}))
save_csv(point_diagnostics, 'point_forecast_diagnostics')
calibration_bins <- bind_rows(lapply(c('LMM', 'LMM smearing', 'XGBoost', 'XGBoost smearing'),
  function(model) predictions[[model]] %>% arrange(predicted_xG_for, match_id, team_id) %>%
    mutate(bin = floor((row_number() - 1) * 10 / n()) + 1) %>% group_by(bin) %>%
    summarise(n = n(), forecast = mean(predicted_xG_for), observed = mean(actual_xG_for),
      .groups = 'drop') %>% mutate(model)))
save_csv(calibration_bins, 'forecast_binned_means')
raw <- read_csv('df.csv', col_select = c(match_id, competition_id, season_id,
  country_name, competition_name, season_name, period), show_col_types = FALSE)
matches <- raw %>% group_by(match_id, competition_id, season_id, country_name,
  competition_name, season_name) %>% summarise(extra_time = any(period %in% c(3, 4)),
    shootout = any(period == 5), .groups = 'drop') %>%
  mutate(excluded = extra_time | shootout, both = extra_time & shootout)
stopifnot(!anyDuplicated(matches$match_id))
save_csv(matches %>% group_by(country_name, competition_name) %>%
  summarise(source_matches = n(), excluded = sum(excluded), extra_time = sum(extra_time),
    shootout = sum(shootout), both = sum(both), percent = 100 * excluded / source_matches,
    .groups = 'drop'), 'exclusions_by_competition')
save_csv(filter(matches, excluded), 'excluded_matches')
save_csv(matches %>% summarise(source_matches = n(), excluded = sum(excluded),
  extra_time = sum(extra_time), shootout = sum(shootout), both = sum(both),
  retained = source_matches - excluded, percent = 100 * excluded / source_matches),
  'exclusion_summary')

# 6. Publication figures ----
message('Figures and tables')
out <- function(name) file.path(output_dir, name)
ink <- '#222222'
gray <- '#888888'
league_colors <- c('Ekstraklasa' = '#0072B2', 'La Liga 2' = '#D55E00')
start_plot <- function(name, height, title) {
  pdf(out(name), width = 9.1, height = height, family = 'Helvetica',
    useDingbats = FALSE, title = title)
  par(mfrow = c(1, 2), mar = c(4.2, 4.3, 2.1, 0.9), mgp = c(2.6, 0.7, 0),
    cex = 0.86, col.axis = ink, col.lab = ink, col.main = ink, fg = ink)
}

df_resid <- residual_data
z <- df_resid$standardized_residual
qq_indices <- unique(round(seq(1, length(z), length.out = 3000)))
start_plot('lmm_residual_diagnostics.pdf', 3.8,
  'Conditional residual diagnostics for the mixed-effects model')
plot(qnorm(ppoints(length(z)))[qq_indices], sort(z)[qq_indices], pch = 16,
  cex = 0.42, col = adjustcolor(ink, alpha.f = 0.55),
  xlab = 'Standard normal quantile', ylab = 'Conditional residual / residual SD',
  main = '(A)  Normal Q-Q plot')
abline(a = 0, b = 1, col = gray, lty = 2, lwd = 1.1)
plot(df_resid$fitted_log, df_resid$residual_log, pch = 16, cex = 0.25,
  col = adjustcolor(ink, alpha.f = 0.11), xaxs = 'i',
  xlab = expression('Fitted ' * log(1 + xG^F)), ylab = 'Conditional residual',
  main = '(B)  Residuals versus fitted values')
abline(h = 0, col = gray, lty = 2)
smooth_x <- seq(min(df_resid$fitted_log), max(df_resid$fitted_log), length.out = 401)
bandwidth <- 0.10
# Gaussian weights estimate the local mean and SD from all observed residuals.
smooth <- vapply(smooth_x, function(x) {
  weights <- exp(-0.5 * ((df_resid$fitted_log - x) / bandwidth)^2)
  weights <- weights / sum(weights)
  mean <- sum(weights * df_resid$residual_log)
  c(mean = mean, sd = sqrt(sum(weights * (df_resid$residual_log - mean)^2)))
}, numeric(2))
stopifnot(all(is.finite(smooth)), all(smooth['sd', ] > 0))
lines(smooth_x, smooth['mean', ], col = 'black', lty = 1, lwd = 1.35)
for (sign in c(-1, 1)) {
  lines(smooth_x, smooth['mean', ] + sign * smooth['sd', ],
    col = 'black', lty = 2, lwd = 1.05)
}
dev.off()

points_data <- by_competition[order(by_competition$n_seasons, by_competition$mae_xgf), ]
offsets <- ave(points_data$n_seasons, points_data$n_seasons,
  FUN = function(x) if (length(x) == 1L) 0 else seq(-0.16, 0.16, length.out = length(x)))
points_data$x_position <- points_data$n_seasons + offsets
points_data$highlight <- ifelse(points_data$competition_name %in% names(league_colors),
  points_data$competition_name, 'Other competitions')
points_data$display_label <- points_data$competition_name
labels <- c('Bundesliga' = 'Bundesliga (AUT)', 'Superliga' = 'Superliga (DEN)',
  'Super Liga' = 'Super Liga (SVK)', 'Primera División' = 'Primera Division (URU)')
matched <- points_data$competition_name %in% names(labels)
points_data$display_label[matched] <- labels[points_data$competition_name[matched]]
style <- theme_classic(base_size = 11, base_family = 'Helvetica') +
  theme(text = element_text(colour = ink),
    axis.text = element_text(colour = ink),
    panel.border = element_rect(colour = ink, fill = NA, linewidth = 0.4),
    axis.line = element_blank(), plot.title = element_text(size = 12),
    plot.margin = margin(12, 8, 8, 6))
left <- ggplot(points_data, aes(x_position, mae_xgf)) +
  geom_point(aes(colour = highlight, shape = highlight), size = 2.1) +
  geom_text_repel(aes(label = display_label),
    colour = ink, size = 3.5, box.padding = 0.20, point.padding = 0.16,
    min.segment.length = 0, segment.colour = '#999999', segment.size = 0.25,
    max.overlaps = Inf, max.iter = 20000, max.time = Inf, seed = 20260910,
    force = 2, force_pull = 0.12) +
  scale_colour_manual(values = c(league_colors, 'Other competitions' = 'black')) +
  scale_shape_manual(values = c('Ekstraklasa' = 16, 'La Liga 2' = 17,
    'Other competitions' = 16)) +
  scale_x_continuous(breaks = 2:8, limits = c(0.5, 9.2),
    expand = expansion(mult = 0)) +
  scale_y_continuous(limits = c(0.47, 0.72), breaks = seq(0.50, 0.70, 0.05),
    expand = expansion(mult = 0)) +
  labs(x = 'Number of observed seasons', y = expression('Test MAE for ' * xG^F),
    title = '(A)  Competition-level error') + style +
  theme(legend.position = 'none')
right <- ggplot(long_panels, aes(season_end, mae_xgf,
  colour = competition_name, linetype = competition_name, shape = competition_name)) +
  geom_line(linewidth = 0.55) + geom_point(size = 2.1) +
  scale_colour_manual(values = league_colors) +
  scale_linetype_manual(values = c('Ekstraklasa' = 1, 'La Liga 2' = 2)) +
  scale_shape_manual(values = c('Ekstraklasa' = 16, 'La Liga 2' = 17)) +
  scale_x_continuous(breaks = 2019:2026) +
  scale_y_continuous(limits = c(0.47, 0.72), breaks = seq(0.50, 0.70, 0.05),
    expand = expansion(mult = 0)) +
  labs(x = 'Season ending year', y = expression('Test MAE for ' * xG^F),
    title = '(B)  Seasonal errors in two leagues') + style +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5),
    legend.position = 'inside', legend.position.inside = c(0.97, 0.97),
    legend.justification = c(1, 1), legend.title = element_blank(),
    legend.background = element_blank(), legend.key.height = grid::unit(0.45, 'cm'))
pdf(out('rolling_error_stability.pdf'), width = 10.6, height = 5.2,
  family = 'Helvetica', useDingbats = FALSE, title = 'Rolling forecast error by coverage and season')
grid::grid.newpage()
left_grob <- ggplotGrob(left)
right_grob <- ggplotGrob(right)
heights <- grid::unit.pmax(left_grob$heights, right_grob$heights)
left_grob$heights <- heights
right_grob$heights <- heights
grid::pushViewport(grid::viewport(x = 0.29, width = 0.58))
grid::grid.draw(left_grob)
grid::popViewport()
grid::pushViewport(grid::viewport(x = 0.79, width = 0.42))
grid::grid.draw(right_grob)
grid::popViewport()
dev.off()

lims <- range(c(calibration_bins$forecast, calibration_bins$observed)) + c(-0.08, 0.08)
cairo_pdf(file.path(output_dir, 'forecast_calibration.pdf'), width = 8.5, height = 3.55,
  family = 'Arial', pointsize = 10)
par(mfrow = c(1, 2), mar = c(4.0, 4.0, 2.5, 0.8), mgp = c(2.4, 0.7, 0),
  tcl = -0.25, las = 1, bty = 'l')
for (i in seq_along(c('LMM', 'XGBoost'))) {
  model <- c('LMM', 'XGBoost')[i]
  plot(NA, xlim = lims, ylim = lims, xlab = 'Mean forecast xG',
    ylab = if (i == 1L) 'Mean observed xG' else '', axes = FALSE)
  abline(h = pretty(lims), v = pretty(lims), col = '#EEEEEE', lwd = 0.5)
  abline(a = 0, b = 1, lty = 2, col = '#888888', lwd = 0.8)
  for (j in 1:2) {
    d <- calibration_bins[calibration_bins$model == paste0(model, c('', ' smearing')[j]), ]
    lines(d$forecast, d$observed, type = 'o', pch = c(16, 15)[j],
      col = c('#0057B8', '#D55E00')[j], lwd = 1.1, cex = 0.7)
  }
  axis(1); axis(2); box(bty = 'l')
  title(main = paste(c('(A)', '(B)')[i], model), adj = 0, cex.main = 1.05)
  legend('topleft', c('Uncorrected', 'Smearing'), col = c('#0057B8', '#D55E00'),
    pch = c(16, 15), lty = 1, lwd = 1.1, bty = 'n', cex = 0.85)
}
invisible(dev.off())

# 7. LaTeX tables (text in reviewer colors; rules remain black) ----
write_table <- function(name, caption, label, columns, header, rows, color) {
  writeLines(c('\\begin{table}[!htbp]', paste0('\\color{', color, '}'),
    '\\arrayrulecolor{black}', paste0('\\captionsetup{labelfont={bf,color=', color,
      '},textfont={color=', color, '}}'), paste0('\\caption{', caption, '}'),
    paste0('\\label{', label, '}'), '\\centering', '\\footnotesize',
    '\\setlength{\\tabcolsep}{4pt}', paste0('\\begin{tabular}{', columns, '}'),
    '\\toprule', header, '\\midrule', rows, '\\bottomrule', '\\end{tabular}',
    '\\end{table}'), file.path(output_dir, 'tables', paste0(name, '.tex')))
}
rows <- character()
for (i in seq_along(comparisons)) {
  title <- c('Context-only LMM minus full LMM',
    'Matched-information XGBoost minus Temporal ConvNet')[i]
  rows <- c(rows, paste0('\\multicolumn{7}{l}{\\textit{', title, '}} \\\\'))
  for (j in seq_along(targets)) {
    p <- filter(inference, comparison == names(comparisons)[i], target == targets[j])
    a <- filter(p, unit == 'match')
    b <- filter(p, unit == 'competition_season')
    rows <- c(rows, sprintf(paste0('$\\mathrm{xG}^{%s}$ & %.5f & $[%.5f, %.5f]$ & %.3f',
      ' & $[%.5f, %.5f]$ & %.3f & %s \\\\'), c('F', 'D')[j], a$delta_mae,
      a$ci_low, a$ci_high, a$p_holm, b$ci_low, b$ci_high, b$p_holm,
      format(a$n_matches, big.mark = ',', trim = TRUE)))
  }
  rows <- c(rows, '\\addlinespace')
}
write_table('reviewer1_comparisons', paste0('Paired MAE differences for history ablation ',
  'and matched-information model comparison. Positive differences favor full LMM or ',
  'the Temporal ConvNet. Intervals are pointwise; Holm adjustment covers four contrasts ',
  'separately for each resampling scheme.'), 'tab:r1_comparisons', 'lrrrrrr',
  'Target & $\\Delta$MAE & Match 95\\% CI & $p_{\\mathrm{Holm}}$ & Panel 95\\% CI & $p_{\\mathrm{Holm}}$ & Matches \\\\',
  rows, 'blue')
rows <- character()
for (model in c('LMM', 'XGBoost')) {
  rows <- c(rows, paste0('\\multicolumn{6}{l}{\\textit{', model, '}} \\\\'))
  for (j in seq_along(c('log', 'smearing', 'raw'))) {
    p <- filter(smearing_metrics, .data$model == .env$model,
      variant == c('log', 'smearing', 'raw')[j])
    a <- filter(p, target == 'xG_for')
    d <- filter(p, target == 'xG_diff')
    rows <- c(rows, sprintf('%s & %.4f & %.4f & $%.4f$ & %.4f & %.4f \\\\',
      c('Log, uncorrected', 'Log, smearing', 'Raw response')[j],
      a$mae, a$rmse, a$bias, d$mae, d$rmse))
  }
  rows <- c(rows, '\\addlinespace')
}
write_table('reviewer1_smearing', paste0('Retransformation sensitivity on the same ',
  format(nrow(xgb_full), big.mark = ',', trim = TRUE),
  ' test observations. Smearing factors were estimated from validation errors.'),
  'tab:smearing', 'lrrrrr', c(' & \\multicolumn{3}{c}{$\\mathrm{xG}^{F}$} & \\multicolumn{2}{c}{$\\mathrm{xG}^{D}$} \\\\',
  'Specification & MAE & RMSE & Bias & MAE & RMSE \\\\'), rows, 'blue')
rows <- with(coverage_metrics, sprintf('%s & %d & %d & %s & %.4f & %.4f & $%.4f$ \\\\',
  gsub('-', '--', length_group, fixed = TRUE), n_competitions, n_panels,
  format(n, big.mark = ',', trim = TRUE), mae_xgf, mae_xgd, bias_xgf))
write_table('rolling_coverage', 'Observation-weighted rolling-forecast error by competition coverage.',
  'tab:coverage_errors', 'lrrrrrr',
  'Coverage & Competitions & Panels & $N$ & MAE $\\mathrm{xG}^{F}$ & MAE $\\mathrm{xG}^{D}$ & Bias $\\mathrm{xG}^{F}$ \\\\',
  rows, 'reviewerThree')
rows <- character()
for (i in seq_along(cohorts)) {
  p <- filter(calendar_metrics, cohort == names(cohorts)[i])
  label <- c('All test matches', 'Previously observed competition--seasons',
    'New competition--seasons')[i]
  rows <- c(rows, paste0('\\multicolumn{5}{l}{\\textit{', label, ' ($n=',
    format(p$n[1], big.mark = ',', trim = TRUE), '$)}} \\\\'))
  for (model in c('Rolling mean', 'LMM context only', 'LMM full')) {
    a <- filter(p, .data$model == .env$model, target == 'xG_for')
    d <- filter(p, .data$model == .env$model, target == 'xG_diff')
    rows <- c(rows, sprintf('%s & %.4f & %.4f & %.4f & %.4f \\\\',
      model, a$mae, a$rmse, d$mae, d$rmse))
  }
  rows <- c(rows, '\\addlinespace')
}
write_table('calendar_forecasting', paste0('Calendar-time test performance with fits frozen on ',
  format(test_start, '%d %B %Y'), '. All models use the same complete match pairs. ',
  'New competition--seasons have no observations in fitting; $n$ counts team--match records.'),
  'tab:calendar', 'lrrrr', c(' & \\multicolumn{2}{c}{$\\mathrm{xG}^{F}$} & \\multicolumn{2}{c}{$\\mathrm{xG}^{D}$} \\\\',
  'Model & MAE & RMSE & MAE & RMSE \\\\'), rows, 'reviewerFour')
rows <- with(point_diagnostics, sprintf('%s & %.3f & %.3f & $%.3f$ & %.2f & %.2f \\\\',
  model, mae, rmse, bias, direction_accuracy, weighted_direction_accuracy))
write_table('point_forecast_diagnostics', sprintf(paste0('Point-forecast diagnostics on %s common ',
  'test records. High-xG errors use %s records with observed $\\mathrm{xG}^{F}\\geq %.3f$, ',
  'the training 90th percentile. Directional scores use all common records; weighted ',
  'accuracy weights matches by observed $|\\mathrm{xG}^{D}|$.'),
  format(nrow(common_keys), big.mark = ',', trim = TRUE),
  format(point_diagnostics$n[1], big.mark = ',', trim = TRUE), threshold),
  'tab:point_diagnostics', 'lrrrrr', c(' & \\multicolumn{3}{c}{High $\\mathrm{xG}^{F}$} & \\multicolumn{2}{c}{Direction accuracy (\\%)} \\\\',
  'Model & MAE & RMSE & Bias & Ordinary & Weighted \\\\'), rows, 'reviewerFour')
capture.output(sessionInfo(), file = out('sessionInfo.txt'))
print(inference)
print(calendar_metrics)
print(point_diagnostics)

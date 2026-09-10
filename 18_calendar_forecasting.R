# Rscript --vanilla 18_calendar_forecasting.R project_dir [output_dir]
# Calendar sensitivity: select windows before the test period, then freeze fits.
suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(purrr)
  library(lme4)
  library(zoo)
  library(jsonlite)
})
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) >= 1L)
project_dir <- normalizePath(args[1])
output_dir <- if (length(args) > 1L) args[2] else file.path(project_dir, 'results', 'calendar')
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
out <- function(name) file.path(output_dir, name)
validation_start <- as.Date('2024-07-01')
test_start <- as.Date('2025-07-01')
rolling_windows <- c(3L, 5L, 8L, 10L)
response_scale <- 'log'
set.seed(20260911)

# Load definitions without executing the original tuning/output pipeline.
for (expr in parse(file.path(project_dir, '5_model_lmm.R'))) {
  if (is.call(expr) && identical(expr[[1]], as.name('<-')) &&
      is.call(expr[[3]]) && identical(expr[[3]][[1]], as.name('function'))) {
    eval(expr, envir = .GlobalEnv)
  }
}
write_json(list(validation_start = as.character(validation_start),
  test_start = as.character(test_start), rolling_windows = rolling_windows,
  response_scale = response_scale, selection_metric = 'validation xGF MAE',
  fits_frozen_at_test_start = TRUE, seed = 20260911L,
  analysis = 'Exploratory calendar sensitivity on the existing snapshot',
  baseline_history = 'Previous eligible team records across seasons, as in the primary rolling model',
  shared_eligibility = 'Complete pairs with all candidate features and rolling forecasts'),
  out('calendar_protocol.json'), auto_unbox = TRUE, pretty = TRUE)

df <- read_csv(file.path(project_dir, 'df_model.csv'), show_col_types = FALSE) %>%
  mutate(match_date = as.Date(match_date),
    original_split = split,
    split = case_when(match_date < validation_start ~ 'train',
      match_date < test_start ~ 'validation', TRUE ~ 'test')) %>%
  arrange(team_id, match_date, match_id)
stopifnot(!anyDuplicated(df[c('match_id', 'team_id')]))

# The historical rolling benchmark carries eligible observations across seasons.
for (k in rolling_windows) {
  df <- df %>% group_by(team_id) %>% mutate(
    !!paste0('baseline_', k) := expm1(rollmeanr(lag(log1p(xG_for)), k, fill = NA_real_)),
    last_history_date = lag(match_date)) %>% ungroup()
}
stopifnot(all(df$last_history_date < df$match_date, na.rm = TRUE))
required <- c(unique(unlist(lapply(rolling_windows, get_predictors))),
  paste0('baseline_', rolling_windows), 'xG_for')
df <- prepare_data(df, 3L) %>% filter(if_all(all_of(required), is.finite)) %>%
  group_by(match_id) %>% filter(n() == 2L) %>% ungroup()
stopifnot(all(table(df$match_id) == 2L),
  all(df %>% group_by(match_id) %>% summarise(n = n_distinct(split)) %>% pull(n) == 1L))
write_csv(df %>% group_by(split) %>% summarise(n = n(), matches = n_distinct(match_id),
  competitions = n_distinct(competition_id), first_date = min(match_date),
  last_date = max(match_date), .groups = 'drop'), out('calendar_split_counts.csv'))
train <- filter(df, split == 'train')
validation <- filter(df, split == 'validation')
fit_data <- filter(df, split != 'test')
test <- filter(df, split == 'test')
stopifnot(max(train$match_date) < min(validation$match_date),
  max(fit_data$match_date) < min(test$match_date))

tuning <- map_dfr(rolling_windows, function(k) {
  message('Calendar validation, LMM K = ', k)
  fitted_model <- fit_model(train, k, reml = FALSE)
  p <- predict_xg_for(fitted_model, validation)
  bind_rows(tibble(model = 'LMM full', k, n = nrow(p),
    mae = mean(abs(p$predicted - p$actual)),
    singular = isSingular(fitted_model$fit),
    convergence = paste(fitted_model$fit@optinfo$conv$lme4$messages, collapse = ' | ')),
    tibble(model = 'Rolling mean', k, n = nrow(validation),
      mae = mean(abs(validation[[paste0('baseline_', k)]] - validation$xG_for)),
      singular = NA, convergence = ''))
})
write_csv(tuning, out('calendar_validation.csv'))
selected <- tuning %>% arrange(mae, k) %>% group_by(model) %>% slice(1) %>% ungroup()
write_csv(selected, out('calendar_selected_windows.csv'))
k_lmm <- selected$k[selected$model == 'LMM full']
k_rolling <- selected$k[selected$model == 'Rolling mean']
message('Final calendar fits; LMM K = ', k_lmm, ', rolling K = ', k_rolling)
full <- fit_model(fit_data, k_lmm, reml = TRUE)
context_formula <- response ~ home_away + (1 | competition_id) +
  (1 | team_season_id) + (1 | opponent_season_id)
contrasts(fit_data$home_away) <- contr.treatment(2, base = 1)
context <- list(fit = lmer(context_formula, data = fit_data, REML = TRUE,
  control = lmerControl(optimizer = 'bobyqa', optCtrl = list(maxfun = 2e5))),
  scaler = tibble(variable = character(), center = double(), scale = double()),
  formula = context_formula, predictors = character(), reml = TRUE)
saveRDS(list(full = full, context = context), out('calendar_fits.rds'))
predictions <- bind_rows(
  'LMM full' = pair_predictions(predict_xg_for(full, test)),
  'LMM context only' = pair_predictions(predict_xg_for(context, test)),
  'Rolling mean' = pair_predictions(test %>% transmute(match_id, team_id, split,
    target = 'xG_for', actual = xG_for, predicted = .data[[paste0('baseline_', k_rolling)]])),
  .id = 'model')
metadata <- test %>% transmute(match_id, team_id, match_date, competition_id,
  season_id, season_name, competition_name, team_season_id, opponent_season_id,
  n_previous_matches, original_split,
  new_team_season = !team_season_id %in% fit_data$team_season_id,
  new_opponent_season = !opponent_season_id %in% fit_data$opponent_season_id,
  new_competition = !competition_id %in% fit_data$competition_id,
  new_competition_season = !paste(competition_id, season_id) %in%
    paste(fit_data$competition_id, fit_data$season_id))
predictions <- left_join(predictions, metadata, by = c('match_id', 'team_id'),
  relationship = 'many-to-one')
stopifnot(nrow(predictions) == 3L * nrow(test),
  all(table(predictions$model, predictions$match_id) == 2L))
write_csv(predictions, out('calendar_test_predictions.csv'))
write_csv(metadata %>% group_by(new_competition_season, new_team_season,
  new_opponent_season, new_competition) %>% summarise(n = n(),
  matches = n_distinct(match_id), team_seasons = n_distinct(team_season_id),
  min_history = min(n_previous_matches), .groups = 'drop'), out('calendar_new_levels.csv'))
write_csv(bind_rows(lapply(list(full = full, context = context), function(m) {
  tibble(formula = paste(deparse(m$formula), collapse = ' '), n = nobs(m$fit),
    reml = isREML(m$fit), singular = isSingular(m$fit),
    convergence = paste(m$fit@optinfo$conv$lme4$messages, collapse = ' | '))
})), out('calendar_fit_diagnostics.csv'))
capture.output(sessionInfo(), file = out('calendar_session_info.txt'))
print(selected)
print(read_csv(out('calendar_new_levels.csv'), show_col_types = FALSE))

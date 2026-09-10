# Reviewer 1: selected LMM refits, history ablation and validation smearing.
# Usage: Rscript --vanilla 12_revision_R1_lmm.R [project_dir] [output_dir]

suppressPackageStartupMessages({
  library(dplyr)
  library(purrr)
  library(readr)
  library(lme4)
})
args <- commandArgs(trailingOnly = TRUE)
project_dir <- if (length(args) >= 1) normalizePath(args[1]) else getwd()
output_dir <- if (length(args) >= 2) args[2] else file.path(project_dir, 'results', 'reviewer1')
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
response_scale <- 'log'

# Reuse only function definitions: sourcing the original script would retune
# models and overwrite the original results.
for (expr in parse(file.path(project_dir, '5_model_lmm.R'))) {
  if (is.call(expr) && identical(expr[[1]], as.name('<-')) &&
      is.call(expr[[3]]) && identical(expr[[3]][[1]], as.name('function'))) {
    eval(expr, envir = .GlobalEnv)
  }
}
output_file <- function(name) file.path(output_dir, paste0('lmm_', name, '.csv'))
df <- read_csv(file.path(project_dir, 'df_model.csv'), show_col_types = FALSE)
config <- read_csv(file.path(project_dir, 'results', 'lmm_log_best_config.csv'), show_col_types = FALSE)
k <- config$rolling_window[1]
df <- prepare_data(df, k)
train <- filter(df, split == 'train')
validation <- filter(df, split == 'validation')
fit_data <- filter(df, split %in% c('train', 'validation'))
test <- filter(df, split == 'test')
stopifnot(length(intersect(train$match_id, validation$match_id)) == 0,
          length(intersect(fit_data$match_id, test$match_id)) == 0)

message('Fitting selected training-only LMM for validation smearing; K = ', k)
validation_model <- fit_model(train, k, reml = FALSE)
validation_log <- as.numeric(predict(validation_model$fit,
  newdata = apply_scaler(validation, validation_model$scaler), allow.new.levels = TRUE))
calibration <- validation %>% transmute(
  match_id, team_id, competition_id, season_id, split,
  actual_xG_for = xG_for, predicted_log = validation_log,
  log_residual = log1p(xG_for) - validation_log
)
smearing_factor <- mean(exp(calibration$log_residual))
stopifnot(is.finite(smearing_factor), smearing_factor > 0)
write_csv(calibration, output_file('validation_calibration'))
write_csv(tibble(smearing_factor, n_calibration = nrow(calibration),
  calibration_split = 'validation', calibration_fit_split = 'train',
  calibration_estimation = 'ML', final_estimation = 'REML', rolling_window = k,
  test_used_for_calibration = FALSE), output_file('smearing_factor'))
saveRDS(validation_model, file.path(output_dir, 'lmm_validation_model.rds'))
message('Validation smearing factor: ', signif(smearing_factor, 9))

message('Refitting selected full LMM with REML on training plus validation')
full_model <- fit_model(fit_data, k, reml = TRUE)
saveRDS(full_model, file.path(output_dir, 'lmm_full_model.rds'))
full_log <- as.numeric(predict(full_model$fit,
  newdata = apply_scaler(test, full_model$scaler), allow.new.levels = TRUE))
full_predictions <- predict_xg_for(full_model, test)
smearing_predictions <- full_predictions %>% mutate(
  predicted = pmax(smearing_factor * exp(full_log) - 1, 0))

message('Fitting context-only LMM on the identical fitting observations')
context_formula <- response ~ home_away + (1 | competition_id) +
  (1 | team_season_id) + (1 | opponent_season_id)
contrasts(fit_data$home_away) <- contr.treatment(n = 2, base = 1)
context_model <- list(
  fit = lmer(context_formula, data = fit_data, REML = TRUE,
    control = lmerControl(optimizer = 'bobyqa', optCtrl = list(maxfun = 2e5))),
  scaler = tibble(variable = character(), center = double(), scale = double()),
  formula = context_formula, predictors = character(), reml = TRUE)
saveRDS(context_model, file.path(output_dir, 'lmm_context_only_model.rds'))
context_predictions <- predict_xg_for(context_model, test)

predictions <- bind_rows(
  full = pair_predictions(full_predictions),
  full_smearing = pair_predictions(smearing_predictions),
  context_only = pair_predictions(context_predictions), .id = 'variant'
)
stopifnot(all(table(predictions$variant, predictions$match_id) == 2))
write_csv(predictions, output_file('test_predictions'))
long <- bind_rows(lapply(split(predictions, predictions$variant), function(p) {
  mutate(make_long_predictions(p), variant = p$variant[1], .before = 1)
}))
write_csv(long, output_file('test_predictions_long'))
metrics <- long %>% group_by(variant, target) %>% summarise(
  n = n(), mae = mean(abs(predicted - actual)),
  rmse = sqrt(mean((predicted - actual)^2)), bias = mean(predicted - actual), .groups = 'drop')
write_csv(metrics, output_file('test_metrics'))

archived <- read_csv(file.path(project_dir, 'results', 'lmm_log_test_predictions.csv'), show_col_types = FALSE)
check <- predictions %>% filter(variant == 'full') %>% select(match_id, team_id, predicted_xG_for) %>%
  inner_join(archived %>% select(match_id, team_id, archived = predicted_xG_for),
             by = c('match_id', 'team_id'), relationship = 'one-to-one')
stopifnot(nrow(check) == nrow(archived))
write_csv(tibble(n = nrow(check),
  max_abs_prediction_difference = max(abs(check$predicted_xG_for - check$archived)),
  mean_abs_prediction_difference = mean(abs(check$predicted_xG_for - check$archived))),
  output_file('reproduction_check'))

fits <- list(validation_full = validation_model, final_full = full_model,
             final_context_only = context_model)
diagnostics <- bind_rows(lapply(names(fits), function(name) {
  model <- fits[[name]]
  messages <- model$fit@optinfo$conv$lme4$messages
  tibble(variant = name, n = nobs(model$fit), reml = model$reml,
    formula = paste(deparse(model$formula), collapse = ' '),
    singular = isSingular(model$fit), residual_sd = sigma(model$fit),
    convergence_message = paste(messages, collapse = ' | '))
}))
write_csv(diagnostics, output_file('fit_diagnostics'))
capture.output(sessionInfo(), file = file.path(output_dir, 'lmm_session_info.txt'))
print(metrics)
print(read_csv(output_file('reproduction_check'), show_col_types = FALSE))

# Usage: Rscript --vanilla 16_model_diagnostics_stability.R project_dir model_file output_dir
# Summarizes saved fits and forecasts; does not fit or tune models.

suppressPackageStartupMessages({
  library(lme4)
  library(dplyr)
  library(readr)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 3L)
project_dir <- normalizePath(args[1])
model_file <- normalizePath(args[2])
output_dir <- args[3]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
out <- function(name) file.path(output_dir, name)
read_result <- function(name) read_csv(file.path(project_dir, 'results', name),
  show_col_types = FALSE)

model <- readRDS(model_file)$fit
stopifnot(inherits(model, 'lmerMod'), isREML(model), nobs(model) == 48671L)
df_resid <- tibble(fitted_log = as.numeric(fitted(model)),
  residual_log = as.numeric(residuals(model))) %>%
  mutate(standardized_residual = residual_log / sigma(model),
    fitted_decile = ntile(fitted_log, 10L))
stopifnot(all(is.finite(as.matrix(df_resid))))
bins <- df_resid %>% group_by(fitted_decile) %>% summarise(
  n = n(), fitted_mean = mean(fitted_log), residual_mean = mean(residual_log),
  residual_sd = sd(residual_log), .groups = 'drop')
z <- df_resid$standardized_residual
centered <- z - mean(z)
moment2 <- mean(centered^2)
diagnostics <- tibble(n = nobs(model), residual_sd = sigma(model),
  mean_residual = mean(df_resid$residual_log),
  skewness = mean(centered^3) / moment2^1.5,
  excess_kurtosis = mean(centered^4) / moment2^2 - 3,
  min_decile_sd = min(bins$residual_sd), max_decile_sd = max(bins$residual_sd),
  decile_sd_ratio = max(bins$residual_sd) / min(bins$residual_sd),
  min_decile_mean = min(bins$residual_mean), max_decile_mean = max(bins$residual_mean),
  fraction_abs_standardized_over_3 = mean(abs(z) > 3))
write_csv(df_resid, out('lmm_conditional_residuals.csv'))
write_csv(bins, out('lmm_residual_fitted_deciles.csv'))
write_csv(diagnostics, out('lmm_residual_diagnostics.csv'))

metadata <- read_csv(file.path(project_dir, 'df_model.csv'),
  col_select = c(match_id, team_id, competition_id, season_id, country_name,
    competition_name, season_name, match_date, split, xG_for, xG_diff),
  show_col_types = FALSE)
stopifnot(!anyDuplicated(metadata[c('match_id', 'team_id')]))
panels <- metadata %>% distinct(competition_id, country_name, competition_name,
  season_id, season_name) %>% mutate(season_end = as.integer(substr(season_name,
    nchar(season_name) - 3L, nchar(season_name))))
competitions <- panels %>% group_by(competition_id, country_name, competition_name) %>%
  summarise(n_seasons = n(), first_season = season_name[which.min(season_end)],
    last_season = season_name[which.max(season_end)], .groups = 'drop')
rolling <- read_result('rolling_log_test_predictions.csv') %>%
  select(match_id, team_id, actual_xG_for, predicted_xG_for, actual_xG_diff,
    predicted_xG_diff) %>%
  left_join(metadata, by = c('match_id', 'team_id'), relationship = 'one-to-one')
stopifnot(nrow(rolling) == 9686L, !anyNA(rolling$competition_id),
  all(rolling$split == 'test'), all(table(rolling$match_id) == 2L),
  max(abs(rolling$actual_xG_for - rolling$xG_for)) < 1e-8,
  max(abs(rolling$actual_xG_diff - rolling$xG_diff)) < 1e-8)
rolling <- rolling %>% mutate(error_xgf = predicted_xG_for - actual_xG_for,
  error_xgd = predicted_xG_diff - actual_xG_diff)
metrics <- function(df, ...) df %>% group_by(...) %>% summarise(
  n = n(), n_matches = n_distinct(match_id),
  mae_xgf = mean(abs(error_xgf)), rmse_xgf = sqrt(mean(error_xgf^2)),
  bias_xgf = mean(error_xgf), mean_xgf = mean(actual_xG_for), sd_xgf = sd(actual_xG_for),
  mae_xgd = mean(abs(error_xgd)), rmse_xgd = sqrt(mean(error_xgd^2)),
  .groups = 'drop')
by_competition <- metrics(rolling, competition_id) %>%
  left_join(competitions, by = 'competition_id', relationship = 'one-to-one')
by_season <- metrics(rolling, competition_id, season_id) %>%
  left_join(panels, by = c('competition_id', 'season_id'), relationship = 'one-to-one')
by_length <- rolling %>% left_join(select(competitions, competition_id, n_seasons),
  by = 'competition_id', relationship = 'many-to-one') %>%
  mutate(length_group = case_when(n_seasons %in% c(2L, 3L) ~ '2-3 seasons',
    n_seasons == 5L ~ '5 seasons', n_seasons %in% c(6L, 8L) ~ '6-8 seasons'))
stopifnot(!anyNA(by_length$length_group))
length_metrics <- metrics(by_length, length_group) %>% left_join(
  by_length %>% group_by(length_group) %>% summarise(
    n_competitions = n_distinct(competition_id),
    n_panels = n_distinct(competition_id, season_id), .groups = 'drop'),
  by = 'length_group', relationship = 'one-to-one')
stopifnot(sum(by_competition$n) == nrow(rolling), sum(by_season$n) == nrow(rolling),
  nrow(by_competition) == 30L, nrow(by_season) == 114L)
write_csv(by_competition, out('rolling_errors_by_competition.csv'))
write_csv(by_season, out('rolling_errors_by_season.csv'))
write_csv(length_metrics, out('rolling_errors_by_coverage.csv'))

long_panels <- by_season %>% filter(
  (country_name == 'Poland' & competition_name == 'Ekstraklasa') |
  (country_name == 'Spain' & competition_name == 'La Liga 2')) %>%
  arrange(country_name, season_end)
stopifnot(sum(long_panels$country_name == 'Poland') == 8L,
  sum(long_panels$country_name == 'Spain') == 6L)
write_csv(long_panels, out('rolling_errors_long_coverage.csv'))
tuning <- read_result('tcn_log_tuning_results.csv') %>% mutate(
  parameter_count = filters * kernel_size + filters +
    (n_blocks - 1) * (filters^2 * kernel_size + filters) +
    (filters + 1) * dense_units + dense_units + dense_units + 1)
selected <- read_result('tcn_log_best_config.csv')
best <- filter(tuning, trial == selected$selected_trial[1])
stopifnot(nrow(tuning) == 100L, nrow(best) == 1L, best$parameter_count == 27105L,
  best$mae == min(tuning$mae))
write_csv(tuning, out('tcn_tuning_capacity.csv'))
write_json(list(n_trials = nrow(tuning), selected_trial = best$trial,
  selected_parameters = best$parameter_count,
  selected_filters = best$filters, selected_blocks = best$n_blocks,
  evaluated_parameter_range = range(tuning$parameter_count),
  selected_mae = best$mae, larger_candidates = sum(tuning$parameter_count > best$parameter_count),
  claim_scope = 'Best evaluated trial; no architecture-wide optimum or overfitting mechanism established'),
  out('tcn_capacity_summary.json'), auto_unbox = TRUE, pretty = TRUE)

effects <- read_result('bootstrap_log_mae_differences_vs_rolling.csv') %>%
  filter(model == 'Mixed-effects model') %>% select(target, n_rows, n_matches,
    model_mae, baseline_mae, mae_improvement, mae_improvement_pct,
    mae_improvement_ci_low, mae_improvement_ci_high)
write_csv(effects, out('paired_effect_sizes.csv'))
capture.output(sessionInfo(), file = out('session_info.txt'))
write_json(list(input_project = project_dir, fitted_model = model_file,
  fitting_observations = nobs(model), test_observations = nrow(rolling),
  test_matches = n_distinct(rolling$match_id),
  competitions = nrow(by_competition), competition_seasons = nrow(by_season),
  test_predictions_recomputed = FALSE, model_refitted = FALSE,
  random_operations = FALSE, residual_type = 'Conditional response residuals',
  display_qq_quantiles = 3000L,
  coverage_analysis = 'Descriptive; observation-weighted errors; shared primary test set'),
  out('analysis_settings.json'), auto_unbox = TRUE, pretty = TRUE)
print(diagnostics)
print(length_metrics)
print(long_panels %>% select(competition_name, season_name, n, mae_xgf, mae_xgd))
print(effects)

script_file <- sub('^--file=', '', grep('^--file=', commandArgs(), value = TRUE)[1])
source(file.path(dirname(normalizePath(script_file)), '17_plot_diagnostics_stability.R'))
render_diagnostics_stability(output_dir)

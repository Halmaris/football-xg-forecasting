# Rscript --vanilla 11_main.R [summary|models|all] [python_executable]
# Run from the project root. Download/preparation are explicit separate steps.
args <- commandArgs(trailingOnly = TRUE)
mode <- if (length(args)) args[1] else 'summary'
if (!mode %in% c('summary', 'models', 'all')) stop('Mode must be summary, models or all')
python <- if (length(args) > 1L) args[2] else Sys.getenv('XG_PYTHON', 'python3')
rscript <- file.path(R.home('bin'), 'Rscript')
run <- function(command, args) {
  status <- system2(command, args = vapply(args, shQuote, character(1)), stdout = '', stderr = '')
  if (status != 0L) stop('Failed: ', paste(args, collapse = ' '), '; exit status ', status)
}
run_r <- function(script) run(rscript, c('--vanilla', script))
stopifnot(file.exists('df_model.csv'))

if (mode %in% c('models', 'all')) {
  for (scale in c('log', 'raw')) {
    Sys.setenv(XG_RESPONSE_SCALE = scale)
    for (script in c('3_model_rolling.R', '3_model_rolling_loco.R',
                     '4_model_arima.R', '5_model_lmm.R')) run_r(script)
  }
  Sys.unsetenv('XG_RESPONSE_SCALE')
  for (script in c('6_model_xgb.py', '6_model_xgb_direct_xgdiff.py',
                   '6_model_xgb_ablation.py', '6_model_xgb_loco.py',
                   '7_model_tcn.py')) run(python, script)
}
if (mode %in% c('summary', 'all')) {
  for (script in c('6_compare_xgdiff_approaches.R', '8_summary.R',
                   '9_bootCI.R', '9_bootCI_pairwise.R', '9_bootCI_cluster.R',
                   '10_tables_figures.R')) run_r(script)
}

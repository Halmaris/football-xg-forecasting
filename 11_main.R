# Run from the project directory. Set the Python interpreter used by the models.
python <- '/Users/Tomek/miniforge3/envs/vscode-python-clean/bin/python'

for (scale in c('log', 'raw')) {
  Sys.setenv(XG_RESPONSE_SCALE = scale)
  source('3_model_rolling.R')
  source('3_model_rolling_loco.R')
  source('4_model_arima.R')
  source('5_model_lmm.R')
}
Sys.unsetenv('XG_RESPONSE_SCALE')

for (script in c('6_model_xgb.py', '6_model_xgb_direct_xgdiff.py',
                 '6_model_xgb_ablation.py', '6_model_xgb_loco.py', '7_model_tcn.py')) {
  stopifnot(system2(python, shQuote(script)) == 0L)
}

source('6_compare_xgdiff_approaches.R')
source('8_summary.R')
source('9_bootCI.R')
source('9_bootCI_pairwise.R')
source('9_bootCI_cluster.R')
source('10_tables_figures.R')
source('revision.R')

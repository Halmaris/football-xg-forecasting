source('3_model_rolling.R')
spurce('3_model_rolling_loco.R')
source('4_model_arima.R')
source('5_model_lmm.R')

python <- '/Users/Tomek/miniforge3/envs/vscode-python-clean/bin/python'
script1 <- '/Users/Tomek/Downloads/6_model_xgb.py'
script2 <- '/Users/Tomek/Downloads/6_model_xgb_direct_xgdiff.py'
script3 <- '/Users/Tomek/Downloads/6_model_xgb_ablation.py'
script4 <- '/Users/Tomek/Downloads/6_model_xgb_loco.py'
script5 <- '/Users/Tomek/Downloads/7_model_tcn.py'

system2(command = python, args = script1, stdout = '', stderr = '')
system2(command = python, args = script2, stdout = '', stderr = '')
system2(command = python, args = script3, stdout = '', stderr = '')
system2(command = python, args = script4, stdout = '', stderr = '')
system2(command = python, args = script5, stdout = '', stderr = '')

source('6_compare_xgdiff_approaches')
source('8_summary.R')
source('9_bootCI.R')

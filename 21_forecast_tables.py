'''python 21_forecast_tables.py calendar_dir diagnostics_dir latex_tables_dir'''
import sys
from pathlib import Path
import pandas as pd

calendar, diagnostics, tables = map(Path, sys.argv[1:4])
tables.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(calendar / 'calendar_metrics.csv')
rows = []
for cohort, label in [('All', 'All test matches'), ('Observed season', 'Previously observed competition--seasons'),
                      ('New season', 'New competition--seasons')]:
    part = df.loc[df.cohort.eq(cohort)]
    n = int(part.n.iloc[0])
    rows.append(r'\multicolumn{5}{l}{\textit{' + label + f' ($n={n:,}$)' + r'}} \\')
    for model in ['Rolling mean', 'LMM context only', 'LMM full']:
        a = part.loc[part.model.eq(model) & part.target.eq('xG_for')].iloc[0]
        d = part.loc[part.model.eq(model) & part.target.eq('xG_diff')].iloc[0]
        rows.append(' & '.join([model, f'{a.mae:.4f}', f'{a.rmse:.4f}', f'{d.mae:.4f}', f'{d.rmse:.4f}']) + r' \\')
    rows.append(r'\addlinespace')
table = r'''\begin{table}[!htbp]
\color{reviewerFour}
\caption{\revFour{Calendar-time test performance with fits frozen on 1 July 2025. All models use the same complete match pairs. New competition--seasons have no observations in fitting; $n$ counts team--match records.}}
\label{tab:calendar}
\centering
\small
\begin{tabular}{lrrrr}
\toprule
 & \multicolumn{2}{c}{$\mathrm{xG}^{F}$} & \multicolumn{2}{c}{$\mathrm{xG}^{D}$} \\
Model & MAE & RMSE & MAE & RMSE \\
\midrule
''' + '\n'.join(rows) + r'''
\bottomrule
\end{tabular}
\end{table}
'''
(tables / 'calendar_forecasting.tex').write_text(table)
df = pd.read_csv(diagnostics / 'point_forecast_diagnostics.csv')
rows = []
for _, r in df.iterrows():
    rows.append(' & '.join([r.model, f'{r.mae:.3f}', f'{r.rmse:.3f}', f'${r.bias:.3f}$',
                f'{r.direction_accuracy:.2f}', f'{r.weighted_direction_accuracy:.2f}']) + r' \\')
table = r'''\begin{table}[!htbp]
\color{reviewerFour}
\caption{\revFour{Point-forecast diagnostics on 9,590 common test records. High-xG errors use 1,075 records with observed $\mathrm{xG}^{F}\geq 2.314$, the training 90th percentile. Directional scores use all common records; weighted accuracy weights matches by observed $|\mathrm{xG}^{D}|$.}}
\label{tab:point_diagnostics}
\centering
\small
\setlength{\tabcolsep}{5pt}
\begin{tabular}{lrrrrr}
\toprule
 & \multicolumn{3}{c}{High $\mathrm{xG}^{F}$} & \multicolumn{2}{c}{Direction accuracy (\%)} \\
Model & MAE & RMSE & Bias & Ordinary & Weighted \\
\midrule
''' + '\n'.join(rows) + r'''
\bottomrule
\end{tabular}
\end{table}
'''
(tables / 'point_forecast_diagnostics.tex').write_text(table)

"""Export the two Reviewer 1 LaTeX tables from completed result CSVs.

Usage: python 15_revision_R1_tables.py [results_dir] [latex_tables_dir]
The containing manuscript defines the blue \revOne macro.
"""
from pathlib import Path
import csv
import sys

results = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('results/reviewer1')
tables = Path(sys.argv[2]) if len(sys.argv) > 2 else results / 'latex_tables'
tables.mkdir(parents=True, exist_ok=True)
with (results / 'paired_mae_comparisons.csv').open() as f:
    comparisons = list(csv.DictReader(f))
rows = []
for comparison in ['LMM history ablation', 'Matched-information comparison']:
    title = 'Context-only LMM minus full LMM' if comparison.startswith('LMM') else 'Matched-information XGBoost minus Temporal ConvNet'
    rows.append(r'\multicolumn{7}{l}{\textit{' + title + r'}} \\')
    for target, label in [('xG_for', r'$\mathrm{xG}^{F}$'), ('xG_diff', r'$\mathrm{xG}^{D}$')]:
        match = next(r for r in comparisons if r['comparison'] == comparison and r['target'] == target and r['unit'] == 'match')
        cluster = next(r for r in comparisons if r['comparison'] == comparison and r['target'] == target and r['unit'] == 'competition_season')
        nums = [label, f"{float(match['delta_mae']):.5f}",
                f"$[{float(match['ci_low']):.5f}, {float(match['ci_high']):.5f}]$",
                f"{float(match['p_holm']):.3f}",
                f"$[{float(cluster['ci_low']):.5f}, {float(cluster['ci_high']):.5f}]$",
                f"{float(cluster['p_holm']):.3f}", f"{int(match['n_matches']):,}"]
        rows.append(' & '.join(nums) + r' \\')
    rows.append(r'\addlinespace')
comparison_table = r'''\begin{table}[!htbp]
\color{blue}
\caption{\revOne{Paired MAE differences for history ablation and matched-information model comparison. Positive differences favor full LMM or the Temporal ConvNet. Intervals are pointwise; Holm adjustment covers four contrasts separately for each resampling scheme.}}
\label{tab:r1_comparisons}
\centering
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lrrrrrr}
\toprule
Target & $\Delta$MAE & Match 95\% CI & $p_{\mathrm{Holm}}$ & Panel 95\% CI & $p_{\mathrm{Holm}}$ & Matches \\
\midrule
''' + '\n'.join(rows) + r'''
\bottomrule
\end{tabular}
\end{table}
'''
(tables / 'reviewer1_comparisons.tex').write_text(comparison_table)

with (results / 'smearing_scale_comparison.csv').open() as f:
    metrics = list(csv.DictReader(f))
rows = []
for model in ['Mixed-effects model', 'XGBoost']:
    rows.append(r'\multicolumn{6}{l}{\textit{' + model + r'}} \\')
    for variant, label in [('log', 'Log, uncorrected'), ('smearing', 'Log, smearing'), ('raw', 'Raw response')]:
        a = next(r for r in metrics if r['model'] == model and r['variant'] == variant and r['target'] == 'xG_for')
        d = next(r for r in metrics if r['model'] == model and r['variant'] == variant and r['target'] == 'xG_diff')
        values = [label] + [f"${float(a[k]):.4f}$" for k in ['mae', 'rmse', 'bias']] + [f"${float(d[k]):.4f}$" for k in ['mae', 'rmse']]
        rows.append(' & '.join(values) + r' \\')
    rows.append(r'\addlinespace')
smearing_table = r'''\begin{table}[!htbp]
\color{blue}
\caption{\revOne{Retransformation sensitivity on the same 9,694 test observations. Smearing factors were estimated from validation errors. Results are shown for uncorrected log-response, smearing-corrected and raw-response specifications.}}
\label{tab:smearing}
\centering
\small
\setlength{\tabcolsep}{5pt}
\begin{tabular}{lrrrrr}
\toprule
 & \multicolumn{3}{c}{$\mathrm{xG}^{F}$} & \multicolumn{2}{c}{$\mathrm{xG}^{D}$} \\
Specification & MAE & RMSE & Bias & MAE & RMSE \\
\midrule
''' + '\n'.join(rows) + r'''
\bottomrule
\end{tabular}
\end{table}
'''
(tables / 'reviewer1_smearing.tex').write_text(smearing_table)

print('Exported two LaTeX tables to', tables)

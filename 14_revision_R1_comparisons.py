"""Reviewer 1: paired MAE inference and consolidated smearing results.

Usage: python 14_revision_R1_comparisons.py [project_dir] [output_dir]
Bootstrap units preserve complete matches or entire competition-season panels.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 20260910
N_BOOT = 5000
KEYS = ['match_id', 'team_id']
TARGETS = ['xG_for', 'xG_diff']


def read_predictions(path, variant=None):
    df = pd.read_csv(path)
    if variant is not None:
        df = df.loc[df['variant'] == variant].copy()
    for col in KEYS:
        assert np.all(df[col] == np.floor(df[col]))
        df[col] = df[col].astype('int64')
    assert not df.duplicated(KEYS).any()
    assert (df.groupby('match_id').size() == 2).all()
    assert np.isfinite(df[[f'{kind}_{target}' for kind in ['actual', 'predicted']
                           for target in TARGETS]].to_numpy()).all()
    for kind in ['actual', 'predicted']:
        assert np.allclose(df[f'{kind}_xG_diff'], df[f'{kind}_xG_for'] - df[f'{kind}_xG_against'], atol=2e-6)
        assert np.allclose(df.groupby('match_id')[f'{kind}_xG_diff'].sum(), 0, atol=2e-6)
    return df


def holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    result = np.empty_like(p)
    result[order] = np.minimum(1, np.maximum.accumulate(p[order] * (len(p) - np.arange(len(p)))))
    return result


def bootstrap(df, groups, rng):
    units = df.groupby(groups, sort=True).agg(
        e1=('error_1', 'sum'), e2=('error_2', 'sum'), n=('error_1', 'size'))
    sums = units[['e1', 'e2', 'n']].to_numpy()
    replicate = np.empty(N_BOOT)
    for start in range(0, N_BOOT, 100):
        indices = rng.integers(0, len(units), size=(min(100, N_BOOT - start), len(units)))
        sampled = sums[indices].sum(axis=1)
        replicate[start:start + len(sampled)] = (sampled[:, 0] - sampled[:, 1]) / sampled[:, 2]
    low, high = np.quantile(replicate, [0.025, 0.975])
    p = min(1., 2 * min((np.count_nonzero(replicate <= 0) + 1) / (N_BOOT + 1),
                       (np.count_nonzero(replicate >= 0) + 1) / (N_BOOT + 1)))
    return {'n_units': len(units), 'ci_low': low, 'ci_high': high, 'p_boot': p}, replicate


def main():
    project = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd()
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else project / 'results' / 'reviewer1'
    metadata = pd.read_csv(project / 'df_model.csv', usecols=KEYS + ['competition_id', 'season_id', 'split'])
    for key in KEYS:
        metadata[key] = metadata[key].astype('int64')
    specs = [
        ('LMM history ablation', 'LMM context only', 'LMM full',
         out / 'lmm_test_predictions.csv', 'context_only', out / 'lmm_test_predictions.csv', 'full'),
        ('Matched-information comparison', 'XGBoost matched information', 'Temporal ConvNet',
         project / 'results' / 'xgb_ablation_log_test_predictions.csv', 'matched_information',
         project / 'results' / 'tcn_log_test_predictions.csv', None),
    ]
    summaries, replicates, errors, metric_rows = [], [], [], []
    for comparison_no, (comparison, name1, name2, path1, variant1, path2, variant2) in enumerate(specs):
        a, b = read_predictions(path1, variant1), read_predictions(path2, variant2)
        common = a.merge(b, on=KEYS, suffixes=('_1', '_2'), validate='one_to_one')
        common = common.merge(metadata, on=KEYS, validate='one_to_one').sort_values(KEYS)
        assert (common['split'] == 'test').all()
        assert (common.groupby('match_id').size() == 2).all()
        for target_no, target in enumerate(TARGETS):
            assert np.allclose(common[f'actual_{target}_1'], common[f'actual_{target}_2'], atol=1e-12)
            paired = common[KEYS + ['competition_id', 'season_id']].copy()
            paired['actual'] = common[f'actual_{target}_1']
            for i, name in [(1, name1), (2, name2)]:
                residual = common[f'predicted_{target}_{i}'] - common[f'actual_{target}_{i}']
                paired[f'error_{i}'] = np.abs(residual)
                metric_rows.append({'comparison': comparison, 'model': name, 'target': target,
                                    'n': len(residual), 'n_matches': common['match_id'].nunique(),
                                    'mae': np.mean(np.abs(residual)),
                                    'rmse': np.sqrt(np.mean(residual**2)), 'bias': np.mean(residual)})
            paired['comparison'] = comparison
            paired['target'] = target
            errors.append(paired)
            for unit_no, (unit, groups) in enumerate([
                ('match', ['match_id']), ('competition_season', ['competition_id', 'season_id'])
            ]):
                seed = SEED + comparison_no * 100 + target_no * 10 + unit_no
                summary, values = bootstrap(paired, groups, np.random.default_rng(seed))
                summaries.append({'comparison': comparison, 'model_1': name1, 'model_2': name2,
                                  'target': target, 'unit': unit, 'n': len(common),
                                  'n_matches': common['match_id'].nunique(),
                                  'mae_1': paired['error_1'].mean(), 'mae_2': paired['error_2'].mean(),
                                  'delta_mae': (paired['error_1'] - paired['error_2']).mean(),
                                  **summary, 'seed': seed, 'n_boot': N_BOOT})
                replicates.append(pd.DataFrame({'comparison': comparison, 'target': target,
                                                'unit': unit, 'replicate': np.arange(1, N_BOOT + 1),
                                                'delta_mae': values}))
    inference = pd.DataFrame(summaries)
    inference['p_holm'] = inference.groupby('unit')['p_boot'].transform(holm)
    inference.to_csv(out / 'paired_mae_comparisons.csv', index=False)
    pd.concat(replicates, ignore_index=True).to_csv(out / 'bootstrap_replicates.csv', index=False)
    pd.concat(errors, ignore_index=True).to_csv(out / 'paired_errors.csv', index=False)
    pd.DataFrame(metric_rows).to_csv(out / 'common_sample_metrics.csv', index=False)

    consolidated, extreme = [], []
    for prefix, label in [('lmm', 'Mixed-effects model'), ('xgb', 'XGBoost')]:
        for variant, display, path, selector in [
            ('log', 'Log, uncorrected', out / f'{prefix}_test_predictions.csv', 'full'),
            ('smearing', 'Log, smearing', out / f'{prefix}_test_predictions.csv', 'full_smearing'),
            ('raw', 'Raw response', project / 'results' / f'{prefix}_raw_test_predictions.csv', None),
        ]:
            frame = read_predictions(path, selector)
            for target in TARGETS:
                error = frame[f'predicted_{target}'] - frame[f'actual_{target}']
                consolidated.append({'model': label, 'variant': variant, 'display': display,
                                     'target': target, 'n': len(error), 'mae': error.abs().mean(),
                                     'rmse': np.sqrt(np.mean(error**2)), 'bias': error.mean(),
                                     'direction_accuracy': np.mean(np.sign(frame['predicted_xG_diff']) ==
                                                                   np.sign(frame['actual_xG_diff']))})
            # Descriptive only: this threshold is not used in fitting/calibration.
            q = frame['actual_xG_for'].quantile(.9)
            upper = frame.loc[frame['actual_xG_for'] >= q]
            extreme.append({'model': label, 'variant': variant, 'threshold': q,
                            'n': len(upper), 'bias': (upper['predicted_xG_for'] - upper['actual_xG_for']).mean()})
    pd.DataFrame(consolidated).to_csv(out / 'smearing_scale_comparison.csv', index=False)
    pd.DataFrame(extreme).to_csv(out / 'upper_decile_bias_descriptive.csv', index=False)
    inputs = [project / 'df_model.csv', project / '5_model_lmm.R', project / '6_model_xgb.py']
    inputs.extend(project / 'results' / name for name in [
        'lmm_log_best_config.csv', 'xgb_log_best_config.csv', 'lmm_log_test_predictions.csv',
        'xgb_log_test_predictions.csv', 'tcn_log_test_predictions.csv', 'xgb_ablation_log_test_predictions.csv',
        'lmm_raw_test_predictions.csv', 'xgb_raw_test_predictions.csv'])
    run_info = {'seed': SEED, 'n_boot': N_BOOT, 'numpy': np.__version__, 'pandas': pd.__version__,
                'contrast': 'MAE(model_1) - MAE(model_2)',
                'ci': 'pointwise 95% percentile bootstrap',
                'holm_family': 'four new contrasts (two comparisons by two targets), separately per resampling unit',
                'cluster_estimand': 'team-match weighted MAE; resample entire competition-season clusters',
                'scope': 'revision-stage exploratory sensitivity; original test set reused',
                'input_sha256': {str(p.relative_to(project)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}}
    (out / 'comparison_settings.json').write_text(json.dumps(run_info, indent=2))
    print(inference.to_string(index=False))
    print(pd.DataFrame(consolidated).to_string(index=False))


if __name__ == '__main__':
    main()

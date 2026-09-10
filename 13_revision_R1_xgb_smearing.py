"""Selected XGBoost refits and validation-based log1p smearing; no tuning.

Usage: python 13_revision_R1_xgb_smearing.py [project_dir] [output_dir]
"""

import ast
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def main():
    project = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd()
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else project / 'results' / 'reviewer1'
    out.mkdir(parents=True, exist_ok=True)
    original = project / '6_model_xgb.py'
    tree = ast.parse(original.read_text())
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name in {'get_features', 'make_model', 'pair_predictions'}]
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(original), 'exec'), globals())
    df = pd.read_csv(project / 'df_model.csv')
    config = pd.read_csv(project / 'results' / 'xgb_log_best_config.csv').iloc[0]
    numeric, categorical = get_features(df, int(config['rolling_window']))
    features = numeric + categorical
    train = df.loc[df['split'] == 'train'].copy()
    validation = df.loc[df['split'] == 'validation'].copy()
    fitting = df.loc[df['split'].isin(['train', 'validation'])].copy()
    test = df.loc[df['split'] == 'test'].copy()
    assert set(train['match_id']).isdisjoint(validation['match_id'])
    assert set(fitting['match_id']).isdisjoint(test['match_id'])
    param_names = ['max_depth', 'learning_rate', 'n_estimators', 'subsample',
                   'colsample_bytree', 'colsample_bylevel', 'min_child_weight',
                   'reg_alpha', 'reg_lambda', 'gamma']
    params = {name: float(config[name]) for name in param_names}
    for name in ['max_depth', 'n_estimators']:
        params[name] = int(params[name])
    params.update(objective='reg:squarederror', eval_metric='rmse', tree_method='hist',
                  device='cpu', n_jobs=4, verbosity=0)
    started = time.monotonic()
    print('Refitting selected training-only XGBoost for validation calibration', flush=True)
    # The saved best-trial seed reproduces the selected validation fit.
    calibration_model = make_model(numeric, categorical, {**params, 'random_state': int(config['seed'])})
    calibration_model.fit(train[features], np.log1p(train['xG_for']))
    validation_log = calibration_model.predict(validation[features])
    residual = np.log1p(validation['xG_for'].to_numpy()) - validation_log
    factor = float(np.mean(np.exp(residual)))
    assert np.isfinite(factor) and factor > 0
    calibration = validation[['match_id', 'team_id', 'competition_id', 'season_id', 'split']].copy()
    calibration['actual_xG_for'] = validation['xG_for']
    calibration['predicted_log'] = validation_log
    calibration['log_residual'] = residual
    calibration.to_csv(out / 'xgb_validation_calibration.csv', index=False)
    pd.DataFrame([{'smearing_factor': factor, 'n_calibration': len(validation),
                   'calibration_split': 'validation', 'calibration_fit_split': 'train',
                   'validation_seed': int(config['seed']), 'final_seed': int(config['scale_seed']),
                   'test_used_for_calibration': False}]).to_csv(out / 'xgb_smearing_factor.csv', index=False)
    print(f'Validation smearing factor: {factor:.9f}', flush=True)
    print('Refitting selected XGBoost on training plus validation', flush=True)
    final = make_model(numeric, categorical, {**params, 'random_state': int(config['scale_seed'])})
    final.fit(fitting[features], np.log1p(fitting['xG_for']))
    test_log = final.predict(test[features])
    uncorrected = np.maximum(np.expm1(test_log), 0)
    corrected = np.maximum(factor * np.exp(test_log.astype(float)) - 1, 0)
    outputs, metric_rows = [], []
    for variant, pred in [('full', uncorrected), ('full_smearing', corrected)]:
        frame = test[['match_id', 'team_id', 'opponent_id', 'split']].copy()
        frame['actual_xG_for'] = test['xG_for'].to_numpy()
        frame['predicted_xG_for'] = pred
        paired, _ = pair_predictions(frame)
        paired.insert(0, 'variant', variant)
        assert (paired.groupby('match_id').size() == 2).all()
        outputs.append(paired)
        for target in ['xG_for', 'xG_against', 'xG_diff']:
            error = paired[f'predicted_{target}'] - paired[f'actual_{target}']
            metric_rows.append({'variant': variant, 'target': target, 'n': len(error),
                                'mae': error.abs().mean(), 'rmse': np.sqrt(np.mean(error**2)),
                                'bias': error.mean()})
    pd.concat(outputs).to_csv(out / 'xgb_test_predictions.csv', index=False)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out / 'xgb_test_metrics.csv', index=False)
    archived = pd.read_csv(project / 'results' / 'xgb_log_test_predictions.csv')
    check = outputs[0][['match_id', 'team_id', 'predicted_xG_for']].merge(
        archived[['match_id', 'team_id', 'predicted_xG_for']],
        on=['match_id', 'team_id'], suffixes=('_refit', '_archived'), validate='one_to_one')
    assert len(check) == len(archived)
    diff = np.abs(check['predicted_xG_for_refit'] - check['predicted_xG_for_archived'])
    reproduction = {'n': len(check), 'max_abs_prediction_difference': float(diff.max()),
                    'mean_abs_prediction_difference': float(diff.mean())}
    pd.DataFrame([reproduction]).to_csv(out / 'xgb_reproduction_check.csv', index=False)
    final.named_steps['xgb'].save_model(out / 'xgb_full_booster.ubj')
    import joblib
    joblib.dump(final, out / 'xgb_full_pipeline.joblib')
    joblib.dump(calibration_model, out / 'xgb_validation_pipeline.joblib')
    metadata = {'python': sys.version, 'platform': platform.platform(),
                'numpy': np.__version__, 'pandas': pd.__version__,
                'sklearn': sklearn.__version__, 'xgboost': xgb.__version__,
                'params': params, 'features': features, 'smearing_factor': factor,
                'validation_seed': int(config['seed']), 'final_seed': int(config['scale_seed']),
                'original_script_sha256': hashlib.sha256(original.read_bytes()).hexdigest(),
                'elapsed_seconds': time.monotonic() - started, 'reproduction': reproduction}
    (out / 'xgb_run_metadata.json').write_text(json.dumps(metadata, indent=2))
    print(metrics.to_string(index=False), flush=True)
    print(json.dumps(reproduction), flush=True)


if __name__ == '__main__':
    main()

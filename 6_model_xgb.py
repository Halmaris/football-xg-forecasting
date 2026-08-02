"""
Fit XGBoost models for team-level xG forecasting.

Input
-----
df_model.csv with xG outcomes, train/validation/test split, lagged and rolling
predictors, match identifiers and team identifiers.

Output
------
All files are saved in results/. For the log and raw response scales, the
script saves tuning results, the selected configuration, test predictions,
test metrics, directional accuracy, feature importance and the feature list.
It also saves combined comparisons of both response scales.
"""

import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# Analysis settings
input_file = 'df_model.csv'
output_dir = Path('results')
response_scales = ['log', 'raw']
rolling_windows = [3, 5, 8, 10]
n_trials = 100
use_gpu = True

output_dir.mkdir(exist_ok=True)
run_seed = int(np.random.SeedSequence().generate_state(1)[0])


def metrics(actual, predicted):
    error = np.asarray(predicted) - np.asarray(actual)
    return {
        'n': len(error),
        'mae': np.mean(np.abs(error)),
        'rmse': np.sqrt(np.mean(error ** 2)),
        'bias': np.mean(error),
    }


def get_features(df, k):
    numeric = [
        'match_week', 'n_previous_matches',
        'lag1_xG_for', 'lag2_xG_for', 'lag3_xG_for', 'lag5_xG_for',
        'lag1_xG_against', 'lag2_xG_against', 'lag3_xG_against',
        'lag5_xG_against',
        'lag1_xG_diff', 'lag2_xG_diff', 'lag3_xG_diff', 'lag5_xG_diff',
        f'rolling{k}_xG_for', f'rolling{k}_xG_against',
        f'rolling{k}_xG_diff',
        f'opponent_rolling{k}_xG_for',
        f'opponent_rolling{k}_xG_against',
        f'opponent_rolling{k}_xG_diff',
    ]
    categorical = [
        'home_away', 'competition_id', 'season_id', 'team_id', 'opponent_id'
    ]
    return (
        [x for x in numeric if x in df.columns],
        [x for x in categorical if x in df.columns],
    )


def make_model(numeric, categorical, params):
    preprocess = ColumnTransformer([
        ('num', SimpleImputer(strategy='median'), numeric),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore')),
        ]), categorical),
    ])
    return Pipeline([
        ('preprocess', preprocess),
        ('xgb', xgb.XGBRegressor(**params)),
    ])


def pair_predictions(predictions):
    opponent = predictions[
        ['match_id', 'team_id', 'actual_xG_for', 'predicted_xG_for']
    ].rename(columns={
        'team_id': 'opponent_id',
        'actual_xG_for': 'actual_xG_against',
        'predicted_xG_for': 'predicted_xG_against',
    })

    paired = predictions.merge(opponent, on=['match_id', 'opponent_id'])
    paired['actual_xG_diff'] = (
        paired['actual_xG_for'] - paired['actual_xG_against']
    )
    paired['predicted_xG_diff'] = (
        paired['predicted_xG_for'] - paired['predicted_xG_against']
    )

    long = []
    for target in ['xG_for', 'xG_against', 'xG_diff']:
        tmp = paired[
            ['match_id', 'team_id', 'opponent_id', 'split',
             f'actual_{target}', f'predicted_{target}']
        ].copy()
        tmp.columns = [
            'match_id', 'team_id', 'opponent_team_id', 'split',
            'actual', 'predicted'
        ]
        tmp['target'] = target
        long.append(tmp)

    return paired, pd.concat(long, ignore_index=True)


df = pd.read_csv(input_file)
if 'xG_diff' not in df:
    df['xG_diff'] = df['xG_for'] - df['xG_against']

train = df[df['split'] == 'train'].copy()
validation = df[df['split'] == 'validation'].copy()
test = df[df['split'] == 'test'].copy()

all_metrics = []
all_direction = []

for scale_no, response_scale in enumerate(response_scales):
    scale_seed = int((run_seed + scale_no) % (2 ** 32 - 1))
    trial_results = []

    def objective(trial):
        k = trial.suggest_categorical('rolling_window', rolling_windows)
        numeric, categorical = get_features(df, k)
        features = numeric + categorical

        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'tree_method': 'hist',
            'device': 'cuda' if use_gpu else 'cpu',
            'max_depth': trial.suggest_int('max_depth', 2, 8),
            'learning_rate': trial.suggest_float(
                'learning_rate', 0.01, 0.20, log=True
            ),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1500),
            'subsample': trial.suggest_float('subsample', 0.60, 1.00),
            'colsample_bytree': trial.suggest_float(
                'colsample_bytree', 0.60, 1.00
            ),
            'colsample_bylevel': trial.suggest_float(
                'colsample_bylevel', 0.60, 1.00
            ),
            'min_child_weight': trial.suggest_float(
                'min_child_weight', 1.0, 50.0, log=True
            ),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
            'reg_lambda': trial.suggest_float(
                'reg_lambda', 0.1, 20.0, log=True
            ),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'random_state': scale_seed + trial.number,
            'n_jobs': -1,
            'verbosity': 0,
        }

        y_train = train['xG_for']
        if response_scale == 'log':
            y_train = np.log1p(y_train)

        model = make_model(numeric, categorical, params)
        model.fit(train[features], y_train)

        pred = model.predict(validation[features])
        if response_scale == 'log':
            pred = np.expm1(pred)
        pred = np.maximum(pred, 0)
        result = metrics(validation['xG_for'], pred)

        trial_results.append({
            'response_scale': response_scale,
            'trial': trial.number,
            'seed': scale_seed + trial.number,
            'rolling_window': k,
            'n_numeric_features': len(numeric),
            'n_categorical_features': len(categorical),
            **trial.params,
            **result,
        })
        print(
            f'{response_scale}: trial {trial.number + 1}/{n_trials}, '
            f'MAE = {result["mae"]:.4f}'
        )
        return result['mae']

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=scale_seed),
    )
    study.optimize(objective, n_trials=n_trials)

    tuning = pd.DataFrame(trial_results).sort_values('mae')
    best = tuning.iloc[0].copy()
    best_k = int(best['rolling_window'])
    numeric, categorical = get_features(df, best_k)
    features = numeric + categorical

    final_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'tree_method': 'hist',
        'device': 'cuda' if use_gpu else 'cpu',
        **{k: v for k, v in study.best_params.items() if k != 'rolling_window'},
        'random_state': scale_seed,
        'n_jobs': -1,
        'verbosity': 0,
    }

    train_validation = df[df['split'].isin(['train', 'validation'])]
    y_train_validation = train_validation['xG_for']
    if response_scale == 'log':
        y_train_validation = np.log1p(y_train_validation)

    final_model = make_model(numeric, categorical, final_params)
    final_model.fit(train_validation[features], y_train_validation)

    pred = final_model.predict(test[features])
    if response_scale == 'log':
        pred = np.expm1(pred)
    pred = np.maximum(pred, 0)

    predictions = test[
        ['match_id', 'team_id', 'opponent_id', 'split']
    ].copy()
    predictions['actual_xG_for'] = test['xG_for'].to_numpy()
    predictions['predicted_xG_for'] = pred
    paired, predictions_long = pair_predictions(predictions)

    test_metrics = []
    for target, group in predictions_long.groupby('target'):
        test_metrics.append({
            'model': 'XGBoost',
            'response_scale': response_scale,
            'target': target,
            'rolling_window': best_k,
            'nrounds': final_params['n_estimators'],
            **metrics(group['actual'], group['predicted']),
        })
    test_metrics = pd.DataFrame(test_metrics)

    direction = pd.DataFrame([{
        'model': 'XGBoost',
        'response_scale': response_scale,
        'target': 'xG_diff',
        'n': len(paired),
        'direction_accuracy': np.mean(
            np.sign(paired['actual_xG_diff']) ==
            np.sign(paired['predicted_xG_diff'])
        ),
    }])

    feature_names = final_model.named_steps['preprocess'].get_feature_names_out()
    importance = pd.DataFrame({
        'response_scale': response_scale,
        'feature': feature_names,
        'gain': final_model.named_steps['xgb'].feature_importances_,
    }).sort_values('gain', ascending=False)

    feature_list = pd.DataFrame({
        'response_scale': response_scale,
        'feature': features,
        'feature_type': (
            ['numeric'] * len(numeric) + ['categorical'] * len(categorical)
        ),
    })

    best['run_seed'] = run_seed
    best['scale_seed'] = scale_seed
    prefix = f'xgb_{response_scale}'
    tuning.to_csv(output_dir / f'{prefix}_tuning_results.csv', index=False)
    pd.DataFrame([best]).to_csv(
        output_dir / f'{prefix}_best_config.csv', index=False
    )
    test_metrics.to_csv(
        output_dir / f'{prefix}_test_metrics.csv', index=False
    )
    direction.to_csv(
        output_dir / f'{prefix}_direction_accuracy.csv', index=False
    )
    paired.to_csv(
        output_dir / f'{prefix}_test_predictions.csv', index=False
    )
    predictions_long.to_csv(
        output_dir / f'{prefix}_test_predictions_long.csv', index=False
    )
    importance.to_csv(
        output_dir / f'{prefix}_importance.csv', index=False
    )
    feature_list.to_csv(
        output_dir / f'{prefix}_feature_list.csv', index=False
    )

    with open(output_dir / f'{prefix}_settings.json', 'w') as file:
        json.dump({
            'run_seed': run_seed,
            'scale_seed': scale_seed,
            'response_scale': response_scale,
            'rolling_windows': rolling_windows,
            'n_trials': n_trials,
            'use_gpu': use_gpu,
        }, file, indent=2)

    all_metrics.append(test_metrics)
    all_direction.append(direction)

pd.concat(all_metrics, ignore_index=True).to_csv(
    output_dir / 'xgb_scale_comparison_test_metrics.csv', index=False
)
pd.concat(all_direction, ignore_index=True).to_csv(
    output_dir / 'xgb_scale_comparison_direction_accuracy.csv', index=False
)

print(f'Run seed: {run_seed}')
print('All files saved in results/.')

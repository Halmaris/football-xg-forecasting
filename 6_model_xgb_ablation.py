"""
Run a feature-set ablation study for the log-response XGBoost model.

Input
-----
df_model.csv with the train/validation/test split and the predictors used by
the primary XGBoost analysis.

Output
------
All files are saved in results/. The script saves tuning results, selected
configurations, test predictions, test metrics, directional accuracy, feature
importance, feature lists and paired bootstrap comparisons with the full model.
"""

from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# Settings
input_file = 'df_model.csv'
output_dir = Path('results')
rolling_windows = [3, 5, 8, 10]
n_trials = 100
n_bootstrap = 5000
use_gpu = True

variants = [
    'full',
    'matched_information',
    'context_only',
    'without_identity',
    'without_opponent_history',
    'without_match_context',
]

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


def get_features(df, k, variant):
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

    if variant == 'matched_information':
        numeric = [
            x for x in numeric
            if (x.startswith('lag') and x.endswith('_xG_for'))
            or x == f'rolling{k}_xG_for'
        ]
        categorical = ['home_away']
    elif variant == 'context_only':
        numeric = []
    elif variant == 'without_identity':
        categorical = ['home_away']
    elif variant == 'without_opponent_history':
        numeric = [x for x in numeric if not x.startswith('opponent_')]
    elif variant == 'without_match_context':
        numeric = [
            x for x in numeric
            if x not in ['match_week', 'n_previous_matches']
        ]
        categorical = [x for x in categorical if x != 'home_away']

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


def holm_adjust(p_values):
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    previous = 0

    for rank, index in enumerate(order):
        value = min((len(p_values) - rank) * p_values[index], 1)
        previous = max(previous, value)
        adjusted[index] = previous

    return adjusted


df = pd.read_csv(input_file)
if 'xG_diff' not in df:
    df['xG_diff'] = df['xG_for'] - df['xG_against']

train = df[df['split'] == 'train']
validation = df[df['split'] == 'validation']
test = df[df['split'] == 'test']
train_validation = df[df['split'].isin(['train', 'validation'])]

tuning_list = []
best_list = []
metrics_list = []
direction_list = []
paired_list = []
prediction_list = []
importance_list = []
feature_list = []

for variant_no, variant in enumerate(variants):
    variant_seed = int((run_seed + variant_no) % (2 ** 32 - 1))
    trial_rows = []

    def objective(trial):
        k = trial.suggest_categorical('rolling_window', rolling_windows)
        numeric, categorical = get_features(df, k, variant)
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
            'random_state': variant_seed + trial.number,
            'n_jobs': -1,
            'verbosity': 0,
        }

        model = make_model(numeric, categorical, params)
        model.fit(train[features], np.log1p(train['xG_for']))
        pred = np.maximum(
            np.expm1(model.predict(validation[features])),
            0,
        )
        result = metrics(validation['xG_for'], pred)

        trial_rows.append({
            'variant': variant,
            'trial': trial.number,
            'seed': variant_seed + trial.number,
            'rolling_window': k,
            'n_numeric_features': len(numeric),
            'n_categorical_features': len(categorical),
            **trial.params,
            **result,
        })

        print(
            f'{variant}: trial {trial.number + 1}/{n_trials}, '
            f'MAE = {result["mae"]:.4f}'
        )
        return result['mae']

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=variant_seed),
    )
    study.optimize(objective, n_trials=n_trials)

    tuning = pd.DataFrame(trial_rows).sort_values('mae')
    best = tuning.iloc[0].copy()
    best_k = int(best['rolling_window'])
    numeric, categorical = get_features(df, best_k, variant)
    features = numeric + categorical

    final_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'tree_method': 'hist',
        'device': 'cuda' if use_gpu else 'cpu',
        **{
            key: value for key, value in study.best_params.items()
            if key != 'rolling_window'
        },
        'random_state': variant_seed,
        'n_jobs': -1,
        'verbosity': 0,
    }

    model = make_model(numeric, categorical, final_params)
    model.fit(
        train_validation[features],
        np.log1p(train_validation['xG_for']),
    )
    pred = np.maximum(
        np.expm1(model.predict(test[features])),
        0,
    )

    predictions = test[
        ['match_id', 'team_id', 'opponent_id', 'split']
    ].copy()
    predictions['actual_xG_for'] = test['xG_for'].to_numpy()
    predictions['predicted_xG_for'] = pred

    paired, predictions_long = pair_predictions(predictions)
    paired.insert(0, 'variant', variant)
    predictions_long.insert(0, 'variant', variant)

    for target, group in predictions_long.groupby('target'):
        metrics_list.append({
            'model': 'XGBoost',
            'variant': variant,
            'response_scale': 'log',
            'target': target,
            'rolling_window': best_k,
            'nrounds': final_params['n_estimators'],
            'run_seed': run_seed,
            'variant_seed': variant_seed,
            **metrics(group['actual'], group['predicted']),
        })

    direction_list.append({
        'model': 'XGBoost',
        'variant': variant,
        'response_scale': 'log',
        'target': 'xG_diff',
        'n': len(paired),
        'direction_accuracy': np.mean(
            np.sign(paired['actual_xG_diff']) ==
            np.sign(paired['predicted_xG_diff'])
        ),
        'run_seed': run_seed,
        'variant_seed': variant_seed,
    })

    importance = pd.DataFrame({
        'variant': variant,
        'feature': model.named_steps[
            'preprocess'
        ].get_feature_names_out(),
        'gain': model.named_steps['xgb'].feature_importances_,
    }).sort_values('gain', ascending=False)

    features_used = pd.DataFrame({
        'variant': variant,
        'feature': features,
        'feature_type': (
            ['numeric'] * len(numeric) +
            ['categorical'] * len(categorical)
        ),
    })

    best['run_seed'] = run_seed
    best['variant_seed'] = variant_seed

    tuning_list.append(tuning)
    best_list.append(best)
    paired_list.append(paired)
    prediction_list.append(predictions_long)
    importance_list.append(importance)
    feature_list.append(features_used)

tuning_results = pd.concat(tuning_list, ignore_index=True)
best_config = pd.DataFrame(best_list)
test_metrics = pd.DataFrame(metrics_list)
direction_accuracy = pd.DataFrame(direction_list)
test_predictions = pd.concat(paired_list, ignore_index=True)
test_predictions_long = pd.concat(prediction_list, ignore_index=True)
importance = pd.concat(importance_list, ignore_index=True)
features_used = pd.concat(feature_list, ignore_index=True)

# Bootstrap differences: ablated model minus full model
rng = np.random.default_rng(run_seed)
bootstrap_rows = []

for variant in variants[1:]:
    for target in ['xG_for', 'xG_diff']:
        full = test_predictions_long.query(
            "variant == 'full' and target == @target"
        )[
            ['match_id', 'team_id', 'actual', 'predicted']
        ].rename(columns={'predicted': 'predicted_full'})

        ablated = test_predictions_long.query(
            'variant == @variant and target == @target'
        )[
            ['match_id', 'team_id', 'predicted']
        ].rename(columns={'predicted': 'predicted_ablated'})

        comparison = full.merge(
            ablated,
            on=['match_id', 'team_id'],
        )
        comparison['difference'] = (
            np.abs(comparison['actual'] - comparison['predicted_ablated']) -
            np.abs(comparison['actual'] - comparison['predicted_full'])
        )
        difference = comparison.groupby(
            'match_id'
        )['difference'].mean().to_numpy()

        bootstrap = np.array([
            rng.choice(
                difference,
                size=len(difference),
                replace=True,
            ).mean()
            for _ in range(n_bootstrap)
        ])

        p_value = min(
            1,
            2 * min(
                (np.sum(bootstrap <= 0) + 1) / (n_bootstrap + 1),
                (np.sum(bootstrap >= 0) + 1) / (n_bootstrap + 1),
            ),
        )

        bootstrap_rows.append({
            'variant': variant,
            'reference': 'full',
            'target': target,
            'n_matches': len(difference),
            'mae_difference': difference.mean(),
            'mae_difference_ci_low': np.quantile(bootstrap, 0.025),
            'mae_difference_ci_high': np.quantile(bootstrap, 0.975),
            'p_value': p_value,
            'run_seed': run_seed,
            'n_bootstrap': n_bootstrap,
        })

bootstrap_results = pd.DataFrame(bootstrap_rows)
bootstrap_results['p_holm'] = (
    bootstrap_results
    .groupby('target')['p_value']
    .transform(lambda x: holm_adjust(x.to_numpy()))
)

prefix = 'xgb_ablation_log'

tuning_results.to_csv(
    output_dir / f'{prefix}_tuning_results.csv',
    index=False,
)
best_config.to_csv(
    output_dir / f'{prefix}_best_config.csv',
    index=False,
)
test_metrics.to_csv(
    output_dir / f'{prefix}_test_metrics.csv',
    index=False,
)
direction_accuracy.to_csv(
    output_dir / f'{prefix}_direction_accuracy.csv',
    index=False,
)
test_predictions.to_csv(
    output_dir / f'{prefix}_test_predictions.csv',
    index=False,
)
test_predictions_long.to_csv(
    output_dir / f'{prefix}_test_predictions_long.csv',
    index=False,
)
importance.to_csv(
    output_dir / f'{prefix}_importance.csv',
    index=False,
)
features_used.to_csv(
    output_dir / f'{prefix}_feature_list.csv',
    index=False,
)
bootstrap_results.to_csv(
    output_dir / f'{prefix}_bootstrap_vs_full.csv',
    index=False,
)

pd.DataFrame([{
    'run_seed': run_seed,
    'response_scale': 'log',
    'n_trials': n_trials,
    'n_bootstrap': n_bootstrap,
    'use_gpu': use_gpu,
    'variants': ', '.join(variants),
}]).to_csv(
    output_dir / f'{prefix}_settings.csv',
    index=False,
)

print(f'Run seed: {run_seed}')
print('All files saved in results/.')

"""LOCO evaluation of the tuned XGBoost model."""

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# Analysis settings
input_file = 'df_model.csv'
output_dir = Path('results')
best_config_file = output_dir / 'xgb_log_best_config.csv'
rolling_results_file = (
    output_dir / 'rolling_loco_log_test_metrics_by_competition.csv'
)
response_scale = 'log'
use_gpu = True

file_prefix = f'xgb_loco_{response_scale}'
output_dir.mkdir(exist_ok=True)


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
        tmp = paired[[
            'held_out_competition_id', 'held_out_competition_name',
            'match_id', 'team_id', 'opponent_id', 'split',
            f'actual_{target}', f'predicted_{target}',
        ]].copy()
        tmp.columns = [
            'held_out_competition_id', 'held_out_competition_name',
            'match_id', 'team_id', 'opponent_team_id', 'split',
            'actual', 'predicted',
        ]
        tmp['target'] = target
        long.append(tmp)

    return paired, pd.concat(long, ignore_index=True)


# Read data and the configuration selected in the main analysis
df = pd.read_csv(input_file)
best = pd.read_csv(best_config_file).iloc[0]

best_k = int(best['rolling_window'])
seed_column = 'scale_seed' if 'scale_seed' in best.index else 'seed'
random_state = int(best[seed_column])

parameter_types = {
    'max_depth': int,
    'learning_rate': float,
    'n_estimators': int,
    'subsample': float,
    'colsample_bytree': float,
    'colsample_bylevel': float,
    'min_child_weight': float,
    'reg_alpha': float,
    'reg_lambda': float,
    'gamma': float,
}

params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'tree_method': 'hist',
    'device': 'cuda' if use_gpu else 'cpu',
    **{name: cast(best[name]) for name, cast in parameter_types.items()},
    'random_state': random_state,
    'n_jobs': -1,
    'verbosity': 0,
}

numeric, categorical = get_features(df, best_k)
features = numeric + categorical

competitions = (
    df.loc[df['split'] == 'test', ['competition_id', 'competition_name']]
    .drop_duplicates()
    .sort_values('competition_name')
)


# Leave one competition out
all_paired = []
all_long = []
competition_metrics = []
competition_direction = []

for _, competition in competitions.iterrows():
    held_out_id = competition['competition_id']
    held_out_name = competition['competition_name']

    train = df[
        df['split'].isin(['train', 'validation'])
        & (df['competition_id'] != held_out_id)
    ]
    test = df[
        (df['split'] == 'test')
        & (df['competition_id'] == held_out_id)
    ]

    model = make_model(numeric, categorical, params)
    model.fit(train[features], np.log1p(train['xG_for']))

    predicted = np.maximum(
        np.expm1(model.predict(test[features])),
        0,
    )

    predictions = test[
        ['match_id', 'team_id', 'opponent_id', 'split']
    ].copy()
    predictions['actual_xG_for'] = test['xG_for'].to_numpy()
    predictions['predicted_xG_for'] = predicted
    predictions['held_out_competition_id'] = held_out_id
    predictions['held_out_competition_name'] = held_out_name

    paired, predictions_long = pair_predictions(predictions)

    for target, group in predictions_long.groupby('target'):
        competition_metrics.append({
            'model': 'XGBoost',
            'response_scale': response_scale,
            'held_out_competition_id': held_out_id,
            'held_out_competition_name': held_out_name,
            'rolling_window': best_k,
            'target': target,
            **metrics(group['actual'], group['predicted']),
        })

    competition_direction.append({
        'model': 'XGBoost',
        'response_scale': response_scale,
        'held_out_competition_id': held_out_id,
        'held_out_competition_name': held_out_name,
        'target': 'xG_diff',
        'n': len(paired),
        'direction_accuracy': np.mean(
            np.sign(paired['actual_xG_diff'])
            == np.sign(paired['predicted_xG_diff'])
        ),
    })

    all_paired.append(paired)
    all_long.append(predictions_long)

    mae = metrics(
        predictions['actual_xG_for'],
        predictions['predicted_xG_for'],
    )['mae']
    print(f'{held_out_name}: MAE = {mae:.4f}')


# Overall results
paired = pd.concat(all_paired, ignore_index=True)
predictions_long = pd.concat(all_long, ignore_index=True)
competition_metrics = pd.DataFrame(competition_metrics)
competition_direction = pd.DataFrame(competition_direction)

test_metrics = pd.DataFrame([
    {
        'model': 'XGBoost',
        'response_scale': response_scale,
        'rolling_window': best_k,
        'target': target,
        **metrics(group['actual'], group['predicted']),
    }
    for target, group in predictions_long.groupby('target')
])

direction_accuracy = pd.DataFrame([{
    'model': 'XGBoost',
    'response_scale': response_scale,
    'target': 'xG_diff',
    'n': len(paired),
    'direction_accuracy': np.mean(
        np.sign(paired['actual_xG_diff'])
        == np.sign(paired['predicted_xG_diff'])
    ),
}])

competition_summary = (
    competition_metrics
    .groupby('target', as_index=False)
    .agg(
        n_competitions=('held_out_competition_id', 'nunique'),
        mean_mae=('mae', 'mean'),
        median_mae=('mae', 'median'),
        q1_mae=('mae', lambda x: x.quantile(0.25)),
        q3_mae=('mae', lambda x: x.quantile(0.75)),
        mean_rmse=('rmse', 'mean'),
        median_rmse=('rmse', 'median'),
        mean_bias=('bias', 'mean'),
        median_bias=('bias', 'median'),
    )
)


# Optional comparison with rolling mean
if rolling_results_file.exists():
    rolling = pd.read_csv(rolling_results_file).rename(columns={
        'n': 'rolling_n',
        'mae': 'rolling_mae',
        'rmse': 'rolling_rmse',
        'bias': 'rolling_bias',
    })
    xgb_results = competition_metrics.rename(columns={
        'n': 'xgb_n',
        'mae': 'xgb_mae',
        'rmse': 'xgb_rmse',
        'bias': 'xgb_bias',
    })

    comparison = xgb_results.merge(
        rolling[[
            'held_out_competition_id', 'target',
            'rolling_n', 'rolling_mae', 'rolling_rmse', 'rolling_bias',
        ]],
        on=['held_out_competition_id', 'target'],
    )
    comparison['delta_mae_xgb_minus_rolling'] = (
        comparison['xgb_mae'] - comparison['rolling_mae']
    )
    comparison['xgb_better_mae'] = (
        comparison['delta_mae_xgb_minus_rolling'] < 0
    )

    comparison_summary = (
        comparison
        .groupby('target', as_index=False)
        .agg(
            n_competitions=('held_out_competition_id', 'nunique'),
            mean_delta_mae=('delta_mae_xgb_minus_rolling', 'mean'),
            median_delta_mae=('delta_mae_xgb_minus_rolling', 'median'),
            q1_delta_mae=(
                'delta_mae_xgb_minus_rolling',
                lambda x: x.quantile(0.25),
            ),
            q3_delta_mae=(
                'delta_mae_xgb_minus_rolling',
                lambda x: x.quantile(0.75),
            ),
            xgb_better_n=('xgb_better_mae', 'sum'),
            xgb_better_proportion=('xgb_better_mae', 'mean'),
        )
    )

    comparison.to_csv(
        output_dir / f'{file_prefix}_vs_rolling_by_competition.csv',
        index=False,
    )
    comparison_summary.to_csv(
        output_dir / f'{file_prefix}_vs_rolling_summary.csv',
        index=False,
    )


# Save results
pd.DataFrame([best]).to_csv(
    output_dir / f'{file_prefix}_best_config.csv', index=False
)
test_metrics.to_csv(
    output_dir / f'{file_prefix}_test_metrics.csv', index=False
)
competition_metrics.to_csv(
    output_dir / f'{file_prefix}_test_metrics_by_competition.csv', index=False
)
direction_accuracy.to_csv(
    output_dir / f'{file_prefix}_direction_accuracy.csv', index=False
)
competition_direction.to_csv(
    output_dir / f'{file_prefix}_direction_accuracy_by_competition.csv',
    index=False,
)
competition_summary.to_csv(
    output_dir / f'{file_prefix}_competition_summary.csv', index=False
)
paired.to_csv(
    output_dir / f'{file_prefix}_test_predictions.csv', index=False
)
predictions_long.to_csv(
    output_dir / f'{file_prefix}_test_predictions_long.csv', index=False
)

print(test_metrics)
print('All files saved in results/.')

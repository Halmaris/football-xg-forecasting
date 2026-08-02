"""
Fit Temporal Convolutional Network models for team-level xG forecasting.

Input
-----
df_model.csv with match identifiers, team identifiers, data split, home/away
status and xG_for. The model uses previous xG_for values and the current
home/away indicator.

Output
------
All files are saved in results/. For the log and raw response scales, the
script saves:
- tuning results and the selected configuration,
- training and validation history for the selected Optuna trial,
- the learning-curve plot,
- final training history,
- test predictions, test metrics and directional accuracy.

The run seed is generated automatically and saved with the results. Therefore,
subsequent runs are not identical.
"""

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# Analysis settings
input_file = 'df_model.csv'
output_dir = Path('results')
response_scales = ['log', 'raw']
lookback_values = [3, 5, 8, 10]
n_trials = 100
max_epochs = 80
patience = 10
min_delta = 1e-5
gradient_clip_norm = 0.5

output_dir.mkdir(exist_ok=True)
run_seed = int(np.random.SeedSequence().generate_state(1)[0])
random.seed(run_seed)
np.random.seed(run_seed)
torch.manual_seed(run_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(run_seed)

device = torch.device(
    'cuda' if torch.cuda.is_available()
    else 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    else 'cpu'
)


def metrics(actual, predicted):
    error = np.asarray(predicted) - np.asarray(actual)
    return {
        'n': len(error),
        'mae': np.mean(np.abs(error)),
        'rmse': np.sqrt(np.mean(error ** 2)),
        'bias': np.mean(error),
    }


def make_examples(df, lookback, response_scale):
    data = df.copy()
    data['home_away_num'] = (
        data['home_away'].astype(str).str.lower()
        .map({'home': 1, 'h': 1, '1': 1, 'true': 1,
              'away': 0, 'a': 0, '0': 0, 'false': 0})
    )

    if 'opponent_id' not in data:
        data['opponent_id'] = np.nan

    if 'team_season_id' not in data:
        if {'competition_id', 'season_id'}.issubset(data.columns):
            data['team_season_id'] = (
                data['competition_id'].astype(str) + '_' +
                data['season_id'].astype(str) + '_' +
                data['team_id'].astype(str)
            )
        else:
            data['team_season_id'] = data['team_id']

    if 'match_date' in data:
        data['time'] = pd.to_datetime(data['match_date'])
    elif 'date' in data:
        data['time'] = pd.to_datetime(data['date'])
    elif 'match_week' in data:
        data['time'] = data['match_week']
    else:
        data['time'] = data['match_id']

    data = data.sort_values(['team_season_id', 'time', 'match_id'])
    x_history, x_static, y, meta = [], [], [], []

    for _, group in data.groupby('team_season_id'):
        group = group.reset_index(drop=True)
        for i in range(lookback, len(group)):
            history = group.loc[i - lookback:i - 1, 'xG_for']
            current = group.loc[i]

            if history.isna().any() or pd.isna(current['xG_for']) \
                    or pd.isna(current['home_away_num']):
                continue

            x_history.append(history.to_numpy(dtype=np.float32)[:, None])
            x_static.append([current['home_away_num']])
            y.append(
                np.log1p(current['xG_for'])
                if response_scale == 'log'
                else current['xG_for']
            )
            meta.append({
                'match_id': current['match_id'],
                'team_id': current['team_id'],
                'opponent_id': current['opponent_id'],
                'split': current['split'],
                'actual_xG_for': current['xG_for'],
            })

    return {
        'x_history': np.asarray(x_history, dtype=np.float32),
        'x_static': np.asarray(x_static, dtype=np.float32),
        'y': np.asarray(y, dtype=np.float32),
        'meta': pd.DataFrame(meta),
    }


def subset_examples(examples, rows):
    return {
        'x_history': examples['x_history'][rows],
        'x_static': examples['x_static'][rows],
        'y': examples['y'][rows],
        'meta': examples['meta'].iloc[rows].reset_index(drop=True),
    }


def fit_scalers(examples):
    history = examples['x_history'].reshape(-1, examples['x_history'].shape[-1])
    return {
        'history_mean': history.mean(axis=0),
        'history_sd': history.std(axis=0),
        'static_mean': examples['x_static'].mean(axis=0),
        'static_sd': examples['x_static'].std(axis=0),
        'target_mean': examples['y'].mean(),
        'target_sd': examples['y'].std(),
    }


def scale_examples(examples, scalers):
    history_sd = np.where(scalers['history_sd'] == 0, 1, scalers['history_sd'])
    static_sd = np.where(scalers['static_sd'] == 0, 1, scalers['static_sd'])
    target_sd = scalers['target_sd'] if scalers['target_sd'] > 0 else 1

    return {
        'x_history': (
            (examples['x_history'] - scalers['history_mean'][None, None, :]) /
            history_sd[None, None, :]
        ).astype(np.float32),
        'x_static': (
            (examples['x_static'] - scalers['static_mean'][None, :]) /
            static_sd[None, :]
        ).astype(np.float32),
        'y': ((examples['y'] - scalers['target_mean']) / target_sd).astype(np.float32),
        'meta': examples['meta'].copy(),
    }


def make_loader(examples, batch_size, shuffle=False):
    dataset = TensorDataset(
        torch.tensor(examples['x_history']),
        torch.tensor(examples['x_static']),
        torch.tensor(examples['y']),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, dilation=dilation
        )

    def forward(self, x):
        return self.conv(F.pad(x, (self.padding, 0)))


class TCN(nn.Module):
    def __init__(self, filters, kernel_size, n_blocks, dropout, dense_units):
        super().__init__()
        layers = []
        in_channels = 1

        for block in range(n_blocks):
            layers.extend([
                CausalConv1d(
                    in_channels, filters, kernel_size, dilation=2 ** block
                ),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_channels = filters

        self.tcn = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(filters + 1, dense_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_units, 1),
        )

    def forward(self, x_history, x_static):
        x = self.tcn(x_history.transpose(1, 2)).mean(dim=2)
        return self.head(torch.cat([x, x_static], dim=1)).squeeze(1)


def fit_model(train, validation, params):
    scalers = fit_scalers(train)
    train = scale_examples(train, scalers)
    validation = scale_examples(validation, scalers) if validation is not None else None

    model = TCN(
        filters=params['filters'],
        kernel_size=params['kernel_size'],
        n_blocks=params['n_blocks'],
        dropout=params['dropout'],
        dense_units=params['dense_units'],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params['learning_rate'],
        weight_decay=params['weight_decay'],
    )
    loss_function = nn.SmoothL1Loss()
    train_loader = make_loader(train, params['batch_size'], shuffle=True)
    validation_loader = (
        make_loader(validation, params['batch_size'])
        if validation is not None else None
    )

    best_loss = np.inf
    best_epoch = params['epochs']
    best_state = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, params['epochs'] + 1):
        model.train()
        train_loss = 0

        for x_history, x_static, y in train_loader:
            x_history = x_history.to(device)
            x_static = x_static.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            loss = loss_function(model(x_history, x_static), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
            train_loss += loss.item() * len(y)

        train_loss /= len(train['y'])
        validation_loss = np.nan

        if validation_loader is not None:
            model.eval()
            validation_loss = 0
            with torch.no_grad():
                for x_history, x_static, y in validation_loader:
                    x_history = x_history.to(device)
                    x_static = x_static.to(device)
                    y = y.to(device)
                    validation_loss += (
                        loss_function(model(x_history, x_static), y).item() * len(y)
                    )
            validation_loss /= len(validation['y'])

            if validation_loss < best_loss - min_delta:
                best_loss = validation_loss
                best_epoch = epoch
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': validation_loss,
        })

        if validation_loader is not None and epochs_without_improvement >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, scalers, best_epoch, best_loss, pd.DataFrame(history)


def predict(model, examples, scalers, batch_size, response_scale):
    examples = scale_examples(examples, scalers)
    model.eval()
    predictions = []

    with torch.no_grad():
        for x_history, x_static, _ in make_loader(examples, batch_size):
            predictions.append(
                model(x_history.to(device), x_static.to(device)).cpu().numpy()
            )

    predictions = np.concatenate(predictions)
    predictions = predictions * scalers['target_sd'] + scalers['target_mean']
    if response_scale == 'log':
        predictions = np.expm1(predictions)
    return np.maximum(predictions, 0)


def pair_predictions(predictions):
    xg_for = predictions.drop(columns='actual_xG_against', errors='ignore')

    if xg_for['opponent_id'].notna().any():
        opponent = xg_for[
            ['match_id', 'split', 'team_id', 'actual_xG_for', 'predicted_xG_for']
        ].rename(columns={
            'team_id': 'opponent_id',
            'actual_xG_for': 'actual_xG_against',
            'predicted_xG_for': 'predicted_xG_against',
        })
        paired = xg_for.merge(opponent, on=['match_id', 'split', 'opponent_id'])
        paired = paired.rename(columns={'opponent_id': 'opponent_team_id'})
    else:
        opponent = xg_for.rename(columns={
            'team_id': 'opponent_team_id',
            'actual_xG_for': 'actual_xG_against',
            'predicted_xG_for': 'predicted_xG_against',
        })
        paired = xg_for.merge(opponent, on=['match_id', 'split'])
        paired = paired[paired['team_id'] != paired['opponent_team_id']]

    paired['actual_xG_diff'] = (
        paired['actual_xG_for'] - paired['actual_xG_against']
    )
    paired['predicted_xG_diff'] = (
        paired['predicted_xG_for'] - paired['predicted_xG_against']
    )

    long = []
    for target in ['xG_for', 'xG_against', 'xG_diff']:
        part = paired[[
            'match_id', 'team_id', 'opponent_team_id', 'split',
            f'actual_{target}', f'predicted_{target}'
        ]].copy()
        part.columns = [
            'match_id', 'team_id', 'opponent_team_id', 'split',
            'actual', 'predicted'
        ]
        part['target'] = target
        long.append(part)

    return paired, pd.concat(long, ignore_index=True)


def save_learning_curve(history, best_epoch, prefix):
    plt.figure(figsize=(7, 4.5))
    plt.plot(history['epoch'], history['train_loss'], label='Training loss')
    plt.plot(history['epoch'], history['val_loss'], label='Validation loss')
    plt.axvline(best_epoch, linestyle='--', label=f'Best epoch: {best_epoch}')
    plt.xlabel('Epoch')
    plt.ylabel('Smooth L1 loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f'{prefix}_selected_trial_learning_curve.pdf')
    plt.savefig(
        output_dir / f'{prefix}_selected_trial_learning_curve.png', dpi=300
    )
    plt.close()


# Data
print(f'Device: {device}')
print(f'Run seed: {run_seed}')
df = pd.read_csv(input_file)

all_test_metrics = []
all_direction_accuracy = []
run_info = []

for response_scale in response_scales:
    prefix = f'tcn_{response_scale}'
    examples_cache = {}
    tuning_rows = []
    histories = {}

    def get_examples(lookback):
        if lookback not in examples_cache:
            examples_cache[lookback] = make_examples(
                df, lookback, response_scale
            )
        return examples_cache[lookback]

    def objective(trial):
        params = {
            'lookback': trial.suggest_categorical('lookback', lookback_values),
            'filters': trial.suggest_categorical('filters', [16, 32, 64]),
            'kernel_size': trial.suggest_categorical('kernel_size', [2, 3]),
            'n_blocks': trial.suggest_categorical('n_blocks', [2, 3]),
            'dropout': trial.suggest_float('dropout', 0.05, 0.30),
            'dense_units': trial.suggest_categorical('dense_units', [16, 32, 64]),
            'learning_rate': trial.suggest_float(
                'learning_rate', 1e-4, 3e-3, log=True
            ),
            'weight_decay': trial.suggest_float(
                'weight_decay', 1e-6, 1e-2, log=True
            ),
            'batch_size': trial.suggest_categorical(
                'batch_size', [128, 256, 512]
            ),
            'epochs': max_epochs,
        }

        examples = get_examples(params['lookback'])
        train = subset_examples(
            examples, examples['meta']['split'].eq('train').to_numpy()
        )
        validation = subset_examples(
            examples, examples['meta']['split'].eq('validation').to_numpy()
        )

        model, scalers, best_epoch, best_val_loss, history = fit_model(
            train, validation, params
        )
        pred = predict(
            model, validation, scalers, params['batch_size'], response_scale
        )
        score = metrics(validation['meta']['actual_xG_for'], pred)

        history.insert(0, 'trial', trial.number)
        histories[trial.number] = history
        tuning_rows.append({
            'trial': trial.number,
            **params,
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
            **score,
        })
        return score['mae']

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=run_seed),
    )
    study.optimize(objective, n_trials=n_trials)

    tuning_results = pd.DataFrame(tuning_rows).sort_values('mae')
    best_trial = study.best_trial.number
    best = tuning_results.loc[tuning_results['trial'].eq(best_trial)].iloc[0]
    best_params = {
        'lookback': int(best['lookback']),
        'filters': int(best['filters']),
        'kernel_size': int(best['kernel_size']),
        'n_blocks': int(best['n_blocks']),
        'dropout': float(best['dropout']),
        'dense_units': int(best['dense_units']),
        'learning_rate': float(best['learning_rate']),
        'weight_decay': float(best['weight_decay']),
        'batch_size': int(best['batch_size']),
        'epochs': int(best['best_epoch']),
    }

    selected_history = histories[best_trial].copy()
    selected_history['is_best_epoch'] = selected_history['epoch'].eq(
        best_params['epochs']
    )
    save_learning_curve(selected_history, best_params['epochs'], prefix)

    examples = get_examples(best_params['lookback'])
    train_validation = subset_examples(
        examples,
        examples['meta']['split'].isin(['train', 'validation']).to_numpy(),
    )
    test = subset_examples(
        examples, examples['meta']['split'].eq('test').to_numpy()
    )

    final_model, scalers, _, _, final_history = fit_model(
        train_validation, None, best_params
    )
    pred_test = predict(
        final_model, test, scalers, best_params['batch_size'], response_scale
    )

    predictions = test['meta'].copy()
    predictions['predicted_xG_for'] = pred_test
    paired, predictions_long = pair_predictions(predictions)

    test_metrics = pd.DataFrame([
        {
            'model': 'Temporal ConvNet',
            'response_scale': response_scale,
            'target': target,
            **metrics(group['actual'], group['predicted']),
        }
        for target, group in predictions_long.groupby('target')
    ])

    direction_accuracy = pd.DataFrame([{
        'model': 'Temporal ConvNet',
        'response_scale': response_scale,
        'target': 'xG_diff',
        'n': len(paired),
        'direction_accuracy': np.mean(
            np.sign(paired['actual_xG_diff']) ==
            np.sign(paired['predicted_xG_diff'])
        ),
    }])

    best_config = pd.DataFrame([{
        'response_scale': response_scale,
        'run_seed': run_seed,
        'selected_trial': best_trial,
        **best_params,
    }])

    tuning_results.to_csv(
        output_dir / f'{prefix}_tuning_results.csv', index=False
    )
    best_config.to_csv(
        output_dir / f'{prefix}_best_config.csv', index=False
    )
    selected_history.to_csv(
        output_dir / f'{prefix}_selected_trial_training_history.csv',
        index=False,
    )
    final_history.to_csv(
        output_dir / f'{prefix}_final_fit_training_history.csv', index=False
    )
    paired.to_csv(
        output_dir / f'{prefix}_test_predictions.csv', index=False
    )
    predictions_long.to_csv(
        output_dir / f'{prefix}_test_predictions_long.csv', index=False
    )
    test_metrics.to_csv(
        output_dir / f'{prefix}_test_metrics.csv', index=False
    )
    direction_accuracy.to_csv(
        output_dir / f'{prefix}_direction_accuracy.csv', index=False
    )

    all_test_metrics.append(test_metrics)
    all_direction_accuracy.append(direction_accuracy)
    run_info.append({
        'response_scale': response_scale,
        'run_seed': run_seed,
        'device': str(device),
        'selected_trial': best_trial,
        'best_epoch': best_params['epochs'],
    })

pd.concat(all_test_metrics, ignore_index=True).to_csv(
    output_dir / 'tcn_scale_comparison_test_metrics.csv', index=False
)
pd.concat(all_direction_accuracy, ignore_index=True).to_csv(
    output_dir / 'tcn_scale_comparison_direction_accuracy.csv', index=False
)
pd.DataFrame(run_info).to_csv(output_dir / 'tcn_run_info.csv', index=False)

print(f'Files saved in {output_dir}/')

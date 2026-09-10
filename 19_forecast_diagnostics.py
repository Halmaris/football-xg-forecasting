'''Audit exclusions and evaluate saved point forecasts; no model fitting.

Usage: python 19_forecast_diagnostics.py project_dir output_dir [calendar_dir]
'''
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ['match_id', 'team_id']
SEED = 20260911


def metrics(actual, predicted):
    e = np.asarray(predicted) - np.asarray(actual)
    return dict(n=len(e), mae=float(np.abs(e).mean()),
                rmse=float(np.sqrt(np.mean(e**2))), bias=float(e.mean()))


def load_predictions(path, variant=None):
    df = pd.read_csv(path)
    if variant is not None:
        df = df.loc[df.variant.eq(variant)].copy()
    assert not df.duplicated(KEYS).any()
    finite = np.isfinite(df[['actual_xG_for', 'predicted_xG_for',
                            'actual_xG_diff', 'predicted_xG_diff']]).all(axis=1)
    df = df.loc[finite].groupby('match_id').filter(lambda g: len(g) == 2)
    assert not df.empty
    assert df.groupby('match_id').size().eq(2).all()
    assert np.allclose(df.predicted_xG_diff,
                       df.predicted_xG_for - df.predicted_xG_against, atol=2e-6)
    return df


def panel_interval(df, rng, n_boot=5000):
    units = df.groupby(['competition_id', 'season_id']).agg(
        difference=('difference', 'sum'), n=('difference', 'size')).to_numpy()
    values = np.empty(n_boot)
    for start in range(0, n_boot, 100):
        sampled = units[rng.integers(0, len(units), (100, len(units)))].sum(axis=1)
        values[start:start + 100] = sampled[:, 0] / sampled[:, 1]
    ci = np.quantile(values, [.025, .975])
    return dict(panels=len(units), ci_low=ci[0], ci_high=ci[1])


def calendar_summary(directory):
    df = pd.read_csv(directory / 'calendar_test_predictions.csv')
    rows, contrasts = [], []
    for cohort, mask in [('All', np.ones(len(df), dtype=bool)),
                         ('Observed season', ~df.new_competition_season),
                         ('New season', df.new_competition_season)]:
        group = df.loc[mask]
        assert not group.empty
        for model, part in group.groupby('model'):
            assert part.groupby('match_id').size().eq(2).all()
            for target in ['xG_for', 'xG_diff']:
                rows.append(dict(cohort=cohort, model=model, target=target,
                                 **metrics(part[f'actual_{target}'], part[f'predicted_{target}'])))
        full = group.loc[group.model.eq('LMM full')]
        for comparator in ['LMM context only', 'Rolling mean']:
            pair = full.merge(group.loc[group.model.eq(comparator)], on=KEYS,
                              suffixes=('', '_comparison'), validate='one_to_one')
            for target in ['xG_for', 'xG_diff']:
                pair['difference'] = (
                    np.abs(pair[f'predicted_{target}_comparison'] - pair[f'actual_{target}'])
                    - np.abs(pair[f'predicted_{target}'] - pair[f'actual_{target}']))
                contrasts.append(dict(cohort=cohort, comparator=comparator, target=target,
                    n=len(pair), difference=pair.difference.mean(),
                    **panel_interval(pair, np.random.default_rng(SEED + len(contrasts)))))
    pd.DataFrame(rows).to_csv(directory / 'calendar_metrics.csv', index=False)
    pd.DataFrame(contrasts).to_csv(directory / 'calendar_paired_differences.csv', index=False)


def main():
    project, out = map(Path, sys.argv[1:3])
    out.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(project / 'df_model.csv')
    raw_cols = ['match_id', 'competition_id', 'season_id', 'competition_name',
                'country_name', 'season_name', 'period']
    raw = pd.read_csv(project / 'df.csv', usecols=raw_cols, low_memory=False)
    assert raw.groupby('match_id')[['competition_id', 'season_id']].nunique().eq(1).all().all()
    raw['extra_time'] = raw.period.isin([3, 4])
    raw['shootout'] = raw.period.eq(5)
    matches = raw.groupby('match_id', sort=True).agg(
        competition_id=('competition_id', 'first'), season_id=('season_id', 'first'),
        country_name=('country_name', 'first'), competition_name=('competition_name', 'first'),
        season_name=('season_name', 'first'), extra_time=('extra_time', 'any'),
        shootout=('shootout', 'any'))
    matches['excluded'] = matches.extra_time | matches.shootout
    matches['both'] = matches.extra_time & matches.shootout
    by_competition = matches.groupby(['country_name', 'competition_name']).agg(
        source_matches=('excluded', 'size'), excluded=('excluded', 'sum'),
        extra_time=('extra_time', 'sum'), shootout=('shootout', 'sum'), both=('both', 'sum'))
    by_competition['percent'] = 100 * by_competition.excluded / by_competition.source_matches
    by_competition.to_csv(out / 'exclusions_by_competition.csv')
    matches.loc[matches.excluded].to_csv(out / 'excluded_matches.csv')
    audit = dict(source_matches=len(matches), excluded=int(matches.excluded.sum()),
        extra_time=int(matches.extra_time.sum()), shootout=int(matches.shootout.sum()),
        both=int(matches.both.sum()), retained=int((~matches.excluded).sum()),
        percent=100 * matches.excluded.mean(),
        extra_time_events=int(raw.extra_time.sum()), shootout_events=int(raw.shootout.sum()),
        stage_field_available=False)
    (out / 'exclusion_summary.json').write_text(json.dumps(audit, indent=2))

    specs = [(name, project / 'results' / f'{prefix}_log_test_predictions.csv', None)
             for name, prefix in [('Rolling mean', 'rolling'), ('ARIMA', 'arima'),
                                  ('LMM', 'lmm'), ('XGBoost', 'xgb'), ('TCN', 'tcn')]]
    specs += [('LMM smearing', project / 'results/reviewer1/lmm_test_predictions.csv', 'full_smearing'),
              ('XGBoost smearing', project / 'results/reviewer1/xgb_test_predictions.csv', 'full_smearing')]
    predictions = {name: load_predictions(path, variant) for name, path, variant in specs}
    common = set.intersection(*(set(map(tuple, df[KEYS].to_numpy())) for df in predictions.values()))
    threshold = float(metadata.loc[metadata.split.eq('train'), 'xG_for'].quantile(.9))
    evaluation, bins = [], []
    for name, df in predictions.items():
        df = df.loc[[tuple(k) in common for k in df[KEYS].to_numpy()]].copy()
        assert df.groupby('match_id').size().eq(2).all()
        observed = df.merge(metadata[KEYS + ['split', 'xG_for', 'xG_diff']], on=KEYS, validate='one_to_one')
        assert observed.split_y.eq('test').all() if 'split_y' in observed else observed.split.eq('test').all()
        assert np.allclose(observed.xG_for, observed.actual_xG_for)
        assert np.allclose(observed.xG_diff, observed.actual_xG_diff)
        extreme = df.loc[df.actual_xG_for.ge(threshold)]
        correct = np.sign(df.actual_xG_diff).eq(np.sign(df.predicted_xG_diff))
        weights = np.abs(df.actual_xG_diff)
        evaluation.append(dict(model=name, common_n=len(df), threshold=threshold,
            **metrics(extreme.actual_xG_for, extreme.predicted_xG_for),
            direction_accuracy=100 * correct.mean(),
            weighted_direction_accuracy=100 * np.average(correct, weights=weights)))
        if name in ['LMM', 'LMM smearing', 'XGBoost', 'XGBoost smearing']:
            # Group by forecast values, not by outcomes. Positive smearing preserves ranks.
            all_df = predictions[name].sort_values(['predicted_xG_for'] + KEYS).copy()
            all_df['bin'] = np.minimum(np.arange(len(all_df)) * 10 // len(all_df) + 1, 10)
            b = all_df.groupby('bin').agg(n=('match_id', 'size'),
                forecast=('predicted_xG_for', 'mean'), observed=('actual_xG_for', 'mean'))
            bins.append(b.reset_index().assign(model=name))
    pd.DataFrame(evaluation).to_csv(out / 'point_forecast_diagnostics.csv', index=False)
    calibration = pd.concat(bins, ignore_index=True)
    calibration.to_csv(out / 'forecast_binned_means.csv', index=False)
    inputs = [project / 'df_model.csv', project / 'df.csv'] + [p for _, p, _ in specs]
    settings = dict(seed=SEED, n_boot=5000, extreme_threshold=threshold,
        threshold_source='90th percentile of primary training xGF; pandas linear quantile',
        common_test_rows=len(common), forecast_bins=10,
        interval_unit='competition-season', interval_type='pointwise percentile; exploratory',
        hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        versions={'python': sys.version, 'numpy': np.__version__, 'pandas': pd.__version__})
    (out / 'diagnostics_settings.json').write_text(json.dumps(settings, indent=2))
    if len(sys.argv) > 3:
        calendar_summary(Path(sys.argv[3]))
    print(json.dumps(audit, indent=2))
    print(pd.DataFrame(evaluation).to_string(index=False))


if __name__ == '__main__':
    main()

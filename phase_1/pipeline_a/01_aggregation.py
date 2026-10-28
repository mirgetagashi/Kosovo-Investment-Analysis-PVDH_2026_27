import numpy as np
import pandas as pd
from src.schema import MONEY
from phase_1.pipeline_a import (WINDOWS, LEVELS, BASE_TABLES, INTERMEDIATE, TABLES,
                                load_input, save_frame, mark_running)


def select_features(data):
    columns = ['year', 'month', 'primary_sector'] + MONEY
    selected = data[columns].copy()
    if selected.isna().any().any() or not np.isfinite(selected[MONEY].to_numpy()).all():
        raise ValueError('Pipeline A requires reviewed, finite source attributes; no implicit filling')
    if not selected.month.between(1, 12).all():
        raise ValueError('Invalid month')
    selected['quarter'] = (selected.month - 1) // 3 + 1
    return selected


def reporting_views(data):
    for window in WINDOWS:
        yield window, data if window == 'reported' else data.loc[data.month.le(6)]


def long_sum(view, keys, window):
    totals = view.groupby(keys)[MONEY].sum()
    counts = view.groupby(keys).size().rename('source_records')
    months = view.groupby(keys[:1]).month.nunique().rename('months')
    frame = totals.stack().rename('value').reset_index().rename(columns={f'level_{len(keys)}': 'measure'})
    frame = frame.merge(counts.reset_index(), on=keys).merge(months.reset_index(), on='year')
    frame['partial_year'] = frame.months.lt(12)
    return frame.assign(window=window)[['window', 'year', *keys[1:], 'measure', 'value', 'source_records', 'months', 'partial_year']]


def aggregate(data):
    frame = select_features(data)
    result = dict(monthly=[], quarterly=[], annual=[], sector_monthly=[])
    for window, view in reporting_views(frame):
        result['monthly'].append(long_sum(view, ['year', 'month'], window))
        result['quarterly'].append(long_sum(view, ['year', 'quarter'], window))
        result['annual'].append(long_sum(view, ['year'], window))
        result['sector_monthly'].append(long_sum(view, ['year', 'month', 'primary_sector'], window))
    result = {name: pd.concat(parts, ignore_index=True) for name, parts in result.items()}
    for window, view in reporting_views(frame):
        source = view.groupby('year')[MONEY].sum()
        for name, table in result.items():
            totals = table.loc[table.window.eq(window)].groupby(['year', 'measure']).value.sum().unstack('measure')[MONEY]
            if not np.allclose(source.to_numpy(), totals.reindex(source.index).to_numpy(), rtol=1e-12, atol=.01):
                raise ValueError(f'Aggregation mismatch: {name}/{window}')
    return result


def variation_by_level(result):
    """Coefficient of variation per level over complete years only, so the three levels are comparable."""
    rows = []
    for level in LEVELS:
        table = result[level]
        complete = table.loc[table.window.eq('reported') & ~table.partial_year]
        for measure, group in complete.groupby('measure', sort=False):
            mean = group.value.mean()
            rows.append(dict(level=level, measure=measure, periods=len(group), mean=mean, std=group.value.std(ddof=1),
                             cv=group.value.std(ddof=1) / mean if mean > 0 else np.nan))
    return pd.DataFrame(rows)


def main(data=None, persist=True):
    if persist:
        mark_running('01_aggregation')
    if data is None:
        data, _ = load_input()
    result = aggregate(data)
    cv = variation_by_level(result)
    if persist:
        for name in BASE_TABLES:
            save_frame(result[name], INTERMEDIATE / f'{name}.csv')
        save_frame(cv, TABLES / 'cv_by_level.csv')
        print('A01: 9 features selected; monthly, quarterly, annual and sector series reconcile to source values.')
        print(cv.pivot(index='measure', columns='level', values='cv').round(3).to_string())
    return result, cv


if __name__ == '__main__':
    main()

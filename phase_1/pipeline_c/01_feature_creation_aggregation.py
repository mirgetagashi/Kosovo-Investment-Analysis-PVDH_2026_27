import numpy as np
import pandas as pd
from src.schema import MONEY
from phase_1.pipeline_c import (WINDOWS, TOTALS, BASE_TABLES, INTERMEDIATE, TABLES, eligible_sector,
                                origin_features, origin_shares, load_input, save_frame, mark_running)


def select_features(data):
    columns = ['year', 'month', 'municipality', 'primary_sector'] + MONEY
    selected = data[columns].copy()
    if selected.isna().any().any() or not np.isfinite(selected[MONEY].to_numpy()).all():
        raise ValueError('Pipeline C requires reviewed, finite source attributes; no implicit filling')
    if not selected.month.between(1, 12).all():
        raise ValueError('Invalid month')
    selected['year'] = selected.year.astype('int64')
    selected['month'] = selected.month.astype('int64')
    return origin_features(selected)


def reporting_views(data):
    for window in WINDOWS:
        yield window, data if window == 'reported' else data.loc[data.month.le(6)]


def summarize(view, keys, window):
    """Sum the origin totals for one aggregation level, then derive shares on the aggregate."""
    view = view.assign(nonzero_records=view.total_investment.ne(0), identified_records=view.import_total.ne(0) | view.domestic_total.ne(0))
    grouped = view.groupby(keys, sort=True)
    table = grouped[TOTALS + ['nonzero_records', 'identified_records']].sum()
    table['source_records'] = grouped.size()
    table = table.reset_index()
    if 'year' in keys:
        months = view.groupby('year').month.nunique().rename('months').reset_index()
        table = table.merge(months, on='year')
        table['partial_year'] = table.months.lt(12)
    else:
        table['years'] = f'{view.year.min()}-{view.year.max()}'
    if 'primary_sector' in keys:
        table['sector_eligible'] = table.primary_sector.map(eligible_sector)
    table = origin_shares(table)
    table.insert(0, 'window', window)
    return table


def aggregate(data):
    frame = select_features(data)
    if not np.allclose(frame[['import_total', 'domestic_total', 'mixed_total']].sum(axis=1), frame.total_investment, atol=1e-6):
        raise ValueError('Origin groups do not partition the six source amounts')
    levels = dict(national=['year'], sector_year=['primary_sector', 'year'], municipal_year=['municipality', 'year'],
                  sector_pooled=['primary_sector'], municipal_pooled=['municipality'])
    result = {name: [] for name in levels}
    for window, view in reporting_views(frame):
        for name, keys in levels.items():
            result[name].append(summarize(view, keys, window))
    result = {name: pd.concat(parts, ignore_index=True) for name, parts in result.items()}
    for window, view in reporting_views(frame):
        source = view[TOTALS].sum()
        for name, table in result.items():
            total = table.loc[table.window.eq(window), TOTALS].sum()
            if not np.allclose(source.to_numpy(), total.to_numpy(), rtol=1e-12, atol=.01):
                raise ValueError(f'Aggregation mismatch: {name}/{window}')
    return result, frame


def row_level_definedness(frame):
    """Why shares are not computed per row: most source rows carry no identified-origin amount at all."""
    rows = []
    for window, view in reporting_views(frame):
        identified = view.import_total + view.domestic_total
        rows.append(dict(window=window, records=len(view),
            zero_total_records=int(view.total_investment.eq(0).sum()),
            zero_total_percent=100 * view.total_investment.eq(0).mean(),
            undefined_share_records=int(identified.le(0).sum()),
            undefined_share_percent=100 * identified.le(0).mean(),
            only_import_or_only_domestic_percent=100 * (view.import_total.ne(0) ^ view.domestic_total.ne(0)).mean()))
    return pd.DataFrame(rows)


def main(data=None, persist=True):
    if persist:
        mark_running('01_feature_creation_aggregation')
    if data is None:
        data, _ = load_input()
    result, frame = aggregate(data)
    definedness = row_level_definedness(frame)
    if persist:
        for name in BASE_TABLES:
            save_frame(result[name], INTERMEDIATE / f'{name}.csv')
        save_frame(definedness, TABLES / 'row_level_definedness.csv')
        print('C01: import/domestic/mixed totals created; national, sector, municipal levels reconcile to source values.')
        print(definedness.round(1).to_string(index=False))
        national = result['national'].loc[result['national'].window.eq('reported'),
                                          ['year', 'import_share', 'import_share_lower', 'import_share_upper', 'mixed_share']]
        print(national.round(3).to_string(index=False))
    return result, definedness


if __name__ == '__main__':
    main()

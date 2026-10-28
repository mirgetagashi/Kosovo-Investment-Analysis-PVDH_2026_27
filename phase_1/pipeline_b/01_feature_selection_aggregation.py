import numpy as np
import pandas as pd
from src.schema import MONEY, norm
from phase_1.pipeline_b import (WINDOWS, EXCLUDED_SECTORS, BASE_TABLES, INTERMEDIATE,
                                load_input, save_frame, mark_running)


def select_features(data):
    columns = ['year', 'month', 'municipality', 'primary_sector'] + MONEY
    selected = data[columns].copy()
    if selected.isna().any().any() or not np.isfinite(selected[MONEY].to_numpy()).all():
        raise ValueError('Pipeline B requires reviewed, finite source attributes; no implicit filling')
    if not selected.month.between(1, 12).all():
        raise ValueError('Invalid month')
    return selected


def reporting_views(data):
    for window in WINDOWS:
        yield window, data if window == 'reported' else data.loc[data.month.le(6)]


def aggregate(data):
    frame = select_features(data)
    municipalities = sorted(frame.municipality.unique())
    sectors = sorted(frame.primary_sector.unique())
    if not any(norm(s) not in EXCLUDED_SECTORS for s in sectors):
        raise ValueError('No eligible sector labels for specialization')
    municipal, sector_totals, cells_all, coverage = [], [], [], []
    index = pd.MultiIndex.from_product([municipalities, sectors], names=['municipality', 'primary_sector'])
    for window, view in reporting_views(frame):
        for year, group in view.groupby('year', sort=True):
            eligible = group.primary_sector.map(lambda s: norm(s) not in EXCLUDED_SECTORS)
            unclassified = group.primary_sector.map(lambda s: norm(s) == 'mungon aktiviteti')
            ambiguous = group.primary_sector.map(lambda s: norm(s) == 'person fizik')
            totals = group.groupby('municipality')[MONEY].sum().reindex(municipalities, fill_value=0)
            counts = group.groupby('municipality').size().reindex(municipalities, fill_value=0)
            for measure in MONEY:
                base = pd.DataFrame(dict(municipality=municipalities, value=totals[measure].to_numpy(),
                    source_records=counts.to_numpy()))
                base = base.assign(window=window, year=int(year), measure=measure,
                    months=int(group.month.nunique()), partial_year=group.month.nunique() < 12,
                    negative_cells=int(group[measure].lt(0).sum()),
                    negative_sum=float(group.loc[group[measure].lt(0), measure].sum()))
                municipal.append(base)
                sector = group.groupby('primary_sector')[measure].sum().rename('value').reset_index()
                sector['sector_eligible'] = sector.primary_sector.map(lambda s: norm(s) not in EXCLUDED_SECTORS)
                sector_totals.append(sector.assign(window=window, year=int(year), measure=measure))
                cells = group.groupby(['municipality', 'primary_sector'])[measure].sum()
                grid = cells.reindex(index, fill_value=0).rename('value').reset_index()
                grid['sector_eligible'] = grid.primary_sector.map(lambda s: norm(s) not in EXCLUDED_SECTORS)
                grid['aggregate_zero_combination'] = ~grid.set_index(['municipality', 'primary_sector']).index.isin(cells.index)
                cells_all.append(grid.assign(window=window, year=int(year), measure=measure))
                all_total = float(group[measure].sum())
                eligible_total = float(group.loc[eligible, measure].sum())
                coverage.append(dict(window=window, year=int(year), measure=measure,
                    all_records=len(group), eligible_records=int(eligible.sum()),
                    unclassified_records=int(unclassified.sum()), ambiguous_records=int(ambiguous.sum()),
                    all_value=all_total, eligible_value=eligible_total,
                    unclassified_value=float(group.loc[unclassified, measure].sum()),
                    ambiguous_value=float(group.loc[ambiguous, measure].sum()),
                    eligible_value_share=eligible_total / all_total if all_total > 0 else np.nan))
    result = dict(municipal_base=pd.concat(municipal, ignore_index=True),
        sectors=pd.concat(sector_totals, ignore_index=True),
        municipal_sector_base=pd.concat(cells_all, ignore_index=True), sector_coverage=pd.DataFrame(coverage))
    for keys, group in result['municipal_base'].groupby(['window', 'year', 'measure'], sort=False):
        window, year, measure = keys
        view = frame if window == 'reported' else frame.loc[frame.month.le(6)]
        source = view.loc[view.year.eq(year), measure].sum()
        for name in ['municipal_base', 'sectors', 'municipal_sector_base']:
            target = result[name]
            total = target.loc[target.window.eq(window) & target.year.eq(year) & target.measure.eq(measure), 'value'].sum()
            if not np.isclose(source, total, rtol=1e-12, atol=.01):
                raise ValueError(f'Aggregation mismatch: {name}/{keys}')
    return result


def main(data=None, persist=True):
    if persist:
        mark_running('01_feature_selection_aggregation')
    if data is None:
        data, _ = load_input()
    result = aggregate(data)
    if persist:
        for name in BASE_TABLES:
            save_frame(result[name], INTERMEDIATE / f'{name}.csv')
        print('B01: selected 10 features; three aggregation levels reconcile to source values.')
    return result


if __name__ == '__main__':
    main()

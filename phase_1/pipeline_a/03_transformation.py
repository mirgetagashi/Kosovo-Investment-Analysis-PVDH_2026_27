import numpy as np
import pandas as pd
from phase_1.pipeline_a import GROUP, INTERMEDIATE, TABLES, BASE_YEAR, read_table, save_frame, mark_running


def base_index(table, base):
    """Index with the base year at 100; the base of the Jan-Jun window is Jan-Jun of the base year."""
    return table.value.div(base) * 100 if base > 0 else pd.Series(np.nan, index=table.index)


def annual_index(annual):
    results = []
    for (window, measure), group in annual.groupby(GROUP, sort=False):
        table = group.sort_values('year').reset_index(drop=True)
        base = table.loc[table.year.eq(BASE_YEAR), 'value']
        table['index_2019'] = base_index(table, float(base.iloc[0]) if len(base) else 0)
        results.append(table)
    return pd.concat(results, ignore_index=True)


def decompose(values, months):
    """Classical additive decomposition: centred 2x12 moving average trend, mean seasonal effect per month, residual."""
    trend = values.rolling(12, center=True).mean().rolling(2).mean().shift(-1)
    detrended = values - trend
    effect = detrended.groupby(months).mean()
    effect = effect - effect.mean()
    seasonal = months.map(effect)
    return trend, seasonal, detrended - seasonal


def standardize(values):
    mean, sd = values.mean(), values.std(ddof=1)
    return (values - mean) / (sd if sd and np.isfinite(sd) and sd > 0 else 1.0), mean, sd


def monthly_transformed(monthly):
    results, skewness, components = [], [], []
    for (window, measure), group in monthly.groupby(GROUP, sort=False):
        table = group.sort_values(['year', 'month']).reset_index(drop=True)
        base = table.loc[table.year.eq(BASE_YEAR), 'value'].mean()
        table['index_2019'] = base_index(table, base)
        if window == 'reported':
            table['trend'], table['seasonal'], table['residual'] = decompose(table.value, table.month)
            table['seasonally_adjusted'] = table.value - table.seasonal
            components.append(table.groupby('month').seasonal.first().rename('seasonal_effect').reset_index().assign(measure=measure))
        else:
            table['trend'] = table['seasonal'] = table['residual'] = table['seasonally_adjusted'] = np.nan
        nonnegative = table.value.ge(0)
        table['value_log1p'] = np.log1p(table.value.where(nonnegative))
        table['z_raw'], mean_raw, sd_raw = standardize(table.value)
        table['z_log1p'], mean_log, sd_log = standardize(table.value_log1p)
        skewness.append(dict(window=window, measure=measure, months=len(table), negative_months=int((~nonnegative).sum()),
            mean_raw=mean_raw, sd_raw=sd_raw, skew_raw=table.value.skew(), skew_log1p=table.value_log1p.skew(),
            skew_seasonally_adjusted=table.seasonally_adjusted.skew() if window == 'reported' else np.nan,
            min_max_ratio=table.value.max() / table.value.min() if table.value.min() > 0 else np.nan))
        results.append(table)
    return pd.concat(results, ignore_index=True), pd.DataFrame(skewness), pd.concat(components, ignore_index=True)


def main(monthly=None, annual=None, persist=True):
    if persist:
        mark_running('03_transformation')
    if monthly is None:
        monthly = read_table('monthly', INTERMEDIATE)
        annual = read_table('annual', INTERMEDIATE)
    transformed, skewness, components = monthly_transformed(monthly)
    yearly = annual_index(annual)
    if persist:
        for name, table in [('monthly_transformed', transformed), ('skewness', skewness),
                            ('seasonal_component', components), ('annual_index', yearly)]:
            save_frame(table, TABLES / f'{name}.csv')
        print('A03: base-year index, seasonal decomposition, log1p and z-scores applied.')
        print(skewness.loc[skewness.window.eq('reported'), ['measure', 'skew_raw', 'skew_log1p', 'skew_seasonally_adjusted']].round(2).to_string(index=False))
    return transformed, skewness, components, yearly


if __name__ == '__main__':
    main()

import numpy as np
import pandas as pd
from phase_1.pipeline_a import (GROUP, INTERMEDIATE, TABLES, EXCLUDED_SECTORS, COVID_START, COVID_END,
                                LOCKDOWN_START, LOCKDOWN_END, BASE_YEAR, period_index, in_period, norm,
                                read_table, save_frame, mark_running)

SEASONS = {12: 'winter', 1: 'winter', 2: 'winter', 3: 'spring', 4: 'spring', 5: 'spring',
           6: 'summer', 7: 'summer', 8: 'summer', 9: 'autumn', 10: 'autumn', 11: 'autumn'}


def pct_change(current, previous):
    return (current - previous).div(previous.where(previous.gt(0))) * 100


def calendar_features(table):
    table = table.copy()
    table['t'] = period_index(table.year, table.month)
    table['quarter'] = (table.month - 1) // 3 + 1
    table['season'] = table.month.map(SEASONS)
    table['is_covid'] = in_period(table.year, table.month, COVID_START, COVID_END)
    table['is_lockdown'] = in_period(table.year, table.month, LOCKDOWN_START, LOCKDOWN_END)
    table['is_full_year'] = ~table.partial_year
    return table


def monthly_features(monthly):
    results = []
    for (window, measure), group in monthly.groupby(GROUP, sort=False):
        table = calendar_features(group.sort_values(['year', 'month']).reset_index(drop=True))
        consecutive = table.t.diff().eq(1)
        previous = table.value.shift(1).where(consecutive)
        table['mom_abs'] = table.value - previous
        table['mom_pct'] = pct_change(table.value, previous)
        last_year = table.set_index(['year', 'month']).value
        same_month = pd.Series([last_year.get((y - 1, m), np.nan) for y, m in zip(table.year, table.month)], index=table.index)
        table['yoy_abs'] = table.value - same_month
        table['yoy_pct'] = pct_change(table.value, same_month)
        if window == 'reported':
            table['rolling_3m'] = table.value.rolling(3).mean()
            table['rolling_12m'] = table.value.rolling(12).mean()
        else:
            table['rolling_3m'] = table['rolling_12m'] = np.nan
        table['ytd_cumulative'] = table.groupby('year').value.cumsum()
        annual_mean = table.groupby('year').value.transform('mean')
        table['ratio_to_annual_mean'] = table.value.div(annual_mean.where(annual_mean.gt(0)))
        results.append(table)
    return pd.concat(results, ignore_index=True)


def seasonality(features):
    """Average month-to-annual-mean ratio over complete reported years: >1 means a systematically strong month."""
    complete = features.loc[features.window.eq('reported') & features.is_full_year]
    table = complete.groupby(['measure', 'month']).agg(seasonality_index=('ratio_to_annual_mean', 'mean'),
        years=('year', 'nunique'), min_ratio=('ratio_to_annual_mean', 'min'), max_ratio=('ratio_to_annual_mean', 'max')).reset_index()
    rank = table.groupby('measure').seasonality_index.rank(ascending=False, method='min').astype(int)
    return table.assign(rank_within_measure=rank)


def annual_features(annual):
    results, slopes = [], []
    for (window, measure), group in annual.groupby(GROUP, sort=False):
        table = group.sort_values('year').reset_index(drop=True)
        previous = table.value.shift(1)
        table['yoy_abs'] = table.value - previous
        table['yoy_pct'] = pct_change(table.value, previous)
        table['comparable'] = table.months.eq(table.months.shift(1)) & previous.notna()
        results.append(table)
        complete = table.loc[~table.partial_year] if window == 'reported' else table
        if len(complete) >= 3 and complete.value.gt(0).any():
            slope, intercept = np.polyfit(complete.year, complete.value, 1)
            fitted = slope * complete.year + intercept
            residual = complete.value - fitted
            total = ((complete.value - complete.value.mean()) ** 2).sum()
            slopes.append(dict(window=window, measure=measure, years=len(complete), first_year=int(complete.year.min()),
                last_year=int(complete.year.max()), slope_per_year=slope,
                slope_pct_of_mean=slope / complete.value.mean() * 100 if complete.value.mean() > 0 else np.nan,
                r_squared=1 - (residual ** 2).sum() / total if total > 0 else np.nan))
    return pd.concat(results, ignore_index=True), pd.DataFrame(slopes)


def recovery(sector_monthly):
    """Months after the pandemic start until the 3-month average regains the 2019 monthly mean, per sector and measure."""
    reported = sector_monthly.loc[sector_monthly.window.eq('reported')].copy()
    national = reported.groupby(['window', 'year', 'month', 'measure'], as_index=False).value.sum().assign(primary_sector='ALL')
    series = pd.concat([reported[national.columns], national], ignore_index=True)
    start = period_index(*COVID_START)
    rows = []
    for (measure, sector), group in series.groupby(['measure', 'primary_sector'], sort=False):
        table = group.sort_values(['year', 'month']).reset_index(drop=True)
        table['t'] = period_index(table.year, table.month)
        table['rolling_3m'] = table.value.rolling(3).mean()
        baseline = table.loc[table.year.eq(BASE_YEAR), 'value'].mean()
        eligible = sector == 'ALL' or norm(sector) not in EXCLUDED_SECTORS
        row = dict(window='reported', measure=measure, primary_sector=sector, sector_eligible=eligible,
                   baseline_2019_monthly=baseline, months_observed=len(table))
        after = table.loc[table.t.ge(start)]
        if not eligible:
            row.update(status='excluded_from_sector_analysis')
        elif not baseline > 0:
            row.update(status='no_positive_2019_baseline')
        elif after.rolling_3m.isna().all():
            row.update(status='insufficient_history')
        else:
            first_year = after.loc[after.t.lt(start + 12)]
            trough = first_year.loc[first_year.rolling_3m.idxmin()]
            regained = after.loc[after.rolling_3m.ge(baseline)]
            row.update(status='valid', trough_year=int(trough.year), trough_month=int(trough.month),
                       trough_rolling_3m=trough.rolling_3m, drop_pct=(trough.rolling_3m / baseline - 1) * 100,
                       recovered=not regained.empty,
                       recovery_months=int(regained.t.iloc[0] - start) if not regained.empty else np.nan,
                       recovery_year=int(regained.year.iloc[0]) if not regained.empty else np.nan,
                       recovery_month=int(regained.month.iloc[0]) if not regained.empty else np.nan)
        rows.append(row)
    columns = ['window', 'measure', 'primary_sector', 'sector_eligible', 'status', 'baseline_2019_monthly', 'months_observed',
               'trough_year', 'trough_month', 'trough_rolling_3m', 'drop_pct', 'recovered', 'recovery_months', 'recovery_year', 'recovery_month']
    return pd.DataFrame(rows).reindex(columns=columns)


def main(monthly=None, annual=None, sector_monthly=None, persist=True):
    if persist:
        mark_running('02_feature_creation')
    if monthly is None:
        monthly = read_table('monthly', INTERMEDIATE)
        annual = read_table('annual', INTERMEDIATE)
        sector_monthly = read_table('sector_monthly', INTERMEDIATE)
    features = monthly_features(monthly)
    season = seasonality(features)
    yearly, slopes = annual_features(annual)
    recovered = recovery(sector_monthly)
    if persist:
        for name, table in [('monthly_features', features), ('seasonality', season), ('annual_features', yearly),
                            ('trend_slope', slopes), ('recovery', recovered)]:
            save_frame(table, TABLES / f'{name}.csv')
        print('A02: calendar, growth, rolling, seasonality, trend slope and pandemic recovery features created.')
        national = recovered.loc[recovered.primary_sector.eq('ALL'), ['measure', 'drop_pct', 'recovery_months']]
        print(national.round(1).to_string(index=False))
    return features, season, yearly, slopes, recovered


if __name__ == '__main__':
    main()

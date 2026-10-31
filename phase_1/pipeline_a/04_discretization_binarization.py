import numpy as np
import pandas as pd
from phase_1.pipeline_a import GROUP, TABLES, GROWTH_BINS, GROWTH_LABELS, read_table, save_frame, mark_running

BINARY = ['growth_positive', 'mom_positive', 'above_trend', 'is_covid', 'is_lockdown', 'is_full_year']


def growth_classes(values):
    """Fixed economic thresholds versus the two mechanical binnings; equal width collapses under skewed growth rates."""
    fixed = pd.cut(values, GROWTH_BINS, labels=GROWTH_LABELS, right=False)
    present = values.dropna()
    width = pd.cut(values, 5, labels=[f'ew{i}' for i in range(1, 6)]) if len(present) else pd.Series(np.nan, index=values.index)
    frequency = pd.qcut(values, 5, labels=[f'q{i}' for i in range(1, 6)], duplicates='drop') if present.nunique() >= 5 else pd.Series(np.nan, index=values.index)
    return fixed.astype(object), width.astype(object), frequency.astype(object)


def discretize_monthly(features):
    results = []
    for _, group in features.groupby(GROUP, sort=False):
        table = group.copy()
        table['growth_class'], table['growth_class_equal_width'], table['growth_class_equal_freq'] = growth_classes(table.yoy_pct)
        results.append(table)
    table = pd.concat(results, ignore_index=True)
    table['growth_positive'] = table.yoy_pct.gt(0).where(table.yoy_pct.notna())
    table['mom_positive'] = table.mom_pct.gt(0).where(table.mom_pct.notna())
    table['above_trend'] = table.value.gt(table.rolling_12m).where(table.rolling_12m.notna())
    for column in BINARY:
        table[column] = table[column].astype('Int8')
    return table


def discretize_annual(yearly):
    table = yearly.copy()
    classes = pd.cut(table.yoy_pct.where(table.comparable), GROWTH_BINS, labels=GROWTH_LABELS, right=False)
    table['growth_class'] = classes.astype(object)
    table['growth_positive'] = table.yoy_pct.gt(0).where(table.comparable & table.yoy_pct.notna()).astype('Int8')
    return table


def class_comparison(monthly):
    rows = []
    for method, column in [('fixed_thresholds', 'growth_class'), ('equal_width', 'growth_class_equal_width'),
                           ('equal_frequency', 'growth_class_equal_freq')]:
        for (window, measure), group in monthly.groupby(GROUP, sort=False):
            counts = group[column].value_counts(dropna=True)
            for label, count in counts.items():
                rows.append(dict(window=window, measure=measure, method=method, growth_class=label,
                                 months=int(count), share=count / counts.sum()))
    table = pd.DataFrame(rows)
    largest = table.groupby(['window', 'measure', 'method']).share.max().rename('largest_class_share').reset_index()
    return table, largest


def main(features=None, yearly=None, persist=True):
    if persist:
        mark_running('04_discretization_binarization')
    if features is None:
        features = read_table('monthly_features')
        yearly = read_table('annual_features')
    monthly = discretize_monthly(features)
    annual = discretize_annual(yearly)
    comparison, largest = class_comparison(monthly)
    if persist:
        for name, table in [('monthly_discretized', monthly), ('annual_discretized', annual),
                            ('growth_class_comparison', comparison), ('growth_class_largest_share', largest)]:
            save_frame(table, TABLES / f'{name}.csv')
        print('A04: growth classes (fixed, equal-width, equal-frequency) and binary indicators assigned.')
        print(largest.pivot_table(index='measure', columns='method', values='largest_class_share', aggfunc='max').round(2).to_string())
    return monthly, annual, comparison


if __name__ == '__main__':
    main()

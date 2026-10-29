import numpy as np
import pandas as pd
from src.schema import MONEY
from phase_1.pipeline_c import (MEASURE_LABELS, ORIGIN_OF, NONZERO_VARIANCE_THRESHOLD, MIN_VALUE_SHARE, TABLES,
                                load_input, save_frame, mark_running)


def profile_measures(data):
    """Which of the six monetary attributes actually carry information?

    Two independent criteria, both computed on source rows (reported window):
    - variance threshold on the binary 'amount is nonzero' indicator, p(1-p) >= .95*.05,
      i.e. a column that is zero in more than 95% of rows is near-constant;
    - value share: the column must hold at least 1% of all invested value.
    A column is kept for the multivariate analysis only if it passes both.
    """
    total = float(data[MONEY].sum().sum())
    rows = []
    for measure in MONEY:
        values = data[measure].astype(float)
        p = float(values.ne(0).mean())
        nonnegative = values.where(values.ge(0))
        rows.append(dict(measure=measure, label=MEASURE_LABELS[measure], origin=ORIGIN_OF[measure], records=len(values),
            nonzero_records=int(values.ne(0).sum()), nonzero_rate=p, indicator_variance=p * (1 - p),
            value=float(values.sum()), value_share=float(values.sum()) / total if total > 0 else np.nan,
            negative_records=int(values.lt(0).sum()), skew_raw=values.skew(), skew_log1p=np.log1p(nonnegative).skew()))
    profile = pd.DataFrame(rows)
    profile['passes_variance_threshold'] = profile.indicator_variance.ge(NONZERO_VARIANCE_THRESHOLD)
    profile['passes_value_share'] = profile.value_share.ge(MIN_VALUE_SHARE)
    profile['selected'] = profile.passes_variance_threshold & profile.passes_value_share
    profile['reason'] = np.select(
        [profile.selected, ~profile.passes_variance_threshold & ~profile.passes_value_share,
         ~profile.passes_variance_threshold],
        ['Informative: frequent and economically material',
         'Near-constant zero indicator and negligible value',
         'Near-constant zero indicator'],
        'Negligible share of invested value')
    if profile.loc[profile.selected, 'origin'].nunique() < 2:
        raise ValueError('Feature selection left fewer than two origin groups')
    return profile


def main(data=None, persist=True):
    if persist:
        mark_running('02_feature_selection')
    if data is None:
        data, _ = load_input()
    profile = profile_measures(data)
    if persist:
        save_frame(profile, TABLES / 'feature_selection.csv')
        print('C02: near-empty monetary attributes removed from the multivariate feature set '
              '(they stay inside the origin totals, so totals still reconcile).')
        print(profile[['measure', 'nonzero_rate', 'indicator_variance', 'value_share', 'selected']].round(4).to_string(index=False))
    return profile


if __name__ == '__main__':
    main()

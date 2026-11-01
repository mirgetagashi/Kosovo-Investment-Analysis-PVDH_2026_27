"""Sampling as data reduction: can a 10% sample reproduce the import shares of the full data?

Stage 05 (bootstrap) asks how STABLE a share is. This stage asks how much DATA a share needs.
Three sample designs are drawn DRAWS times each, all with the same total sample size (10% of rows):

- simple_random:          10% of all source rows, every row with the same chance;
- stratified_year_sector: 10% inside every year x sector stratum (at least one row per stratum);
- take_all_large:         the 1% largest rows (by identified investment) are always included,
                          the remaining sample is stratified by year x sector as above.

Investment is extremely concentrated: the 1% largest rows hold about half of the identified value,
so a share estimate mostly depends on whether those few rows are drawn. Stratifying by sector does
not control that; a take-all stratum for the largest units (standard in business surveys) does.

Each sampled row gets an expansion weight (rows in its stratum / sampled rows), so weighted sums
estimate the full-data sums. The estimated import shares are compared with the full-data shares
nationally per year and per sector (pooled 2019-2026): absolute error in percentage points, and
whether the sample puts the sector in the same origin class (stage 04 thresholds).
"""
import argparse
import numpy as np
import pandas as pd
from phase_1.pipeline_c import (CLASS_BINS, CLASS_LABELS, SEED, TABLES, eligible_sector, origin_features,
                                load_input, save_frame, mark_running)

FRACTION = .1
DRAWS = 50
TAKE_ALL = .01
METHODS = ['simple_random', 'stratified_year_sector', 'take_all_large']
PARTS = ['import_total', 'domestic_total']


def share(imported, domestic):
    """import / (import + domestic); undefined for an empty denominator or a negative component (as in origin_shares)."""
    identified = imported + domestic
    valid = (identified > 0) & (imported >= 0) & (domestic >= 0)
    return np.where(valid, imported / np.where(identified > 0, identified, 1.), np.nan)


def classes(values):
    return pd.cut(pd.Series(values), bins=CLASS_BINS, labels=CLASS_LABELS, include_lowest=True).astype('string')


def simple_weights(rows, fraction, rng):
    size = max(1, round(rows * fraction))
    weights = np.zeros(rows)
    weights[rng.choice(rows, size=size, replace=False)] = rows / size
    return weights


def stratified_weights(strata, rows, fraction, rng, weights=None):
    weights = np.zeros(rows) if weights is None else weights
    for index in strata:
        size = max(1, round(len(index) * fraction))
        weights[rng.choice(index, size=size, replace=False)] = len(index) / size
    return weights


def sample_sensitivity(data, fraction=FRACTION, draws=DRAWS, seed=SEED):
    if not 0 < fraction <= 1:
        raise ValueError('Sampling fraction must be in (0, 1]')
    frame = origin_features(data.assign(year=data.year.astype('int64'))).reset_index(drop=True)
    years, year_code = np.unique(frame.year.to_numpy(), return_inverse=True)
    sectors, sector_code = np.unique(frame.primary_sector.astype(str).to_numpy(), return_inverse=True)
    parts = frame[PARTS].to_numpy(float)

    def sums(codes, size, weights):
        return [np.bincount(codes, weights=parts[:, k] * weights, minlength=size) for k in range(len(PARTS))]

    full = np.ones(len(frame))
    true_national = share(*sums(year_code, len(years), full))
    true_sector = share(*sums(sector_code, len(sectors), full))
    stratum = year_code * len(sectors) + sector_code
    order = np.argsort(stratum, kind='stable')
    strata = np.split(order, np.flatnonzero(np.diff(stratum[order])) + 1)
    # Take-all stratum: the largest rows by identified investment, kept with weight 1. The rest is
    # stratified by year x sector at the fraction that keeps the total sample size equal to the others.
    size = np.abs(parts).sum(axis=1)
    large = np.zeros(len(frame), bool)
    large[np.argsort(-size, kind='stable')[:round(len(frame) * TAKE_ALL)]] = True
    target = max(1, round(len(frame) * fraction))
    rest_fraction = max(0., (target - large.sum()) / max(1, (~large).sum()))
    rest_strata = [index[~large[index]] for index in strata]
    rest_strata = [index for index in rest_strata if len(index)]

    rng = np.random.default_rng(seed)
    national, sector, sizes = [], [], []
    for draw in range(draws):
        for method in METHODS:
            if method == 'simple_random':
                weights = simple_weights(len(frame), fraction, rng)
            elif method == 'stratified_year_sector':
                weights = stratified_weights(strata, len(frame), fraction, rng)
            else:
                weights = stratified_weights(rest_strata, len(frame), rest_fraction, rng, large.astype(float))
            sizes.append(dict(method=method, draw=draw, sampled_records=int((weights > 0).sum())))
            national.append(pd.DataFrame(dict(method=method, draw=draw, year=years, true_share=true_national,
                                              sample_share=share(*sums(year_code, len(years), weights)))))
            sector.append(pd.DataFrame(dict(method=method, draw=draw, primary_sector=sectors, true_share=true_sector,
                                            sample_share=share(*sums(sector_code, len(sectors), weights)))))
    national, sector, sizes = pd.concat(national), pd.concat(sector), pd.DataFrame(sizes)
    records = frame.groupby('primary_sector').size()
    sector = sector.loc[sector.primary_sector.map(eligible_sector) & sector.true_share.notna()].copy()
    sector['records'] = sector.primary_sector.map(records)
    return summarize(national, sector, sizes, fraction, draws, seed)


def errors(table, keys):
    table = table.assign(abs_error_pp=(table.sample_share - table.true_share).abs() * 100,
                         defined=table.sample_share.notna())
    return table, table.groupby(keys).agg(true_share=('true_share', 'first'), sample_share_mean=('sample_share', 'mean'),
        mean_abs_error_pp=('abs_error_pp', 'mean'), p95_abs_error_pp=('abs_error_pp', lambda v: v.quantile(.95)),
        max_abs_error_pp=('abs_error_pp', 'max'), defined_draws=('defined', 'sum')).reset_index()


def summarize(national, sector, sizes, fraction, draws, seed):
    national, by_year = errors(national, ['method', 'year'])
    sector['same_class'] = classes(sector.sample_share.to_numpy()).eq(classes(sector.true_share.to_numpy())).fillna(False).to_numpy()
    sector, by_sector = errors(sector, ['method', 'primary_sector'])
    agreement = sector.groupby(['method', 'primary_sector']).agg(records=('records', 'first'), class_agreement=('same_class', 'mean'))
    by_sector = by_sector.merge(agreement.reset_index(), on=['method', 'primary_sector'])
    by_sector['true_class'] = classes(by_sector.true_share.to_numpy()).to_numpy()
    rows = []
    for method in METHODS:
        n = sizes.loc[sizes.method.eq(method), 'sampled_records']
        for level, table in [('national_year', national), ('sector_pooled', sector)]:
            view = table.loc[table.method.eq(method)]
            row = dict(method=method, level=level, fraction=fraction, draws=draws, seed=seed,
                       sampled_records_mean=float(n.mean()), estimates=len(view),
                       undefined_rate=float(1 - view.defined.mean()),
                       mean_abs_error_pp=view.abs_error_pp.mean(), median_abs_error_pp=view.abs_error_pp.median(),
                       p95_abs_error_pp=view.abs_error_pp.quantile(.95), max_abs_error_pp=view.abs_error_pp.max())
            row['class_agreement'] = float(view.same_class.mean()) if level == 'sector_pooled' else np.nan
            rows.append(row)
    return pd.DataFrame(rows), by_year, by_sector.sort_values(['method', 'records'], ascending=[True, False])


def main(data=None, fraction=FRACTION, draws=DRAWS, seed=SEED, persist=True):
    if persist:
        mark_running('05b_sample_sensitivity')
    if data is None:
        data, _ = load_input()
    summary, by_year, by_sector = sample_sensitivity(data, fraction, draws, seed)
    if persist:
        save_frame(summary, TABLES / 'sample_sensitivity_summary.csv')
        save_frame(by_year, TABLES / 'sample_sensitivity_national.csv')
        save_frame(by_sector, TABLES / 'sample_sensitivity_sectors.csv')
        print(f'C05b: {draws} draws of {fraction:.0%} samples per design (simple random, stratified by year x sector, '
              f'take-all {TAKE_ALL:.0%} largest + stratified rest), seed={seed}.')
        print(summary[['method', 'level', 'sampled_records_mean', 'mean_abs_error_pp', 'p95_abs_error_pp',
                       'max_abs_error_pp', 'class_agreement', 'undefined_rate']].round(3).to_string(index=False))
    return summary, by_year, by_sector


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-fraction', type=float, default=FRACTION)
    parser.add_argument('--draws', type=int, default=DRAWS)
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()
    main(fraction=args.sample_fraction, draws=args.draws, seed=args.seed)

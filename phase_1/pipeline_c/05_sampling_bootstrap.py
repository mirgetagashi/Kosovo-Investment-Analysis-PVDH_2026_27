import numpy as np
import pandas as pd
from phase_1.pipeline_c import (CLASS_BINS, CLASS_LABELS, DOMINANCE_THRESHOLD, BOOTSTRAP_ITERATIONS, CONFIDENCE, SEED,
                                TABLES, eligible_sector, origin_features, load_input, save_frame, mark_running)


def stratified_bootstrap(frame, rng, iterations=BOOTSTRAP_ITERATIONS):
    """Resample source rows with replacement inside each year, so every replicate keeps the yearly structure.

    Returns the replicated import, domestic and mixed sums; the shares are ratios of these sums.
    """
    sums = {name: np.zeros(iterations) for name in ['import_total', 'domestic_total', 'mixed_total']}
    for _, stratum in frame.groupby('year', sort=True):
        index = rng.integers(0, len(stratum), size=(iterations, len(stratum)))
        for name in sums:
            sums[name] += stratum[name].to_numpy()[index].sum(axis=1)
    return sums


def class_of(value):
    return pd.cut(pd.Series([value]), bins=CLASS_BINS, labels=CLASS_LABELS, include_lowest=True).astype('string').iloc[0]


def sector_intervals(data, iterations=BOOTSTRAP_ITERATIONS, seed=SEED):
    frame = origin_features(data.assign(year=data.year.astype('int64')))
    rng = np.random.default_rng(seed)
    alpha = (1 - CONFIDENCE) / 2
    rows = []
    for sector, group in frame.groupby('primary_sector', sort=True):
        imported, domestic, mixed = (float(group[c].sum()) for c in ['import_total', 'domestic_total', 'mixed_total'])
        top = group.import_total.nlargest(5).sum()
        row = dict(primary_sector=sector, sector_eligible=eligible_sector(sector), records=len(group),
                   years=group.year.nunique(), identified_total=imported + domestic,
                   top5_records_import_share=top / imported if imported > 0 else np.nan)
        if imported + domestic <= 0 or min(imported, domestic, mixed) < 0:
            rows.append({**row, 'status': 'undefined_share'})
            continue
        sums = stratified_bootstrap(group, rng, iterations)
        identified = sums['import_total'] + sums['domestic_total']
        total = identified + sums['mixed_total']
        share = np.divide(sums['import_total'], identified, out=np.full(iterations, np.nan), where=identified > 0)
        upper = np.divide(sums['import_total'] + sums['mixed_total'], total, out=np.full(iterations, np.nan), where=total > 0)
        valid = share[np.isfinite(share)]
        point = imported / (imported + domestic)
        low, high = np.quantile(valid, [alpha, 1 - alpha])
        upper_low, upper_high = np.nanquantile(upper, [alpha, 1 - alpha])
        point_class = class_of(point)
        rows.append({**row, 'status': 'valid', 'import_share': point, 'bootstrap_mean': valid.mean(),
            'bootstrap_se': valid.std(ddof=1), 'ci_low': low, 'ci_high': high, 'ci_width': high - low,
            'import_share_upper': (imported + mixed) / (imported + domestic + mixed),
            'upper_ci_low': upper_low, 'upper_ci_high': upper_high,
            'valid_iterations': len(valid), 'origin_class': point_class,
            'dominant_stable': bool(low > DOMINANCE_THRESHOLD or high <= DOMINANCE_THRESHOLD),
            'class_stable': bool(class_of(low) == point_class == class_of(high))})
    result = pd.DataFrame(rows)
    result['stability'] = np.select([result.status.ne('valid'), result.class_stable.eq(True), result.dominant_stable.eq(True)],
                                    ['undefined', 'class_stable', 'dominance_stable_only'], 'unstable')
    return result


def main(data=None, persist=True):
    if persist:
        mark_running('05_sampling_bootstrap')
    if data is None:
        data, _ = load_input()
    intervals = sector_intervals(data)
    if persist:
        save_frame(intervals, TABLES / 'bootstrap_ci.csv')
        view = intervals.loc[intervals.sector_eligible & intervals.status.eq('valid')].sort_values('import_share')
        print(f'C05: {BOOTSTRAP_ITERATIONS} year-stratified bootstrap replicates per sector, {CONFIDENCE:.0%} percentile intervals.')
        print(view[['primary_sector', 'records', 'import_share', 'ci_low', 'ci_high', 'top5_records_import_share', 'stability']]
              .assign(primary_sector=lambda t: t.primary_sector.str.slice(0, 40)).round(3).to_string(index=False))
    return intervals


if __name__ == '__main__':
    main()

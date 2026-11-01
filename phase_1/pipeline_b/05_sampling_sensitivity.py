import argparse
from importlib import import_module
import pandas as pd
from phase_1.pipeline_b import MONEY, TABLES, read_table, load_input, save_frame, mark_running
aggregation = import_module('phase_1.pipeline_b.01_feature_selection_aggregation')
geography = import_module('phase_1.pipeline_b.02_geographic_share_hhi')
reporting_views = aggregation.reporting_views
concentration = geography.concentration


def stratified_sensitivity(frame, full_concentration, fraction=.2, seed=42):
    if not 0 < fraction <= 1:
        raise ValueError('Sampling fraction must be in (0, 1]')
    strata, comparisons = [], []
    municipalities = sorted(frame.municipality.unique())
    for window, view in reporting_views(frame):
        samples = []
        for (year, municipality), group in view.groupby(['year', 'municipality'], sort=True):
            n = max(1, round(len(group) * fraction))
            sample = group.sample(n=n, random_state=seed).copy()
            weight = len(group) / n
            sample[MONEY] = sample[MONEY].astype(float) * weight
            samples.append(sample)
            strata.append(dict(window=window, year=int(year), municipality=municipality,
                population_records=len(group), sampled_records=n, expansion_weight=weight, seed=seed))
        sampled = pd.concat(samples, ignore_index=True)
        for year, group in sampled.groupby('year', sort=True):
            weighted = group.groupby('municipality')[MONEY].sum().reindex(municipalities, fill_value=0)
            for measure in MONEY:
                _, hhi, status = concentration(weighted[measure])
                full = full_concentration.loc[(full_concentration.window == window) &
                    (full_concentration.year == year) & (full_concentration.measure == measure)].iloc[0]
                comparisons.append(dict(window=window, year=int(year), measure=measure,
                    sample_fraction=fraction, seed=seed, population_records=len(view.loc[view.year.eq(year)]),
                    sampled_records=len(group), full_hhi=full.hhi, sampled_weighted_hhi=hhi,
                    hhi_difference=hhi-full.hhi, full_total=full.total,
                    sampled_weighted_total=float(weighted[measure].sum()), sample_status=status))
    return pd.DataFrame(strata), pd.DataFrame(comparisons)


def main(frame=None, summary=None, fraction=.2, seed=42, persist=True):
    if persist:
        mark_running('05_sampling_sensitivity')
    if frame is None:
        data, _ = load_input()
        frame = aggregation.select_features(data)
        summary = read_table('concentration')
    strata, sensitivity = stratified_sensitivity(frame, summary, fraction, seed)
    if persist:
        save_frame(strata, TABLES / 'sample_strata.csv')
        save_frame(sensitivity, TABLES / 'sample_sensitivity.csv')
        print(f'B05: {100*fraction:g}% weighted stratified sensitivity saved; seed={seed}.')
    return strata, sensitivity


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-fraction', type=float, default=.2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(fraction=args.sample_fraction, seed=args.seed)

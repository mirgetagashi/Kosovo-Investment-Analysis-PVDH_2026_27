import argparse
import json
from importlib import import_module
import numpy as np
import pandas as pd
from phase_1.pipeline_b import (ROOT, RAW, COMMON, MONEY, TABLES, FIGURES,
    INTERMEDIATE, VALIDATION, load_input, digest, save_frame, save_json, norm)
aggregation = import_module('phase_1.pipeline_b.01_feature_selection_aggregation')
geography = import_module('phase_1.pipeline_b.02_geographic_share_hhi')
specialization = import_module('phase_1.pipeline_b.03_sector_specialization_lq')
features = import_module('phase_1.pipeline_b.04_discretization_binarization')
sampling = import_module('phase_1.pipeline_b.05_sampling_sensitivity')
visualization = import_module('phase_1.pipeline_b.06_visualization')
select_features = aggregation.select_features
reporting_views = aggregation.reporting_views


def analyse(data, sample_fraction=.2, seed=42, persist=False):
    base = aggregation.main(data, persist=persist)
    municipal, summary = geography.main(base['municipal_base'], persist=persist)
    lq = specialization.main(base['municipal_sector_base'], base['sectors'], base['sector_coverage'], persist=persist)
    municipal, lq, profiles, top = features.main(municipal, lq, persist=persist)
    strata, sensitivity = sampling.main(select_features(data), summary, sample_fraction, seed, persist=persist)
    results = dict(municipal=municipal, concentration=summary, sectors=base['sectors'], specialization=lq,
        sector_coverage=base['sector_coverage'], profiles=profiles, top_specialization=top,
        sample_strata=strata, sample_sensitivity=sensitivity)
    if persist:
        visualization.main(results)
    return results


def validate_results(data, results):
    frame = select_features(data)
    checks = []
    for window, view in reporting_views(frame):
        for year, group in view.groupby('year'):
            for measure in MONEY:
                key = lambda table: table.loc[(table.window == window) & (table.year == year) & (table.measure == measure)]
                municipal = key(results['municipal'])
                sectors = key(results['sectors'])
                coverage = key(results['sector_coverage']).iloc[0]
                summary = key(results['concentration']).iloc[0]
                lq = key(results['specialization'])
                for name, actual in [('municipal', municipal.value.sum()), ('sector', sectors.value.sum()),
                                     ('municipal-sector', lq.value.sum())]:
                    if not np.isclose(group[measure].sum(), actual, rtol=1e-12, atol=.01):
                        raise ValueError(f'{name} aggregation changed values: {window}/{year}/{measure}')
                if coverage.all_records != len(group) or coverage.eligible_records + coverage.unclassified_records + coverage.ambiguous_records != len(group):
                    raise ValueError('Sector coverage does not account for every record')
                if not np.isclose(coverage.all_value, coverage.eligible_value + coverage.unclassified_value + coverage.ambiguous_value, atol=.01):
                    raise ValueError('Excluded-sector amounts do not reconcile')
                if summary.status == 'valid':
                    if not np.isclose(municipal.municipal_share.sum(), 1) or not 1/len(municipal)-1e-12 <= summary.hhi <= 1+1e-12:
                        raise ValueError('Invalid geographic shares/HHI')
                elif municipal.municipal_share.notna().any() or np.isfinite(summary.hhi):
                    raise ValueError('Undefined geographic indicators must remain missing')
                valid = lq.lq_status.eq('valid')
                if lq.loc[~valid, 'location_quotient'].notna().any() or lq.loc[~valid, 'high_specialization'].notna().any():
                    raise ValueError('Undefined LQ/flags must remain missing')
                if not np.isfinite(lq.loc[valid, 'location_quotient']).all():
                    raise ValueError('Valid LQ must be finite')
                if valid.any():
                    expected = lq.loc[valid, 'local_sector_share'] / lq.loc[valid, 'national_sector_share']
                    if not np.allclose(expected, lq.loc[valid, 'location_quotient']):
                        raise ValueError('LQ formula mismatch')
                checks.append(dict(window=window, year=int(year), measure=measure, records=len(group),
                    source_total=float(group[measure].sum()), aggregate_total=float(municipal.value.sum()),
                    hhi_status=summary.status, undefined_lq_cells=int((~valid).sum()), status='PASS'))
    return pd.DataFrame(checks)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Run Pipeline B stages 01-06 in order')
    parser.add_argument('--sample-fraction', type=float, default=.2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    save_json(dict(status='RUNNING'), VALIDATION)
    try:
        data, common_status = load_input()
        protected = {str(p.relative_to(ROOT)):digest(p)
            for directory in [RAW, ROOT/'data/intermediate', ROOT/'data/processed']
            for p in directory.rglob('*') if p.is_file() and not p.is_relative_to(INTERMEDIATE)}
        results = analyse(data, args.sample_fraction, args.seed, persist=True)
        validation = validate_results(data, results)
        save_frame(validation, TABLES / 'validation.csv')
        latest = int(results['concentration'].loc[lambda t:t.window.eq('reported') & t.months.eq(12), 'year'].max())
        if any(digest(ROOT/name)!=value for name,value in protected.items()):
            raise ValueError('An input/common-preprocessing file was modified')
        warnings = ['Currency/scale and official activity codebook remain unconfirmed.',
            '2026 is partial; Jan-Jun comparisons use the same months for all years.',
            'LQ uses the eligible-sector subset and is conditional on its disclosed coverage.',
            'Person Fizik is ambiguous, not automatically recoded or declared missing.',
            'Static 2017-represented boundaries are not historical boundary harmonization.',
            f'One weighted {100*args.sample_fraction:g}% stratified sample is descriptive sensitivity, not a confidence interval.',
            'Historical common baseline archive is absent.' if common_status['checks'].get('REVIEW') else 'Common baseline available.']
        output_hashes = {str(p.relative_to(ROOT)):digest(p) for folder in [TABLES, FIGURES] for p in folder.glob('*') if p.is_file()}
        save_json(dict(status='PASS', input_sha256=digest(COMMON), records=len(data),
            municipalities=int(data.municipality.nunique()), unclassified_records=int(data.sector_unclassified.sum()),
            ambiguous_records=int(data.primary_sector.map(norm).eq('person fizik').sum()), latest_complete_year=latest,
            monetary_measures=MONEY, analysis_checks=len(validation), map_matches=int(data.municipality.nunique()),
            hhi_status_counts=results['concentration'].status.value_counts().to_dict(),
            lq_status_counts=results['specialization'].lq_status.value_counts().to_dict(),
            sample_fraction=args.sample_fraction, seed=args.seed, protected_input_hashes=protected,
            output_hashes=output_hashes, warnings=warnings,
            stages=['01_feature_selection_aggregation','02_geographic_share_hhi','03_sector_specialization_lq',
                    '04_discretization_binarization','05_sampling_sensitivity','06_visualization']), VALIDATION)
        print(f'Pipeline B: PASS; {len(data):,} source records; {len(validation)} reconciliations; six stages completed.')
    except Exception as error:
        save_json(dict(status='FAIL', reason=str(error)), VALIDATION)
        raise


if __name__ == '__main__':
    main()

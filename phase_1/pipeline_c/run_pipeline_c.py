import importlib
from phase_1.pipeline_c import VALIDATION, SEED, BOOTSTRAP_ITERATIONS, LOW_COVERAGE, load_input, save_json


def stage(name):
    return importlib.import_module(f'phase_1.pipeline_c.{name}')


def main():
    """Run Pipeline C end to end on the validated common dataset.

    python -m phase_1.run_common_preprocessing   (once, produces the common dataset)
    python -m phase_1.pipeline_c.run_pipeline_c
    """
    try:
        data, status = load_input()
        base, definedness = stage('01_feature_creation_aggregation').main(data)
        selection = stage('02_feature_selection').main(data)
        scores, loadings, explained, contrast = stage('03_transformation_pca').main(data, selection)
        classified, counts, ranking = stage('04_discretization_binarization').main(base)
        intervals = stage('05_sampling_bootstrap').main(data)
        sampling, _, _ = stage('05b_sample_sensitivity').main(data)
        figures = stage('06_visualization').main()
        national = classified['national'].query("window == 'reported'")
        sectors = classified['sector_pooled'].query("window == 'reported' and sector_eligible and share_status == 'valid'")
        low_coverage = sectors.loc[sectors.identified_coverage.lt(LOW_COVERAGE)]
        save_json(dict(status='PASS', input_records=len(data), input_sha256=status['dataset_sha256'],
            selected_features=selection.loc[selection.selected, 'measure'].tolist(),
            dropped_features=selection.loc[~selection.selected, 'measure'].tolist(),
            pca_rows=len(scores), pca_explained_variance=explained.set_index('component').explained_variance_ratio.round(4).to_dict(),
            pc2_separates_import_from_domestic=bool(contrast.set_index('component').loc['PC2', 'separates_import_from_domestic']),
            national_import_share=national.set_index('year').import_share.round(4).to_dict(),
            bootstrap=dict(iterations=BOOTSTRAP_ITERATIONS, seed=SEED,
                           stability=intervals.loc[intervals.sector_eligible].stability.value_counts().to_dict()),
            sample_sensitivity_mean_abs_error_pp={f'{row.method}/{row.level}': round(float(row.mean_abs_error_pp), 3)
                                                  for row in sampling.itertuples()},
            low_coverage_threshold=LOW_COVERAGE,
            low_coverage_sectors={row.primary_sector: round(float(row.identified_coverage), 3) for row in low_coverage.itertuples()},
            class_robust_sectors=f'{int(sectors.class_robust.sum())} of {len(sectors)}',
            ranked_municipalities=len(ranking), figures=[path.name for path in figures]), VALIDATION)
        print('Pipeline C: PASS')
    except Exception as error:
        save_json(dict(status='FAIL', reason=str(error)), VALIDATION)
        raise


if __name__ == '__main__':
    main()

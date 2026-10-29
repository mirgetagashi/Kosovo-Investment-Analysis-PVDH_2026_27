import importlib
from phase_1.pipeline_c import VALIDATION, SEED, BOOTSTRAP_ITERATIONS, load_input, save_json


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
        figures = stage('06_visualization').main()
        national = classified['national'].query("window == 'reported'")
        save_json(dict(status='PASS', input_records=len(data), input_sha256=status['dataset_sha256'],
            selected_features=selection.loc[selection.selected, 'measure'].tolist(),
            dropped_features=selection.loc[~selection.selected, 'measure'].tolist(),
            pca_rows=len(scores), pca_explained_variance=explained.set_index('component').explained_variance_ratio.round(4).to_dict(),
            pc2_separates_import_from_domestic=bool(contrast.set_index('component').loc['PC2', 'separates_import_from_domestic']),
            national_import_share=national.set_index('year').import_share.round(4).to_dict(),
            bootstrap=dict(iterations=BOOTSTRAP_ITERATIONS, seed=SEED,
                           stability=intervals.loc[intervals.sector_eligible].stability.value_counts().to_dict()),
            ranked_municipalities=len(ranking), figures=[path.name for path in figures]), VALIDATION)
        print('Pipeline C: PASS')
    except Exception as error:
        save_json(dict(status='FAIL', reason=str(error)), VALIDATION)
        raise


if __name__ == '__main__':
    main()

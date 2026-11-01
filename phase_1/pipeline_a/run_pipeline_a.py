import importlib
from src.io_utils import digest
from phase_1.pipeline_a import TABLES, FIGURES, VALIDATION, load_input, save_json, mark_running

STAGES = ['01_aggregation', '02_feature_creation', '03_transformation', '04_discretization_binarization',
          '05_sampling_sensitivity', '06_visualization']
EXPECTED_TABLES = ['cv_by_level', 'monthly_features', 'seasonality', 'annual_features', 'trend_slope', 'recovery',
                   'monthly_transformed', 'skewness', 'seasonal_component', 'annual_index', 'monthly_discretized',
                   'annual_discretized', 'growth_class_comparison', 'growth_class_largest_share', 'bootstrap_ci', 'sample_sensitivity']


def main():
    mark_running('start')
    try:
        data, status = load_input()
        for stage in STAGES:
            importlib.import_module(f'phase_1.pipeline_a.{stage}').main()
        missing = [name for name in EXPECTED_TABLES if not (TABLES / f'{name}.csv').exists()]
        figures = sorted(path.name for path in FIGURES.glob('*.png'))
        if missing or len(figures) < 9:
            raise FileNotFoundError(f'Missing outputs: tables {missing}, figures {figures}')
        save_json(dict(status='PASS', stages=STAGES, input_records=len(data), input_sha256=status['dataset_sha256'],
                       tables={name: digest(TABLES / f'{name}.csv') for name in EXPECTED_TABLES}, figures=figures), VALIDATION)
        print(f'Pipeline A: {len(STAGES)} stages, {len(EXPECTED_TABLES)} tables, {len(figures)} figures; status PASS.')
    except Exception as error:
        save_json(dict(status='FAIL', reason=str(error)), VALIDATION)
        raise


if __name__ == '__main__':
    main()

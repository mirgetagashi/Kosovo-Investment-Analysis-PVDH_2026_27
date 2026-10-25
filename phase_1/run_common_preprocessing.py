import importlib
from src.io_utils import STATUS, save_json


def main():
    save_json(dict(mandatory_status='RUNNING'),STATUS)
    try:
        annual,layouts=importlib.import_module('phase_1.01_data_inspection').main()
        importlib.import_module('phase_1.02_data_quality').main(annual)
        importlib.import_module('phase_1.03_data_integration').main(annual,layouts)
        importlib.import_module('phase_1.04_data_cleaning').main()
        importlib.import_module('phase_1.05_missing_values').main()
        importlib.import_module('phase_1.06_common_validation').main()
    except Exception as error:
        save_json(dict(mandatory_status='FAIL',reason=str(error)),STATUS)
        raise


if __name__=='__main__':
    main()

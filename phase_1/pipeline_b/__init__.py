import json
from pathlib import Path
import pandas as pd
from src.io_utils import ROOT, RAW, COMMON, STATUS, MANIFEST, digest, load_dataset, save_frame, save_json
from src.schema import MONEY, PROVENANCE_KEY, norm
from src.validation import check_time

WINDOWS = ('reported', 'jan_jun')
GROUP = ['window', 'year', 'measure']
EXCLUDED_SECTORS = {'mungon aktiviteti', 'person fizik'}
MEASURE_LABELS = {
    MONEY[0]: 'Purchases/imports without VAT',
    MONEY[1]: 'Purchases/imports: non-deductible VAT',
    MONEY[2]: 'Investment imports: 18%',
    MONEY[3]: 'Investment imports: 8%',
    MONEY[4]: 'Domestic investment purchases: 18%',
    MONEY[5]: 'Domestic investment purchases: 8%',
}
TABLES = ROOT / 'outputs/tables/pipeline_b'
FIGURES = ROOT / 'outputs/figures/pipeline_b'
REFERENCE = ROOT / 'data/reference'
INTERMEDIATE = ROOT / 'data/intermediate/pipeline_b'
VALIDATION = ROOT / 'outputs/pipeline_b_validation.json'
BASE_TABLES = ['municipal_base', 'sectors', 'municipal_sector_base', 'sector_coverage']


def load_input():
    if not COMMON.exists() or not STATUS.exists():
        raise FileNotFoundError('First run: python -m phase_1.run_common_preprocessing')
    status = json.loads(STATUS.read_text(encoding='utf-8'))
    if status.get('mandatory_status') != 'PASS' or status.get('dataset_sha256') != digest(COMMON):
        raise ValueError('Common dataset must have a matching PASS validation status and hash')
    data = load_dataset(COMMON)
    check_time(data)
    if data.duplicated(PROVENANCE_KEY).any() or status['records'] != len(data):
        raise ValueError('Common source provenance/count mismatch')
    layouts = json.loads(MANIFEST.read_text(encoding='utf-8'))
    if any(digest(RAW / item['file']) != item['raw_sha256'] for item in layouts):
        raise ValueError('Raw Excel hash differs from Common Preprocessing manifest')
    if not data.primary_sector.map(lambda value: norm(value) == 'mungon aktiviteti').equals(data.sector_unclassified):
        raise ValueError('Common sector-unclassified flag mismatch')
    return data, status


def read_table(name, directory=TABLES):
    path = directory / f'{name}.csv'
    if not path.exists():
        raise FileNotFoundError(f'Missing stage output {path.name}; run the preceding Pipeline B stages')
    data = pd.read_csv(path, float_precision='round_trip')
    for column in ['top_contributor', 'high_specialization']:
        if column in data:
            data[column] = data[column].astype('Int8')
    return data


def mark_running(stage):
    save_json(dict(status='RUNNING', current_stage=stage), VALIDATION)

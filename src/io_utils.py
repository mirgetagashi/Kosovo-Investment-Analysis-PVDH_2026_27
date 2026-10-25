from pathlib import Path
import hashlib
import json
import pandas as pd
from src.schema import DTYPES

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw'
INTERMEDIATE = ROOT / 'data/intermediate'
PROCESSED = ROOT / 'data/processed'
TABLES = ROOT / 'outputs/tables'
FIGURES = ROOT / 'outputs/figures'
INTEGRATED = INTERMEDIATE / '03_integrated.csv'
CLEANED = INTERMEDIATE / '04_cleaned.csv'
COMMON = PROCESSED / 'common_dataset.csv'
MANIFEST = INTERMEDIATE / 'source_manifest.json'
STATUS = PROCESSED / 'common_validation.json'
BASELINE = ROOT / 'data/baseline/common_preprocessing_before.zip'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_frame(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    frame.to_csv(temporary, index=False, encoding='utf-8-sig', lineterminator='\n')
    temporary.replace(path)

def save_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + '\n', encoding='utf-8')

def load_dataset(path):
    header = pd.read_csv(path, nrows=0).columns
    dtypes = {c:t for c,t in DTYPES.items() if c in header and t not in ('float64', 'Int8', 'Int16', 'Int64')}
    frame = pd.read_csv(path, dtype=dtypes, keep_default_na=False, float_precision='round_trip')
    for c in frame:
        if c in DTYPES and DTYPES[c] in ('float64', 'Int8', 'Int16', 'Int64'):
            frame[c] = pd.to_numeric(frame[c].replace('', pd.NA), errors='raise').astype(DTYPES[c])
    return frame


def save_table(frame, name):
    save_frame(frame, TABLES / f'{name}.csv')

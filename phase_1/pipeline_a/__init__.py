import numpy as np
import pandas as pd
from src.io_utils import ROOT, save_frame, save_json
from src.schema import MONEY, norm
from phase_1.pipeline_b import load_input, MEASURE_LABELS, WINDOWS, EXCLUDED_SECTORS

LEVELS = ('monthly', 'quarterly', 'annual')
GROUP = ['window', 'measure']
BASE_YEAR = 2019
COVID_START, COVID_END = (2020, 3), (2021, 6)
LOCKDOWN_START, LOCKDOWN_END = (2020, 3), (2020, 5)
SEED = 42
GROWTH_BINS = [-float('inf'), -20, -5, 5, 20, float('inf')]
GROWTH_LABELS = ['strong_decline', 'decline', 'stable', 'growth', 'strong_growth']
INTERMEDIATE = ROOT / 'data/intermediate/pipeline_a'
TABLES = ROOT / 'outputs/tables/pipeline_a'
FIGURES = ROOT / 'outputs/figures/pipeline_a'
VALIDATION = ROOT / 'outputs/pipeline_a_validation.json'
BASE_TABLES = ['monthly', 'quarterly', 'annual', 'sector_monthly']


def widen(values):
    """Source columns arrive as Int8/Int16, which overflow in year*100; scalars stay scalars."""
    return int(values) if np.isscalar(values) else values.astype('int64')


def period_index(year, month):
    """Months elapsed since January of the base year; the time axis for slopes and windows."""
    return (widen(year) - BASE_YEAR) * 12 + (widen(month) - 1)


def in_period(year, month, start, end):
    stamp = widen(year) * 100 + widen(month)
    return stamp.between(start[0] * 100 + start[1], end[0] * 100 + end[1])


def read_table(name, directory=TABLES):
    path = directory / f'{name}.csv'
    if not path.exists():
        raise FileNotFoundError(f'Missing stage output {path.name}; run the preceding Pipeline A stages')
    data = pd.read_csv(path, float_precision='round_trip')
    for column in ['growth_positive', 'above_trend', 'mom_positive', 'is_covid', 'is_lockdown', 'is_full_year', 'recovered']:
        if column in data:
            data[column] = data[column].map({True: 1, False: 0, 'True': 1, 'False': 0, 1: 1, 0: 0}).astype('Int8')
    return data


def mark_running(stage):
    save_json(dict(status='RUNNING', current_stage=stage), VALIDATION)

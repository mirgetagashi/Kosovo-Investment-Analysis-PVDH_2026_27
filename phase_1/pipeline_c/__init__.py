import numpy as np
import pandas as pd
from src.io_utils import ROOT, save_frame, save_json
from src.schema import MONEY, norm
from phase_1.pipeline_b import load_input, MEASURE_LABELS, WINDOWS, EXCLUDED_SECTORS

# Pipeline C: how much do Kosovo business investments depend on imports,
# and which sectors invest most in the domestic economy?
#
# The six ATK amounts split into three origin groups. The first two source columns are
# "Blerjet dhe importet investive ..." (purchases AND imports), so their origin cannot be
# separated. They are kept as a separate MIXED group instead of being counted as imports.
IMPORT = MONEY[2:4]      # Importet investive me normë 18% / 8%
DOMESTIC = MONEY[4:6]    # Blerjet investive vendore me normë 18% / 8%
MIXED = MONEY[0:2]       # Blerjet dhe importet investive pa TVSH / me TVSH jo të zbritshme
ORIGIN = dict(import_total=IMPORT, domestic_total=DOMESTIC, mixed_total=MIXED)
ORIGIN_OF = {**dict.fromkeys(IMPORT, 'import'), **dict.fromkeys(DOMESTIC, 'domestic'), **dict.fromkeys(MIXED, 'mixed')}
TOTALS = ['import_total', 'domestic_total', 'mixed_total', 'total_investment']
SHARES = ['import_share', 'import_share_lower', 'import_share_upper', 'mixed_share']
SHORT_LABELS = {
    MONEY[0]: 'Mixed: no VAT',
    MONEY[1]: 'Mixed: non-deductible VAT',
    MONEY[2]: 'Imports 18%',
    MONEY[3]: 'Imports 8%',
    MONEY[4]: 'Domestic 18%',
    MONEY[5]: 'Domestic 8%',
}

DOMINANCE_THRESHOLD = .5
CLASS_BINS = [0, 1 / 3, 2 / 3, 1]
CLASS_LABELS = ['mostly_domestic', 'mixed', 'mostly_import']
RELATIVE_LABELS = ['lower_import_reliance', 'middle_import_reliance', 'higher_import_reliance']
# Feature selection: a binary "nonzero" indicator with p > 95% or p < 5% has variance below .95*.05
NONZERO_VARIANCE_THRESHOLD = .95 * .05
MIN_VALUE_SHARE = .01
MIN_MUNICIPAL_VALUE = 1_000_000   # EUR of identified-origin investment before a municipality is ranked
RANKING_SIZE = 10
BOOTSTRAP_ITERATIONS = 1000
CONFIDENCE = .95
SEED = 42

INTERMEDIATE = ROOT / 'data/intermediate/pipeline_c'
TABLES = ROOT / 'outputs/tables/pipeline_c'
FIGURES = ROOT / 'outputs/figures/pipeline_c'
VALIDATION = ROOT / 'outputs/pipeline_c_validation.json'
BASE_TABLES = ['national', 'sector_year', 'municipal_year', 'sector_pooled', 'municipal_pooled']


def eligible_sector(label):
    return norm(label) not in EXCLUDED_SECTORS


def origin_features(data):
    """Row-level origin totals; every source amount belongs to exactly one origin group."""
    data = data.copy()
    for name, columns in ORIGIN.items():
        data[name] = data[columns].sum(axis=1)
    data['total_investment'] = data[MONEY].sum(axis=1)
    return data


def origin_shares(table):
    """Shares are defined only on aggregates; a negative component or empty denominator leaves them undefined.

    import_share       = import / (import + domestic)          identified-origin share (main measure)
    import_share_lower = import / total                         if every mixed euro were domestic
    import_share_upper = (import + mixed) / total               if every mixed euro were imported
    import_share always lies between the lower and upper bound.
    """
    table = table.copy()
    negative = table[['import_total', 'domestic_total', 'mixed_total']].lt(0).any(axis=1)
    identified = table.import_total + table.domestic_total
    total = table.total_investment
    table['identified_total'] = identified
    table['share_status'] = np.select([negative, identified.le(0)], ['negative_component', 'zero_identified'], 'valid')
    bounded = ~negative & total.gt(0)
    table['import_share'] = table.import_total.div(identified.where(table.share_status.eq('valid')))
    table['import_share_lower'] = table.import_total.div(total.where(bounded))
    table['import_share_upper'] = (table.import_total + table.mixed_total).div(total.where(bounded))
    table['mixed_share'] = table.mixed_total.div(total.where(bounded))
    return table


def short_sector(label, width=34):
    text = str(label).split(';')[0].strip()
    return text if len(text) <= width else text[:width - 1] + '…'


def read_table(name, directory=TABLES):
    path = directory / f'{name}.csv'
    if not path.exists():
        raise FileNotFoundError(f'Missing stage output {path.name}; run the preceding Pipeline C stages')
    data = pd.read_csv(path, float_precision='round_trip')
    for column in ['import_dominant', 'import_dominant_upper']:
        if column in data:
            data[column] = data[column].astype('Int8')
    for column in ['partial_year', 'sector_eligible', 'class_robust', 'selected', 'dominant_stable', 'class_stable']:
        if column in data and data[column].dtype == object:
            data[column] = data[column].map({'True': True, 'False': False, True: True, False: False})
    return data


def mark_running(stage):
    save_json(dict(status='RUNNING', current_stage=stage), VALIDATION)

import unicodedata
import pandas as pd
SOURCE_COLUMNS = [
    'year', 'month', 'primary_sector', 'registration_status', 'municipality',
    'taxpayer_count', 'investment_purchases_imports_without_vat',
    'investment_purchases_imports_nondeductible_vat', 'investment_imports_18pct',
    'investment_imports_8pct', 'domestic_investment_purchases_18pct',
    'domestic_investment_purchases_8pct',
]
CATEGORIES = SOURCE_COLUMNS[2:5]
MONEY = SOURCE_COLUMNS[6:]
NUMERIC = SOURCE_COLUMNS[:2] + SOURCE_COLUMNS[5:]
BUSINESS_KEY = SOURCE_COLUMNS[:5]
PROVENANCE = ['source_file', 'source_year', 'source_sheet', 'source_excel_row']
PROVENANCE_KEY = ['source_file', 'source_sheet', 'source_excel_row']
RAW_CATEGORIES = [f'{c}_raw' for c in CATEGORIES]
FLAGS = ['review_negative_monetary', 'review_statistical_extreme',
         'review_category_domain_unverified', 'review_candidate_key_duplicate', 'review_missing',
         'sector_unclassified']
INTEGRATED_COLUMNS = SOURCE_COLUMNS + PROVENANCE + RAW_CATEGORIES
COMMON_COLUMNS = INTEGRATED_COLUMNS + FLAGS
DTYPES = dict.fromkeys(CATEGORIES + RAW_CATEGORIES + ['source_file', 'source_sheet'], 'string')
DTYPES.update(year='Int16', month='Int8', taxpayer_count='Int64', source_year='Int16', source_excel_row='Int64')
DTYPES.update(dict.fromkeys(MONEY, 'float64'))
DTYPES.update(dict.fromkeys(FLAGS, 'bool'))


YEARS_EXPECTED = tuple(range(2019, 2027))
EXPECTED_FILES = tuple(f'Investimet-{year}.xlsx' for year in YEARS_EXPECTED)
EXPECTED_MONTHS = {year:set(range(1, 7 if year == 2026 else 13)) for year in YEARS_EXPECTED}
PLACEHOLDERS = {'na', 'n/a', 'null', 'none', 'unknown', '-', 'e panjohur'}
COLUMN_PLACEHOLDERS = {'primary_sector': {'mungon aktiviteti'}}



def norm(value):
    return ''.join(c for c in unicodedata.normalize('NFKD', str(value).strip().casefold())
                   if not unicodedata.combining(c))

def standard_name(label):
    s = norm(label)
    names = SOURCE_COLUMNS
    tokens = [('year','year'), ('month','month'), ('sector','primary_sector'),
              ('registration status','registration_status'), ('municipality','municipality'),
              ('taxpayer','taxpayer_count'), ('without vat',names[6]), ('deductible vat',names[7])]
    for token, name in tokens:
        if token in s:
            return name
    if '18%' in s and ('domestic' in s or 'vendore' in s): return names[10]
    if '8%' in s and ('domestic' in s or 'vendore' in s): return names[11]
    if '18%' in s: return names[8]
    if '8%' in s: return names[9]
    return 'unmapped_' + s.replace('\n', '_').replace(' ', '_')


def missing_masks(series):
    """Detect missing information without replacing source cells or labels."""
    null = series.isna()
    empty = series.map(lambda value: isinstance(value, str) and not value.strip())
    placeholder = series.map(lambda value: isinstance(value, str) and norm(value) in PLACEHOLDERS)
    labels = COLUMN_PLACEHOLDERS.get(series.name, set())
    unavailable = series.map(lambda value: isinstance(value, str) and norm(value) in labels)
    return null, empty, placeholder, unavailable


def missing_mask(series):
    null, empty, placeholder, unavailable = missing_masks(series)
    return null | empty | placeholder | unavailable


def category_interpretation(column, value):
    """Interpret observed labels; an unfamiliar category is not automatically missing."""
    if pd.isna(value) or not str(value).strip():
        return 'literal missing', 'Empty meaningful attribute; retain and review'
    label = norm(value)
    if label in COLUMN_PLACEHOLDERS.get(column, set()):
        return 'domain-specific unavailable', 'Sector unavailable/unclassified'
    if label in PLACEHOLDERS:
        return 'generic placeholder', 'Missing-information placeholder'
    if column == 'primary_sector' and label == 'person fizik':
        return 'ambiguous category', 'Person/legal-type label in sector field; codebook needed'
    if column == 'registration_status' and label == 'tjeter':
        return 'catch-all category', 'Other legal form; not evidence of missing information'
    return 'observed category', 'No missing meaning identified; domain still unverified'

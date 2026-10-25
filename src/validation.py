import pandas as pd
from src.schema import (SOURCE_COLUMNS, NUMERIC, CATEGORIES, PROVENANCE,
                        PROVENANCE_KEY, YEARS_EXPECTED, EXPECTED_MONTHS)


def check_records(before, after, trim_categories=False):
    required = set(SOURCE_COLUMNS + PROVENANCE)
    if not required.issubset(before) or not required.issubset(after):
        raise ValueError('Missing source attributes or provenance')
    if len(before) != len(after):
        raise ValueError('Record count changed')
    if before.duplicated(PROVENANCE_KEY).any() or after.duplicated(PROVENANCE_KEY).any():
        raise ValueError('Duplicated source provenance')
    old = before.sort_values(PROVENANCE_KEY).reset_index(drop=True)
    new = after.sort_values(PROVENANCE_KEY).reset_index(drop=True)
    for column in PROVENANCE:
        if not old[column].eq(new[column]).fillna(False).all():
            raise ValueError(f'Source provenance changed: {column}')
    for column in NUMERIC:
        left = pd.to_numeric(old[column], errors='raise')
        right = pd.to_numeric(new[column], errors='raise')
        equal = left.eq(right) | (left.isna() & right.isna())
        if not equal.fillna(False).all():
            raise ValueError(f'Numeric values changed: {column}')
    for column in CATEGORIES:
        expected = old[column]
        if trim_categories:
            expected = expected.map(lambda value: value.strip() if isinstance(value, str) else value)
        equal = expected.eq(new[column]) | (expected.isna() & new[column].isna())
        if not equal.fillna(False).all():
            raise ValueError(f'Unexpected categorical correction: {column}')
        if f'{column}_raw' in new:
            original = old[f'{column}_raw'] if f'{column}_raw' in old else old[column]
            equal = original.eq(new[f'{column}_raw']) | (original.isna() & new[f'{column}_raw'].isna())
            if not equal.fillna(False).all():
                raise ValueError(f'Original category label lost: {column}')


def check_time(data):
    year = pd.to_numeric(data.year, errors='raise')
    month = pd.to_numeric(data.month, errors='raise')
    if year.isna().any() or not year.eq(data.source_year).all():
        raise ValueError('Source year is absent or disagrees with the workbook year')
    if not year.isin(YEARS_EXPECTED).all() or (year % 1 != 0).any():
        raise ValueError('Invalid year')
    if month.isna().any() or not month.between(1,12).all() or (month % 1 != 0).any():
        raise ValueError('Invalid month')
    coverage = data.groupby('source_year').month.agg(set).to_dict()
    if coverage != EXPECTED_MONTHS:
        raise ValueError(f'Unexpected reporting coverage: {coverage}; expected {EXPECTED_MONTHS}')

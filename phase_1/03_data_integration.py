import numpy as np
import pandas as pd
from src.io_utils import RAW, MANIFEST, INTEGRATED, save_frame, save_json, save_table
from src.excel_reader import read_annual_files
from src.schema import (SOURCE_COLUMNS, PROVENANCE, NUMERIC, CATEGORIES, DTYPES,
                        INTEGRATED_COLUMNS, YEARS_EXPECTED)
from src.validation import check_records, check_time


def normalize_numeric(series, column):
    values = pd.to_numeric(series.mask(series.map(lambda v:isinstance(v,str) and not v.strip())),errors='raise')
    if not np.isfinite(values.dropna()).all():
        raise ValueError(f'Nonfinite {column}')
    if DTYPES[column].startswith('Int') and values.dropna().mod(1).ne(0).any():
        raise ValueError(f'Nonintegral {column}')
    if column in ['year','source_year'] and not values.isin(YEARS_EXPECTED).all():
        raise ValueError(f'Invalid or missing {column}')
    if column=='month' and (values.isna().any() or not values.between(1,12).all()):
        raise ValueError('Invalid or missing month')
    if column=='taxpayer_count' and values.dropna().lt(0).any():
        raise ValueError('Negative taxpayer count requires source review')
    return values.astype(DTYPES[column])


def integrate_sources(annual_frames):
    for frame in annual_frames:
        if set(frame.columns)!=set(SOURCE_COLUMNS+PROVENANCE):
            raise ValueError('Source schema does not match the twelve known attributes plus provenance')
    source = pd.concat(annual_frames,ignore_index=True)
    data = source.copy()
    for column in NUMERIC+['source_year','source_excel_row']:
        data[column] = normalize_numeric(data[column],column)
    for column in CATEGORIES:
        if not data[column].dropna().map(lambda value:isinstance(value,str)).all():
            raise ValueError(f'Expected text labels: {column}')
        data[f'{column}_raw'] = data[column].astype('string')
    data = data[INTEGRATED_COLUMNS].astype({c:DTYPES[c] for c in INTEGRATED_COLUMNS})
    check_time(data)
    check_records(source,data)
    return data


def main(annual_frames=None, layouts=None):
    if annual_frames is None:
        annual_frames, layouts = read_annual_files(RAW)
    data = integrate_sources(annual_frames)
    save_frame(data,INTEGRATED)
    save_json(layouts,MANIFEST)
    summary = data.groupby('source_year').agg(records=('year','size'),files=('source_file','nunique'),
        months=('month','nunique'),source_sheets=('source_sheet','nunique')).reset_index()
    save_table(summary,'03_integration_summary')
    print(summary.to_string(index=False))
    print(f'03: {len(annual_frames)} source sheets -> {len(data):,} integrated records; raw labels retained.')
    return data


if __name__=='__main__':
    main()

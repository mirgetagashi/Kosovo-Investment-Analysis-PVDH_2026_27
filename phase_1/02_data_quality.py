import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.io_utils import RAW, FIGURES, save_table
from src.excel_reader import read_annual_files
from src.schema import SOURCE_COLUMNS, CATEGORIES, NUMERIC, MONEY, BUSINESS_KEY, missing_masks, category_interpretation


def duplicate_masks(data):
    return data.duplicated(SOURCE_COLUMNS,keep=False), data.duplicated(BUSINESS_KEY,keep=False)


def assess_quality(annual_frames):
    rows, categories = [], []
    for data in annual_frames:
        check_time_for_source = data.year.eq(data.source_year).all()
        if not check_time_for_source:
            raise ValueError('Year in a source record disagrees with the file year')
        year = int(data.source_year.iloc[0])
        for column in SOURCE_COLUMNS:
            values = data[column]
            null, blank, placeholder, unavailable = missing_masks(values)
            missing = int((null|blank|placeholder|unavailable).sum())
            row = dict(year=year, source_file=data.source_file.iloc[0], attribute=column,
                pandas_dtype=str(values.dtype), distinct=int(values.nunique()),
                nulls=int(values.isna().sum()), blanks=int(blank.sum()), placeholders=int(placeholder.sum()),
                domain_unavailable=int(unavailable.sum()), missing=missing,
                missing_percent=100*missing/len(data),
                domain_unavailable_percent=100*unavailable.mean(),
                whitespace=0, zeros=0, negatives=0, parse_errors=0, nontext=0)
            if column in CATEGORIES:
                whitespace = values.map(lambda value:isinstance(value,str) and value!=value.strip())
                row['whitespace'] = int(whitespace.sum())
                row['nontext'] = int(values.dropna().map(lambda value:not isinstance(value,str)).sum())
                for value, frequency in values.value_counts(dropna=False).items():
                    interpretation, reason = category_interpretation(column, value)
                    categories.append(dict(year=year, attribute=column, value=value, records=int(frequency),
                        edge_whitespace=isinstance(value,str) and value!=value.strip(),
                        interpretation=interpretation, review_reason=reason,
                        example_file=data.source_file.iloc[0],
                        example_excel_row=int(data.loc[values.isna() if pd.isna(value) else values.eq(value),'source_excel_row'].iloc[0]),
                        domain_status='UNRESOLVED: no authoritative codebook'))
            else:
                numbers = pd.to_numeric(values,errors='coerce')
                row.update(parse_errors=int((numbers.isna()&~(null|blank)).sum()),
                    nonfinite=int((~np.isfinite(numbers.dropna())).sum()),
                    zeros=int(numbers.eq(0).sum()), negatives=int(numbers.lt(0).sum()),
                    minimum=numbers.min(), maximum=numbers.max())
            rows.append(row)
        exact, key = duplicate_masks(data)
        print(f'{year}: records={len(data):,}; exact duplicate members={exact.sum()}; candidate-key members={key.sum()}')
    profile = pd.DataFrame(rows)
    return profile, pd.DataFrame(categories)


def main(annual_frames=None):
    if annual_frames is None:
        annual_frames, _ = read_annual_files(RAW)
    profile, categories = assess_quality(annual_frames)
    save_table(profile, '02_quality_summary')
    save_table(categories, '02_categories')
    if profile.parse_errors.sum() or profile.nonfinite.sum() or profile.nontext.sum():
        raise ValueError('Invalid numeric/text representation; inspect 02_quality_summary.csv')
    print('02: whitespace is a confirmed formatting issue; negatives need review; zeros are valid observations.')
    print('Categorical domains, currency and accounting meanings remain UNRESOLVED.')
    summary = profile.groupby('year')[['missing','whitespace','negatives']].sum()
    print(summary.to_string())
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    for ax, column, label in [(axes[0],'negatives','Negative monetary cells retained'),
                              (axes[1],'whitespace','Categorical cells with edge whitespace')]:
        ax.bar(summary.index.astype(str),summary[column],color='#397790')
        ax.set(title=label,xlabel='Source year (2026: Jan-Jun)',ylabel='Cells')
        ax.tick_params(axis='x',rotation=45)
    fig.tight_layout(); FIGURES.mkdir(parents=True,exist_ok=True)
    fig.savefig(FIGURES/'02_quality_findings.png',dpi=160); plt.close(fig)
    return profile


if __name__=='__main__':
    main()

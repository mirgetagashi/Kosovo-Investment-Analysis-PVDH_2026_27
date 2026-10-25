import pandas as pd
from src.io_utils import CLEANED, load_dataset, save_table
from src.schema import SOURCE_COLUMNS, YEARS_EXPECTED, EXPECTED_MONTHS, COLUMN_PLACEHOLDERS, missing_masks


def assess_missingness(data):
    rows=[]
    for column in SOURCE_COLUMNS:
        absent = column not in data
        if absent:
            null = blank = placeholder = unavailable = pd.Series(False,index=data.index)
        else:
            null, blank, placeholder, unavailable = missing_masks(data[column])
        missing = int((null|blank|placeholder|unavailable).sum())
        rows.append(dict(attribute=column,missing_attribute=absent,nulls=int(null.sum()),
            blanks=int(blank.sum()),placeholders=int(placeholder.sum()),
            domain_unavailable=int(unavailable.sum()),missing_cells=missing,
            missing_percent=100*missing/len(data) if len(data) else 0,
            encoded_missing_rule='Generic placeholders' + ('; sector: Mungon aktiviteti (activity unavailable)' if column in COLUMN_PLACEHOLDERS else ''),
            decision='Retain and review; no supported imputation' if absent or missing else 'No imputation necessary',
            imputed_cells=0))
    periods=[]
    for year in YEARS_EXPECTED:
        observed=set(data.loc[data.year==year,'month'])
        for month in range(1,13):
            status='observed' if month in observed else 'outside supplied Jan-Jun 2026 coverage' if year==2026 and month>6 else 'missing expected period'
            periods.append(dict(year=year,month=month,status=status,records_generated=0))
    return pd.DataFrame(rows),pd.DataFrame(periods)


def main():
    data = load_dataset(CLEANED)
    assessment, periods = assess_missingness(data)
    save_table(assessment,'05_missing_values')
    if assessment.missing_attribute.any():
        raise ValueError('Required attributes are missing')
    if periods.status.eq('missing expected period').any():
        raise ValueError('An expected reporting period is missing')
    print(assessment.to_string(index=False))
    print('05: nulls, blanks, generic placeholders and domain-specific unavailable values are counted separately.')
    print('Unavailable sectors are retained with their financial values; no justified sector imputation exists.')
    print('Zeros are observed values; structural applicability is unconfirmed.')
    print('July-December 2026 is outside supplied coverage; no zero-investment or future records are created.')
    return data


if __name__=='__main__':
    main()

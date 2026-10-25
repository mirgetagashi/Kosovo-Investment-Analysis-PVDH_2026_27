import pandas as pd
from src.io_utils import INTEGRATED, CLEANED, load_dataset, save_frame, save_table
from src.schema import SOURCE_COLUMNS, CATEGORIES, MONEY, BUSINESS_KEY, COMMON_COLUMNS, missing_mask, missing_masks
from src.validation import check_records


def clean_records(integrated):
    data = integrated.copy()
    changes = []
    for column in CATEGORIES:
        trimmed = data[column].astype('string').str.strip()
        changed = data[column].ne(trimmed).fillna(False)
        for (year,old,new),group in data.loc[changed].assign(new=trimmed[changed]).groupby(['source_year',column,'new']):
            changes.append(dict(rule='TRIM_EDGE_WHITESPACE',attribute=column,year=int(year),
                original_value=old,new_value=new,affected_records=len(group),
                reason='Non-semantic leading/trailing whitespace; original label preserved'))
        data[column] = trimmed

    data['review_negative_monetary'] = data[MONEY].lt(0).any(axis=1)
    data['review_statistical_extreme'] = False
    for _, group in data.groupby('year',dropna=False):
        q1 = group[MONEY].quantile(.25)
        q3 = group[MONEY].quantile(.75)
        flagged = group[MONEY].gt(q3+1.5*(q3-q1)).any(axis=1)
        data.loc[group.index,'review_statistical_extreme'] = flagged
    data['review_category_domain_unverified'] = True
    data['review_candidate_key_duplicate'] = data.duplicated(BUSINESS_KEY,keep=False)
    data['review_missing'] = data[SOURCE_COLUMNS].apply(missing_mask).any(axis=1)
    data['sector_unclassified'] = missing_masks(data.primary_sector)[3]
    columns=['rule','attribute','year','original_value','new_value','affected_records','reason']
    return data[COMMON_COLUMNS],pd.DataFrame(changes,columns=columns)


def main():
    before = load_dataset(INTEGRATED)
    data, audit = clean_records(before)
    check_records(before,data,trim_categories=True)
    repeated, _ = clean_records(data)
    if not data.equals(repeated):
        raise ValueError('Cleaning is not idempotent')
    save_frame(data,CLEANED)
    save_table(audit,'04_cleaning_audit')
    print(audit.to_string(index=False))
    print(f'04: {int(audit.affected_records.sum()):,} categorical cells trimmed; no rows or numbers removed/corrected.')
    return data


if __name__=='__main__':
    main()

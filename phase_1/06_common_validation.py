from io import BytesIO
import importlib
import json
import math
import zipfile
import openpyxl
import numpy as np
import pandas as pd
from src.io_utils import (RAW, MANIFEST, INTEGRATED, CLEANED, COMMON, STATUS, BASELINE,
                          load_dataset, save_frame, save_table, save_json, digest, TABLES)
from src.excel_reader import read_annual_files
from src.schema import (SOURCE_COLUMNS, MONEY, DTYPES, COMMON_COLUMNS, EXPECTED_FILES,
                        PROVENANCE_KEY, PROVENANCE, NUMERIC, CATEGORIES, PLACEHOLDERS,
                        standard_name, missing_mask, missing_masks)
from src.validation import check_records, check_time


def verify_original_workbooks(final, integrated, layouts):
    """Independent check of these eight supplied layouts, without the Excel parser.

    Read the physical cells directly. Select records by their reporting period,
    then separately prove the final summary using its nine sums. A changed or
    ambiguous layout fails this audit rather than silently dropping a row.
    """
    results=[]
    positions=[0,1,2,4,5,6,7,9,10,12,13,15]
    for name in EXPECTED_FILES:
        path=RAW/name; original_hash=digest(path); year=int(path.stem.rsplit('-',1)[1])
        wb=openpyxl.load_workbook(path,read_only=True,data_only=False)
        try:
            if len(wb.worksheets)!=1:
                raise ValueError(f'Independent audit: unexpected sheets in {name}')
            ws=wb.worksheets[0]; rows=list(ws.values)
            if ws.max_column!=16 or [standard_name(rows[5][j]) for j in positions]!=SOURCE_COLUMNS:
                raise ValueError(f'Independent audit: source header/layout changed in {name}')
            records=[]; indexes=[]; excluded=[]
            counts=dict(blank_rows=0,header_rows=0,metadata_rows=0,footer_rows=0)
            for i,row in enumerate(rows,1):
                if all(v is None for v in row):
                    counts['blank_rows']+=1
                elif i==6:
                    counts['header_rows']+=1
                elif i<6:
                    counts['metadata_rows']+=1
                else:
                    values=[row[j] for j in positions]
                    if values[0]==year and values[1] in range(1,13):
                        records.append(values); indexes.append(i)
                    else:
                        excluded.append((i,values))
            if len(excluded)!=1:
                raise ValueError(f'Independent audit: unexpected non-period rows in {name}')
            footer_row,footer=excluded[0]
            last_occupied=max(i for i,row in enumerate(rows,1) if any(v is not None for v in row))
            numeric=[SOURCE_COLUMNS.index(c) for c in NUMERIC]
            if (footer_row!=last_occupied or any(footer[k] not in (None,'') for k in [2,3,4])
                    or not all(math.isclose(float(footer[k]),math.fsum(float(r[k]) for r in records
                        if r[k] is not None and str(r[k]).strip()),rel_tol=1e-10,abs_tol=.02) for k in numeric)):
                raise ValueError(f'Independent audit: footer not proven in {name}')
            counts['footer_rows']=1
            raw=pd.DataFrame(records,columns=SOURCE_COLUMNS)
            raw['source_file']=name; raw['source_year']=year
            raw['source_sheet']=ws.title; raw['source_excel_row']=indexes
            check_records(raw,integrated.loc[integrated.source_file.eq(name)])
            check_records(raw,final.loc[final.source_file.eq(name)],trim_categories=True)
            cells=[v for record in records for v in record]
            nulls=sum(v is None for v in cells)
            blanks=sum(isinstance(v,str) and not v.strip() for v in cells)
            placeholders=sum(isinstance(v,str) and v.strip().casefold() in PLACEHOLDERS for v in cells)
            unclassified=[str(r[2]).strip().casefold()=='mungon aktiviteti' for r in records]
            actual=final.loc[final.source_file.eq(name)].sort_values('source_excel_row')
            if actual.sector_unclassified.tolist()!=unclassified:
                raise ValueError(f'Independent audit: sector flag differs from original C cells in {name}')
            layout=next(item for item in layouts if item['file']==name)
            expected_counts=dict(blank_rows=layout['row_counts'].get('blank',0),
                header_rows=layout['row_counts'].get('header',0),metadata_rows=layout['row_counts'].get('metadata',0),
                footer_rows=layout['row_counts'].get('summary',0))
            if counts!=expected_counts or len(records)!=layout['row_counts'].get('record',0):
                raise ValueError(f'Independent audit: row partition differs in {name}')
            results.append(dict(year=year,file=name,records=len(records),meaningful_attributes=12,
                **counts,footer_excel_row=footer_row,nulls=nulls,blanks=blanks,placeholders=placeholders,
                domain_unavailable=sum(unclassified),unclassified_percent=100*sum(unclassified)/len(records),
                person_fizik=sum(r[2]=='Person Fizik' for r in records),status='PASS'))
        finally:
            wb.close()
        if digest(path)!=original_hash:
            raise ValueError(f'Independent audit: raw bytes changed in {name}')
    return pd.DataFrame(results)


def compare_baseline(final, integrated, cleaned, audit, missing):
    if not BASELINE.exists():
        return pd.DataFrame([dict(comparison='Baseline availability',status='REVIEW',detail='No historical archive; current raw-source checks still apply')])
    rows=[]
    def record(name, condition, detail='Identical'):
        rows.append(dict(comparison=name,status='PASS' if condition else 'FAIL',detail=detail))
    with zipfile.ZipFile(BASELINE) as archive:
        summary=json.loads(archive.read('baseline_summary.json'))
        def dataset(name):
            return pd.read_csv(BytesIO(archive.read(name)),dtype=DTYPES,keep_default_na=False,float_precision='round_trip')
        old=dataset('data/processed/common_dataset.csv')
        record('Records per year',old.groupby('year').size().to_dict()==final.groupby('year').size().to_dict())
        record('Columns and dtypes; one documented indicator added',
               list(old.columns)+['sector_unclassified']==list(final.columns)
               and old.dtypes.equals(final[old.columns].dtypes)
               and str(final.sector_unclassified.dtype)=='bool')
        record('Year/month coverage',old.groupby(['year','month']).size().to_dict()==final.groupby(['year','month']).size().to_dict())
        record('Every monetary value',old[MONEY].equals(final[MONEY]))
        record('Each monetary total/year',old.groupby('year')[MONEY].sum().equals(final.groupby('year')[MONEY].sum()))
        record('Integrated dataset',dataset('data/intermediate/03_integrated.csv').equals(integrated))
        def corrected_flags(frame):
            frame=frame.copy()
            frame['review_missing']=frame[SOURCE_COLUMNS].apply(missing_mask).any(axis=1)
            frame['sector_unclassified']=missing_masks(frame.primary_sector)[3]
            return frame
        old_cleaned=dataset('data/intermediate/04_cleaned.csv')
        record('Cleaned dataset with corrected missing flag',corrected_flags(old_cleaned).equals(cleaned))
        record('Entire final dataset with corrected missing flag',corrected_flags(old).equals(final))
        changed=old.review_missing.ne(final.review_missing)
        sector_missing=old.primary_sector.eq('Mungon aktiviteti').astype(bool)
        record('Only documented semantic flag changes',changed.equals(sector_missing)
               and not old.loc[changed,'review_missing'].any() and final.loc[changed,'review_missing'].all(),
               f'{int(changed.sum()):,} review_missing flags False -> True; sector_unclassified added; all source values identical')
        old_audit=pd.read_csv(BytesIO(archive.read('outputs/tables/04_cleaning_audit.csv')),keep_default_na=False)
        columns=['attribute','year','original_value','new_value','affected_records']
        record('Cleaning corrections',old_audit[columns].equals(audit[columns]))
        old_missing=pd.read_csv(BytesIO(archive.read('outputs/tables/05_missing_strategy.csv')))
        before=old_missing[['actual_nulls','empty_cells','placeholder_strings','imputed_cells']].sum().tolist()
        after=missing[['nulls','blanks','placeholders','imputed_cells']].sum().tolist()
        domain=int(missing.domain_unavailable.sum())
        record('Missing-value findings with encoded sectors',before[0:2]==after[0:2]
               and before[3]==after[3]==0 and after[2]==before[2]
               and domain==int(sector_missing.sum()),
               f'Null/blank/generic/imputed unchanged: {after}; domain unavailable recognized: {domain}')
        record('Raw file bytes',all(digest(RAW/name)==sha for name,sha in summary['raw_hashes'].items()))
    return pd.DataFrame(rows)


def main():
    checks=[]
    save_json(dict(mandatory_status='RUNNING'),STATUS)
    try:
        annual_frames, layouts=read_annual_files(RAW)
        source=pd.concat(annual_frames,ignore_index=True)
        integrated=load_dataset(INTEGRATED)
        cleaned=load_dataset(CLEANED)
        saved_layouts=json.loads(MANIFEST.read_text(encoding='utf-8'))
        if layouts!=saved_layouts:
            raise ValueError('Raw layout/hash manifest changed since inspection/integration')
        checks.append(dict(check='Eight source files and unchanged raw hashes',status='PASS'))
        if list(cleaned.columns)!=COMMON_COLUMNS or any(str(cleaned[c].dtype)!=DTYPES[c] for c in COMMON_COLUMNS):
            raise ValueError('Final schema/dtypes differ from the expected common dataset')
        checks.append(dict(check='Common schema and dtypes',status='PASS'))
        check_time(cleaned)
        checks.append(dict(check='2019-2025 full months; 2026 Jan-Jun only',status='PASS'))
        check_records(source,integrated)
        checks.append(dict(check='Integration: source rows, provenance and all numeric values preserved',status='PASS'))
        check_records(integrated,cleaned,trim_categories=True)
        checks.append(dict(check='Cleaning: only allowed edge whitespace changed',status='PASS'))
        for layout in layouts:
            subset=cleaned[(cleaned.source_file==layout['file'])&(cleaned.source_sheet==layout['sheet'])]
            forbidden={r['excel_row'] for r in layout['nonrecord_rows']}
            if subset.source_excel_row.isin(forbidden).any() or len(subset)!=layout['row_counts'].get('record',0):
                raise ValueError('Header/footer included or source count changed')
        checks.append(dict(check='No headers, metadata or report totals included',status='PASS'))
        independent=verify_original_workbooks(cleaned,integrated,layouts)
        save_table(independent,'06_source_verification')
        checks.append(dict(check='Independent physical-cell audit: row partition, all values and unclassified flags',status='PASS'))
        clean_stage=importlib.import_module('phase_1.04_data_cleaning')
        expected,audit=clean_stage.clean_records(integrated)
        saved_audit=pd.read_csv(TABLES/'04_cleaning_audit.csv',keep_default_na=False)
        if not expected.equals(cleaned) or not saved_audit.equals(audit):
            raise ValueError('Cleaning audit or review flags do not match the declared rules')
        checks.append(dict(check='Cleaning audit, review flags and idempotence',status='PASS'))
        missing_stage=importlib.import_module('phase_1.05_missing_values')
        missing,_=missing_stage.assess_missingness(cleaned)
        saved_missing=pd.read_csv(TABLES/'05_missing_values.csv',keep_default_na=False)
        if saved_missing.to_csv(index=False)!=missing.to_csv(index=False) or missing.imputed_cells.sum()!=0:
            raise ValueError('Missingness decisions differ from the actual dataset')
        checks.append(dict(check='Missingness assessed; no unsupported imputation',status='PASS'))
        expected_missing=cleaned[SOURCE_COLUMNS].apply(missing_mask).any(axis=1)
        raw_encoded=source.primary_sector.eq('Mungon aktiviteti')
        encoded=cleaned.primary_sector.eq('Mungon aktiviteti')
        if (not expected_missing.equals(cleaned.review_missing)
                or not missing_masks(cleaned.primary_sector)[3].equals(cleaned.sector_unclassified)
                or int(raw_encoded.sum())!=int(encoded.sum())
                or not cleaned.loc[encoded,'primary_sector_raw'].eq('Mungon aktiviteti').all()):
            raise ValueError('Encoded missing sectors lost, relabelled or not flagged')
        quality=pd.read_csv(TABLES/'02_quality_summary.csv')
        for column in ['nulls','blanks','placeholders','domain_unavailable']:
            if not (int(independent[column].sum())==int(quality[column].sum())==int(missing[column].sum())):
                raise ValueError(f'Independent/raw/quality/missingness counts disagree: {column}')
        checks.append(dict(check=f'Encoded missing sectors retained and flagged: {int(encoded.sum()):,} ({100*encoded.mean():.4f}%)',status='PASS'))

        before=source.groupby('source_year')[MONEY].sum()
        after=cleaned.groupby('source_year')[MONEY].sum()
        reconciliation=[]
        for year in before.index:
            for measure in MONEY:
                difference=after.loc[year,measure]-before.loc[year,measure]
                reconciliation.append(dict(year=year,measure=measure,source_total=before.loc[year,measure],
                    final_total=after.loc[year,measure],difference=difference))
        reconciliation=pd.DataFrame(reconciliation)
        save_table(reconciliation,'06_reconciliation')
        if not np.allclose(before.to_numpy(),after.to_numpy(),rtol=1e-10,atol=.02):
            raise ValueError('An individual measure/year monetary sum changed')
        checks.append(dict(check='48 individual financial measure/year sums preserved',status='PASS'))

        comparison=compare_baseline(cleaned,integrated,cleaned,audit,missing)
        save_table(comparison,'06_baseline_comparison')
        if comparison.status.eq('FAIL').any():
            raise ValueError('Unexplained difference from the saved baseline')
        checks.append(dict(check='Saved baseline: records, schema, values, cleaning and missingness',
                           status='PASS' if comparison.status.eq('PASS').all() else 'REVIEW'))
        pending=COMMON.with_name('common_dataset.pending.csv')
        save_frame(cleaned,pending)
        if not load_dataset(pending).equals(cleaned):
            raise ValueError('Final CSV values or dtypes did not round-trip exactly')
        pending.replace(COMMON)
        checks.append(dict(check='Final dataset CSV round-trip',status='PASS'))
        for topic in ['Category domains/codebooks','Currency, units and VAT-category overlap',
                      'Taxpayer counting and candidate aggregate grain','Negative adjustments and statistical extremes',
                      'Source revisions, applicability and suppression',
                      'Unavailable sectors: retain unclassified bucket and disclose analysis denominator']:
            checks.append(dict(check=topic,status='REVIEW/UNRESOLVED'))
        save_table(pd.DataFrame(checks),'06_validation')
        save_json(dict(mandatory_status='PASS',domain_status='REVIEW/UNRESOLVED',records=len(cleaned),
            dataset_sha256=digest(COMMON),checks=pd.DataFrame(checks).status.value_counts().to_dict()),STATUS)
        print(pd.DataFrame(checks).to_string(index=False))
        print('Records per stage/year (same counts):')
        print(cleaned.groupby('year').size().to_string())
        print(f'06: {len(cleaned):,} records validated; domain questions remain unresolved.')
        return cleaned
    except Exception as error:
        checks.append(dict(check=str(error),status='FAIL'))
        save_table(pd.DataFrame(checks),'06_validation')
        save_json(dict(mandatory_status='FAIL',reason=str(error)),STATUS)
        raise


if __name__=='__main__':
    main()

from collections import Counter
from pathlib import Path
import json
import math
import openpyxl
import pandas as pd
from src.schema import SOURCE_COLUMNS as NAMES, PROVENANCE, EXPECTED_FILES
from src.schema import norm, standard_name

def inspect_workbook(path):
    wb = openpyxl.load_workbook(path, read_only=True, data_only=False)
    sheets, frames, audits = [], [], []
    for ws in wb:
        rows = list(ws.values)
        occupied = [(i+1,r) for i,r in enumerate(rows) if any(v is not None for v in r)]
        headers = [i for i,r in occupied if any('year' in norm(v) for v in r if v is not None)
                   and any('month' in norm(v) for v in r if v is not None)]
        h = headers[0] if headers else None
        labels = [(j,str(v)) for j,v in enumerate(rows[h-1]) if v is not None] if h else []
        columns = [standard_name(v) for _,v in labels]
        records, indexes, kinds = [], [], Counter()
        footer_evidence=[]
        for i,row in enumerate(rows,1):
            present = [v for v in row if v is not None]
            if not present: kind = 'blank'
            elif i in headers: kind = 'header'
            elif h and i < h: kind = 'metadata'
            elif h and 'year' in columns and norm(row[labels[columns.index('year')][0]]) in {'total','totali','gjithsej','ukupno','grand total'}: kind = 'summary'
            elif not h: kind = 'unclassified'
            else:
                candidate=[row[j] for j,_ in labels]
                numeric_positions=[k for k,c in enumerate(columns) if c in NAMES[:2]+NAMES[5:]]
                categorical_positions=[k for k,c in enumerate(columns) if c in NAMES[2:5]]
                is_footer=bool(records and i==occupied[-1][0] and len(numeric_positions)==9
                    and len(categorical_positions)==3
                    and all(candidate[k] in (None,'') for k in categorical_positions))
                evidence={}
                if is_footer:
                    for k in numeric_positions:
                        try:
                            total=math.fsum(float(r[k]) for r in records if r[k] is not None
                                            and not (isinstance(r[k],str) and not r[k].strip()))
                            agrees=math.isclose(float(candidate[k]),total,rel_tol=1e-10,abs_tol=.02)
                        except (TypeError,ValueError,OverflowError):
                            agrees=False; total=None
                        evidence[columns[k]]={'footer':candidate[k],'record_sum':total,'agrees':agrees}
                    is_footer=all(v['agrees'] for v in evidence.values())
                if is_footer:
                    kind='summary'
                    footer_evidence.append({'excel_row':i,'method':'unlabelled final row; all numeric sums reconcile; categorical dimensions blank','checks':evidence})
                else:
                    kind = 'record'
                    records.append(candidate); indexes.append(i)
            kinds[kind] += 1
            if kind not in {'blank', 'record'}:
                audits.append({'file':path.name,'sheet':ws.title,'excel_row':i,'kind':kind,
                    'values':json.dumps(list(row),ensure_ascii=False,default=str) if kind!='record' else ''})
        df = pd.DataFrame(records,columns=columns,index=pd.Index(indexes,name='excel_row'))
        frames.append((ws.title,df,labels))
        sheets.append({'file':path.name,'sheet':ws.title,'physical_rows':ws.max_row,'physical_columns':ws.max_column,
            'header_row':h,'header_candidates':headers,'row_counts':dict(kinds),'footer_evidence':footer_evidence,
            'trailing_blank_rows':ws.max_row-occupied[-1][0] if occupied else ws.max_row,
            'columns':[{'excel_column':openpyxl.utils.get_column_letter(j+1),'original':v,'standardized':standard_name(v)} for j,v in labels],
            'empty_columns':[openpyxl.utils.get_column_letter(j+1) for j in range(ws.max_column) if all(r[j] is None for r in rows)],
            'separator_columns':[openpyxl.utils.get_column_letter(j+1) for j in range(ws.max_column)
                if h and rows[h-1][j] is None and all(r[j] is None for r in rows[h:])],
            'metadata':[{'row':i,'values':list(r)} for i,r in occupied if h and i<h],
            'unlabelled_nonempty_columns':[openpyxl.utils.get_column_letter(j+1) for j in range(ws.max_column)
                if h and rows[h-1][j] is None and any(r[j] is not None for r in rows[h:])]})
    wb.close()
    return sheets,frames,audits


def read_annual_files(raw_directory):
    from src.io_utils import digest
    paths = sorted(Path(raw_directory).glob('*.xlsx'))
    if {p.name for p in paths} != set(EXPECTED_FILES):
        raise ValueError('Expected exactly the eight Investimet-2019.xlsx ... Investimet-2026.xlsx files')
    annual_frames, layouts = [], []
    for path in paths:
        year = int(path.stem.rsplit('-', 1)[1])
        original_hash = digest(path)
        details, frames, audit = inspect_workbook(path)
        for info, (sheet, frame, _) in zip(details, frames):
            info.update(year=year, raw_sha256=original_hash,
                        nonrecord_rows=[row for row in audit if row['sheet'] == sheet])
            layouts.append(info)
            if not info['header_row']:
                if info['row_counts'].get('unclassified', 0):
                    raise ValueError(f'Nonempty sheet without a header: {path.name}/{sheet}')
                continue
            if set(frame.columns) != set(NAMES) or len(frame.columns) != len(NAMES):
                raise ValueError(f'Unknown or duplicate source attributes: {path.name}/{sheet}')
            if len(info['header_candidates']) != 1 or info['unlabelled_nonempty_columns']:
                raise ValueError(f'Ambiguous header/unlabelled data: {path.name}/{sheet}')
            info['pandas_dtypes'] = {c:str(t) for c,t in frame.dtypes.items()}
            item = frame[NAMES].copy()
            item['source_file'] = path.name
            item['source_year'] = year
            item['source_sheet'] = sheet
            item['source_excel_row'] = frame.index
            annual_frames.append(item.reset_index(drop=True))
        if digest(path) != original_hash:
            raise ValueError(f'Raw file changed during reading: {path.name}')
    if not annual_frames:
        raise ValueError('No data records found')
    return annual_frames, layouts

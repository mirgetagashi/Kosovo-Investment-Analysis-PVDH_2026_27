from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.io_utils import RAW, MANIFEST, FIGURES, save_json, save_table
from src.excel_reader import read_annual_files
from src.schema import SOURCE_COLUMNS, CATEGORIES, MONEY


def main(annual_frames=None, layouts=None):
    if annual_frames is None:
        annual_frames, layouts = read_annual_files(RAW)

    example = annual_frames[0][SOURCE_COLUMNS]
    print('First source: shape', example.shape)
    print(example.head(3).to_string(index=False))
    example.info()
    print(example.dtypes.to_string())
    print(example.nunique().to_string())
    print(example.describe().to_string())

    overview = []
    for layout in layouts:
        counts = layout['row_counts']
        if sum(counts.values()) != layout['physical_rows']:
            raise ValueError('Physical rows do not reconcile with the row classification')
        matching = [frame for frame in annual_frames
                    if frame.source_file.iloc[0] == layout['file']
                    and frame.source_sheet.iloc[0] == layout['sheet']]
        months = sorted(set().union(*(set(frame.month) for frame in matching)))
        overview.append(dict(year=layout['year'], file=layout['file'], sheet=layout['sheet'],
            physical_rows=layout['physical_rows'], physical_columns=layout['physical_columns'],
            header_row=layout['header_row'], records=counts.get('record',0),
            summary_rows=counts.get('summary',0), metadata_rows=counts.get('metadata',0),
            blank_rows=counts.get('blank',0), separators=', '.join(layout['separator_columns']),
            months=', '.join(str(m) for m in months),
            reporting_scope='January-June only' if layout['year']==2026 else 'Twelve represented months'))
    summary = pd.DataFrame(overview)
    save_table(summary, '01_overview')
    save_json(layouts, MANIFEST)

    attributes = []
    for column in SOURCE_COLUMNS:
        scale = 'nominal' if column in CATEGORIES else 'interval' if column=='year' else 'ordinal/cyclic' if column=='month' else 'ratio'
        attributes.append(dict(attribute=column, scale=scale,
            values='continuous (finite precision)' if column in MONEY else 'discrete',
            example_pandas_dtype=str(example[column].dtype),
            units='Unconfirmed monetary unit/scale' if column in MONEY else 'taxpayers in aggregate' if column=='taxpayer_count' else 'calendar/category'))
    save_table(pd.DataFrame(attributes), '01_attribute_types')

    counts = pd.concat([frame.groupby(['year','month']).size().rename('records')
                        for frame in annual_frames]).groupby(level=[0,1]).sum().unstack('month')
    fig, ax = plt.subplots(figsize=(10,4))
    image = ax.imshow(counts.to_numpy(), cmap='Blues', aspect='auto', vmin=0)
    ax.set(xticks=range(12), xticklabels=range(1,13), yticks=range(len(counts)),
           yticklabels=counts.index, xlabel='Month', ylabel='Year',
           title='Source records: 2026 covers January-June only')
    maximum = np.nanmax(counts.to_numpy())
    for row in range(len(counts)):
        for col in range(12):
            value = counts.iloc[row,col]
            ax.text(col,row,'N/S' if pd.isna(value) else str(int(value)),ha='center',va='center',
                    fontsize=8,color='white' if pd.notna(value) and value>maximum/2 else 'black')
    fig.colorbar(image,ax=ax,label='Records'); fig.tight_layout()
    FIGURES.mkdir(parents=True,exist_ok=True)
    fig.savefig(FIGURES/'01_month_coverage.png',dpi=160); plt.close(fig)
    print(summary.to_string(index=False))
    print('01: inspection complete; source meanings/units still require documentation.')
    return annual_frames, layouts


if __name__=='__main__':
    main()

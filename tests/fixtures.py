import pandas as pd
from src.schema import SOURCE_COLUMNS, PROVENANCE, YEARS_EXPECTED, EXPECTED_MONTHS


def source_fixture():
    records=[]
    for year in YEARS_EXPECTED:
        for month in EXPECTED_MONTHS[year]:
            records.append([year,month,'Sector',' Form ','Unfamiliar Town',2,
                -0.1 if month==1 else 0.1,0.2,0.3,0.0,100000000.01,0.00001,
                f'Investimet-{year}.xlsx',year,'Report',month+6])
    return pd.DataFrame(records,columns=SOURCE_COLUMNS+PROVENANCE)

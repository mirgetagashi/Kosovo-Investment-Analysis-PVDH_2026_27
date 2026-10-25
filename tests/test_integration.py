import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from src.schema import SOURCE_COLUMNS, MONEY, INTEGRATED_COLUMNS, PROVENANCE_KEY
from src.schema import standard_name
import importlib
stage=importlib.import_module('phase_1.03_data_integration')
integrate_sources=stage.integrate_sources
normalize_numeric=stage.normalize_numeric
from src.validation import check_records
from src.io_utils import save_frame,load_dataset
from tests.fixtures import source_fixture


class IntegrationTests(unittest.TestCase):
    def test_header_mapping_and_schema(self):
        self.assertEqual(standard_name('Importet investive me normë 18%'),MONEY[2])
        self.assertEqual(standard_name('Domestic investment purchases at 8%'),MONEY[5])
        self.assertTrue(standard_name('unrecognized measure').startswith('unmapped_'))
        frame=integrate_sources([source_fixture()])
        self.assertEqual(list(frame.columns),INTEGRATED_COLUMNS)
        self.assertEqual(str(frame.month.dtype),'Int8')

    def test_unknown_schema_fails(self):
        frame=source_fixture().rename(columns={'municipality':'unmapped_town'})
        with self.assertRaises(ValueError): integrate_sources([frame])
        frame=source_fixture(); frame['municipality']=42
        with self.assertRaises(ValueError): integrate_sources([frame])

    def test_record_counts_provenance_and_amounts(self):
        source=source_fixture(); result=integrate_sources([source])
        check_records(source,result)
        self.assertEqual(len(source),len(result))
        self.assertFalse(result.duplicated(PROVENANCE_KEY).any())
        self.assertEqual(result.iloc[0].source_excel_row,7)
        self.assertEqual(result.iloc[0].municipality,'Unfamiliar Town')

    def test_year_and_month_normalization(self):
        self.assertEqual(normalize_numeric(pd.Series(['01','12.0']), 'month').tolist(),[1,12])
        self.assertEqual(normalize_numeric(pd.Series(['2026']), 'year').tolist(),[2026])
        for value in ['0','13','1.5','invalid',None]:
            with self.subTest(value=value), self.assertRaises((ValueError,TypeError)):
                normalize_numeric(pd.Series([value]),'month')

    def test_source_year_mismatch_fails(self):
        source=source_fixture(); source.loc[0,'year']=2020
        with self.assertRaises(ValueError): integrate_sources([source])

    def test_numeric_parse_failure_not_zero(self):
        with self.assertRaises(ValueError): normalize_numeric(pd.Series(['bad']),MONEY[0])
        with self.assertRaises(ValueError): normalize_numeric(pd.Series([float('inf')]),MONEY[0])

    def test_lost_and_duplicated_records_fail(self):
        source=source_fixture(); result=integrate_sources([source])
        for broken in [result.iloc[1:].copy(),pd.concat([result,result.iloc[:1]],ignore_index=True)]:
            with self.assertRaises(ValueError): check_records(source,broken)

    def test_numeric_corruption_fails_reconciliation(self):
        source=source_fixture(); result=integrate_sources([source])
        result.loc[0,MONEY[0]]+=1
        with self.assertRaises(ValueError): check_records(source,result)

    def test_csv_roundtrip_is_exact(self):
        frame=integrate_sources([source_fixture()])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'checkpoint.csv'; save_frame(frame,path)
            self.assertTrue(frame.equals(load_dataset(path)))

    def test_mandatory_failure_nonzero_exit(self):
        command="import pandas as pd; from src.validation import check_time; check_time(pd.DataFrame(dict(year=[2026],month=[13],source_year=[2026])))"
        result=subprocess.run([sys.executable,'-c',command],capture_output=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn(b'Invalid month',result.stderr)

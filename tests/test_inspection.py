import tempfile
from io import BytesIO
import unittest
from unittest.mock import patch
import importlib
from pathlib import Path
import openpyxl
import pandas as pd
from src.excel_reader import inspect_workbook
from src.schema import SOURCE_COLUMNS

HEADERS=['Year','Month','Primary Sector','Registration Status','Municipality','Taxpayers',
         'without VAT','non-deductible VAT','imports 18%','imports 8%',
         'domestic 18%','domestic 8%']


class InspectionTests(unittest.TestCase):
    def test_independent_audit_checks_values_flags_and_incomplete_records(self):
        stage=importlib.import_module('phase_1.06_common_validation')
        name='Investimet-2026.xlsx'
        positions=[0,1,2,4,5,6,7,9,10,12,13,15]
        records=[[2026,1,'Mungon aktiviteti','Form','Town',1,None,0,0,0,0,0],
                 [2026,1,'Total','Form','Town',1,10,0,0,0,0,0]]
        integrated=pd.DataFrame(records,columns=SOURCE_COLUMNS)
        integrated['source_file']=name; integrated['source_year']=2026
        integrated['source_sheet']='Report'; integrated['source_excel_row']=[7,8]
        final=integrated.assign(sector_unclassified=[True,False])
        layouts=[dict(file=name,row_counts=dict(blank=4,header=1,metadata=1,summary=1,record=2))]
        with tempfile.TemporaryDirectory() as directory:
            wb=openpyxl.Workbook(); ws=wb.active; ws.title='Report'
            ws.append(['Title'])
            for _ in range(4): ws.append([])
            for values in [HEADERS,*records,[4052,2,'','','',2,10,0,0,0,0,0]]:
                row=[None]*16
                for j,value in zip(positions,values): row[j]=value
                ws.append(row)
            wb.save(Path(directory)/name); wb.close()
            with patch.object(stage,'RAW',Path(directory)),patch.object(stage,'EXPECTED_FILES',(name,)):
                result=stage.verify_original_workbooks(final,integrated,layouts).iloc[0]
                self.assertEqual((result.records,result.nulls,result.domain_unavailable,result.footer_excel_row),(2,1,1,9))
                for column,value in [('sector_unclassified',False),(SOURCE_COLUMNS[6],99)]:
                    broken=final.copy(); broken.loc[0,column]=value
                    with self.subTest(column=column),self.assertRaises(ValueError):
                        stage.verify_original_workbooks(broken,integrated,layouts)

    def inspect_rows(self, rows):
        stream=BytesIO(); stream.name='synthetic.xlsx'
        wb=openpyxl.Workbook(); ws=wb.active; ws.append(HEADERS)
        for row in rows:
            ws.append(row)
        wb.save(stream); wb.close(); stream.seek(0)
        return inspect_workbook(stream)

    def test_total_is_allowed_in_category_even_with_missing_month(self):
        for month in [1,None]:
            with self.subTest(month=month):
                details,frames,_=self.inspect_rows([[2026,month,'Total','Form','Town',1,10,0,0,0,0,0]])
                self.assertEqual(len(frames[0][1]),1)
                self.assertEqual(details[0]['row_counts'].get('summary',0),0)

    def test_partially_empty_record_is_retained(self):
        _,frames,_=self.inspect_rows([[2026,1,None,'Form','Town',1,None,0,0,0,0,0]])
        data=frames[0][1]
        self.assertEqual(list(data.index),[2])
        self.assertEqual(int(data.isna().sum().sum()),2)
        self.assertEqual(data.iloc[0].investment_imports_18pct,0)

    def test_footer_reconciles_past_missing_numeric_cell_without_filling_it(self):
        for empty in [None,'','  ']:
            with self.subTest(empty=empty):
                details,frames,_=self.inspect_rows([
                    [2026,1,'Sector','Form','Town',1,empty,0,0,0,0,0],
                    [2026,1,'Sector','Form','Town',1,10,0,0,0,0,0],
                    [4052,2,'','','',2,10,0,0,0,0,0]])
                data=frames[0][1]
                self.assertEqual(list(data.index),[2,3])
                self.assertEqual(details[0]['row_counts']['summary'],1)
                value=data.iloc[0].investment_purchases_imports_without_vat
                self.assertTrue(value==empty if isinstance(empty,str) and empty else pd.isna(value))

    def test_invalid_numeric_value_does_not_prove_footer(self):
        details,frames,_=self.inspect_rows([
            [2026,1,'Sector','Form','Town',1,'invalid',0,0,0,0,0],
            [2026,1,'','','',1,0,0,0,0,0,0]])
        self.assertEqual(len(frames[0][1]),2)
        self.assertFalse(details[0]['footer_evidence'])

    def inspect(self, footer):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'example.xlsx'
            wb=openpyxl.Workbook(); ws=wb.active
            ws.append(['Report title']); ws.append([]); ws.append(HEADERS)
            ws.append([2026,1,'Sector','Form','Town',2,10,20,30,40,50,60])
            ws.append(footer)
            wb.create_sheet('Empty')
            wb.save(path); wb.close()
            return inspect_workbook(path)

    def test_unlabelled_totals_need_all_sums(self):
        details,frames,_=self.inspect([2026,1,'','','',2,10,20,30,40,50,60])
        self.assertEqual(details[0]['header_row'],3)
        self.assertEqual(len(frames[0][1]),1)
        self.assertEqual(details[0]['row_counts']['summary'],1)
        self.assertTrue(all(v['agrees'] for v in details[0]['footer_evidence'][0]['checks'].values()))
        self.assertIsNone(details[1]['header_row'])

    def test_bad_footer_remains_candidate_record(self):
        details,frames,_=self.inspect([2026,1,'','','',2,10,20,30,40,50,999])
        self.assertEqual(len(frames[0][1]),2)
        self.assertFalse(details[0]['footer_evidence'])

    def test_labeled_summary(self):
        details,frames,_=self.inspect(['Totali',None,None,None,None,2,10,20,30,40,50,60])
        self.assertEqual(len(frames[0][1]),1)
        self.assertEqual(details[0]['row_counts']['summary'],1)


if __name__=='__main__':
    unittest.main()

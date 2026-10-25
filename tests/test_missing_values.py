import unittest
import pandas as pd
from src.schema import missing_masks,missing_mask,category_interpretation,SOURCE_COLUMNS
import importlib
integrate_sources=importlib.import_module('phase_1.03_data_integration').integrate_sources
clean_records=importlib.import_module('phase_1.04_data_cleaning').clean_records
assess_missingness=importlib.import_module('phase_1.05_missing_values').assess_missingness
from tests.fixtures import source_fixture


class MissingTests(unittest.TestCase):
    def test_sector_rule_is_specific_and_preserves_valid_categories(self):
        values=['Mungon aktiviteti',' MUNGON AKTIVITETI ','Person Fizik','Tjeter',
                'Aktivitetet e tjera sherbyese',None,'',0]
        sector=pd.Series(values,name='primary_sector')
        null,blank,generic,encoded=missing_masks(sector)
        self.assertEqual(int(null.sum()),1); self.assertEqual(int(blank.sum()),1)
        self.assertEqual(encoded.tolist(),[True,True,False,False,False,False,False,False])
        self.assertEqual(int(generic.sum()),0)
        self.assertFalse(missing_mask(pd.Series(values[:2],name='municipality')).any())
        self.assertEqual(category_interpretation('primary_sector',pd.NA)[0],'literal missing')
        self.assertEqual(category_interpretation('primary_sector',values[0])[0],'domain-specific unavailable')
        self.assertEqual(category_interpretation('registration_status','Tjeter')[0],'catch-all category')
        self.assertEqual(category_interpretation('primary_sector','Person Fizik')[0],'ambiguous category')

    def test_encoded_sector_flags_reports_and_original_values_agree(self):
        source=source_fixture(); source.loc[0,'primary_sector']='Mungon aktiviteti'
        integrated=integrate_sources([source]); cleaned,_=clean_records(integrated)
        self.assertEqual(len(cleaned),len(source))
        self.assertEqual(cleaned.loc[0,'primary_sector_raw'],'Mungon aktiviteti')
        self.assertEqual(cleaned.loc[0,'primary_sector'],'Mungon aktiviteti')
        self.assertTrue(integrated[SOURCE_COLUMNS].equals(cleaned[SOURCE_COLUMNS].assign(
            registration_status=integrated.registration_status)))
        self.assertEqual(int(cleaned.review_missing.sum()),1)
        self.assertEqual(int(cleaned.sector_unclassified.sum()),1)
        strategy,_=assess_missingness(cleaned)
        row=strategy.set_index('attribute').loc['primary_sector']
        self.assertEqual((row.nulls,row.blanks,row.placeholders,row.domain_unavailable,row.missing_cells),(0,0,0,1,1))
        self.assertAlmostEqual(row.missing_percent,100/len(cleaned))
        self.assertEqual(int(strategy.imputed_cells.sum()),0)
        annual=[g for _,g in source.groupby('source_year')]
        quality=importlib.import_module('phase_1.02_data_quality')
        profile,categories=quality.assess_quality(annual)
        self.assertEqual(int(profile.missing.sum()),1)
        self.assertEqual(int(categories.loc[categories.interpretation.eq('domain-specific unavailable'),'records'].sum()),1)
        annual_sector=profile.query('year == 2019 and attribute == "primary_sector"').iloc[0]
        self.assertAlmostEqual(annual_sector.domain_unavailable_percent,100/12)
        twice,_=clean_records(cleaned)
        self.assertTrue(twice.equals(cleaned))

    def test_null_empty_placeholder_and_zero_are_distinct(self):
        s=pd.Series([None,'','  ','NA','n/a',0,'0','valid'])
        null,empty,placeholder,domain=missing_masks(s)
        self.assertEqual(int(null.sum()),1); self.assertEqual(int(empty.sum()),2)
        self.assertEqual(int(placeholder.sum()),2); self.assertEqual(int(missing_mask(s).sum()),5)
        self.assertFalse(missing_mask(s).iloc[5])
        self.assertEqual(int(domain.sum()),0)

    def test_four_missing_kinds_and_specific_flag_are_distinct(self):
        source=source_fixture()
        source.loc[:4,'primary_sector']=[None,'  ','unknown','Mungon aktiviteti','Person Fizik']
        integrated=integrate_sources([source]); cleaned,_=clean_records(integrated)
        self.assertEqual(cleaned.review_missing.iloc[:5].tolist(),[True,True,True,True,False])
        self.assertEqual(cleaned.sector_unclassified.iloc[:5].tolist(),[False,False,False,True,False])
        from src.schema import MONEY
        self.assertTrue(cleaned[MONEY].equals(integrated[MONEY]))
        self.assertEqual(len(source),len(cleaned))
        report,_=assess_missingness(cleaned)
        row=report.set_index('attribute').loc['primary_sector']
        self.assertEqual((row.nulls,row.blanks,row.placeholders,row.domain_unavailable,row.missing_cells),
                         (1,1,1,1,4))

    def test_no_imputation_and_partial_year(self):
        source=source_fixture(); cleaned,_=clean_records(integrate_sources([source]))
        before=cleaned.copy(deep=True); strategy,periods=assess_missingness(cleaned)
        self.assertTrue(cleaned.equals(before))
        self.assertEqual(int(strategy.missing_cells.sum()),0)
        self.assertTrue(strategy.decision.eq('No imputation necessary').all())
        future=periods[(periods.year==2026)&(periods.month>6)]
        self.assertEqual(len(future),6); self.assertTrue(future.records_generated.eq(0).all())
        self.assertFalse(((cleaned.year==2026)&(cleaned.month>6)).any())

    def test_missing_attribute_and_expected_period_reported(self):
        frame=integrate_sources([source_fixture()]).drop(columns=['primary_sector'])
        frame=frame[~((frame.year==2025)&(frame.month==12))]
        strategy,periods=assess_missingness(frame)
        self.assertTrue(strategy.set_index('attribute').loc['primary_sector','missing_attribute'])
        self.assertEqual(periods.query('year == 2025 and month == 12').iloc[0].status,'missing expected period')

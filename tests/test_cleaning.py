import unittest
import pandas as pd
import importlib
integrate_sources=importlib.import_module('phase_1.03_data_integration').integrate_sources
clean_records=importlib.import_module('phase_1.04_data_cleaning').clean_records
quality=importlib.import_module('phase_1.02_data_quality')
duplicate_masks=quality.duplicate_masks
from src.schema import MONEY
from tests.fixtures import source_fixture


class CleaningTests(unittest.TestCase):
    def test_only_whitespace_changes_and_raw_preserved(self):
        source=source_fixture(); integrated=integrate_sources([source])
        cleaned,audit=clean_records(integrated)
        self.assertTrue(cleaned.registration_status.eq('Form').all())
        self.assertTrue(cleaned.registration_status_raw.eq(' Form ').all())
        self.assertTrue(cleaned.municipality.eq('Unfamiliar Town').all())
        self.assertEqual(int(audit.affected_records.sum()),len(source))
        self.assertEqual(set(audit.rule),{'TRIM_EDGE_WHITESPACE'})

    def test_idempotent_and_negative_values_retained(self):
        integrated=integrate_sources([source_fixture()]); cleaned,_=clean_records(integrated)
        twice,audit=clean_records(cleaned)
        self.assertTrue(twice.equals(cleaned))
        self.assertTrue(audit.empty)
        self.assertTrue(integrated[MONEY].equals(cleaned[MONEY]))
        self.assertEqual(int(cleaned.review_negative_monetary.sum()),8)

    def test_duplicate_detection_ignores_provenance(self):
        source=source_fixture(); extra=source.iloc[:1].copy(); extra['source_excel_row']=999
        source=pd.concat([source,extra],ignore_index=True)
        exact,key=duplicate_masks(source)
        self.assertEqual(int(exact.sum()),2); self.assertEqual(int(key.sum()),2)
        cleaned,_=clean_records(integrate_sources([source]))
        self.assertEqual(len(cleaned),len(source)); self.assertEqual(int(cleaned.review_candidate_key_duplicate.sum()),2)

    def test_candidate_key_conflict_is_not_exact_duplicate(self):
        source=source_fixture(); extra=source.iloc[:1].copy(); extra[MONEY[0]]=55; extra['source_excel_row']=999
        exact,key=duplicate_masks(pd.concat([source,extra],ignore_index=True))
        self.assertEqual(int(exact.sum()),0); self.assertEqual(int(key.sum()),2)

    def test_unfamiliar_domain_remains_unresolved(self):
        profile,categories=quality.assess_quality([group for _,group in source_fixture().groupby('source_year')])
        self.assertTrue(categories.domain_status.str.startswith('UNRESOLVED').all())
        cleaned,_=clean_records(integrate_sources([source_fixture()]))
        self.assertTrue(cleaned.municipality.eq('Unfamiliar Town').all())
        self.assertTrue(cleaned.review_category_domain_unverified.all())
        self.assertEqual(int(profile.negatives.sum()),8)

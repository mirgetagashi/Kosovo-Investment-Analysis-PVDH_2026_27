import unittest
import numpy as np
import pandas as pd
from src.schema import MONEY
from importlib import import_module
aggregation = import_module('phase_1.pipeline_b.01_feature_selection_aggregation')
geography = import_module('phase_1.pipeline_b.02_geographic_share_hhi')
sampling = import_module('phase_1.pipeline_b.05_sampling_sensitivity')
runner = import_module('phase_1.pipeline_b.run_pipeline_b')
concentration = geography.concentration
select_features = aggregation.select_features
municipal_metrics = geography.municipal_metrics
stratified_sensitivity = sampling.stratified_sensitivity
analyse = runner.analyse
validate_results = runner.validate_results


def fixture():
    data = pd.DataFrame(dict(year=[2025]*4, month=[1]*4,
        municipality=['A', 'A', 'B', 'B'], primary_sector=['S1', 'S2', 'S1', 'S2']))
    for measure in MONEY:
        data[measure] = [40., 60., 10., 90.]
    return data


class PipelineBTests(unittest.TestCase):
    def test_known_shares_hhi_and_lq(self):
        data = fixture(); original = data.copy(deep=True)
        result = analyse(data, sample_fraction=1)
        self.assertTrue(data.equals(original))
        self.assertTrue(result['concentration'].hhi.eq(.5).all())
        self.assertTrue(result['municipal'].municipal_share.eq(.5).all())
        lq = result['specialization']
        self.assertTrue(np.allclose(lq.loc[lq.municipality.eq('A') & lq.primary_sector.eq('S1'), 'location_quotient'], 1.6))
        self.assertTrue(np.allclose(lq.loc[lq.municipality.eq('B') & lq.primary_sector.eq('S1'), 'location_quotient'], .4))
        self.assertTrue(result['sample_sensitivity'].hhi_difference.eq(0).all())
        self.assertEqual(len(validate_results(data, result)), 12)
        profile = result['profiles'].query('municipality == "A"').iloc[0]
        self.assertAlmostEqual(profile.sector_hhi, .52)

    def test_hhi_limits_and_invalid_inputs(self):
        self.assertAlmostEqual(concentration([10, 10, 10, 10])[1], .25)
        self.assertAlmostEqual(concentration([0, 0, 40])[1], 1)
        for values, reason in [([0, 0], 'zero_total'), ([-1, 5], 'negative_municipal_total')]:
            shares, hhi, status = concentration(values)
            self.assertTrue(np.isnan(shares).all()); self.assertTrue(np.isnan(hhi))
            self.assertEqual(status, reason)

    def test_zero_and_negative_blocks_are_not_silently_repaired(self):
        data = fixture(); data[MONEY] = 0.
        result = analyse(data)
        self.assertTrue(result['specialization'].location_quotient.isna().all())
        self.assertTrue(result['specialization'].high_specialization.isna().all())
        self.assertTrue(result['municipal'].top_contributor.isna().all())
        data = fixture(); data.loc[0, MONEY] = -1.
        original = data.copy(deep=True)
        result = analyse(data)
        self.assertTrue(data.equals(original))
        self.assertTrue(result['concentration'].status.eq('valid').all())
        self.assertTrue(result['specialization'].lq_status.eq('negative_net_sector_cell').all())
        self.assertTrue(result['specialization'].location_quotient.isna().all())

    def test_excluded_labels_preserved_with_consistent_denominators(self):
        data = fixture()
        extra = data.iloc[:2].copy()
        extra.primary_sector = ['Mungon aktiviteti', 'Person Fizik']
        extra[MONEY] = 50.
        data = pd.concat([data, extra], ignore_index=True)
        result = analyse(data)
        self.assertTrue(result['concentration'].total.eq(300).all())
        coverage = result['sector_coverage']
        self.assertTrue(coverage.all_records.eq(6).all())
        self.assertTrue(coverage.eligible_records.eq(4).all())
        self.assertTrue(coverage.unclassified_records.eq(1).all())
        self.assertTrue(coverage.ambiguous_records.eq(1).all())
        lq = result['specialization']
        self.assertTrue(lq.loc[~lq.sector_eligible, 'location_quotient'].isna().all())
        a = lq.loc[lq.municipality.eq('A') & lq.primary_sector.eq('S1')].iloc[0]
        self.assertAlmostEqual(a.location_quotient, 1.6)
        self.assertEqual(a.municipal_eligible_value, 100)
        self.assertTrue(validate_results(data, result).status.eq('PASS').all())

    def test_aggregate_zero_pair_and_zero_municipality(self):
        data = fixture().drop(index=3)
        result = analyse(data)
        pair = result['specialization'].query('municipality == "B" and primary_sector == "S2"')
        self.assertTrue(pair.value.eq(0).all())
        self.assertTrue(pair.location_quotient.eq(0).all())
        self.assertTrue(pair.aggregate_zero_combination.all())
        data.loc[data.municipality.eq('B'), MONEY] = 0.
        result = analyse(data)
        pair = result['specialization'].query('municipality == "B"')
        self.assertTrue(pair.location_quotient.isna().all())
        self.assertTrue(pair.high_specialization.isna().all())

    def test_balanced_jan_jun_and_no_future_records(self):
        data = fixture()
        second = data.copy(); second['month'] = 7
        future = data.copy(); future['year'] = 2026
        data = pd.concat([data, second, future], ignore_index=True)
        original = data.copy(deep=True)
        result = analyse(data)
        totals = result['concentration']
        self.assertTrue(totals.query('year == 2025 and window == "reported"').total.eq(400).all())
        self.assertTrue(totals.query('year == 2025 and window == "jan_jun"').total.eq(200).all())
        self.assertTrue(totals.query('year == 2026').months.eq(1).all())
        self.assertTrue(data.equals(original))

    def test_sampling_reproducible_and_strata_accounted_for(self):
        data = pd.concat([fixture()]*20, ignore_index=True)
        _, summary = municipal_metrics(aggregation.aggregate(data)['municipal_base'])
        first, comparison = stratified_sensitivity(data, summary, fraction=.25, seed=7)
        second, repeated = stratified_sensitivity(data, summary, fraction=.25, seed=7)
        pd.testing.assert_frame_equal(first, second)
        pd.testing.assert_frame_equal(comparison, repeated)
        self.assertTrue(first.sampled_records.eq(10).all())
        self.assertTrue(first.expansion_weight.eq(4).all())
        self.assertTrue((first.sampled_records * first.expansion_weight).eq(first.population_records).all())
        with self.assertRaises(ValueError): stratified_sensitivity(data, summary, fraction=0)

    def test_missing_nonfinite_and_corruption_fail(self):
        for value in [np.nan, np.inf]:
            data = fixture(); data.loc[0, MONEY[0]] = value
            with self.assertRaises(ValueError): select_features(data)
        data = fixture(); result = analyse(data)
        result['municipal'].loc[0, 'value'] += 1
        with self.assertRaises(ValueError): validate_results(data, result)

    def test_measure_separation_and_documented_band_boundaries(self):
        data = fixture(); data[MONEY[0]] *= 10
        result = analyse(data)
        summary = result['concentration']
        self.assertTrue(summary.loc[summary.measure.eq(MONEY[0]), 'total'].eq(2000).all())
        self.assertTrue(summary.loc[summary.measure.eq(MONEY[1]), 'total'].eq(200).all())
        data = fixture()
        data.municipality = ['A', 'B', 'C', 'D']
        for measure in MONEY: data[measure] = [0., 1., 5., 94.]
        base = aggregation.aggregate(data)
        municipal, _ = municipal_metrics(base['municipal_base'])
        self.assertNotIn('share_band', municipal.columns)
        lq = runner.specialization.sector_metrics(base['municipal_sector_base'], base['sector_coverage'])
        municipal, _ = runner.features.discretize(municipal, lq)
        first = municipal.query('window == "reported"').drop_duplicates('municipality').set_index('municipality')
        self.assertEqual(first.loc['A', 'share_band'], 'zero')
        self.assertEqual(first.loc['B', 'share_band'], 'medium')
        self.assertEqual(first.loc['C', 'share_band'], 'high')
        self.assertEqual(first.loc['B', 'top_contributor'], 0)
        self.assertEqual(first.loc['C', 'top_contributor'], 1)

    def test_equal_specialization_is_not_flagged_above_national(self):
        data = fixture(); data[MONEY] = 10.
        result = analyse(data)
        self.assertTrue(result['specialization'].location_quotient.eq(1).all())
        self.assertTrue(result['specialization'].high_specialization.eq(0).all())


if __name__ == '__main__':
    unittest.main()

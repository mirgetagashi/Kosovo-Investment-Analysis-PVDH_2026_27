import unittest
import importlib
import numpy as np
import pandas as pd
from src.schema import MONEY
from phase_1.pipeline_a import MEASURES, TOTAL, with_total
from tests.fixtures import source_fixture
integrate_sources = importlib.import_module('phase_1.03_data_integration').integrate_sources
clean_records = importlib.import_module('phase_1.04_data_cleaning').clean_records
aggregation = importlib.import_module('phase_1.pipeline_a.01_aggregation')
features = importlib.import_module('phase_1.pipeline_a.02_feature_creation')
transformation = importlib.import_module('phase_1.pipeline_a.03_transformation')
discretization = importlib.import_module('phase_1.pipeline_a.04_discretization_binarization')
sampling = importlib.import_module('phase_1.pipeline_a.05_sampling_sensitivity')


def common_fixture():
    cleaned, _ = clean_records(integrate_sources([source_fixture()]))
    return cleaned


class PipelineATests(unittest.TestCase):
    def setUp(self):
        self.data = common_fixture()
        self.result, self.cv = aggregation.main(self.data, persist=False)

    def test_levels_reconcile_to_source_and_have_expected_shapes(self):
        source = self.data.groupby('year')[MONEY].sum()
        for name, table in self.result.items():
            reported = table.loc[table.window.eq('reported')].groupby(['year', 'measure']).value.sum().unstack('measure')[MONEY]
            self.assertTrue(np.allclose(source.to_numpy(), reported.reindex(source.index).to_numpy()), name)
        monthly = self.result['monthly']
        self.assertEqual(len(monthly.loc[monthly.window.eq('reported')]), 90 * len(MEASURES))
        self.assertEqual(len(monthly.loc[monthly.window.eq('jan_jun')]), 48 * len(MEASURES))
        self.assertTrue(monthly.loc[monthly.year.eq(2026), 'partial_year'].all())
        constant = self.cv.loc[self.cv.measure.eq(MONEY[4]), 'cv']
        self.assertTrue(np.allclose(constant, 0, atol=1e-9))
        self.assertTrue(self.cv.loc[self.cv.measure.eq(MONEY[0]) & self.cv.level.eq('monthly'), 'cv'].gt(0).all())

    def test_total_investment_is_the_sum_of_the_six_amounts(self):
        annual = self.result['annual'].pivot_table(index=['window', 'year'], columns='measure', values='value')
        self.assertTrue(np.allclose(annual[MONEY].sum(axis=1), annual[TOTAL]))

    def test_constant_series_yields_zero_growth_and_unit_ratios(self):
        table = features.monthly_features(self.result['monthly'])
        constant = table.loc[table.measure.eq(MONEY[4]) & table.window.eq('reported')]
        self.assertTrue(constant.yoy_pct.dropna().eq(0).all())
        self.assertTrue(constant.mom_pct.dropna().eq(0).all())
        self.assertTrue(np.allclose(constant.rolling_12m.dropna(), 100000000.01))
        self.assertTrue(np.allclose(constant.ratio_to_annual_mean, 1))
        self.assertEqual(int(constant.is_covid.sum()), 16)
        self.assertEqual(int(constant.is_lockdown.sum()), 3)
        self.assertEqual(constant.t.iloc[0], 0)
        self.assertEqual(constant.t.iloc[-1], 89)

    def test_recovery_of_a_series_that_never_dropped_is_immediate(self):
        recovery = features.recovery(self.result['sector_monthly'])
        national = recovery.loc[recovery.primary_sector.eq('ALL') & recovery.measure.eq(MONEY[4])].iloc[0]
        self.assertEqual(national.status, 'no_decline')
        self.assertEqual(national.recovery_months, 0)
        self.assertTrue(bool(national.recovered))
        self.assertAlmostEqual(national.drop_pct, 0)

    def test_recovery_is_measured_after_the_trough_without_pre_pandemic_months(self):
        months = [(year, month) for year in range(2019, 2027) for month in range(1, 13 if year < 2026 else 7)]
        def value(year, month):
            if (year, month) in [(2020, 1), (2020, 2)]:
                return 1000.0   # a strong start of 2020 must not count as recovery
            if year == 2020 and 4 <= month <= 10:
                return 10.0
            return 100.0
        series = pd.DataFrame([dict(window='reported', year=y, month=m, primary_sector='Sector', measure=MONEY[4], value=value(y, m))
                               for y, m in months if (y, m) != (2020, 8)])   # a month without records counts as zero
        row = features.recovery(series).loc[lambda t: t.primary_sector.eq('Sector')].iloc[0]
        self.assertEqual(row.status, 'valid')
        self.assertEqual((row.trough_year, row.trough_month), (2020, 8))   # (10 + 10 + 0) / 3
        self.assertAlmostEqual(row.drop_pct, -100 + 20 / 3)
        self.assertEqual((row.recovery_year, row.recovery_month), (2021, 1))   # first 100 + 100 + 100 after the trough
        self.assertEqual(row.recovery_months, 10)

    def test_partial_year_is_never_a_significant_change(self):
        intervals = sampling.bootstrap_intervals(with_total(self.data), np.random.default_rng(0))
        partial = intervals.loc[intervals.window.eq('reported') & intervals.year.eq(2026)]
        self.assertFalse(partial.comparable.any())
        self.assertFalse(partial.significant_decline.any() or partial.significant_growth.any())
        self.assertTrue(partial.yoy_pct.dropna().lt(0).all())   # six months against twelve look like a fall

    def test_decomposition_of_a_constant_series_has_no_seasonal_effect(self):
        transformed, skewness, components = transformation.monthly_transformed(self.result['monthly'])
        constant = transformed.loc[transformed.measure.eq(MONEY[4]) & transformed.window.eq('reported')]
        self.assertTrue(np.allclose(constant.seasonal, 0))
        self.assertTrue(np.allclose(constant.trend.dropna(), 100000000.01))
        self.assertTrue(np.allclose(constant.index_2019, 100))
        self.assertEqual(int(skewness.loc[skewness.measure.eq(MONEY[0]), 'negative_months'].sum()), 8 + 8)

    def test_growth_classes_follow_fixed_thresholds(self):
        fixed, width, frequency = discretization.growth_classes(pd.Series([-50.0, -10.0, 0.0, 10.0, 50.0, np.nan]))
        self.assertEqual(fixed.dropna().tolist(), ['strong_decline', 'decline', 'stable', 'growth', 'strong_growth'])
        self.assertEqual(frequency.dropna().nunique(), 5)
        self.assertTrue(pd.isna(fixed.iloc[-1]))


if __name__ == '__main__':
    unittest.main()

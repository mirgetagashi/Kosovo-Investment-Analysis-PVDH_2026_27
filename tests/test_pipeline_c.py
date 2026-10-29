import importlib
import unittest
import numpy as np
import pandas as pd
from src.schema import MONEY
from phase_1.pipeline_c import origin_features, origin_shares, IMPORT, DOMESTIC, MIXED

c01 = importlib.import_module('phase_1.pipeline_c.01_feature_creation_aggregation')
c02 = importlib.import_module('phase_1.pipeline_c.02_feature_selection')
c03 = importlib.import_module('phase_1.pipeline_c.03_transformation_pca')
c04 = importlib.import_module('phase_1.pipeline_c.04_discretization_binarization')
c05 = importlib.import_module('phase_1.pipeline_c.05_sampling_bootstrap')


def investment_fixture(seed=0):
    """Two complete years plus one Jan-Jun year; sector A buys mostly at home, sector B mostly abroad."""
    rng = np.random.default_rng(seed)
    rows = []
    for year, months in [(2019, 12), (2020, 12), (2021, 6)]:
        for month in range(1, months + 1):
            for sector, imported, domestic in [('A', 10., 90.), ('B', 80., 20.)]:
                for town in ['X', 'Y']:
                    noise = rng.uniform(.5, 1.5)
                    amounts = [5. * noise, 3. * noise, imported * noise, 0., domestic * noise, 0.]
                    rows.append([year, month, town, sector, *amounts])
                rows.append([year, month, 'X', sector, 0., 0., 0., 0., 0., 0.])
    return pd.DataFrame(rows, columns=['year', 'month', 'municipality', 'primary_sector', *MONEY])


class PipelineCTests(unittest.TestCase):
    def test_origin_groups_partition_the_six_amounts(self):
        self.assertEqual(sorted(IMPORT + DOMESTIC + MIXED), sorted(MONEY))
        frame = origin_features(investment_fixture())
        parts = frame[['import_total', 'domestic_total', 'mixed_total']].sum(axis=1)
        self.assertTrue(np.allclose(parts, frame.total_investment))

    def test_shares_bounds_and_undefined_cases(self):
        table = pd.DataFrame(dict(import_total=[30., 0., 10., 5.], domestic_total=[70., 0., -1., 5.],
                                  mixed_total=[100., 4., 0., 0.]))
        table['total_investment'] = table.sum(axis=1)
        shares = origin_shares(table)
        self.assertAlmostEqual(shares.import_share[0], .3)
        self.assertAlmostEqual(shares.import_share_lower[0], .15)
        self.assertAlmostEqual(shares.import_share_upper[0], .65)
        self.assertTrue((shares.import_share_lower[0] <= shares.import_share[0] <= shares.import_share_upper[0]))
        self.assertEqual(shares.share_status.tolist(), ['valid', 'zero_identified', 'negative_component', 'valid'])
        self.assertTrue(np.isnan(shares.import_share[1]) and np.isnan(shares.import_share[2]))
        self.assertAlmostEqual(shares.import_share_upper[1], 1.)

    def test_aggregation_reconciles_and_jan_jun_window(self):
        data = investment_fixture()
        result, frame = c01.aggregate(data)
        for name, table in result.items():
            reported = table.loc[table.window.eq('reported'), 'total_investment'].sum()
            self.assertAlmostEqual(reported, data[MONEY].to_numpy().sum(), places=6, msg=name)
        national = result['national'].set_index(['window', 'year'])
        self.assertTrue(national.loc[('reported', 2021), 'partial_year'])
        self.assertEqual(national.loc[('jan_jun', 2019), 'months'], 6)
        sectors = result['sector_pooled'].query("window == 'reported'").set_index('primary_sector')
        self.assertLess(sectors.loc['A', 'import_share'], .2)
        self.assertGreater(sectors.loc['B', 'import_share'], .7)
        definedness = c01.row_level_definedness(frame)
        self.assertTrue((definedness.zero_total_records > 0).all())

    def test_feature_selection_drops_near_empty_columns(self):
        profile = c02.profile_measures(investment_fixture()).set_index('measure')
        self.assertFalse(profile.loc[MONEY[3], 'selected'])
        self.assertFalse(profile.loc[MONEY[5], 'selected'])
        self.assertTrue(profile.loc[[MONEY[0], MONEY[2], MONEY[4]], 'selected'].all())

    def test_pca_is_complete_orthogonal_and_signed(self):
        data = investment_fixture()
        selected = [MONEY[0], MONEY[1], MONEY[2], MONEY[4]]
        values, origin, exclusions = c03.sector_year_matrix(data, selected)
        self.assertEqual(sorted(values.index.get_level_values('year').unique()), [2019, 2020])
        _, z, skewness = c03.transform(values)
        scores, loadings, _, explained = c03.pca(z)
        self.assertAlmostEqual(explained.explained_variance_ratio.sum(), 1.)
        self.assertTrue(np.allclose(loadings.T @ loadings, np.eye(len(selected)), atol=1e-10))
        self.assertGreaterEqual(loadings.PC1.sum(), 0)
        self.assertTrue(np.allclose(np.corrcoef(scores.T.to_numpy())[0, 1], 0, atol=1e-8))

    def test_binarization_and_discretization_boundaries(self):
        table = pd.DataFrame(dict(import_share=[.2, 1 / 3, .5, .51, .9, np.nan], import_share_lower=[.1, .3, .4, .5, .8, np.nan],
                                  import_share_upper=[.3, .4, .6, .6, .95, np.nan]))
        result = c04.classify(table)
        self.assertEqual(result.import_dominant.tolist()[:5], [0, 0, 0, 1, 1])
        self.assertTrue(pd.isna(result.import_dominant.iloc[5]))
        self.assertEqual(result.origin_class.tolist()[:5], ['mostly_domestic', 'mostly_domestic', 'mixed', 'mixed', 'mostly_import'])
        self.assertEqual(result.class_robust.tolist(), [True, False, True, True, True, False])
        relative = c04.relative_class(table.assign(window='reported'), ['window'])
        self.assertEqual(relative.iloc[0], 'lower_import_reliance')
        self.assertEqual(relative.iloc[4], 'higher_import_reliance')

    def test_bootstrap_is_reproducible_and_brackets_the_estimate(self):
        data = investment_fixture()
        first = c05.sector_intervals(data, iterations=200, seed=7).set_index('primary_sector')
        second = c05.sector_intervals(data, iterations=200, seed=7).set_index('primary_sector')
        pd.testing.assert_frame_equal(first, second)
        for sector in ['A', 'B']:
            row = first.loc[sector]
            self.assertLessEqual(row.ci_low, row.import_share)
            self.assertGreaterEqual(row.ci_high, row.import_share)
        self.assertEqual(first.loc['B', 'stability'], 'class_stable')


if __name__ == '__main__':
    unittest.main()

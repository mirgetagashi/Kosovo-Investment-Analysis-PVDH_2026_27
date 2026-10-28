import numpy as np
import pandas as pd
from phase_1.pipeline_b import GROUP, INTERMEDIATE, TABLES, read_table, save_frame, mark_running


def sector_metrics(base, coverage):
    results = []
    for (window, year, measure), group in base.groupby(GROUP, sort=False):
        eligible_total = coverage.loc[coverage.window.eq(window) & coverage.year.eq(year) & coverage.measure.eq(measure), 'eligible_value'].iloc[0]
        table = group[['municipality', 'primary_sector', 'value', 'sector_eligible', 'aggregate_zero_combination']].copy().reset_index(drop=True)
        table['municipal_eligible_value'] = table.value.where(table.sector_eligible, 0).groupby(table.municipality).transform('sum')
        table['national_sector_value'] = table.groupby('primary_sector').value.transform('sum')
        table['national_eligible_value'] = eligible_total
        table['lq_status'] = 'valid'
        if table.loc[table.sector_eligible, 'value'].lt(0).any():
            table['lq_status'] = 'negative_net_sector_cell'
        elif eligible_total <= 0:
            table['lq_status'] = 'zero_national_eligible_total'
        else:
            table.loc[table.municipal_eligible_value.le(0), 'lq_status'] = 'zero_municipal_eligible_total'
            table.loc[table.national_sector_value.le(0), 'lq_status'] = 'zero_national_sector_total'
        table.loc[~table.sector_eligible, 'lq_status'] = 'excluded_from_sector_analysis'
        valid = table.lq_status.eq('valid')
        table['local_sector_share'] = table.value.div(table.municipal_eligible_value.where(table.municipal_eligible_value.gt(0))).where(table.sector_eligible)
        table['national_sector_share'] = table.national_sector_value/eligible_total if eligible_total > 0 else np.nan
        table['national_sector_share'] = table.national_sector_share.where(table.sector_eligible)
        table['location_quotient'] = table.local_sector_share.div(table.national_sector_share.where(table.national_sector_share.gt(0))).where(valid)
        results.append(table.assign(window=window, year=int(year), measure=measure))
    return pd.concat(results, ignore_index=True)


def main(base=None, sectors=None, coverage=None, persist=True):
    if persist:
        mark_running('03_sector_specialization_lq')
    if base is None:
        base = read_table('municipal_sector_base', INTERMEDIATE)
        sectors = read_table('sectors', INTERMEDIATE)
        coverage = read_table('sector_coverage', INTERMEDIATE)
    lq = sector_metrics(base, coverage)
    if persist:
        for name, table in [('sectors', sectors), ('specialization', lq), ('sector_coverage', coverage)]:
            save_frame(table, TABLES / f'{name}.csv')
        print('B03: LQ calculated; unavailable/ambiguous labels retained with undefined LQ.')
    return lq


if __name__ == '__main__':
    main()

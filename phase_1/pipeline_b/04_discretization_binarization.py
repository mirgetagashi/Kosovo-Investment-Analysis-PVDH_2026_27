import numpy as np
import pandas as pd
from phase_1.pipeline_b import GROUP, TABLES, read_table, save_frame, mark_running


def discretize(municipal, lq):
    municipal = municipal.copy()
    lq = lq.copy()
    bands = pd.cut(municipal.municipal_share, [0, .01, .05, np.inf], right=False,
                   labels=['low', 'medium', 'high']).astype('string')
    municipal['share_band'] = bands.mask(municipal.municipal_share.eq(0), 'zero')
    municipal['top_contributor'] = municipal.municipal_share.ge(.05).astype('Int8').mask(municipal.municipal_share.isna())
    municipal = municipal[['municipality', 'value', 'source_records', 'municipal_share', 'rank',
        'cumulative_share', 'share_band', 'top_contributor', 'status', 'window', 'year', 'measure', 'months', 'partial_year']]
    valid = lq.lq_status.eq('valid')
    lq['high_specialization'] = lq.location_quotient.gt(1).astype('Int8').mask(~valid)
    lq['specialization_band'] = pd.cut(lq.location_quotient, [-np.inf, 1, 2, np.inf],
        labels=['not_above_national', 'above_national', 'strong_relative']).astype('string')
    lq = lq[['municipality', 'primary_sector', 'value', 'sector_eligible', 'municipal_eligible_value',
        'national_sector_value', 'national_eligible_value', 'lq_status', 'local_sector_share',
        'national_sector_share', 'location_quotient', 'high_specialization', 'specialization_band',
        'aggregate_zero_combination', 'window', 'year', 'measure']]
    return municipal, lq


def municipality_profiles(lq, municipal):
    rows = []
    for keys, group in lq.groupby(GROUP + ['municipality'], sort=True):
        group = group.loc[group.sector_eligible]
        valid = group.lq_status.eq('valid')
        positive = group.value.gt(0)
        has_profile = valid.any() and group.lq_status.ne('negative_net_sector_cell').all()
        best = group.loc[valid].sort_values(['location_quotient', 'primary_sector'], ascending=[False, True])
        dominant = group.sort_values(['value', 'primary_sector'], ascending=[False, True]).iloc[0]
        row = dict(zip(GROUP + ['municipality'], keys))
        row.update(eligible_value=float(group.value.sum()), active_sectors=int(positive.sum()),
            sector_hhi=float(group.local_sector_share.pow(2).sum()) if has_profile else np.nan,
            specialized_sectors=int(group.high_specialization.fillna(0).sum()) if has_profile else np.nan,
            max_lq=float(best.location_quotient.iloc[0]) if len(best) else np.nan,
            max_lq_sector=best.primary_sector.iloc[0] if len(best) else None,
            max_lq_cell_value=float(best.value.iloc[0]) if len(best) else np.nan,
            dominant_sector=dominant.primary_sector if has_profile else None,
            profile_status='valid' if has_profile else 'undefined_sector_distribution')
        rows.append(row)
    result = pd.DataFrame(rows)
    return result.merge(municipal[GROUP + ['municipality', 'value', 'municipal_share', 'rank']],
                        on=GROUP + ['municipality'], validate='one_to_one')


def top_specialization(lq):
    top = lq.loc[lq.lq_status.eq('valid') & lq.location_quotient.gt(1)].sort_values(
        GROUP + ['primary_sector', 'location_quotient', 'value', 'municipality'],
        ascending=[True, True, True, True, False, False, True])
    return top.groupby(GROUP + ['primary_sector'], sort=False).head(5).reset_index(drop=True)


def main(municipal=None, lq=None, persist=True):
    if persist:
        mark_running('04_discretization_binarization')
    if municipal is None:
        municipal, lq = read_table('municipal'), read_table('specialization')
    municipal, lq = discretize(municipal, lq)
    profiles = municipality_profiles(lq, municipal)
    top = top_specialization(lq)
    if persist:
        for name, table in [('municipal', municipal), ('specialization', lq), ('profiles', profiles), ('top_specialization', top)]:
            save_frame(table, TABLES / f'{name}.csv')
        print('B04: descriptive bands, nullable binary flags, profiles and top specialization saved.')
    return municipal, lq, profiles, top


if __name__ == '__main__':
    main()

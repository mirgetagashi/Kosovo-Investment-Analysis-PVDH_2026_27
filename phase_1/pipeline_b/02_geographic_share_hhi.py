import numpy as np
import pandas as pd
from phase_1.pipeline_b import GROUP, INTERMEDIATE, TABLES, read_table, save_frame, mark_running


def concentration(values):
    """Nonnegative aggregate components and a positive total are necessary for shares."""
    numbers = np.asarray(values, dtype=float)
    total = float(numbers.sum())
    status = 'negative_municipal_total' if (numbers < 0).any() else 'zero_total' if total <= 0 else 'valid'
    if status != 'valid':
        return np.full(len(numbers), np.nan), np.nan, status
    shares = numbers / total
    return shares, float(np.square(shares).sum()), status


def municipal_metrics(base):
    results, summaries = [], []
    for (window, year, measure), group in base.groupby(GROUP, sort=False):
        values = group.value
        shares, hhi, status = concentration(values)
        table = group[['municipality', 'value', 'source_records']].copy().reset_index(drop=True)
        table['municipal_share'] = shares
        table = table.sort_values(['value', 'municipality'], ascending=[False, True]).reset_index(drop=True)
        table['rank'] = table.value.rank(method='min', ascending=False).astype(int)
        table['cumulative_share'] = table.municipal_share.cumsum()
        table['status'] = status
        months = int(group.months.iloc[0])
        partial = bool(group.partial_year.iloc[0])
        table = table.assign(window=window, year=int(year), measure=measure, months=months, partial_year=partial)
        results.append(table)
        reach = np.flatnonzero(table.cumulative_share.to_numpy() >= .8)
        summaries.append(dict(window=window, year=int(year), measure=measure, months=months, partial_year=partial,
            total=float(values.sum()), municipalities=len(group), positive_municipalities=int(values.gt(0).sum()),
            hhi=hhi, hhi_10000=hhi*10000, equal_share_reference=1/len(group),
            effective_municipalities=1/hhi if np.isfinite(hhi) else np.nan,
            top1_share=table.municipal_share.iloc[0], top5_share=table.municipal_share.head(5).sum(min_count=1),
            municipalities_for_80pct=int(reach[0]+1) if len(reach) else np.nan,
            negative_cells=int(group.negative_cells.iloc[0]), negative_sum=float(group.negative_sum.iloc[0]), status=status))
    return pd.concat(results, ignore_index=True), pd.DataFrame(summaries)


def main(base=None, persist=True):
    if persist:
        mark_running('02_geographic_share_hhi')
    if base is None:
        base = read_table('municipal_base', INTERMEDIATE)
    municipal, summary = municipal_metrics(base)
    if persist:
        save_frame(municipal, TABLES / 'municipal.csv')
        save_frame(summary, TABLES / 'concentration.csv')
        print('B02: municipal shares, HHI, ranking and Pareto coverage calculated.')
    return municipal, summary


if __name__ == '__main__':
    main()

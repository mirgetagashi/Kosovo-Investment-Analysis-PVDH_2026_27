import pandas as pd
from phase_1.pipeline_c import (DOMINANCE_THRESHOLD, CLASS_BINS, CLASS_LABELS, RELATIVE_LABELS, MIN_MUNICIPAL_VALUE, RANKING_SIZE,
                                BASE_TABLES, INTERMEDIATE, TABLES, read_table, save_frame, mark_running)


def discretize(share):
    return pd.cut(share, bins=CLASS_BINS, labels=CLASS_LABELS, include_lowest=True).astype('string')


def classify(table):
    """Binarization (> 50% import) and three origin classes on the identified-origin share.

    The same rules are applied to the lower and upper bounds: a class is 'robust' only when it holds
    whether the mixed purchases/imports were all domestic or all imported.
    """
    table = table.copy()
    table['import_dominant'] = table.import_share.gt(DOMINANCE_THRESHOLD).astype('Int8').where(table.import_share.notna())
    table['import_dominant_upper'] = table.import_share_upper.gt(DOMINANCE_THRESHOLD).astype('Int8').where(table.import_share_upper.notna())
    table['origin_class'] = discretize(table.import_share)
    table['origin_class_lower'] = discretize(table.import_share_lower)
    table['origin_class_upper'] = discretize(table.import_share_upper)
    table['class_robust'] = table.origin_class_lower.eq(table.origin_class_upper).fillna(False).astype(bool)
    return table


def relative_class(table, keys):
    """Equal-frequency discretization (tertiles of the rank) within each window/year.

    The absolute 33/67% classes answer 'is it import-dominated?'; the relative classes answer
    'which units rely on imports more than the others?', which stays informative even when
    almost every unit falls in the same absolute class. Excluded sector labels are not ranked.
    """
    eligible = table.sector_eligible.astype(bool) if 'sector_eligible' in table else pd.Series(True, index=table.index)
    shares = table.import_share.where(eligible)
    ranks = shares.groupby([table[k] for k in keys]).rank(pct=True) if keys else shares.rank(pct=True)
    return pd.cut(ranks, bins=[0, 1 / 3, 2 / 3, 1], labels=RELATIVE_LABELS, include_lowest=True).astype('string')


def class_counts(sector_year):
    eligible = sector_year.loc[sector_year.sector_eligible.astype(bool) & sector_year.origin_class.notna()]
    counts = eligible.groupby(['window', 'year', 'origin_class']).size().unstack('origin_class', fill_value=0)
    return counts.reindex(columns=CLASS_LABELS, fill_value=0).reset_index()


def municipal_ranking(municipal):
    """Most domestic vs most import-oriented municipalities; small totals are excluded as unstable."""
    pooled = municipal.loc[municipal.window.eq('reported') & municipal.import_share.notna()]
    ranked = pooled.loc[pooled.identified_total.ge(MIN_MUNICIPAL_VALUE)]
    columns = ['municipality', 'import_share', 'import_share_lower', 'import_share_upper', 'identified_total',
               'total_investment', 'source_records', 'origin_class']
    domestic = ranked.nsmallest(RANKING_SIZE, 'import_share')[columns].assign(side='most_domestic')
    imported = ranked.nlargest(RANKING_SIZE, 'import_share')[columns].assign(side='most_import')
    result = pd.concat([domestic, imported], ignore_index=True)
    result['rank'] = result.groupby('side').cumcount() + 1
    excluded = len(pooled) - len(ranked)
    return result, excluded


def main(base=None, persist=True):
    if persist:
        mark_running('04_discretization_binarization')
    if base is None:
        base = {name: read_table(name, INTERMEDIATE) for name in BASE_TABLES}
    classified = {name: classify(table) for name, table in base.items()}
    for name, table in classified.items():
        keys = ['window'] if name in ('national', 'sector_pooled', 'municipal_pooled') else ['window', 'year']
        table['relative_class'] = relative_class(table, keys)
    counts = class_counts(classified['sector_year'])
    ranking, excluded = municipal_ranking(classified['municipal_pooled'])
    if persist:
        for name, table in classified.items():
            save_frame(table, TABLES / f'{name}_classes.csv')
        save_frame(counts, TABLES / 'class_counts_by_year.csv')
        save_frame(ranking, TABLES / 'municipal_ranking.csv')
        sectors = classified['sector_pooled']
        view = sectors.loc[sectors.window.eq('reported') & sectors.sector_eligible.astype(bool) & sectors.import_share.notna()]
        print(f'C04: import_dominant (> {DOMINANCE_THRESHOLD:.0%}) and three origin classes assigned; '
              f'{excluded} municipalities below {MIN_MUNICIPAL_VALUE:,} EUR left out of the ranking.')
        print(view.sort_values('import_share')[['primary_sector', 'import_share', 'origin_class', 'relative_class', 'class_robust']]
              .assign(primary_sector=lambda t: t.primary_sector.str.slice(0, 40)).round(3).to_string(index=False))
    return classified, counts, ranking


if __name__ == '__main__':
    main()

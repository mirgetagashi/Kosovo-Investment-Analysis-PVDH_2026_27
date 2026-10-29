import numpy as np
import pandas as pd
from phase_1.pipeline_c import (ORIGIN_OF, SHORT_LABELS, INTERMEDIATE, TABLES, eligible_sector, origin_features,
                                origin_shares, read_table, load_input, save_frame, mark_running)


def sector_year_matrix(data, selected):
    """Sector x complete-year matrix of the selected monetary attributes.

    Complete years only (the Jan-Jun 2026 levels are not comparable), eligible sectors only.
    A cell with a negative aggregate cannot be log-transformed and an all-zero row carries no information,
    so both are excluded and counted, never silently altered.
    """
    frame = origin_features(data.assign(year=data.year.astype('int64'), month=data.month.astype('int64')))
    complete = frame.groupby('year').month.nunique().loc[lambda m: m.eq(12)].index
    frame = frame.loc[frame.year.isin(complete) & frame.primary_sector.map(eligible_sector)]
    grouped = frame.groupby(['primary_sector', 'year'])
    values = grouped[selected].sum()
    origin = origin_shares(grouped[['import_total', 'domestic_total', 'mixed_total', 'total_investment']].sum())
    status = pd.Series('valid', index=values.index)
    status[values.sum(axis=1).le(0)] = 'zero_total'
    status[values.lt(0).any(axis=1)] = 'negative_aggregate'
    exclusions = status.value_counts().rename_axis('status').rename('rows').reset_index()
    keep = status.eq('valid')
    return values.loc[keep], origin.loc[keep, ['import_share', 'mixed_share', 'share_status']], exclusions


def transform(values):
    logged = np.log1p(values)
    mean, sd = logged.mean(), logged.std(ddof=1)
    if sd.le(0).any():
        raise ValueError('Constant attribute after log1p; cannot standardize')
    z = (logged - mean) / sd
    skewness = pd.DataFrame(dict(measure=values.columns, skew_raw=values.skew().to_numpy(), skew_log1p=logged.skew().to_numpy(),
                                 mean_log1p=mean.to_numpy(), sd_log1p=sd.to_numpy()))
    return logged, z, skewness


def pca(z):
    """PCA through the SVD of the standardized matrix (equivalent to sklearn.decomposition.PCA).

    Sign convention, so results are reproducible and readable:
    PC1 points towards larger investment (positive loading sum);
    later components point towards the import side (import loadings minus domestic loadings >= 0).
    """
    matrix = z.to_numpy()
    _, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    eigen = singular ** 2 / (len(matrix) - 1)
    origin = np.array([ORIGIN_OF[c] for c in z.columns])
    for k in range(len(vt)):
        contrast = vt[k, origin == 'import'].mean() - vt[k, origin == 'domestic'].mean()
        reference = vt[k].sum() if k == 0 else contrast
        if reference < 0:
            vt[k] = -vt[k]
    names = [f'PC{k + 1}' for k in range(len(vt))]
    scores = pd.DataFrame(matrix @ vt.T, index=z.index, columns=names)
    loadings = pd.DataFrame(vt.T, index=z.columns, columns=names)
    correlation = loadings * np.sqrt(eigen)
    explained = pd.DataFrame(dict(component=names, eigenvalue=eigen, explained_variance_ratio=eigen / eigen.sum(),
                                  cumulative=np.cumsum(eigen) / eigen.sum()))
    return scores, loadings, correlation, explained


def origin_contrast(loadings):
    """Does a component separate imports from domestic purchases? Opposite signs on the two pure groups."""
    origin = loadings.index.map(ORIGIN_OF)
    rows = []
    for component in loadings.columns:
        imported = loadings.loc[origin == 'import', component].mean()
        domestic = loadings.loc[origin == 'domestic', component].mean()
        mixed = loadings.loc[origin == 'mixed', component].mean()
        rows.append(dict(component=component, import_loading=imported, domestic_loading=domestic, mixed_loading=mixed,
                         import_minus_domestic=imported - domestic,
                         separates_import_from_domestic=bool(np.sign(imported) != np.sign(domestic))))
    return pd.DataFrame(rows)


def main(data=None, selection=None, persist=True):
    if persist:
        mark_running('03_transformation_pca')
    if data is None:
        data, _ = load_input()
    if selection is None:
        selection = read_table('feature_selection')
    selected = selection.loc[selection.selected.astype(bool), 'measure'].tolist()
    values, origin, exclusions = sector_year_matrix(data, selected)
    logged, z, skewness = transform(values)
    scores, loadings, correlation, explained = pca(z)
    contrast = origin_contrast(loadings)
    scores = scores.join(origin).reset_index()
    loadings_table = loadings.add_prefix('loading_').join(correlation.add_prefix('correlation_')).rename_axis('measure').reset_index()
    loadings_table.insert(1, 'origin', loadings_table.measure.map(ORIGIN_OF))
    loadings_table.insert(2, 'label', loadings_table.measure.map(SHORT_LABELS))
    if persist:
        save_frame(z.add_suffix('_z').join(logged.add_suffix('_log1p')).reset_index(), INTERMEDIATE / 'pca_matrix.csv')
        for name, table in [('transformation_skewness', skewness), ('pca_explained_variance', explained),
                            ('pca_loadings', loadings_table), ('pca_scores', scores), ('pca_origin_contrast', contrast),
                            ('pca_exclusions', exclusions)]:
            save_frame(table, TABLES / f'{name}.csv')
        print(f'C03: log1p + z-score on {len(selected)} attributes, PCA on {len(z)} sector-year rows (complete years).')
        print(explained.round(3).to_string(index=False))
        print(contrast.round(3).to_string(index=False))
    return scores, loadings_table, explained, contrast


if __name__ == '__main__':
    main()

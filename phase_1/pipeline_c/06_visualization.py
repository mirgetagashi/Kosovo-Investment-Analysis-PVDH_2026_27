import importlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Patch, PathPatch
from phase_1.pipeline_c import (CLASS_LABELS, RELATIVE_LABELS, MIN_MUNICIPAL_VALUE, LOW_COVERAGE, FIGURES, TABLES,
                                short_sector, read_table, mark_running)

# Same boundary file, hash check and name crosswalk as Pipeline B, so both maps use one verified reference.
geography = importlib.import_module('phase_1.pipeline_b.06_visualization')

SURFACE, INK, SECONDARY, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
DOMESTIC, IMPORT, NEUTRAL, UNKNOWN = '#2a78d6', '#e34948', '#a3a199', '#d6d4cc'
SERIES = ['#2a78d6', '#eb6834']
CLASS_COLORS = dict(zip(CLASS_LABELS, [DOMESTIC, NEUTRAL, IMPORT]))
CLASS_NAMES = dict(zip(CLASS_LABELS, ['Mostly domestic (< 33%)', 'Balanced (33–67%)', 'Mostly import (> 67%)']))

plt.rcParams.update({'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.edgecolor': AXIS, 'axes.labelcolor': SECONDARY, 'xtick.color': MUTED, 'ytick.color': SECONDARY,
    'text.color': INK, 'axes.titlesize': 13, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
    'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': False, 'legend.frameon': False})


def percent_axis(axis, which='x'):
    formatter = plt.FuncFormatter(lambda v, _: f'{v:.0%}')
    (axis.xaxis if which == 'x' else axis.yaxis).set_major_formatter(formatter)


def class_legend(axis, **kwargs):
    axis.legend(handles=[Patch(color=CLASS_COLORS[c], label=CLASS_NAMES[c]) for c in CLASS_LABELS], **kwargs)


def save(figure, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f'{name}.png'
    figure.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(figure)
    return path


def sector_order(bootstrap):
    view = bootstrap.loc[bootstrap.sector_eligible.astype(bool) & bootstrap.status.eq('valid')].copy()
    return view.sort_values('import_share').reset_index(drop=True)


def sector_share_chart(bootstrap, sectors):
    view = sector_order(bootstrap)
    coverage = sectors.loc[sectors.window.eq('reported')].set_index('primary_sector').identified_coverage
    view['coverage'] = view.primary_sector.map(coverage)
    low = view.coverage.lt(LOW_COVERAGE)
    figure, axis = plt.subplots(figsize=(10, 0.38 * len(view) + 1.6))
    y = np.arange(len(view))
    bars = axis.barh(y, view.import_share, height=.62, color=view.origin_class.map(CLASS_COLORS), edgecolor=SURFACE, linewidth=1)
    for bar, flagged in zip(bars, low):
        if flagged:   # a share that describes only a small part of the sector's money: hatched, lighter
            bar.set_alpha(.45)
            bar.set_hatch('////')
            bar.set_edgecolor(INK)
            bar.set_linewidth(0)
    axis.errorbar(view.import_share, y, xerr=[view.import_share - view.ci_low, view.ci_high - view.import_share],
                  fmt='none', ecolor=INK, elinewidth=1, capsize=2.5)
    for k, row in view.iterrows():
        note = f'   origin known for only {row.coverage:.0%} of the money' if low[k] else ''
        axis.text(row.ci_high + .012, k, f'{row.import_share:.0%}{note}', va='center', fontsize=8.5,
                  color=IMPORT if low[k] else SECONDARY)
    for line in (1 / 3, 2 / 3):
        axis.axvline(line, color=GRID, linewidth=1, zorder=0)
    axis.axvline(.5, color=MUTED, linewidth=1, linestyle='--', zorder=0)
    axis.set_yticks(y, view.primary_sector.map(short_sector))
    axis.set_xlim(0, 1.08)
    percent_axis(axis)
    axis.set_xlabel('Import share of identified-origin investment  =  imports / (imports + domestic purchases)')
    axis.set_title('Import share of business investment by sector, 2019–2026')
    handles = [Patch(color=CLASS_COLORS[c], label=CLASS_NAMES[c]) for c in CLASS_LABELS]
    handles.append(Patch(facecolor=DOMESTIC, alpha=.45, hatch='////', edgecolor=INK, linewidth=0,
                         label=f'Origin known for < {LOW_COVERAGE:.0%} of the money: not comparable'))
    axis.legend(handles=handles, loc='center right', fontsize=8.5)
    figure.tight_layout()
    figure.text(.01, -.01, 'Whiskers: 95% year-stratified bootstrap interval (random variation only, not the unknown origin '
                'of the mixed columns). Dashed line: 50% (import-dominant). Unclassified and "Person Fizik" labels excluded.',
                fontsize=8, color=MUTED)
    return save(figure, 'c1_sector_import_share')


def composition_chart(sectors, bootstrap):
    order = sector_order(bootstrap).primary_sector
    view = sectors.loc[sectors.window.eq('reported')].set_index('primary_sector').loc[order]
    parts = pd.DataFrame(dict(imports=view.import_total, unknown=view.mixed_total, domestic=view.domestic_total))
    parts = parts.div(parts.sum(axis=1), axis=0)
    figure, axis = plt.subplots(figsize=(10, 0.38 * len(view) + 1.6))
    y = np.arange(len(view))
    left = np.zeros(len(view))
    for column, color, label in [('imports', IMPORT, 'Investment imports (18% + 8%)'),
                                 ('unknown', UNKNOWN, 'Purchases & imports, origin not separable (no VAT / non-deductible VAT)'),
                                 ('domestic', DOMESTIC, 'Domestic investment purchases (18% + 8%)')]:
        axis.barh(y, parts[column], left=left, height=.62, color=color, edgecolor=SURFACE, linewidth=1.5, label=label)
        left += parts[column].to_numpy()
    axis.set_yticks(y, [short_sector(s) for s in order])
    axis.set_xlim(0, 1)
    percent_axis(axis)
    axis.set_xlabel('Share of all reported investment')
    axis.set_title('Where the investment money goes: import, domestic or not separable')
    axis.legend(loc='upper center', bbox_to_anchor=(.42, -.1), ncol=1, fontsize=8.5)
    return save(figure, 'c2_sector_origin_composition')


def national_chart(national):
    figure, axis = plt.subplots(figsize=(9, 4.8))
    reported = national.loc[national.window.eq('reported')].sort_values('year')
    jan_jun = national.loc[national.window.eq('jan_jun')].sort_values('year')
    axis.fill_between(reported.year, reported.import_share_lower, reported.import_share_upper, color=GRID, alpha=.7, linewidth=0,
                      label='Bounds: mixed amounts all domestic … all imported')
    for table, color, label in [(reported, SERIES[0], 'Reported year (2026 = Jan–Jun)'), (jan_jun, SERIES[1], 'January–June only')]:
        axis.plot(table.year, table.import_share, color=color, linewidth=2, marker='o', markersize=6, label=label)
    partial = reported.loc[reported.partial_year.astype(bool)]
    axis.scatter(partial.year, partial.import_share, s=60, facecolor=SURFACE, edgecolor=SERIES[0], linewidth=2, zorder=3)
    axis.axhline(.5, color=MUTED, linewidth=1, linestyle='--', zorder=0)
    axis.set_ylim(0, 1)
    axis.grid(axis='y', color=GRID, linewidth=.8)
    axis.set_axisbelow(True)
    percent_axis(axis, 'y')
    axis.set_xticks(reported.year)
    axis.set_ylabel('Import share of identified-origin investment')
    axis.set_title('Import share of business investment in Kosovo, 2019–2026')
    axis.legend(loc='upper right', fontsize=8.5)
    return save(figure, 'c3_national_import_share')


def municipal_chart(ranking):
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True)
    for axis, side, color, title in [(axes[0], 'most_domestic', DOMESTIC, 'Most domestic'),
                                     (axes[1], 'most_import', IMPORT, 'Most import-oriented')]:
        view = ranking.loc[ranking.side.eq(side)].sort_values('rank', ascending=False)
        axis.barh(view.municipality, view.import_share, height=.62, color=color, edgecolor=SURFACE)
        for k, value in enumerate(view.import_share):
            axis.text(value + .01, k, f'{value:.0%}', va='center', fontsize=8.5, color=SECONDARY)
        axis.axvline(.5, color=MUTED, linewidth=1, linestyle='--', zorder=0)
        axis.set_xlim(0, 1.05)
        percent_axis(axis)
        axis.set_title(title, fontsize=11)
    figure.suptitle('Municipalities by import share of investment, 2019–2026', x=.06, ha='left', fontweight='bold', fontsize=13)
    figure.supxlabel('Import share of identified-origin investment (municipalities with ≥ 1 M EUR identified investment)',
                     fontsize=9, color=SECONDARY)
    figure.tight_layout()
    return save(figure, 'c4_municipal_ranking')


def biplot(scores, loadings, explained):
    figure, axis = plt.subplots(figsize=(8.5, 7))
    valid = scores.loc[scores.import_share.notna()]
    classes = pd.cut(valid.import_share, bins=[0, 1 / 3, 2 / 3, 1], labels=CLASS_LABELS, include_lowest=True)
    axis.scatter(valid.PC1, valid.PC2, s=40, c=classes.map(CLASS_COLORS).astype(str), edgecolor=SURFACE, linewidth=1, alpha=.9)
    scale = np.abs(valid[['PC1', 'PC2']]).max().min() * .9
    tips = pd.DataFrame(dict(label=loadings.label, x=loadings.correlation_PC1 * scale, y=loadings.correlation_PC2 * scale))
    for _, row in tips.iterrows():
        axis.annotate('', xy=(row.x, row.y), xytext=(0, 0), arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.5))
    tips = tips.assign(ty=tips.y * 1.12).sort_values('ty')
    gap = .09 * (axis.get_ylim()[1] - axis.get_ylim()[0])
    for k in range(1, len(tips)):   # push labels apart vertically when arrows point the same way
        if abs(tips.x.iloc[k] - tips.x.iloc[k - 1]) < 1.5 and tips.ty.iloc[k] - tips.ty.iloc[k - 1] < gap:
            tips.iloc[k, tips.columns.get_loc('ty')] = tips.ty.iloc[k - 1] + gap
    for _, row in tips.iterrows():
        axis.text(row.x * 1.12, row.ty, row.label, fontsize=9, color=INK, ha='left', va='center',
                  bbox=dict(boxstyle='round,pad=.2', facecolor=SURFACE, edgecolor='none', alpha=.85))
    axis.axhline(0, color=GRID, linewidth=1, zorder=0)
    axis.axvline(0, color=GRID, linewidth=1, zorder=0)
    ratio = explained.set_index('component').explained_variance_ratio
    axis.set_xlabel(f'PC1 ({ratio["PC1"]:.0%} of variance)')
    axis.set_ylabel(f'PC2 ({ratio["PC2"]:.0%} of variance)')
    axis.set_title('PCA of sector-year investment (log1p, standardized), 2019–2025')
    class_legend(axis, loc='lower left', fontsize=8.5, title='Sector-year import share', title_fontsize=8.5)
    return save(figure, 'c5_pca_biplot')


def ring_centroid(ring):
    """Area-weighted centroid of a polygon ring (shoelace formula)."""
    x, y = np.asarray(ring, float).T
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    area = cross.sum() / 2
    if abs(area) < 1e-12:
        return x.mean(), y.mean()
    return ((x + np.roll(x, -1)) * cross).sum() / (6 * area), ((y + np.roll(y, -1)) * cross).sum() / (6 * area)


def feature_polygons(feature):
    geometry = feature['geometry']
    return [geometry['coordinates']] if geometry['type'] == 'Polygon' else geometry['coordinates']


def label_point(feature):
    """Centroid of the largest polygon's outer ring, so the label sits inside the municipality."""
    def area(ring):
        x, y = np.asarray(ring, float).T
        return abs((x * np.roll(y, -1) - np.roll(x, -1) * y).sum()) / 2
    return ring_centroid(max((rings[0] for rings in feature_polygons(feature)), key=area))


# Small municipalities whose labels would overlap a neighbour: label placed outside with a leader line (degrees).
LABEL_OFFSETS = {'JUNIK': (-.2, -.06), 'MAMUSHË': (-.22, -.1), 'KLLOKOT': (.3, -.06), 'PARTESH': (.3, .05),
                 'MITROVICË VERIORE': (-.5, .17)}


def draw_map(axis, boundaries, names, colors, labels):
    for feature in boundaries['features']:
        municipality = names[feature['properties']['shapeName']]
        for rings in feature_polygons(feature):
            axis.add_patch(PathPatch(geography.polygon_path(rings), facecolor=colors[municipality],
                                     edgecolor=SURFACE, linewidth=.8))
        x, y = label_point(feature)
        if municipality in LABEL_OFFSETS:
            dx, dy = LABEL_OFFSETS[municipality]
            axis.annotate(labels[municipality], xy=(x, y), xytext=(x + dx, y + dy), ha='center', va='center',
                          fontsize=5.6, color=INK, linespacing=.95,
                          arrowprops=dict(arrowstyle='-', color=SECONDARY, linewidth=.6, shrinkA=2, shrinkB=0))
        else:
            axis.text(x, y, labels[municipality], ha='center', va='center', fontsize=5.6, color=INK, linespacing=.95)
    axis.autoscale_view()
    axis.set_aspect(1 / np.cos(np.deg2rad(42.6)))
    axis.axis('off')


def municipal_map(municipal):
    """Pooled 2019-2026 import share per municipality, absolute (left) and relative tertiles (right)."""
    view = municipal.loc[municipal.window.eq('reported')].set_index('municipality')
    boundaries, crosswalk = geography.load_boundaries(view.index)
    names = crosswalk.set_index('boundary_name').municipality.to_dict()
    shown = view.import_share.notna() & view.identified_total.ge(MIN_MUNICIPAL_VALUE)
    national = view.import_total.sum() / (view.import_total.sum() + view.domestic_total.sum())
    top = max(.5, np.ceil(view.loc[shown, 'import_share'].max() * 10) / 10)
    cmap = LinearSegmentedColormap.from_list('origin', [DOMESTIC, '#f2f1ec', IMPORT])
    color_norm = TwoSlopeNorm(vmin=0, vcenter=national, vmax=top)
    short = {m: m.title().replace(' ', '\n', 1) if len(m) > 11 else m.title() for m in view.index}

    figure, axes = plt.subplots(1, 2, figsize=(15, 8.4))
    colors = {m: cmap(color_norm(view.import_share[m])) if shown[m] else UNKNOWN for m in view.index}
    labels = {m: f'{short[m]}\n{view.import_share[m]:.0%}' if shown[m] else short[m] for m in view.index}
    draw_map(axes[0], boundaries, names, colors, labels)
    axes[0].set_title('Import share of identified-origin investment', fontsize=12)
    bar = figure.colorbar(plt.cm.ScalarMappable(norm=color_norm, cmap=cmap), ax=axes[0], orientation='horizontal',
                          fraction=.04, pad=.02, format=plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
    bar.ax.axvline(national, color=INK, linewidth=1.5)
    bar.set_label(f'Colour centre = Kosovo average ({national:.0%}); blue below, red above', color=SECONDARY, fontsize=9)
    bar.outline.set_visible(False)

    relative = dict(zip(RELATIVE_LABELS, [DOMESTIC, NEUTRAL, IMPORT]))
    relative_names = dict(zip(RELATIVE_LABELS, ['Lower third', 'Middle third', 'Upper third']))
    colors = {m: relative[view.relative_class[m]] if shown[m] and isinstance(view.relative_class[m], str) else UNKNOWN
              for m in view.index}
    draw_map(axes[1], boundaries, names, colors, short)
    axes[1].set_title('Relative import reliance (equal-frequency tertiles)', fontsize=12)
    axes[1].legend(handles=[Patch(color=relative[c], label=relative_names[c]) for c in RELATIVE_LABELS]
                   + [Patch(color=UNKNOWN, label=f'< {MIN_MUNICIPAL_VALUE / 1e6:g} M EUR identified: not compared')],
                   loc='upper right', fontsize=8.5)
    figure.suptitle('Where business investment relies on imports, by municipality (2019–2026 pooled)',
                    x=.03, ha='left', fontweight='bold', fontsize=14)
    figure.text(.03, .01, 'Import share = imports / (imports + domestic purchases); the mixed columns are excluded from both. '
                'Boundaries: geoBoundaries / OpenStreetMap (CC BY-SA 2.0), as in Pipeline B.', fontsize=8, color=MUTED)
    figure.subplots_adjust(left=.02, right=.98, top=.9, bottom=.08, wspace=.04)
    return save(figure, 'c6_municipal_import_share_map')


def sample_sensitivity_chart(summary, sectors):
    """How far a 10% sample lands from the full-data import share, per design."""
    designs = ['simple_random', 'stratified_year_sector', 'take_all_large']
    colors = dict(zip(designs, [NEUTRAL, SERIES[0], SERIES[1]]))
    names = dict(zip(designs, ['Simple random 10%', 'Stratified by year × sector', 'Take-all 1% largest + stratified rest']))
    figure, (left, right) = plt.subplots(1, 2, figsize=(14, 7.5), gridspec_kw=dict(width_ratios=[1, 2.3]))
    levels = ['national_year', 'sector_pooled']
    x = np.arange(len(levels))
    for k, design in enumerate(designs):
        view = summary.set_index(['method', 'level']).loc[design].reindex(levels)
        left.bar(x + (k - 1) * .26, view.mean_abs_error_pp, width=.24, color=colors[design], label=names[design])
        for xi, value in zip(x + (k - 1) * .26, view.mean_abs_error_pp):
            left.text(xi, value + .05, f'{value:.1f}', ha='center', fontsize=8.5, color=SECONDARY)
    left.set_xticks(x, ['National,\nper year', 'Sector,\n2019–2026 pooled'])
    left.set_ylabel('Mean absolute error (percentage points)')
    left.set_title('Average error of the import share', fontsize=11)
    left.set_ylim(0, summary.mean_abs_error_pp.max() * 1.45)
    left.legend(loc='upper left', fontsize=8.5)

    table = sectors.pivot(index='primary_sector', columns='method', values='mean_abs_error_pp')
    records = sectors.groupby('primary_sector').records.first()
    table = table.loc[records.sort_values().index]
    y = np.arange(len(table))
    right.hlines(y, table[designs].min(axis=1), table[designs].max(axis=1), color=GRID, linewidth=2, zorder=0)
    for design in designs:
        right.scatter(table[design], y, s=36, color=colors[design], label=names[design], zorder=3, edgecolor=SURFACE)
    right.set_yticks(y, [f'{short_sector(s, 30)}  ({records[s]:,})' for s in table.index], fontsize=8.5)
    right.set_xlabel('Mean absolute error (percentage points)')
    right.set_title('Per sector (rows in the full data); smallest sectors at the bottom', fontsize=11)
    right.grid(axis='x', color=GRID, linewidth=.8)
    right.set_axisbelow(True)
    figure.suptitle('Sampling: how close does a 10% sample get to the full-data import share?',
                    x=.03, ha='left', fontweight='bold', fontsize=14)
    first = summary.iloc[0]
    figure.text(.03, -.01, f'{int(first.draws)} draws per design, the same ~{first.sampled_records_mean:,.0f} rows each, seed {int(first.seed)}. '
                'The 1% largest rows hold about half of the identified investment, so including them always matters more than balancing sectors.',
                fontsize=8, color=MUTED)
    figure.tight_layout()
    return save(figure, 'c7_sample_sensitivity')


def main(persist=True):
    if persist:
        mark_running('06_visualization')
    bootstrap = read_table('bootstrap_ci')
    sectors = read_table('sector_pooled_classes')
    national = read_table('national_classes')
    ranking = read_table('municipal_ranking')
    scores, loadings, explained = read_table('pca_scores'), read_table('pca_loadings'), read_table('pca_explained_variance')
    municipal = read_table('municipal_pooled_classes')
    sampling, sampling_sectors = read_table('sample_sensitivity_summary'), read_table('sample_sensitivity_sectors')
    paths = [sector_share_chart(bootstrap, sectors), composition_chart(sectors, bootstrap), national_chart(national),
             municipal_chart(ranking), biplot(scores, loadings, explained), municipal_map(municipal),
             sample_sensitivity_chart(sampling, sampling_sectors)]
    print(f'C06: {len(paths)} figures saved to {FIGURES.relative_to(FIGURES.parents[2])}.')
    return paths


if __name__ == '__main__':
    main()

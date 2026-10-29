import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from phase_1.pipeline_c import (CLASS_LABELS, FIGURES, TABLES, short_sector, read_table, mark_running)

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


def sector_share_chart(bootstrap):
    view = sector_order(bootstrap)
    figure, axis = plt.subplots(figsize=(10, 0.38 * len(view) + 1.6))
    y = np.arange(len(view))
    axis.barh(y, view.import_share, height=.62, color=view.origin_class.map(CLASS_COLORS), edgecolor=SURFACE, linewidth=1)
    axis.errorbar(view.import_share, y, xerr=[view.import_share - view.ci_low, view.ci_high - view.import_share],
                  fmt='none', ecolor=INK, elinewidth=1, capsize=2.5)
    for k, row in view.iterrows():
        axis.text(row.ci_high + .012, k, f'{row.import_share:.0%}', va='center', fontsize=8.5, color=SECONDARY)
    for line in (1 / 3, 2 / 3):
        axis.axvline(line, color=GRID, linewidth=1, zorder=0)
    axis.axvline(.5, color=MUTED, linewidth=1, linestyle='--', zorder=0)
    axis.set_yticks(y, view.primary_sector.map(short_sector))
    axis.set_xlim(0, 1.08)
    percent_axis(axis)
    axis.set_xlabel('Import share of identified-origin investment  =  imports / (imports + domestic purchases)')
    axis.set_title('Import share of business investment by sector, 2019–2026')
    class_legend(axis, loc='lower right', fontsize=8.5)
    figure.tight_layout()
    figure.text(.01, -.01, 'Whiskers: 95% year-stratified bootstrap interval. Dashed line: 50% (import-dominant). '
                'Unclassified and "Person Fizik" labels excluded.', fontsize=8, color=MUTED)
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


def main(persist=True):
    if persist:
        mark_running('06_visualization')
    bootstrap = read_table('bootstrap_ci')
    sectors = read_table('sector_pooled_classes')
    national = read_table('national_classes')
    ranking = read_table('municipal_ranking')
    scores, loadings, explained = read_table('pca_scores'), read_table('pca_loadings'), read_table('pca_explained_variance')
    paths = [sector_share_chart(bootstrap), composition_chart(sectors, bootstrap), national_chart(national),
             municipal_chart(ranking), biplot(scores, loadings, explained)]
    print(f'C06: {len(paths)} figures saved to {FIGURES.relative_to(FIGURES.parents[2])}.')
    return paths


if __name__ == '__main__':
    main()

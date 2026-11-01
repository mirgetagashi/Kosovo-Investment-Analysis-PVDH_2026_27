import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.path import Path as PlotPath
from matplotlib.patches import PathPatch
from phase_1.pipeline_b import (MONEY, MEASURE_LABELS, EXCLUDED_SECTORS, TABLES,
    FIGURES, REFERENCE, read_table, save_frame, digest, norm, mark_running)


def load_boundaries(municipalities):
    metadata = json.loads((REFERENCE / 'kosovo_municipalities_source.json').read_text(encoding='utf-8'))
    geometry = REFERENCE / 'kosovo_municipalities.geojson'
    if digest(geometry) != metadata['download_sha256']:
        raise ValueError('Boundary file hash changed')
    boundaries = json.loads(geometry.read_text(encoding='utf-8'))
    crosswalk = pd.read_csv(REFERENCE / 'municipality_crosswalk.csv')
    names = [f['properties']['shapeName'] for f in boundaries['features']]
    if (crosswalk.municipality.duplicated().any() or crosswalk.boundary_name.duplicated().any()
            or len(names) != len(set(names)) or set(crosswalk.municipality) != set(municipalities)
            or set(names) != set(crosswalk.boundary_name)):
        raise ValueError('Map must match every source municipality to exactly one real polygon')
    if any(f['geometry']['type'] not in ['Polygon', 'MultiPolygon'] for f in boundaries['features']):
        raise ValueError('Unexpected boundary geometry')
    return boundaries, crosswalk


def finish(fig, name, title, note):
    fig.suptitle(title, fontsize=17, fontweight='bold')
    fig.text(.5, .012, note, ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .04, 1, .95])
    fig.savefig(FIGURES / name, dpi=160)
    plt.close(fig)


def polygon_path(rings):
    vertices, codes = [], []
    for ring in rings:
        vertices.extend(ring)
        codes.extend([PlotPath.MOVETO] + [PlotPath.LINETO] * (len(ring)-2) + [PlotPath.CLOSEPOLY])
    return PlotPath(vertices, codes)


def plot_maps(municipal, year, boundaries, crosswalk):
    fig, axes = plt.subplots(2, 3, figsize=(16, 12))
    cmap = plt.get_cmap('YlGnBu')
    period = municipal.loc[(municipal.window == 'reported') & (municipal.year == year)]
    vmax = max(.01, float(period.municipal_share.max()) * 100)
    color_norm = Normalize(0, vmax)
    names = crosswalk.set_index('boundary_name').municipality.to_dict()
    for ax, measure in zip(axes.flat, MONEY):
        shares = period.loc[period.measure.eq(measure)].set_index('municipality').municipal_share
        for feature in boundaries['features']:
            share = shares.loc[names[feature['properties']['shapeName']]]
            color = cmap(color_norm(share*100)) if np.isfinite(share) else '#cccccc'
            geometry = feature['geometry']
            polygons = [geometry['coordinates']] if geometry['type'] == 'Polygon' else geometry['coordinates']
            for rings in polygons:
                ax.add_patch(PathPatch(polygon_path(rings), facecolor=color, edgecolor='white', linewidth=.45))
        ax.autoscale_view()
        ax.set_aspect(1/np.cos(np.deg2rad(42.6)))
        ax.set_title(MEASURE_LABELS[measure], fontsize=11)
        ax.axis('off')
    fig.colorbar(plt.cm.ScalarMappable(norm=color_norm, cmap=cmap), ax=list(axes.flat),
                 location='bottom', fraction=.025, pad=.03, label='Municipal share (%)')
    fig.suptitle(f'Geographic concentration | {year}, full reporting year', fontsize=17)
    fig.text(.5, .018, 'geoBoundaries / OpenStreetMap / Wambacher; 2017-represented boundaries; CC BY-SA 2.0. Static display, no population adjustment.', ha='center', fontsize=8)
    fig.subplots_adjust(top=.92, bottom=.13, hspace=.12, wspace=.05)
    fig.savefig(FIGURES / '01_geographic_share.png', dpi=160)
    plt.close(fig)


def plot_pareto(municipal, year):
    fig, axes = plt.subplots(2, 3, figsize=(21, 12))
    for ax, measure in zip(axes.flat, MONEY):
        table = municipal.loc[(municipal.window == 'reported') & (municipal.year == year) & municipal.measure.eq(measure)].sort_values(['value', 'municipality'], ascending=[False, True])
        positions = np.arange(len(table))
        ax.bar(positions, table.value / 1e6, color='#287e8e')
        twin = ax.twinx()
        twin.plot(positions, 100*table.cumulative_share, color='#d85c38', linewidth=1.7)
        twin.axhline(80, color='#d85c38', linestyle=':', linewidth=1)
        twin.set(ylim=(0, 105), ylabel='Cumulative share (%)')
        ax.set(xticks=positions, xticklabels=table.municipality, ylabel='Reported value (millions; unit unconfirmed)', title=MEASURE_LABELS[measure])
        ax.tick_params(axis='x', labelrotation=90, labelsize=6)
    finish(fig, '02_municipal_pareto.png', f'Municipal Pareto distributions | {year}',
           'Each VAT-related measure is separate. The cumulative line includes the municipality crossing 80%.')


def plot_hhi(summary):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, measure in zip(axes.flat, MONEY):
        group = summary.loc[summary.measure.eq(measure)]
        annual = group.loc[group.window.eq('reported') & ~group.partial_year]
        balanced = group.loc[group.window.eq('jan_jun')]
        partial = group.loc[group.window.eq('reported') & group.partial_year]
        ax.plot(annual.year, annual.hhi, 'o-', label='Full years', color='#287e8e')
        ax.plot(balanced.year, balanced.hhi, 's--', label='Jan-Jun, all years', color='#d85c38', markersize=4)
        ax.scatter(partial.year, partial.hhi, facecolors='none', edgecolors='#287e8e', s=90, label='Partial reported year')
        ax.axhline(group.equal_share_reference.iloc[0], color='grey', linestyle=':', label='Equal shares across 38')
        ax.set(title=MEASURE_LABELS[measure], ylabel='HHI (fraction scale: 0-1)', xticks=balanced.year)
        ax.tick_params(axis='x', labelrotation=45)
    axes[0, 0].legend(fontsize=7)
    finish(fig, '03_hhi_evolution.png', 'Geographic concentration over time',
           '2026 = Jan-Jun. Do not compare its six-month totals with twelve-month years. No competition-policy thresholds are applied.')


def plot_specialization(lq, dictionary, year):
    fig, axes = plt.subplots(2, 3, figsize=(23, 18))
    eligible = dictionary.loc[dictionary.sector_eligible]
    codes = eligible.set_index('primary_sector').sector_code.to_dict()
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad('#dedede')
    for ax, measure in zip(axes.flat, MONEY):
        group = lq.loc[lq.sector_eligible & lq.window.eq('reported') & lq.year.eq(year) & lq.measure.eq(measure)]
        matrix = group.pivot(index='municipality', columns='primary_sector', values='location_quotient').sort_index()
        transformed = np.log2(matrix.clip(lower=.125, upper=8))
        image = ax.imshow(transformed, cmap=cmap, vmin=-3, vmax=3, aspect='auto')
        ax.set(title=MEASURE_LABELS[measure], xticks=range(len(matrix.columns)),
               xticklabels=[codes[s] for s in matrix.columns], yticks=range(len(matrix)), yticklabels=matrix.index)
        ax.tick_params(axis='x', labelrotation=90, labelsize=7)
        ax.tick_params(axis='y', labelsize=7)
        if matrix.isna().all().all():
            reason = group.lq_status.value_counts().index[0].replace('_', ' ')
            ax.text(.5, .5, 'LQ undefined\n' + reason, transform=ax.transAxes,
                    ha='center', va='center', fontsize=11, color='#444444')
    fig.subplots_adjust(left=.09, right=.95, top=.93, bottom=.1, hspace=.13, wspace=.39)
    cax = fig.add_axes([.25, .045, .5, .012])
    colorbar = fig.colorbar(image, cax=cax, orientation='horizontal', ticks=[-3, -2, -1, 0, 1, 2, 3])
    colorbar.ax.set_xticklabels(['<=0.125', '0.25', '0.5', '1', '2', '4', '>=8'])
    colorbar.set_label('Location quotient; log2 colour scale only. Grey = undefined.')
    fig.suptitle(f'Conditional sector specialization | {year}', fontsize=18)
    fig.text(.5, .012, 'Codes: sector_dictionary.csv. Unavailable/ambiguous sectors excluded from both denominators; original amounts retained. Any negative eligible cell invalidates that period/measure LQ block.', ha='center', fontsize=9)
    fig.savefig(FIGURES / '04_sector_specialization.png', dpi=160)
    plt.close(fig)


def plot_comparison(profiles, year):
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    for ax, measure in zip(axes.flat, MONEY):
        table = profiles.loc[profiles.window.eq('reported') & profiles.year.eq(year) & profiles.measure.eq(measure)]
        valid = table.loc[table.profile_status.eq('valid') & table.max_lq.gt(0)]
        ax.scatter(valid.value, valid.max_lq, c=valid.sector_hhi, cmap='viridis', vmin=0, vmax=1, s=45)
        for row in pd.concat([valid.nlargest(2, 'value'), valid.nlargest(2, 'max_lq')]).drop_duplicates('municipality').itertuples():
            ax.annotate(row.municipality, (row.value, row.max_lq), xytext=(3, 3), textcoords='offset points', fontsize=7)
        ax.set_xscale('symlog', linthresh=1)
        if valid.empty:
            ax.set_ylim(.5, 2)
        ax.set_yscale('log')
        ax.set(title=MEASURE_LABELS[measure], xlabel='All-sector municipal value (symlog; unit unconfirmed)', ylabel='Maximum eligible-sector LQ (log)')
        if valid.empty:
            ax.text(.5, .5, 'LQ undefined for this measure/year', transform=ax.transAxes, ha='center', fontsize=9)
    fig.suptitle(f'Absolute reported value versus relative specialization | {year}', fontsize=17)
    fig.tight_layout(rect=[0, .12, 1, .95])
    cax = fig.add_axes([.3, .06, .4, .015])
    fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap='viridis'), cax=cax,
                 orientation='horizontal', label='Eligible-sector HHI: lower = more dispersed sector mix')
    fig.text(.5, .015, 'High LQ can reflect a small base; inspect amounts and coverage in profiles.csv and top_specialization.csv.', ha='center', fontsize=9)
    fig.savefig(FIGURES / '05_municipal_profiles.png', dpi=160)
    plt.close(fig)


def main(results=None):
    mark_running('06_visualization')
    if results is None:
        results = {name:read_table(name) for name in ['municipal', 'concentration', 'specialization', 'profiles']}
    boundaries, crosswalk = load_boundaries(results['municipal'].municipality.unique())
    dictionary = pd.DataFrame({'primary_sector':sorted(results['specialization'].primary_sector.unique())})
    dictionary['sector_code'] = [f'S{i:02d}' for i in range(1, len(dictionary)+1)]
    dictionary['sector_eligible'] = dictionary.primary_sector.map(lambda s:norm(s) not in EXCLUDED_SECTORS)
    save_frame(dictionary, TABLES / 'sector_dictionary.csv')
    years = results['concentration'].loc[lambda t:t.window.eq('reported') & t.months.eq(12), 'year']
    if years.empty:
        raise ValueError('No complete year available for static comparison figures')
    latest = int(years.max())
    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_maps(results['municipal'], latest, boundaries, crosswalk)
    plot_pareto(results['municipal'], latest)
    plot_hhi(results['concentration'])
    plot_specialization(results['specialization'], dictionary, latest)
    plot_comparison(results['profiles'], latest)
    print(f'B06: five figures saved for six separate measures; {len(crosswalk)} map matches.')
    return latest, len(crosswalk)


if __name__ == '__main__':
    main()

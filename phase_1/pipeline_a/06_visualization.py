import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import offset_copy
from src.schema import MONEY
from phase_1.pipeline_a import (FIGURES, MEASURES, MEASURE_LABELS, TOTAL, GROWTH_LABELS, BASE_YEAR,
                                LOCKDOWN_START, LOCKDOWN_END, COVID_START, COVID_END, read_table, mark_running)

MAIN = [TOTAL, MONEY[4]]

# Same palette and typography as Pipeline C, so the three pipelines read as one report.
SURFACE, INK, SECONDARY, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
BLUE, RED, NEUTRAL, LIGHT = '#2a78d6', '#e34948', '#a3a199', '#d6d4cc'
SERIES = ['#2a78d6', '#eb6834']
# Muted, distinct colours for the six source amounts, which stay in the background behind the total
COMPONENTS = dict(zip(MONEY, ['#b5b2a8', '#c9a27e', '#e8a0a0', '#b7a6d6', '#7fbfae', '#d9c27a']))
SHORT_LABELS = dict(zip(MONEY, ['Mixed: no VAT', 'Mixed: non-deductible VAT', 'Imports 18%', 'Imports 8%',
                                'Domestic 18%', 'Domestic 8%']))
GROWTH_COLORS = dict(zip(GROWTH_LABELS, [RED, '#f0a3a2', NEUTRAL, '#95bbea', BLUE]))
GROWTH_NAMES = dict(zip(GROWTH_LABELS, ['Strong decline\n< −20%', 'Decline\n−20% to −5%', 'Stable\n±5%',
                                        'Growth\n5% to 20%', 'Strong growth\n> 20%']))
SOURCE = 'Source: ATK open data, investment reported in VAT declarations, 2019 to June 2026.'
MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

plt.rcParams.update({'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.edgecolor': AXIS, 'axes.labelcolor': SECONDARY, 'xtick.color': MUTED, 'ytick.color': SECONDARY,
    'text.color': INK, 'axes.titlesize': 12, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
    'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': False, 'legend.frameon': False})


def headline(figure, title, subtitle):
    """Figure title states the finding; the subtitle says what is measured. Both sit above the plotting area in
    points, so they never collide with panel titles whatever the figure height."""
    above = lambda points: offset_copy(figure.transFigure, fig=figure, y=points, units='points')
    figure.text(.01, 1, subtitle, transform=above(6), ha='left', va='bottom', fontsize=10, color=SECONDARY)
    figure.text(.01, 1, title, transform=above(24), ha='left', va='bottom', fontsize=15, fontweight='bold')


def footnote(figure, text):
    figure.text(.01, -.01, f'{text} {SOURCE}'.strip(), ha='left', va='top', fontsize=8, color=MUTED, wrap=True)


def save(figure, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / f'{name}.png', dpi=160, bbox_inches='tight')
    plt.close(figure)


def y_grid(axis):
    axis.grid(axis='y', color=GRID, linewidth=.8)
    axis.set_axisbelow(True)


def short_sector(label, width=34):
    text = str(label).split(';')[0].split(',')[0].strip()
    return text if len(text) <= width else text[:width - 1] + '…'


def index_lines(annual_index):
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    complete = annual_index.loc[annual_index.window.eq('reported') & ~annual_index.partial_year]
    last = complete.loc[complete.measure.eq(TOTAL)].sort_values('year').iloc[-1]
    for axis, window, title in [(axes[0], 'reported', 'Complete years, 2019–2025'),
                                (axes[1], 'jan_jun', 'January–June of every year, 2019–2026')]:
        table = complete if window == 'reported' else annual_index.loc[annual_index.window.eq(window)]
        for measure in MONEY:
            series = table.loc[table.measure.eq(measure)]
            axis.plot(series.year, series.index_2019, color=COMPONENTS[measure], linewidth=1.1, marker='o', markersize=2.5,
                      alpha=.85)
        total = table.loc[table.measure.eq(TOTAL)].sort_values('year')
        axis.plot(total.year, total.index_2019, color=BLUE, linewidth=2.4, marker='o', markersize=5, zorder=3)
        end = total.iloc[-1]
        axis.annotate(f'{end.index_2019:.0f}', (end.year, end.index_2019), textcoords='offset points', xytext=(8, 0),
                      va='center', fontsize=11, fontweight='bold', color=BLUE)
        axis.axhline(100, color=MUTED, linewidth=1, linestyle='--', zorder=0)
        axis.set(title=title, xlabel='Year', xticks=total.year)
        y_grid(axis)
    axes[0].set_ylabel('Index, 2019 = 100')
    handles = [Line2D([], [], color=BLUE, linewidth=2.4, marker='o', label=MEASURE_LABELS[TOTAL])]
    handles += [Line2D([], [], color=COMPONENTS[m], linewidth=1.5, label=MEASURE_LABELS[m]) for m in MONEY]
    figure.tight_layout()
    figure.legend(handles=handles, loc='center left', fontsize=8.5, bbox_to_anchor=(1, .5))
    headline(figure, f'Total investment grew {last.index_2019 / 100:.2f}× from 2019 to {int(last.year)}',
             'Index with 2019 = 100: blue = total of the six amounts, light colours = each amount')
    footnote(figure, 'Dashed line: 2019 level. 2026 appears only in the right panel, where every year covers the same six months.')
    save(figure, 'a01_index_2019')


def year_bars(table, name, ylabel, title, subtitle, note):
    """Total per year as a large panel with amount and growth on every bar, the six amounts as small panels;
    the latest year is blue."""
    total = table.loc[table.measure.eq(TOTAL)].sort_values('year')
    latest = total.year.max()
    figure = plt.figure(figsize=(16, 6.5))
    grid = figure.add_gridspec(2, 5, width_ratios=[1.6, 1.6, 1, 1, 1], hspace=.55, wspace=.35)
    axis = figure.add_subplot(grid[:, :2])
    bars = axis.bar(total.year.astype(str), total.value / 1e6, color=[BLUE if y == latest else NEUTRAL for y in total.year], width=.7)
    labels = [f'{v / 1e6:,.0f}' + (f'\n{g:+.0f}%' if pd.notna(g) else '') for v, g in zip(total.value, total.yoy_pct)]
    axis.bar_label(bars, labels=labels, padding=3, fontsize=9, color=SECONDARY)
    axis.set(title=MEASURE_LABELS[TOTAL], ylabel=ylabel, ylim=(0, total.value.max() / 1e6 * 1.15))
    y_grid(axis)
    for position, measure in enumerate(MONEY):
        axis = figure.add_subplot(grid[position // 3, 2 + position % 3])
        series = table.loc[table.measure.eq(measure)].sort_values('year')
        axis.bar(series.year.astype(str).str[-2:], series.value / 1e6, color=[BLUE if y == latest else LIGHT for y in series.year])
        axis.set_title(SHORT_LABELS[measure], fontsize=9.5)
        axis.tick_params(labelsize=7)
    headline(figure, title, subtitle)
    figure.subplots_adjust(top=.93, bottom=.08)
    footnote(figure, ' '.join(filter(None, [f'Blue: {int(latest)}. Labels: EUR million and growth against the previous year. '
                                            'Small panels: the six source amounts, years as two digits.', note])))
    save(figure, name)


def ytd_bars(annual):
    table = annual.loc[annual.window.eq('jan_jun')]
    total = table.loc[table.measure.eq(TOTAL)].sort_values('year')
    latest, previous = total.iloc[-1], total.iloc[-2]
    year_bars(table, 'a02_jan_jun_totals', 'EUR million, January–June',
              f'January–June {int(latest.year)}: EUR {latest.value / 1e6:,.0f} million, '
              f'{latest.yoy_pct:+.0f}% against January–June {int(previous.year)}',
              'Same six months of every year, so the partial year 2026 is compared fairly (EUR million)', '')


def full_year_bars(annual):
    table = annual.loc[annual.window.eq('reported') & ~annual.partial_year]
    total = table.loc[table.measure.eq(TOTAL)].sort_values('year')
    first, latest = total.iloc[0], total.iloc[-1]
    year_bars(table, 'a09_full_year_totals', 'EUR million, full year',
              f'Total investment rose from EUR {first.value / 1e6:,.0f} million in {int(first.year)} '
              f'to EUR {latest.value / 1e6:,.0f} million in {int(latest.year)}',
              f'Complete years only, all twelve months ({int(first.year)}–{int(latest.year)}), EUR million',
              '2026 is left out because only January–June is published; see a02 for that comparison.')


def decomposition(transformed, measure):
    table = transformed.loc[transformed.window.eq('reported') & transformed.measure.eq(measure)].reset_index(drop=True)
    time = table.year + (table.month - 1) / 12
    trend = table.dropna(subset=['trend'])
    december = table.loc[table.month.eq(12), 'seasonal'].iloc[0] / 1e6
    peak = table.loc[table.residual.idxmax()]
    figure, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    panels = [('value', 'Observed monthly total', INK), ('trend', 'Trend: centred 12-month moving average', BLUE),
              ('seasonal', 'Seasonal effect: the same pattern every year', SERIES[1]), ('residual', 'Residual: what is left', MUTED)]
    for axis, (column, title, color) in zip(axes, panels):
        axis.plot(time, table[column] / 1e6, color=color, linewidth=1.6)
        axis.set(title=title, ylabel='EUR million')
        y_grid(axis)
    covid = (COVID_START[0] + (COVID_START[1] - 1) / 12, COVID_END[0] + (COVID_END[1] - 1) / 12)
    axes[0].axvspan(*covid, color=LIGHT, alpha=.6, zorder=0)
    axes[0].text(covid[0] + .05, axes[0].get_ylim()[1] * .92, 'COVID period', fontsize=8.5, color=SECONDARY, va='top')
    for axis in axes[2:]:
        axis.axhline(0, color=AXIS, linewidth=1)
    peak_time = peak.year + (peak.month - 1) / 12
    axes[3].annotate(f'Largest residual: {MONTHS[int(peak.month) - 1]} {int(peak.year)}, +{peak.residual / 1e6:.0f} M\n'
                     'outlier candidate for Phase II', (peak_time, peak.residual / 1e6), textcoords='offset points',
                     xytext=(-40, -12), ha='right', va='center', fontsize=8.5, color=RED,
                     arrowprops=dict(arrowstyle='->', color=RED))
    axes[-1].set_xlabel('Year')
    headline(figure, f'Trend rose from {trend.trend.iloc[0] / 1e6:.0f} to {trend.trend.iloc[-1] / 1e6:.0f} '
                     f'million EUR a month; December adds about {december:.0f} million',
             f'Additive decomposition of {MEASURE_LABELS[measure].lower()}: observed = trend + seasonal effect + residual')
    figure.tight_layout()
    footnote(figure, 'The trend needs six months on each side, so it starts in mid-2019 and ends in late 2025.')
    save(figure, 'a03_decomposition')


def seasonality_heatmap(seasonality):
    grid = seasonality.pivot(index='measure', columns='month', values='seasonality_index').reindex(MEASURES)
    december = grid.loc[TOTAL, 12]
    weakest = int(grid.loc[TOTAL].idxmin())
    colormap = LinearSegmentedColormap.from_list('season', [BLUE, '#ffffff', RED])
    figure, axis = plt.subplots(figsize=(12, 5))
    image = axis.imshow(grid.to_numpy(), cmap=colormap, norm=TwoSlopeNorm(vmin=.4, vcenter=1, vmax=2), aspect='auto')
    labels = [MEASURE_LABELS[m] for m in MEASURES]
    axis.set(xticks=range(12), xticklabels=MONTHS, yticks=range(len(MEASURES)), yticklabels=labels)
    axis.get_yticklabels()[-1].set_fontweight('bold')
    axis.axhline(len(MEASURES) - 1.5, color=INK, linewidth=1.5)
    for r in range(len(MEASURES)):
        for c in range(12):
            value = grid.iloc[r, c]
            axis.text(c, r, f'{value:.2f}' if pd.notna(value) else 'n/a', ha='center', va='center', fontsize=8,
                      fontweight='bold' if MEASURES[r] == TOTAL else 'normal',
                      color='white' if pd.notna(value) and (value > 1.6 or value < .55) else INK)
    for side in axis.spines.values():
        side.set_visible(False)
    colorbar = figure.colorbar(image, ax=axis, fraction=.03, pad=.02)
    colorbar.set_label('1 = an average month', color=SECONDARY)
    colorbar.outline.set_visible(False)
    headline(figure, f'December is the strongest month: total investment {december:.2f}× an average month',
             f'Month total divided by the monthly mean of its year, averaged over 2019–2025; weakest month: {MONTHS[weakest - 1]}')
    figure.tight_layout()
    footnote(figure, 'Red: above an average month, blue: below. Complete years only.')
    save(figure, 'a04_seasonality_heatmap')


def recovery_bars(recovery, measure):
    table = recovery.loc[recovery.measure.eq(measure) & recovery.status.isin(['valid', 'no_decline'])
                         & recovery.primary_sector.ne('ALL')].copy()
    table = table.sort_values(['recovery_months', 'drop_pct'], ascending=[True, False], na_position='last').reset_index(drop=True)
    ceiling = np.nanmax(table.recovery_months) + 3 if table.recovery_months.notna().any() else 12
    colors = [NEUTRAL if status == 'no_decline' else BLUE if recovered else RED
              for status, recovered in zip(table.status, table.recovered.fillna(False))]
    figure, axis = plt.subplots(figsize=(11, .36 * len(table) + 2))
    y = np.arange(len(table))
    # A sector that never fell below its baseline gets a short grey stub instead of an invisible zero bar
    axis.barh(y, table.recovery_months.fillna(ceiling).clip(lower=.4), color=colors, height=.65)
    for k, row in table.iterrows():
        text = 'no decline' if row.status == 'no_decline' else f'{row.recovery_months:.0f} months  (drop {row.drop_pct:.0f}%)'
        axis.text(max(row.recovery_months if pd.notna(row.recovery_months) else ceiling, .4) + .3, k, text,
                  va='center', fontsize=8, color=SECONDARY)
    axis.set_yticks(y, [short_sector(s) for s in table.primary_sector])
    axis.invert_yaxis()
    axis.set_xlim(0, ceiling + 6)
    axis.set_xlabel('Months after March 2020 until the 3-month average, after the trough, regained the 2019 monthly mean')
    axis.grid(axis='x', color=GRID, linewidth=.8)
    axis.set_axisbelow(True)
    valid = table.loc[table.status.eq('valid') & table.recovered.astype(bool)]
    fast, slow = valid.iloc[0], valid.iloc[-1]
    headline(figure, f'{short_sector(fast.primary_sector)} recovered in {fast.recovery_months:.0f} months, '
                     f'{short_sector(slow.primary_sector)} needed {slow.recovery_months:.0f}',
             f'Pandemic recovery by sector: {MEASURE_LABELS[measure].lower()}')
    figure.tight_layout()
    handles = [Patch(color=BLUE, label='Recovered'), Patch(color=RED, label='Not recovered'), Patch(color=NEUTRAL, label='No decline')]
    axis.legend(handles=handles, loc='upper right', fontsize=8.5)
    footnote(figure, 'Drop: lowest 3-month average in the first pandemic year against the 2019 monthly mean. '
                     '"Mungon aktiviteti" and "Person Fizik" are not sectors and are excluded.')
    save(figure, 'a05_recovery_by_sector')


def bootstrap_errorbars(bootstrap):
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for axis, window, title in [(axes[0], 'reported', 'Complete years'), (axes[1], 'jan_jun', 'January–June of every year')]:
        for offset, measure, color in zip([-0.15, 0.15], MAIN, SERIES):
            table = bootstrap.loc[bootstrap.window.eq(window) & bootstrap.measure.eq(measure) & bootstrap.comparable]
            axis.errorbar(table.year + offset, table.yoy_pct, yerr=[table.yoy_pct - table.yoy_ci_low, table.yoy_ci_high - table.yoy_pct],
                          fmt='none', ecolor=color, elinewidth=1.4, capsize=3)
            clear = table.significant_decline | table.significant_growth
            # Filled marker: the whole 95% interval is on one side of zero; hollow: the change could be zero
            axis.scatter(table.year[clear] + offset, table.yoy_pct[clear], color=color, s=45, zorder=3)
            axis.scatter(table.year[~clear] + offset, table.yoy_pct[~clear], facecolor=SURFACE, edgecolor=color, s=45, linewidth=1.5, zorder=3)
        axis.axhline(0, color=MUTED, linewidth=1)
        axis.set(title=title, xlabel='Year', xticks=sorted(bootstrap.loc[bootstrap.window.eq(window) & bootstrap.comparable, 'year'].unique()))
        y_grid(axis)
    axes[0].set_ylabel('Growth against the previous year, %')
    total = bootstrap.loc[bootstrap.measure.eq(TOTAL)].set_index(['window', 'year'])
    handles = [Line2D([], [], color=c, marker='o', linestyle='none', label=MEASURE_LABELS[m]) for m, c in zip(MAIN, SERIES)]
    handles += [Line2D([], [], color=INK, marker='o', linestyle='none', label='Clear change (interval excludes 0)'),
                Line2D([], [], markerfacecolor=SURFACE, markeredgecolor=INK, marker='o', linestyle='none', label='Not clear (interval includes 0)')]
    headline(figure, f'2020: total investment {total.loc[("jan_jun", 2020), "yoy_pct"]:+.0f}% in January–June, '
                     f'{total.loc[("reported", 2020), "yoy_pct"]:+.0f}% over the full year; '
                     f'2021: {total.loc[("reported", 2021), "yoy_pct"]:+.0f}%',
             'Year-over-year growth with 95% bootstrap intervals (200 resamples of the source records within each year)')
    figure.tight_layout()
    figure.legend(handles=handles, loc='center left', fontsize=8.5, bbox_to_anchor=(1, .5))
    footnote(figure, 'The data are the full population of declarations, so the interval shows how stable a change is '
                     'when the records vary, not sampling error. 2026 is compared only January–June.')
    save(figure, 'a06_yoy_bootstrap_ci')


def class_bars(comparison, measure):
    table = comparison.loc[comparison.window.eq('reported') & comparison.measure.eq(measure)]
    width = table.loc[table.method.eq('equal_width')]
    crowded = width.share.max()
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
    panels = [('fixed_thresholds', 'Fixed thresholds (chosen)'), ('equal_width', 'Equal width'), ('equal_frequency', 'Equal frequency')]
    for axis, (method, title) in zip(axes, panels):
        series = table.loc[table.method.eq(method)].set_index('growth_class').months
        if method == 'fixed_thresholds':
            order, names, colors = GROWTH_LABELS, [GROWTH_NAMES[g] for g in GROWTH_LABELS], [GROWTH_COLORS[g] for g in GROWTH_LABELS]
        else:
            order = sorted(series.index)
            names, colors = [f'{"Bin" if method == "equal_width" else "Quintile"} {k + 1}' for k in range(len(order))], [LIGHT] * len(order)
        bars = axis.bar(names, series.reindex(order).fillna(0), color=colors, width=.7)
        axis.bar_label(bars, fmt='%d', padding=2, fontsize=9, color=SECONDARY)
        axis.set_title(title)
        axis.tick_params(axis='x', labelsize=8)
        y_grid(axis)
    axes[0].set_ylabel('Months')
    headline(figure, f'Fixed thresholds keep the economic meaning; equal width puts {crowded:.0%} of the months in one bin',
             f'Monthly year-over-year growth of {MEASURE_LABELS[measure].lower()}, discretised three ways')
    figure.tight_layout()
    footnote(figure, 'Equal width: five bins of the same width over the observed range, so a few extreme months stretch the scale. '
                     'Equal frequency: the same number of months per bin, but cut points without economic meaning.')
    save(figure, 'a07_growth_class_methods')


def covid_overview(features, recovery):
    """One page for the pandemic: the monthly shock against 2019, drop and recovery per measure, and per sector."""
    national = recovery.loc[recovery.primary_sector.eq('ALL')].set_index('measure').reindex(MEASURES)
    total = national.loc[TOTAL]
    figure = plt.figure(figsize=(15, 11))
    grid = figure.add_gridspec(2, 2, height_ratios=[1, 1.2], hspace=.4, wspace=.5)

    axis = figure.add_subplot(grid[0, :])
    months = features.loc[features.window.eq('reported') & features.measure.eq(TOTAL)]
    for year, color, width in [(BASE_YEAR, NEUTRAL, 1.8), (2020, RED, 3), (2021, BLUE, 1.8)]:
        series = months.loc[months.year.eq(year)]
        axis.plot(series.month, series.value / 1e6, marker='o', color=color, linewidth=width, label=str(year))
    axis.axvspan(LOCKDOWN_START[1] - .5, LOCKDOWN_END[1] + .5, color=LIGHT, alpha=.6, zorder=0, label='Lockdown, March–May 2020')
    baseline = months.loc[months.year.eq(BASE_YEAR), 'value'].mean() / 1e6
    axis.axhline(baseline, color=MUTED, linestyle='--', linewidth=1, label=f'2019 monthly mean ({baseline:.0f})')
    for _, row in months.loc[months.year.eq(2020) & months.month.between(LOCKDOWN_START[1], LOCKDOWN_END[1])].iterrows():
        axis.annotate(f'{row.yoy_pct:+.0f}%', (row.month, row.value / 1e6), textcoords='offset points', xytext=(0, -18),
                      ha='center', fontsize=10, color=RED, fontweight='bold')
    axis.set(xticks=range(1, 13), xticklabels=MONTHS, ylabel='EUR million', ylim=(0, None),
             title='Total investment per month (red labels: 2020 against the same month of 2019)')
    y_grid(axis)
    axis.legend(fontsize=8.5, ncol=5, loc='upper left')

    axis = figure.add_subplot(grid[1, 0])
    order = national.iloc[::-1]
    y = np.arange(len(order))
    axis.barh(y, order.drop_pct, color=[INK if m == TOTAL else RED for m in order.index], height=.65)
    for position, (drop, months_needed) in enumerate(zip(order.drop_pct, order.recovery_months)):
        axis.text(drop - 1.5, position, f'{drop:.0f}%  ·  back after {months_needed:.0f} months', ha='right', va='center',
                  fontsize=8.5, color=SECONDARY)
    labels = [MEASURE_LABELS[m] for m in order.index]
    axis.set_yticks(y, labels)
    axis.get_yticklabels()[0].set_fontweight('bold')
    axis.set(xlim=(order.drop_pct.min() * 2.2, 0), xlabel='Lowest 3-month average against the 2019 monthly mean, %',
             title='Drop and recovery per amount')
    axis.tick_params(axis='y', labelsize=8.5)

    axis = figure.add_subplot(grid[1, 1])
    sectors = recovery.loc[recovery.measure.eq(TOTAL) & recovery.status.eq('valid') & recovery.primary_sector.ne('ALL')]
    sectors = sectors.sort_values(['recovery_months', 'drop_pct'], ascending=[False, False])
    y = np.arange(len(sectors))
    axis.barh(y, sectors.recovery_months, color=BLUE, height=.65)
    for position, (drop, months_needed) in enumerate(zip(sectors.drop_pct, sectors.recovery_months)):
        axis.text(months_needed + .3, position, f'{drop:.0f}%', va='center', fontsize=7.5, color=RED)
    axis.set_yticks(y, [short_sector(s) for s in sectors.primary_sector])
    axis.set(xlim=(0, sectors.recovery_months.max() + 4), xlabel='Months after March 2020 until recovered (red: drop, %)',
             title='Sectors, total investment')
    axis.tick_params(axis='y', labelsize=7.5)
    axis.grid(axis='x', color=GRID, linewidth=.8)
    axis.set_axisbelow(True)

    headline(figure, f'COVID-19 cut total investment by {-total.drop_pct:.0f}%; it was back at the 2019 level after '
                     f'{total.recovery_months:.0f} months',
             'Business investment in Kosovo before, during and after the 2020 lockdown')
    figure.subplots_adjust(top=.95)
    footnote(figure, 'Recovery: first month after the trough whose 3-month average is at or above the 2019 monthly mean. '
                     'Domestic purchases recovered faster than imports.')
    save(figure, 'a08_covid')


def main(persist=True):
    if persist:
        mark_running('06_visualization')
    index_lines(read_table('annual_index'))
    ytd_bars(read_table('annual_features'))
    decomposition(read_table('monthly_transformed'), MAIN[0])
    seasonality_heatmap(read_table('seasonality'))
    recovery_bars(read_table('recovery'), MAIN[0])
    bootstrap_errorbars(read_table('bootstrap_ci'))
    class_bars(read_table('growth_class_comparison'), MAIN[0])
    covid_overview(read_table('monthly_features'), read_table('recovery'))
    full_year_bars(read_table('annual_features'))
    print(f'A06: 9 figures written to {FIGURES.relative_to(FIGURES.parents[2])}.')


if __name__ == '__main__':
    main()

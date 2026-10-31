import numpy as np
import pandas as pd
from src.schema import MONEY
from phase_1.pipeline_a import WINDOWS, TABLES, SEED, load_input, save_frame, mark_running

REPLICATES = 200
FRACTION = 0.1
DRAWS = 20


def reporting_views(data):
    for window in WINDOWS:
        yield window, data if window == 'reported' else data.loc[data.month.le(6)]


def bootstrap_totals(view, rng):
    """Resample source records within each year; annual totals and year-over-year growth per replicate."""
    years = sorted(view.year.unique())
    totals = np.empty((REPLICATES, len(years), len(MONEY)))
    for index, year in enumerate(years):
        values = view.loc[view.year.eq(year), MONEY].to_numpy()
        draws = rng.integers(0, len(values), size=(REPLICATES, len(values)))
        totals[:, index, :] = values[draws].sum(axis=1)
    return years, totals


def bootstrap_intervals(data, rng):
    rows = []
    for window, view in reporting_views(data):
        years, totals = bootstrap_totals(view, rng)
        observed = view.groupby('year')[MONEY].sum().reindex(years).to_numpy()
        months = view.groupby('year').month.nunique().reindex(years).to_numpy()
        previous = np.where(totals[:, :-1, :] > 0, totals[:, :-1, :], np.nan)
        growth = np.full(totals.shape, np.nan)
        growth[:, 1:, :] = (totals[:, 1:, :] / previous - 1) * 100
        observed_growth = np.full(observed.shape, np.nan)
        base = np.where(observed[:-1] > 0, observed[:-1], np.nan)
        observed_growth[1:] = (observed[1:] / base - 1) * 100
        for y, year in enumerate(years):
            for m, measure in enumerate(MONEY):
                low, high = np.nanpercentile(growth[:, y, m], [2.5, 97.5]) if y else (np.nan, np.nan)
                rows.append(dict(window=window, year=int(year), measure=measure, months=int(months[y]),
                    comparable=bool(y and months[y] == months[y - 1]), records=int(view.year.eq(year).sum()),
                    total=observed[y, m], total_ci_low=np.percentile(totals[:, y, m], 2.5), total_ci_high=np.percentile(totals[:, y, m], 97.5),
                    total_relative_half_width=(np.percentile(totals[:, y, m], 97.5) - np.percentile(totals[:, y, m], 2.5)) / 2 / observed[y, m] * 100 if observed[y, m] > 0 else np.nan,
                    yoy_pct=observed_growth[y, m], yoy_ci_low=low, yoy_ci_high=high,
                    significant_decline=bool(y and np.isfinite(high) and high < 0), significant_growth=bool(y and np.isfinite(low) and low > 0)))
    return pd.DataFrame(rows)


def sample_sensitivity(data, rng):
    """Simple random versus month-stratified 10% samples, scaled up and compared with the population series."""
    rows = []
    for window, view in reporting_views(data):
        population = view.groupby('year')[MONEY].sum()
        growth = population.pct_change() * 100
        for method in ['simple_random', 'stratified_by_period']:
            total_errors, growth_errors = [], []
            for _ in range(DRAWS):
                if method == 'simple_random':
                    sample = view.sample(frac=FRACTION, random_state=int(rng.integers(0, 2**31)))
                else:
                    sample = view.groupby(['year', 'month'], group_keys=False).sample(frac=FRACTION, random_state=int(rng.integers(0, 2**31)))
                estimate = sample.groupby('year')[MONEY].sum().reindex(population.index) / FRACTION
                total_errors.append((estimate - population).abs().div(population.where(population.gt(0))) * 100)
                growth_errors.append((estimate.pct_change() * 100 - growth).abs())
            total_error = pd.concat(total_errors).groupby(level=0).mean()
            growth_error = pd.concat(growth_errors).groupby(level=0).mean()
            for measure in MONEY:
                rows.append(dict(window=window, method=method, fraction=FRACTION, draws=DRAWS, measure=measure,
                    mean_abs_relative_error_total_pct=total_error[measure].mean(), max_abs_relative_error_total_pct=total_error[measure].max(),
                    mean_abs_error_yoy_pp=growth_error[measure].mean(), max_abs_error_yoy_pp=growth_error[measure].max()))
    return pd.DataFrame(rows)


def main(data=None, persist=True):
    if persist:
        mark_running('05_sampling_sensitivity')
    if data is None:
        data, _ = load_input()
    rng = np.random.default_rng(SEED)
    intervals = bootstrap_intervals(data, rng)
    sensitivity = sample_sensitivity(data, rng)
    if persist:
        save_frame(intervals, TABLES / 'bootstrap_ci.csv')
        save_frame(sensitivity, TABLES / 'sample_sensitivity.csv')
        print(f'A05: {REPLICATES} bootstrap replicates per year; {DRAWS} draws of {FRACTION:.0%} samples compared with the population.')
        print(intervals.loc[intervals.year.eq(2020) & intervals.window.eq('reported'), ['measure', 'yoy_pct', 'yoy_ci_low', 'yoy_ci_high', 'significant_decline']].round(1).to_string(index=False))
    return intervals, sensitivity


if __name__ == '__main__':
    main()

"""
02b · Kontrolli i integrimit
============================
Verifikon që 02a e bëri punën saktë. Nëse ndonjë kontroll i detyrueshëm
dështon (❌), scripti ndalet me exit code 1 dhe pipeline-i s'vazhdon.

KONTROLLET:
  ✔ Numri i rreshtave = shuma e rreshtave të mbetur për çdo fajll (s'u humb/dyfishua asgjë)
  ✔ Krejt vitet 2019–2026 janë prezent
  ✔ S'ka mbetur asnjë rresht "Totali"/"Gjithsej"
  ✔ S'ka kolona pa emër (header i zbuluar gabim)
  ✔ Komunat: ≤ 38 vlera unike, krejt të njohura
  ⚠ Kolona që janë krejt bosh në ndonjë vit (shenjë e skemës së ndryshme)
  ⚠ Mbulimi viti × muaj: a mungon ndonjë muaj?
  ⚠ Duplikate të plota (s'largohen këtu, por te 05a)

OUTPUT:
  outputs/tables/02b_integration_checks.csv
  outputs/tables/02b_missing_by_year.csv
  outputs/tables/02b_unknown_komuna.csv         (vetëm nëse ka)
  outputs/figures/02b_rows_per_year.png
  outputs/figures/02b_coverage_year_month.png   (nëse ka kolonë muaji)

EKZEKUTIMI:
  python -m phase_1.02b_integration_check
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.checks import CheckReport
from src.config import INCOMPLETE_YEAR, N_KOMUNA_KOSOVE, YEARS_EXPECTED
from src.io_utils import load_json, load_step, save_fig, save_table, summary_row_mask
from src.mappings import KOMUNAT

KOMUNA_COL = "komuna"
MUAJI_COL = "muaji"


def check_rows(rep, df, manifest):
    expected = sum(f["rows_kept"] for f in manifest["files"])
    rep.check(len(df) == expected, "Numri i rreshtave përputhet me fajllat",
              f"{len(df):,} në dataset vs {expected:,} nga fajllat")
    removed = sum(f["summary_rows"] + f["note_rows"] for f in manifest["files"])
    rep.warn_if(removed > 0, "Rreshta totali/shënime u larguan gjatë integrimit", f"{removed:,} gjithsej")


def check_years(rep, df):
    present = sorted(df["viti"].unique().tolist())
    missing = sorted(set(YEARS_EXPECTED) - set(present))
    rep.check(not missing, "Krejt vitet janë prezent", f"mungojnë: {missing}" if missing else f"{present}")


def check_no_summary_rows(rep, df):
    left = int(summary_row_mask(df.drop(columns=[c for c in df.columns if c.endswith("_raw")])).sum())
    rep.check(left == 0, "S'ka mbetur asnjë rresht 'Totali'", f"{left} rreshta" if left else "")


def check_columns(rep, df):
    bad = [c for c in df.columns if c.startswith("kolona_") or c.startswith("unnamed")]
    rep.check(not bad, "Krejt kolonat kanë emër", f"pa emër: {bad}" if bad else f"{df.shape[1]} kolona")

    # Kolona krejt bosh në ndonjë vit → zakonisht emër tjetër në atë vit (shto te COL_MAP)
    miss = df.drop(columns=["viti"]).isna().groupby(df["viti"]).mean().mul(100).round(1)
    save_table(miss.T, "02b_missing_by_year")
    fully_empty = {c: miss.index[miss[c] == 100].tolist() for c in miss.columns if (miss[c] == 100).any()}
    for c, years in fully_empty.items():
        rep.warn_if(True, f"Kolona '{c}' është krejt bosh në disa vite", f"{years} → kontrollo COL_MAP")


def check_komunat(rep, df):
    if KOMUNA_COL not in df.columns:
        rep.warn_if(True, f"S'ka kolonë '{KOMUNA_COL}'", "kontrollo emrat te COL_MAP")
        return
    vals = df[KOMUNA_COL].dropna()
    n = vals.nunique()
    rep.check(n <= N_KOMUNA_KOSOVE, f"Komuna unike ≤ {N_KOMUNA_KOSOVE}", f"{n} komuna")

    unknown = vals[~vals.isin(KOMUNAT)].value_counts()
    if len(unknown):
        save_table(unknown.rename("rreshta").to_frame(), "02b_unknown_komuna")
    rep.check(unknown.empty, "Krejt komunat janë të njohura",
              f"{len(unknown)} emra të panjohur, p.sh. {unknown.index[:5].tolist()} → shtoji te KOMUNA_ALIASES"
              if len(unknown) else "")
    rep.warn_if(n < N_KOMUNA_KOSOVE, "Disa komuna s'kanë asnjë rresht",
                f"mungojnë: {sorted(set(KOMUNAT) - set(vals.unique()))}")
    rep.warn_if(df[KOMUNA_COL].isna().any(), "Rreshta pa komunë", f"{df[KOMUNA_COL].isna().sum():,}")


def check_coverage(rep, df):
    if MUAJI_COL not in df.columns:
        rep.warn_if(True, f"S'ka kolonë '{MUAJI_COL}'", "mbulimi viti×muaj s'u kontrollua")
        return
    cov = df.pivot_table(index="viti", columns=MUAJI_COL, values="burimi_fajlli",
                         aggfunc="size", fill_value=0)
    cov = cov.reindex(columns=range(1, 13), fill_value=0)
    save_table(cov, "02b_coverage_year_month")

    fig, ax = plt.subplots(figsize=(11, 4))
    sns.heatmap(cov, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=ax, annot_kws={"size": 7})
    ax.set_title("Numri i rreshtave për vit × muaj (0 = muaj që mungon)")
    ax.set_xlabel("Muaji"), ax.set_ylabel("Viti")
    save_fig(fig, "02b_coverage_year_month")

    for y, row in cov.iterrows():
        empty = [m for m, v in row.items() if v == 0]
        if y == INCOMPLETE_YEAR:
            rep.warn_if(bool(empty), f"{y} është vit jo i plotë", f"muajt pa të dhëna: {empty}")
        else:
            rep.check(not empty, f"{y}: krejt 12 muajt kanë të dhëna", f"mungojnë: {empty}" if empty else "")

    if "viti_ne_fajll" in df.columns:   # a përputhet viti brenda Excel-it me emrin e fajllit?
        mism = (pd.to_numeric(df["viti_ne_fajll"], errors="coerce") != df["viti"]).sum()
        rep.warn_if(mism > 0, "Rreshta ku viti në Excel ≠ viti i fajllit (mbivendosje?)", f"{mism:,}")


def check_duplicates(rep, df):
    d = int(df.drop(columns=["burimi_fajlli"]).duplicated().sum())
    rep.warn_if(d > 0, "Ka rreshta krejtësisht identikë", f"{d:,} (vendimi merret te 05a, jo këtu)")


def plot_rows_per_year(df):
    counts = df["viti"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 3.5))
    colors = ["#9DB7E8" if y == INCOMPLETE_YEAR else "#2F6FDE" for y in counts.index]
    ax.bar(counts.index.astype(str), counts.values, color=colors)
    for x, v in zip(counts.index.astype(str), counts.values):
        ax.text(x, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
    ax.set_title(f"Rreshta për vit pas integrimit ({INCOMPLETE_YEAR} = vit jo i plotë)")
    ax.spines[["top", "right"]].set_visible(False)
    save_fig(fig, "02b_rows_per_year")


def main():
    df = load_step("02a_integrated")
    manifest = load_json("02a_manifest")

    rep = CheckReport("02b · KONTROLLI I INTEGRIMIT")
    check_rows(rep, df, manifest)
    check_years(rep, df)
    check_no_summary_rows(rep, df)
    check_columns(rep, df)
    check_komunat(rep, df)
    check_coverage(rep, df)
    check_duplicates(rep, df)

    plot_rows_per_year(df)
    print("\nRreshta për vit:\n" + df["viti"].value_counts().sort_index().to_string())

    save_table(rep.to_frame(), "02b_integration_checks", index=False)
    rep.finish()


if __name__ == "__main__":
    main()

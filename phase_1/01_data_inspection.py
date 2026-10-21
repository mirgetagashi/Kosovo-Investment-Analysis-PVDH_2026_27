
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import YEARS_EXPECTED
from src.io_utils import list_raw_files, read_raw_year, save_fig, save_json, save_table
from src.mappings import COL_MAP


def inspect_files():
    reports, columns_by_year, samples = [], {}, {}
    for year, path in list_raw_files():
        df, rep = read_raw_year(path, drop_summary=False)
        rep["year"] = year
        reports.append(rep)
        columns_by_year[year] = [COL_MAP.get(c, c) for c in df.columns]
        samples[year] = df.head(3)
    return reports, columns_by_year, samples


def print_file_report(rep: dict):
    print(f"\n📁 {rep['file']}")
    print(f"   sheets: {rep['sheets']}  (me të dhëna: {rep['sheets_with_data']})")
    for d in rep["sheet_details"]:
        if d["empty"]:
            print(f"   · sheet '{d['sheet']}': bosh")
            continue
        print(f"   · sheet '{d['sheet']}': header në rreshtin {d['header_row_excel']} të Excel-it "
              f"({d['rows_above_header']} rreshta titull/shënime sipër)")
        print(f"     {d['rows_read']:,} rreshta · {len(d['columns'])} kolona")
        if d["summary_rows"]:
            print(f"     ️  {d['summary_rows']} rreshta përmbledhës (Totali). Shembull: {d['summary_examples'][:1]}")
        if d["note_rows"]:
            print(f"     ⚠  {d['note_rows']} rreshta shënimesh (vetëm 1 qelizë e mbushur)")
        print(f"     kolonat origjinale: {d['original_columns']}")


def schema_table(columns_by_year: dict) -> pd.DataFrame:
    """Tabela kolonë × vit: True nëse kolona ekziston atë vit."""
    all_cols = sorted({c for cols in columns_by_year.values() for c in cols})
    return pd.DataFrame(
        {y: [c in cols for c in all_cols] for y, cols in columns_by_year.items()},
        index=pd.Index(all_cols, name="kolona"),
    )


def plot_schema(tbl: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(1.0 * tbl.shape[1] + 3, 0.35 * tbl.shape[0] + 1.5))
    sns.heatmap(tbl.astype(int), cmap=["#f2f2f2", "#2F6FDE"], cbar=False,
                linewidths=0.5, linecolor="white", ax=ax)
    ax.set_title("Kolonat sipas vitit (blu = ekziston)")
    ax.set_xlabel("Viti")
    ax.set_ylabel("")
    return fig


def main():
    print("=" * 70 + "\n01 · INSPEKTIMI I TË DHËNAVE\n" + "=" * 70)

    reports, columns_by_year, samples = inspect_files()
    for rep in reports:
        print_file_report(rep)

    found = [r["year"] for r in reports]
    missing = sorted(set(YEARS_EXPECTED) - set(found))
    print(f"\n Vitet e gjetura: {found}")
    if missing:
        print(f"  Mungojnë vitet: {missing}")

    overview = pd.DataFrame([{
        "viti": r["year"], "fajlli": r["file"],
        "sheets": len(r["sheets"]), "sheets_me_te_dhena": len(r["sheets_with_data"]),
        "rreshta_te_lexuar": r["rows_read"], "rreshta_totali": r["summary_rows"],
        "rreshta_shenime": r["note_rows"], "kolona": len(columns_by_year[r["year"]]),
    } for r in reports]).set_index("viti")
    print("\n Përmbledhja:\n" + overview.to_string())
    save_table(overview, "01_files_overview")

    tbl = schema_table(columns_by_year)
    save_table(tbl, "01_schema_by_year")
    save_fig(plot_schema(tbl), "01_schema_heatmap")

    not_everywhere = tbl[~tbl.all(axis=1)]
    if not_everywhere.empty:
        print("\n Krejt fajllat kanë kolonat e njëjta (pas normalizimit të emrave).")
    else:
        print("\n️  Kolona që NUK ekzistojnë në krejt vitet:")
        for col, row in not_everywhere.iterrows():
            print(f"   · {col:35s} vetëm në: {[y for y, v in row.items() if v]}")
        print("   → Nëse janë e njëjta kolonë me emër tjetër, shtoji te src/mappings.py → COL_MAP")

    first_year = min(samples)
    print(f"\n 3 rreshtat e parë ({first_year}):")
    print(samples[first_year].to_string())

    save_json({"reports": reports, "columns_by_year": columns_by_year}, "01_inspection_report")
    print("\n✔ 01 përfundoi.")


if __name__ == "__main__":
    main()
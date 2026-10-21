import pandas as pd

from src.io_utils import list_raw_files, read_raw_year, save_json, save_step
from src.mappings import COL_MAP, month_to_int, standardize_komuna

KOMUNA_COL = "komuna"
MUAJI_COL = "muaji"


def load_year(year: int, path) -> tuple[pd.DataFrame, dict]:
    df, rep = read_raw_year(path, drop_summary=True)

    df = df.rename(columns=COL_MAP)

    if "viti" in df.columns:
        df = df.rename(columns={"viti": "viti_ne_fajll"})

    df["viti"] = year
    df["burimi_fajlli"] = path.name

    print(f"  {path.name}: {rep['rows_read']:>7,} lexuar · "
          f"-{rep['summary_rows']} totali · -{rep['note_rows']} shënime · "
          f"{rep['rows_kept']:>7,} mbetën · {df.shape[1]} kolona")
    return df, rep


def standardize_komunat(df: pd.DataFrame) -> pd.DataFrame:
    """Entity identification: 'Prishtina'/'PRISHTINE'/'Pristina' → 'Prishtinë'."""
    if KOMUNA_COL not in df.columns:
        print(f" S'ka kolonë '{KOMUNA_COL}'. Kontrollo COL_MAP te src/mappings.py")
        return df
    raw = df[KOMUNA_COL]
    std = raw.map(standardize_komuna)
    n_variants = raw.dropna().astype(str).str.strip().nunique()
    n_std = std.dropna().nunique()

    df[f"{KOMUNA_COL}_raw"] = raw
    df[KOMUNA_COL] = std.fillna(raw)
    print(f"  Komunat: {n_variants} variante emrash → {n_std} komuna standarde "
          f"({std.isna().sum() - raw.isna().sum():,} rreshta me emër të panjohur)")
    return df


def standardize_muaji(df: pd.DataFrame) -> pd.DataFrame:
    """Data value conflicts: 'Janar' / 'Jan' / '01' → 1."""
    if MUAJI_COL not in df.columns:
        return df
    before = df[MUAJI_COL].dropna().astype(str).unique()[:5]
    df[MUAJI_COL] = df[MUAJI_COL].map(month_to_int).astype("Int8")
    print(f"  Muaji: p.sh. {list(before)} → numra 1–12 "
          f"({df[MUAJI_COL].isna().sum():,} vlera të panjohura)")
    return df


def main():
    print("=" * 70 + "\n02a · INTEGRIMI\n" + "=" * 70)

    frames, manifest = [], []
    for year, path in list_raw_files():
        df, rep = load_year(year, path)
        frames.append(df)
        manifest.append({
            "year": year, "file": path.name, "rows_read": rep["rows_read"],
            "summary_rows": rep["summary_rows"], "note_rows": rep["note_rows"],
            "rows_kept": rep["rows_kept"], "columns": [c for c in df.columns],
        })

    colsets = {m["year"]: set(m["columns"]) for m in manifest}
    common = set.intersection(*colsets.values())
    for y, cs in colsets.items():
        extra = cs - common
        if extra:
            print(f"  {y} ka kolona që s'janë në krejt vitet: {sorted(extra)}")

    data = pd.concat(frames, ignore_index=True, sort=False)
    print(f"\n  ➕ concat: {len(frames)} fajlla → {len(data):,} rreshta × {data.shape[1]} kolona")

    data = standardize_komunat(data)
    data = standardize_muaji(data)

    front = ["viti", "burimi_fajlli"]
    data = data[front + [c for c in data.columns if c not in front]]

    save_step(data, "02a_integrated")
    save_json({"files": manifest, "total_rows": len(data)}, "02a_manifest")
    print("\n✔ 02a përfundoi.")


if __name__ == "__main__":
    main()
import json
from pathlib import Path
import pandas as pd


def json_to_df(json_path: Path) -> pd.DataFrame:
    """
    Input JSON format:
      [
        [f, prefix(f), F, prefix(F)],
        ...
      ]

    Output DataFrame columns:
      - expression (f)
      - integral (F)
    """
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for item in data:
        if not isinstance(item, list) or len(item) < 4:
            continue
        rows.append({"expression": item[0], "integral": item[2]})

    return pd.DataFrame(rows)


def convert_folder(input_dir: str, output_dir: str) -> None:
    """
    Converts each JSON file in input_dir to a CSV in output_dir,
    and also writes a combined CSV containing all rows across files.

    Optional subset selection can be changed INSIDE this function:
      - set SELECT_MODE / SELECT_FILES / LIMIT_PER_FILE / SAMPLE_FRAC / RANDOM_STATE
    """
    # -------------------------------
    # Subset selection options (edit here, not in call params)
    # -------------------------------
    SELECT_MODE = "all"   # "all" | "first_n_files" | "only_these_files" | "sample_files"
    FIRST_N_FILES = 10    # used when SELECT_MODE == "first_n_files"
    ONLY_THESE_FILES = {"algebra.json", "trig.json"}  # used when SELECT_MODE == "only_these_files"
    SAMPLE_FILE_FRAC = 0.25  # used when SELECT_MODE == "sample_files"
    RANDOM_STATE = 42

    # Optional row-level subsetting (applies per file before combining)
    LIMIT_PER_FILE = None   # e.g., 1000 or None
    SAMPLE_ROW_FRAC = None  # e.g., 0.1 or None

    # Output combined file settings
    COMBINED_FILENAME = "RUBI_all_combined.csv"
    # -------------------------------

    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(in_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No .json files found in: {in_dir.resolve()}")

    # Select subset of files
    if SELECT_MODE == "first_n_files":
        json_files = json_files[:FIRST_N_FILES]
    elif SELECT_MODE == "only_these_files":
        json_files = [p for p in json_files if p.name in ONLY_THESE_FILES]
    elif SELECT_MODE == "sample_files":
        json_files = (
            pd.Series(json_files)
            .sample(frac=SAMPLE_FILE_FRAC, random_state=RANDOM_STATE)
            .tolist()
        )
    elif SELECT_MODE != "all":
        raise ValueError(f"Unknown SELECT_MODE: {SELECT_MODE}")

    combined_dfs = []
    converted = 0

    for jp in json_files:
        df = json_to_df(jp)

        # Optional row-level subsetting
        if LIMIT_PER_FILE is not None:
            df = df.iloc[:LIMIT_PER_FILE]
        if SAMPLE_ROW_FRAC is not None:
            df = df.sample(frac=SAMPLE_ROW_FRAC, random_state=RANDOM_STATE)

        # Save per-file CSV
        df.to_csv(out_dir / f"{jp.stem}.csv", index=False)
        combined_dfs.append(df)
        converted += 1

    # Save combined CSV
    if combined_dfs:
        combined = pd.concat(combined_dfs, ignore_index=True)
        combined.to_csv(out_dir / COMBINED_FILENAME, index=False)
        print(f"Saved combined dataset: {len(combined):,} rows -> {out_dir / COMBINED_FILENAME}")

    print(f"Done. Converted {converted} file(s).")


if __name__ == "__main__":
    convert_folder(
        input_dir="RUBI/",
        output_dir="RUBI_csv"
    )
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, List, Sequence, Optional

import pandas as pd


# -------------------------------------------------
# Salvage loader for truncated JSON list datasets
# -------------------------------------------------
def load_json_dataset_salvage(json_path: Path) -> List[Sequence[Any]]:
    text = json_path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return []

    # First try normal JSON
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # Salvage mode: extract complete top-level list entries
    if "[" not in text:
        raise ValueError(f"{json_path.name}: not a JSON list")

    start = text.find("[")
    s = text[start:]

    entries: List[Sequence[Any]] = []
    depth = 0
    in_str = False
    esc = False
    entry_start = None
    i = 0

    # Assumes dataset is a list of entries (each entry is a list)
    while i < len(s):
        ch = s[i]

        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            i += 1
            continue

        if ch == '"':
            in_str = True
            i += 1
            continue

        if ch == "[":
            depth += 1
            # depth==2 => start of one entry
            if depth == 2:
                entry_start = i
            i += 1
            continue

        if ch == "]":
            if depth == 2 and entry_start is not None:
                entry_text = s[entry_start : i + 1]
                try:
                    entry = json.loads(entry_text)
                    if isinstance(entry, list):
                        entries.append(entry)
                except json.JSONDecodeError:
                    pass
                entry_start = None

            depth -= 1
            if depth <= 0:
                break
            i += 1
            continue

        i += 1

    print(f"⚠️ Salvaged {len(entries)} complete entries from {json_path.name}")
    return entries


# -------------------------------------------------
# Extract only dataset + function, with arc->a normalization
# -------------------------------------------------
def normalize_function(s: str) -> str:
    # Replace every occurrence of "arc" with "a"
    # e.g. arcsin -> asin, arctan -> atan
    s=s.replace("^","**")
    return s.replace("arc", "a")


def extract_rows(raw: List[Sequence[Any]], dataset_name: str) -> pd.DataFrame:
    functions = []
    skipped = 0

    for entry in raw:
        if not isinstance(entry, list) or len(entry) < 1:
            skipped += 1
            continue

        func = entry[0]
        if not isinstance(func, str) or not func.strip():
            skipped += 1
            continue

        functions.append(normalize_function(func))

    df = pd.DataFrame({"dataset": dataset_name, "expression": functions})
    print(f"{dataset_name}: kept {len(df)} rows, skipped {skipped} malformed")
    return df


def load_dataset_df(json_path: Path) -> pd.DataFrame:
    dataset_name = json_path.stem.replace("_train", "").replace("_test", "")
    raw = load_json_dataset_salvage(json_path)
    return extract_rows(raw, dataset_name)


# -------------------------------------------------
# Fixed-size split: 10k train + 1k test per dataset
# -------------------------------------------------
def fixed_size_split(
    combined: pd.DataFrame,
    train_per_dataset: int = 5_000,
    test_per_dataset: int = 5_000,
    seed: int = 43,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)

    train_parts = []
    test_parts = []

    for name, g in combined.groupby("dataset", sort=False):
        g = g.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        needed = train_per_dataset + test_per_dataset

        if len(g) < needed:
            raise ValueError(
                f"Dataset {name} has only {len(g)} rows, but you requested {needed} "
                f"({train_per_dataset} train + {test_per_dataset} test)."
            )

        train_parts.append(g.iloc[:train_per_dataset])
        test_parts.append(g.iloc[train_per_dataset:train_per_dataset + test_per_dataset])

        print(f"{name}: train={train_per_dataset}, test={test_per_dataset}")

    train_df = pd.concat(train_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_df = pd.concat(test_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return train_df, test_df


# -------------------------------------------------
# Main pipeline
# -------------------------------------------------
def build_train_test_only_and_cleanup(
    folder: Path,
    json_filenames = (
        "BWD_train.json",
        "FWD_train.json",
        "IBP_train.json",
        "LIOUVILLE_train.json",
        "RISCH_train.json",
        "SUB_train.json",
    ),
    out_dir: Optional[Path] = None,
    seed: int = 42,
):
    folder = folder.resolve()
    out_dir = (folder / "csv_out") if out_dir is None else out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load and combine datasets (only dataset + function)
    dfs = []
    for fname in json_filenames:
        p = folder / fname
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")
        dfs.append(load_dataset_df(p))

    combined = pd.concat(dfs, axis=0).reset_index(drop=True)

    # Build fixed-size splits: 60k train, 6k test total
    train_df, test_df = fixed_size_split(
        combined,
        train_per_dataset=50_000,
        test_per_dataset=5_000,
        seed=seed,
    )

    # Write only the final split files
    train_path = out_dir / "ALL_train.csv"
    test_path = out_dir / "ALL_test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    # Build additional train dataset containing only FWD entries
    fwd_train_only = train_df[train_df["dataset"] == "FWD"].reset_index(drop=True)

    # Sanity check (should be exactly 10,000 with your fixed split)
    print("FWD train-only rows:", len(fwd_train_only))

    fwd_train_only_path = out_dir / "FWD_train_only.csv"
    fwd_train_only.to_csv(fwd_train_only_path, index=False)
    print(f"✅ Wrote {fwd_train_only_path} ({len(fwd_train_only)} rows)")

    # --- Write per-dataset TEST files (keep these) ---
    for name, g in test_df.groupby("dataset", sort=False):
        test_out = out_dir / f"{name}_test.csv"
        g.reset_index(drop=True).to_csv(test_out, index=False)
        print(f"✅ Wrote {test_out} ({len(g)} rows)")

    # --- (Optional) Write per-dataset TRAIN files (we will delete these after if desired) ---
    for name, g in train_df.groupby("dataset", sort=False):
        train_out = out_dir / f"{name}_train.csv"
        g.reset_index(drop=True).to_csv(train_out, index=False)
        print(f"✅ Wrote {train_out} ({len(g)} rows)")

    # --- Delete only per-dataset TRAIN files; keep per-dataset TEST files ---
    for name in train_df["dataset"].unique():
        train_out = out_dir / f"{name}_train.csv"
        if train_out.exists():
            train_out.unlink()
            print(f"🗑️ Deleted {train_out}")

    # --- Delete any old full per-dataset CSVs if they exist (still safe) ---
    for fname in json_filenames:
        stem = Path(fname).stem  # e.g. "BWD_train"
        full_csv = out_dir / f"{stem}.csv"
        if full_csv.exists():
            full_csv.unlink()
            print(f"🗑️ Deleted {full_csv}")



if __name__ == "__main__":
    build_train_test_only_and_cleanup(folder=Path("."), seed=42)

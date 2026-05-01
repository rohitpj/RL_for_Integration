import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


SUCCESS_STATUSES = {"Correct"}
METHOD_PREFIXES = ("rl", "transformer", "manual")


def load_results_for_framework(results_dir: Path) -> pd.DataFrame:
    """Load combined results if present; otherwise build them from per-method CSV outputs."""
    combined_path = results_dir / "all_integration_results.csv"
    if combined_path.exists():
        return pd.read_csv(combined_path)

    per_method_paths = {
        "rl": results_dir / "rl_test_results.csv",
        "transformer": results_dir / "transformer_test_results.csv",
        "manual": results_dir / "manual_test_results.csv",
    }

    available = {k: p for k, p in per_method_paths.items() if p.exists()}
    if not available:
        raise FileNotFoundError(
            f"No model_testing output CSVs found in {results_dir}. "
            "Expected all_integration_results.csv or per-method test result CSVs."
        )

    merged_df: Optional[pd.DataFrame] = None
    for _, path in available.items():
        df = pd.read_csv(path)
        if "expression" not in df.columns:
            raise KeyError(f"Missing 'expression' column in {path}")

        if merged_df is None:
            merged_df = df
        else:
            merged_df = merged_df.merge(df, on="expression", how="outer")

    return merged_df if merged_df is not None else pd.DataFrame()


def load_fwd_expression_filter(fwd_dataset_csv: Optional[Path], expr_col: str) -> Optional[set]:
    if fwd_dataset_csv is None:
        return None
    if not fwd_dataset_csv.exists():
        print(f"[warn] FWD dataset file not found: {fwd_dataset_csv}. Using full results without FWD filtering.")
        return None

    fwd_df = pd.read_csv(fwd_dataset_csv)
    if expr_col not in fwd_df.columns:
        print(
            f"[warn] Expression column '{expr_col}' not found in {fwd_dataset_csv}. "
            "Using full results without FWD filtering."
        )
        return None

    return set(fwd_df[expr_col].astype(str))


def filter_to_fwd(df: pd.DataFrame, fwd_exprs: Optional[set], expr_col: str) -> pd.DataFrame:
    if fwd_exprs is None:
        return df.copy()
    out = df[df[expr_col].astype(str).isin(fwd_exprs)].copy()
    print(f"FWD-filtered rows: {len(out)} / {len(df)}")
    return out


def plot_branch_histograms(df: pd.DataFrame, out_png: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    model_specs = [
        ("rl_branches", "RL branches"),
        ("transformer_branches", "Transformer branches"),
        ("manual_branches", "Manual branches"),
    ]

    for ax, (col, title) in zip(axes, model_specs):
        if col not in df.columns:
            ax.set_title(f"{title}\n(missing column)")
            ax.axis("off")
            continue

        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if vals.empty:
            ax.set_title(f"{title}\n(no data)")
            ax.axis("off")
            continue

        bins = min(40, max(10, int(vals.nunique())))
        ax.hist(vals, bins=bins, alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel("Branches")
        ax.grid(alpha=0.2)

    axes[0].set_ylabel("Frequency")
    fig.suptitle("Branch-frequency histograms (FWD subset)")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"Saved histogram figure: {out_png}")


def write_method_time_summary(df: pd.DataFrame, out_csv: Path) -> None:
    rows = []
    for prefix in METHOD_PREFIXES:
        time_col = f"{prefix}_time_seconds"
        status_col = f"{prefix}_status"
        if time_col not in df.columns:
            continue

        times = pd.to_numeric(df[time_col], errors="coerce")
        valid_times = times.dropna()
        status_series = df[status_col].astype(str) if status_col in df.columns else pd.Series(dtype=str)

        rows.append(
            {
                "method": prefix,
                "n_rows": int(len(df)),
                "n_timed": int(valid_times.shape[0]),
                "avg_time_seconds": float(valid_times.mean()) if not valid_times.empty else None,
                "p95_time_seconds": float(valid_times.quantile(0.95)) if not valid_times.empty else None,
                "correct_count": int(status_series.isin(SUCCESS_STATUSES).sum()) if not status_series.empty else 0,
                "dontknow_count": int((status_series == "DontKnowRule").sum()) if not status_series.empty else 0,
                "error_count": int((status_series == "Error").sum()) if not status_series.empty else 0,
            }
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved method timing summary: {out_csv}")


def extract_rl_only_successes(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = ["rl_status", "manual_status"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in combined results: {missing}")

    condition = (
        df["rl_status"].astype(str).isin(SUCCESS_STATUSES)
        & ~df["manual_status"].astype(str).isin(SUCCESS_STATUSES)
    )
    return df[condition].copy()


def scan_all_combined_results(repo_root: Path) -> list[Path]:
    files = sorted(repo_root.rglob("all_integration_results.csv"))
    return [f for f in files if f.is_file()]


def aggregate_rl_only_successes(repo_root: Path) -> pd.DataFrame:
    all_files = scan_all_combined_results(repo_root)
    chunks = []

    for path in all_files:
        try:
            df = pd.read_csv(path)
            subset = extract_rl_only_successes(df)
            if not subset.empty:
                subset.insert(0, "source_file", str(path))
                chunks.append(subset)
        except Exception as exc:
            print(f"[warn] Skipping {path}: {exc}")

    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Analyze model_testing result CSVs")
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory containing model_testing outputs (combined and/or per-method CSV files).",
    )
    parser.add_argument(
        "--fwd-dataset",
        default="Data/csv_out/FWD_test.csv",
        help="Optional FWD dataset CSV used to filter expressions before plotting.",
    )
    parser.add_argument("--expr-col", default="function", help="Expression column name in FWD dataset.")
    parser.add_argument(
        "--hist-out",
        default="results/fwd_branch_histograms.png",
        help="Output path for branch histogram image.",
    )
    parser.add_argument(
        "--timing-summary-out",
        default="results/method_timing_summary.csv",
        help="Output CSV path for per-method timing/status summary.",
    )
    parser.add_argument(
        "--rl-only-out",
        default="results/rl_only_success_examples.csv",
        help="Output CSV path for RL-success / others-fail examples (all discovered datasets).",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    hist_out = Path(args.hist_out)
    timing_summary_out = Path(args.timing_summary_out)
    rl_only_out = Path(args.rl_only_out)

    combined_df = load_results_for_framework(results_dir)
    fwd_exprs = load_fwd_expression_filter(Path(args.fwd_dataset), args.expr_col)
    fwd_df = filter_to_fwd(combined_df, fwd_exprs, expr_col="expression")

    plot_branch_histograms(fwd_df, hist_out)
    write_method_time_summary(combined_df, timing_summary_out)

    repo_root = Path(".").resolve()
    rl_only_df = aggregate_rl_only_successes(repo_root)
    rl_only_out.parent.mkdir(parents=True, exist_ok=True)
    rl_only_df.to_csv(rl_only_out, index=False)
    print(f"Saved RL-only success examples: {rl_only_out} ({len(rl_only_df)} rows)")


if __name__ == "__main__":
    main()

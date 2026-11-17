"""Plot validation ARP and catalog coverage against UPD for a calibration run."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def load_overall_metrics(run_dir: Path) -> pd.DataFrame:
    results = pd.read_csv(run_dir / "results.csv")
    popularity = pd.read_csv(run_dir / "metrics-popularity-diversity.csv")
    results = results[(results["groupType"] == "all") & (results["group"] == "all")]
    popularity = popularity[
        (popularity["groupType"] == "all") & (popularity["group"] == "all")
    ]
    if results["setting"].duplicated().any() or popularity["setting"].duplicated().any():
        raise ValueError("Overall metrics must contain at most one row per setting.")
    merged = results[["setting", "method", "n0", "upd"]].merge(
        popularity[["setting", "arp", "coverage"]],
        on="setting",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != 21:
        raise ValueError(
            f"Expected overall UPD and popularity metrics for all 21 settings; found {len(merged)}."
        )
    if merged[["upd", "arp", "coverage"]].isna().any().any():
        raise ValueError("UPD, ARP, and coverage must be present for every setting.")
    return merged


def plot_metrics(metrics: pd.DataFrame, output_path: Path) -> None:
    families = [("CP", metrics[metrics["method"].isin(["baseline", "cp"])])]
    n0_values = sorted(metrics.loc[metrics["method"] == "ha", "n0"].unique())
    families.extend(
        (f"History-aware N0={n0}", metrics[(metrics["method"] == "ha") & (metrics["n0"] == n0)])
        for n0 in n0_values
    )
    if len(families) < 2:
        raise ValueError("Expected CP and history-aware settings in the calibration results.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    for label, subset in families:
        subset = subset.sort_values("upd")
        if subset.empty:
            raise ValueError(f"No settings found for {label}.")
        axes[0].plot(subset["upd"], subset["arp"], marker="o", label=label)
        axes[1].plot(subset["upd"], subset["coverage"], marker="o", label=label)

    axes[0].set_ylabel("Average recommended popularity (training rating count)")
    axes[1].set_ylabel("Catalog coverage (fraction of trained movies)")
    for ax in axes:
        ax.set_xlabel("Validation UPD (lower = better aligned)")
        ax.grid(alpha=0.3)
    axes[0].set_title("ARP vs UPD")
    axes[1].set_title("Aggregate coverage vs UPD")
    axes[0].legend()
    fig.suptitle("Popularity and aggregate diversity across calibration settings")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration_run", type=Path, help="Calibration output run directory")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "arp_coverage_vs_upd.png",
        help="Output PNG path (default: analysis/arp_coverage_vs_upd.png)",
    )
    args = parser.parse_args()
    if not args.calibration_run.is_dir():
        raise FileNotFoundError(f"Calibration run directory not found: {args.calibration_run}")
    metrics = load_overall_metrics(args.calibration_run)
    plot_metrics(metrics, args.output)
    print(f"Plot written to {args.output}")


if __name__ == "__main__":
    main()

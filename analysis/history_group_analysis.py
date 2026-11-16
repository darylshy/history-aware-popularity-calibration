import argparse
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRICS = ("upd", "precision", "recall", "ndcg")
GROUP_KEYS = ("setting", "method", "lambda", "n0", "groupType", "group")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Summarize calibration results, optionally averaging across split seeds."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="One or more calibration results.csv files. The legacy second positional output directory is supported.",
    )
    parser.add_argument("--out", help="Output directory (default: analysis beside the first input).")
    args = parser.parse_args()

    paths = list(args.paths)
    output = args.out
    if output is None and len(paths) == 2 and Path(paths[1]).suffix.lower() != ".csv":
        output = paths.pop()
    if not paths:
        parser.error("Provide at least one results.csv path.")
    if output is None:
        output = str(Path(paths[0]).resolve().parent / "analysis")
    return paths, output


def read_split_seed(results_path, run_index, total_runs):
    manifest_path = Path(results_path).resolve().parent / "manifest.txt"
    if manifest_path.is_file():
        entries = {}
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                entries[key] = value
        if "splitSeed" not in entries:
            raise ValueError(f"{manifest_path} does not record splitSeed.")
        seed = entries["splitSeed"].strip()
        if not seed.isdigit():
            raise ValueError(f"{manifest_path} records splitSeed={entries['splitSeed']!r}, which is not an integer.")
        return seed
    if total_runs == 1:
        return "single"
    raise ValueError(
        f"{results_path}: no manifest.txt beside this results file, so its split seed is unknown. "
        "Multi-run aggregation requires each results.csv to sit next to a manifest.txt that records splitSeed."
    )


def load_runs(paths):
    required = set(GROUP_KEYS) | set(METRICS)
    frames = []
    seen_seeds = set()
    seen_paths = set()
    for index, path in enumerate(paths):
        resolved = Path(path).resolve()
        if resolved in seen_paths:
            raise ValueError(f"{path} was provided more than once; provide each seed's results.csv exactly once.")
        seen_paths.add(resolved)
        frame = pd.read_csv(path)
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        seed = read_split_seed(path, index, len(paths))
        if seed in seen_seeds:
            raise ValueError(f"More than one input uses splitSeed={seed}; provide one run per seed.")
        seen_seeds.add(seed)
        frame["seed"] = seed
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def curve(sub, x_metric="upd", y_metric="ndcg"):
    """Return x/y values sorted by x for interpolation or plotting."""
    sub = sub.sort_values(x_metric)
    return sub[x_metric].to_numpy(), sub[y_metric].to_numpy()


def family(data, name):
    """Rows for a method family, each including the lambda=0 baseline anchor."""
    base = data[data.method == "baseline"]
    if name == "cp":
        return pd.concat([base, data[data.method == "cp"]])
    n0 = int(name.split("=")[1])
    return pd.concat([base, data[(data.method == "ha") & (data.n0 == n0)]])


def aggregate_metrics(data):
    aggregations = {}
    for metric in METRICS:
        aggregations[f"{metric}_mean"] = (metric, "mean")
        aggregations[f"{metric}_std"] = (metric, lambda values: values.std(ddof=1))
    aggregations["seedCount"] = ("seed", "nunique")
    summary = data.groupby(list(GROUP_KEYS), dropna=False).agg(**aggregations).reset_index()
    std_columns = [f"{metric}_std" for metric in METRICS]
    summary[std_columns] = summary[std_columns].fillna(0.0)
    return summary.sort_values(["setting", "groupType", "group"]).reset_index(drop=True)


def matched_gain_by_seed(data, groups, n0_values):
    rows = []
    for seed in sorted(data.seed.unique()):
        seed_data = data[data.seed == seed]
        for group_type, group in groups:
            group_data = seed_data[
                (seed_data.groupType == group_type) & (seed_data.group == group)
            ]
            cp_x, cp_y = curve(family(group_data, "cp"))
            for n0 in n0_values:
                ha_x, ha_y = curve(family(group_data, f"n0={n0}"))
                low, high = max(cp_x.min(), ha_x.min()), min(cp_x.max(), ha_x.max())
                row = {
                    "seed": seed,
                    "groupType": group_type,
                    "group": group,
                    "n0": n0,
                    "overlap": "none",
                    "mean_ndcg_gain": np.nan,
                    "min_gain": np.nan,
                    "max_gain": np.nan,
                    "share_grid_ha_better": np.nan,
                }
                if low < high:
                    grid = np.linspace(low, high, 50)
                    differences = np.interp(grid, ha_x, ha_y) - np.interp(grid, cp_x, cp_y)
                    row.update(
                        overlap=f"{low:.3f}-{high:.3f}",
                        mean_ndcg_gain=float(differences.mean()),
                        min_gain=float(differences.min()),
                        max_gain=float(differences.max()),
                        share_grid_ha_better=float((differences > 0).mean()),
                    )
                rows.append(row)
    return pd.DataFrame(rows)


def summarize_matched_gain(per_seed):
    valid = per_seed.dropna(subset=["mean_ndcg_gain"])
    if valid.empty:
        return pd.DataFrame(
            columns=[
                "groupType",
                "group",
                "n0",
                "overlap",
                "mean_ndcg_gain",
                "std_ndcg_gain",
                "seeds_ha_better",
                "seedCount",
            ]
        )
    summary = (
        valid.groupby(["groupType", "group", "n0"], dropna=False)
        .agg(
            overlap=("overlap", lambda values: ";".join(sorted(set(values)))),
            mean_ndcg_gain=("mean_ndcg_gain", "mean"),
            std_ndcg_gain=("mean_ndcg_gain", lambda values: values.std(ddof=1)),
            seeds_ha_better=("mean_ndcg_gain", lambda values: int((values > 0).sum())),
            seedCount=("seed", "nunique"),
        )
        .reset_index()
    )
    summary["std_ndcg_gain"] = summary["std_ndcg_gain"].fillna(0.0)
    return summary


def main():
    input_paths, output_dir = parse_arguments()
    os.makedirs(output_dir, exist_ok=True)
    data = load_runs(input_paths)
    summary = aggregate_metrics(data)
    summary.to_csv(os.path.join(output_dir, "all_settings_by_group.csv"), index=False)
    data.to_csv(os.path.join(output_dir, "all_settings_by_group_by_seed.csv"), index=False)

    n0_values = sorted(data.loc[data.method == "ha", "n0"].dropna().astype(int).unique())
    groups = sorted(
        {
            (row.groupType, row.group)
            for row in summary[["groupType", "group"]].itertuples(index=False)
        }
    )
    families = ["cp"] + [f"ha N0={value}" for value in n0_values]

    plot_columns = min(3, len(groups))
    plot_rows = math.ceil(len(groups) / plot_columns)
    figure, axes = plt.subplots(
        plot_rows, plot_columns, figsize=(5 * plot_columns, 4 * plot_rows), sharex=False
    )
    axes = np.asarray(axes).reshape(-1)
    for index, (group_type, group) in enumerate(groups):
        axis = axes[index]
        group_data = summary[
            (summary.groupType == group_type) & (summary.group == group)
        ]
        for family_name in families:
            subset = family(group_data, "cp" if family_name == "cp" else family_name.replace("ha ", ""))
            subset = subset.sort_values("upd_mean")
            axis.errorbar(
                subset["upd_mean"],
                subset["ndcg_mean"],
                xerr=subset["upd_std"],
                yerr=subset["ndcg_std"],
                marker="o",
                capsize=2,
                label=family_name,
            )
        axis.set_title(f"{group_type}: {group}")
        axis.set_xlabel("Mean UPD (lower = better aligned)")
        axis.set_ylabel("Mean NDCG@10")
        axis.grid(alpha=0.3)
    for axis in axes[len(groups) :]:
        axis.set_visible(False)
    axes[0].legend()
    figure.suptitle("Mean relevance vs popularity alignment across split seeds (error bars = 1 SD)")
    figure.tight_layout()
    figure.savefig(os.path.join(output_dir, "ndcg_vs_upd.png"), dpi=150)
    plt.close(figure)

    per_seed = matched_gain_by_seed(data, groups, n0_values)
    per_seed.to_csv(os.path.join(output_dir, "matched_upd_by_seed.csv"), index=False)
    matched = summarize_matched_gain(per_seed)
    matched.to_csv(os.path.join(output_dir, "matched_upd_comparison.csv"), index=False)

    history = matched[matched.groupType == "history"]
    if not history.empty:
        buckets = [name for name in ("short", "medium", "long") if name in set(history.group)]
        n0s = sorted(history.n0.unique())
        positions = np.arange(len(buckets))
        width = 0.8 / max(1, len(n0s))
        figure, axis = plt.subplots(figsize=(7, 4))
        for index, n0 in enumerate(n0s):
            rows = history[history.n0 == n0].set_index("group").reindex(buckets)
            axis.bar(
                positions + (index - (len(n0s) - 1) / 2) * width,
                rows["mean_ndcg_gain"],
                width,
                yerr=rows["std_ndcg_gain"],
                capsize=3,
                label=f"N0={n0}",
            )
        axis.axhline(0, color="k", lw=0.8)
        axis.set_xticks(positions, buckets)
        axis.set_ylabel("Mean NDCG@10 gain over CP at equal UPD")
        axis.set_xlabel("History bucket")
        axis.set_title("History-aware vs fixed CP (error bars = 1 SD)")
        axis.legend()
        figure.tight_layout()
        figure.savefig(os.path.join(output_dir, "gain_by_history_bucket.png"), dpi=150)
        plt.close(figure)

        print("\nMatched-UPD NDCG@10 gain by history bucket (mean +/- SD):")
        print(
            history[["n0", "group", "mean_ndcg_gain", "std_ndcg_gain", "seeds_ha_better", "seedCount"]]
            .sort_values(["n0", "group"])
            .round(5)
            .to_string(index=False)
        )
        print("seeds_ha_better counts splits where the per-seed matched-UPD mean gain is positive.")

    print(f"\nAveraged {data.seed.nunique()} split seed(s) from {len(input_paths)} results file(s).")
    print(f"Outputs written to {output_dir}")


if __name__ == "__main__":
    main()

import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

path = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(path), "analysis")
os.makedirs(out, exist_ok=True)
df = pd.read_csv(path)

def curve(sub):
    """UPD-sorted (upd, ndcg) points for one method family."""
    sub = sub.sort_values("upd")
    return sub["upd"].to_numpy(), sub["ndcg"].to_numpy()

def family(d, name):
    """Rows for a method family, each including the lambda=0 baseline as the anchor."""
    base = d[d.method == "baseline"]
    if name == "cp":
        return pd.concat([base, d[d.method == "cp"]])
    n0 = int(name.split("=")[1])
    return pd.concat([base, d[(d.method == "ha") & (d.n0 == n0)]])

n0s = sorted(df.loc[df.method == "ha", "n0"].unique())
families = ["cp"] + [f"ha N0={n}" for n in n0s]
groups = [("all", "all")] + [("history", b) for b in ("short", "medium", "long")] \
         + [("segment", s) for s in ("blockbuster-focused", "diverse", "niche-focused")]

# --- Tables ---------------------------------------------------------------
tbl = df[df.groupType.isin(["all", "history", "segment"])].copy()
tbl.to_csv(os.path.join(out, "all_settings_by_group.csv"), index=False)

# --- NDCG vs UPD curves ----------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=False)
for ax, (gt, g) in zip(axes.ravel(), groups):
    d = df[(df.groupType == gt) & (df.group == g)]
    for fam in families:
        sub = family(d, "cp" if fam == "cp" else fam.replace("ha ", ""))
        x, y = curve(sub)
        ax.plot(x, y, marker="o", label=fam)
    ax.set_title(f"{gt}: {g}")
    ax.set_xlabel("UPD (lower = better aligned)")
    ax.set_ylabel("NDCG@10")
    ax.grid(alpha=.3)
axes[0, 0].legend()
fig.suptitle("Relevance vs popularity alignment across the lambda sweep")
fig.tight_layout()
fig.savefig(os.path.join(out, "ndcg_vs_upd.png"), dpi=150)

# --- Matched-UPD comparison: history-aware minus CP ------------------------
rows = []
for gt, g in groups:
    d = df[(df.groupType == gt) & (df.group == g)]
    cx, cy = curve(family(d, "cp"))
    for n0 in n0s:
        hx, hy = curve(family(d, f"n0={n0}"))
        lo, hi = max(cx.min(), hx.min()), min(cx.max(), hx.max())
        if lo >= hi:
            rows.append(dict(groupType=gt, group=g, n0=n0, overlap="none"))
            continue
        grid = np.linspace(lo, hi, 50)
        diff = np.interp(grid, hx, hy) - np.interp(grid, cx, cy)
        rows.append(dict(groupType=gt, group=g, n0=n0, overlap=f"{lo:.3f}-{hi:.3f}",
                         mean_ndcg_gain=diff.mean(), min_gain=diff.min(), max_gain=diff.max(),
                         share_grid_ha_better=float((diff > 0).mean())))
matched = pd.DataFrame(rows)
matched.to_csv(os.path.join(out, "matched_upd_comparison.csv"), index=False)

# --- Matched-UPD gain by history bucket (conjecture check) ------------------
hist = matched[matched.groupType == "history"]
if "mean_ndcg_gain" in hist:
    pivot = hist.pivot(index="n0", columns="group", values="mean_ndcg_gain")
    pivot = pivot.reindex(columns=[c for c in ("short", "medium", "long") if c in pivot])
    pivot.to_csv(os.path.join(out, "gain_by_history_bucket.csv"))
    ax = pivot.plot(kind="bar", figsize=(7, 4))
    ax.axhline(0, color="k", lw=.8)
    ax.set_ylabel("Mean NDCG@10 gain over CP at equal UPD")
    ax.set_xlabel("N0")
    ax.set_title("History-aware vs fixed CP, by history bucket")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "gain_by_history_bucket.png"), dpi=150)
    print("\nMean NDCG@10 gain of history-aware over CP at matched UPD:")
    print(pivot.round(5).to_string())
    print("\nProposal criterion: gain > 0, concentrated in the short bucket.")
print(f"\nOutputs written to {out}")

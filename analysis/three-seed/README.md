# Three-seed summary (seeds 42, 43, 44)

This folder combines the three calibration runs. Every seed uses the same code and the same settings. Only `splitSeed` changes, and `alsSeed` is set to the same value.

## Runs used

| splitSeed | baselineRunId | profileRunId | calibrationRunId |
|---|---|---|---|
| 42 | `2026-11-16T01-44-41.706431Z` | `2026-11-16T01-50-34.019005Z` | `2026-11-16T01-57-11.331215Z` |
| 43 | `2026-11-27T18-58-31.586034700Z` | `2026-11-27T20-34-44.309675700Z` | `2026-11-27T21-21-27.917536900Z` |
| 44 | `2026-11-27T19-49-07.737696900Z` | `2026-11-27T20-45-10.152502100Z` | `2026-11-27T21-48-32.725206900Z` |

Seed 42 is the existing team run. Seeds 43 and 44 were run with the `-seed43` and `-seed44` copies of `als-baseline.ipynb`, `popularity-profiles.ipynb` and `history-aware-calibration.ipynb` in `notebooks/`. The Parquet outputs of these runs stay local and are not in Git.

All three seeds selected the same settings on validation: the baseline, fixed CP with lambda=0.9, history-aware with N0=50 and lambda=0.9 (best validation NDCG@10), and history-aware with N0=100 and lambda=0.9 (matched-UPD criterion).

## Files

Validation results, written by `analysis/history_group_analysis.py` from the three `results.csv` files:

- `all_settings_by_group.csv`: mean and standard deviation over the three seeds for all 21 settings, overall and by history bucket and segment.
- `all_settings_by_group_by_seed.csv`: the same rows for each seed.
- `matched_upd_by_seed.csv` and `matched_upd_comparison.csv`: NDCG@10 gain of history-aware over fixed CP at equal UPD, per seed and summarized. These are interpolated validation estimates.
- `ndcg_vs_upd.png` and `gain_by_history_bucket.png`: plots of the two tables above with error bars of one standard deviation.

Test results, taken from the three `selected_on_validation.csv` files:

- `selected_test_by_seed.csv`: the test rows of the four selected settings for each seed.
- `selected_test_summary.csv`: mean and sample standard deviation of the test metrics over the three seeds.

The validation files were produced with:

```bash
python analysis/history_group_analysis.py \
  outputs/calibration/2026-11-16T01-57-11.331215Z/results.csv \
  outputs/calibration/2026-11-27T21-21-27.917536900Z/results.csv \
  outputs/calibration/2026-11-27T21-48-32.725206900Z/results.csv \
  --out analysis/three-seed
```

## Test results, all users (mean ± SD over three seeds)

| Setting | UPD | Precision@10 | Recall@10 | NDCG@10 |
|---|---|---|---|---|
| ALS baseline | 0.3093 ± 0.0230 | 0.0605 ± 0.0046 | 0.0422 ± 0.0033 | 0.0562 ± 0.0054 |
| Fixed CP, lambda=0.9 | 0.0280 ± 0.0053 | 0.0869 ± 0.0035 | 0.0581 ± 0.0034 | 0.0814 ± 0.0046 |
| History-aware, N0=50, lambda=0.9 | 0.0779 ± 0.0095 | 0.0850 ± 0.0034 | 0.0540 ± 0.0029 | 0.0782 ± 0.0045 |
| History-aware, N0=100, lambda=0.9 | 0.1287 ± 0.0140 | 0.0814 ± 0.0034 | 0.0499 ± 0.0026 | 0.0742 ± 0.0045 |

## Test results, short-history users (fewer than 50 training ratings)

| Setting | UPD | Precision@10 | Recall@10 | NDCG@10 |
|---|---|---|---|---|
| ALS baseline | 0.3070 ± 0.0182 | 0.0274 ± 0.0026 | 0.0524 ± 0.0050 | 0.0356 ± 0.0036 |
| Fixed CP, lambda=0.9 | 0.0316 ± 0.0044 | 0.0372 ± 0.0032 | 0.0713 ± 0.0074 | 0.0490 ± 0.0045 |
| History-aware, N0=50, lambda=0.9 | 0.1685 ± 0.0160 | 0.0320 ± 0.0030 | 0.0601 ± 0.0060 | 0.0402 ± 0.0041 |
| History-aware, N0=100, lambda=0.9 | 0.2449 ± 0.0202 | 0.0288 ± 0.0025 | 0.0549 ± 0.0048 | 0.0369 ± 0.0035 |

## Matched-UPD NDCG@10 gain of history-aware over fixed CP (validation)

| N0 | All users | Short history | Medium history | Long history |
|---|---|---|---|---|
| 50 | +0.00066 ± 0.00019 (3 of 3 seeds) | +0.00003 ± 0.00009 (2 of 3 seeds) | +0.00000 ± 0.00000 (0 of 3 seeds) | +0.00000 ± 0.00000 (0 of 3 seeds) |
| 100 | +0.00101 ± 0.00027 (3 of 3 seeds) | +0.00002 ± 0.00008 (2 of 3 seeds) | +0.00037 ± 0.00001 (3 of 3 seeds) | +0.00000 ± 0.00000 (0 of 3 seeds) |
| 200 | +0.00089 ± 0.00032 (3 of 3 seeds) | +0.00000 ± 0.00003 (1 of 3 seeds) | +0.00022 ± 0.00004 (3 of 3 seeds) | +0.00000 ± 0.00000 (0 of 3 seeds) |

The number in brackets is how many seeds have a positive gain. The same table by segment is in `matched_upd_comparison.csv`.

For long-history users the gain is exactly zero because their history-aware weight equals the fixed weight. The same holds for medium-history users when N0=50.

## What the three seeds show

- For all users, the order of the four settings is the same in every seed, for UPD and for all three accuracy metrics.
- On test, fixed CP with lambda=0.9 has the lowest UPD and also the highest Precision@10, Recall@10 and NDCG@10.
- The selected history-aware settings lower UPD less than fixed CP and have lower test accuracy than fixed CP, most clearly for short-history users.
- At equal UPD on validation, history-aware has a small NDCG@10 gain over fixed CP for all users (+0.0007 to +0.0010, positive in all three seeds). For short-history users the gain is smaller than its standard deviation, for medium-history users it is at most +0.0004, and for long-history users it is zero.

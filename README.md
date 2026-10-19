# History-Aware Popularity Calibration in Recommender Systems

This CS 7265 project investigates whether popularity calibration in movie recommendations should depend on the amount of rating history available for each user. It uses MovieLens 1M and Apache Spark ALS in Scala to compare unadjusted recommendations, fixed popularity calibration, and history-aware calibration.

## Repository contents

| Path                             | Purpose                                                                                                         |
| -------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `notebooks/als-baseline.ipynb` | Scala notebook for Spark setup, MovieLens rating loading, splitting, ALS training, and baseline recommendations |
| `notebooks/upd-evaluation.ipynb` | UPD evaluation of prepared recommendation lists, with input validation and synthetic tests |
| `data/ml-1m/`                  | Local MovieLens 1M dataset; excluded from version control                                                       |
| `.gitignore`                   | Excludes local data, generated models, and runtime files from version control                                   |
| `.gitattributes`               | Keeps text-file line endings consistent                                                                         |

The notebook implements the environment check, local MovieLens rating loading with a serializable parser, a DataFrame schema and row preview, and checks of dataset counts, rating values, identifier ranges, duplicate user–movie pairs, per-user history counts, basic timestamp validity, and a reproducible per-user training/test split. It trains an explicit-feedback ALS baseline and inspects its learned user and movie profiles. It also produces and validates 100 candidates and 10 unadjusted recommendations per user. The separate UPD notebook implements popularity-alignment evaluation and tests it with synthetic inputs. Real-data UPD results require the prepared profiles and category assignments described below. Calibration and relevance evaluation are not implemented here.

## Requirements

- Java 17
- A Scala 2.12.18 Jupyter kernel supporting Ammonite `$ivy` imports
- Any compatible Jupyter notebook editor
- Internet access on the first run to resolve Spark dependencies that are not already cached

The notebook loads **Spark 3.5.9** through its dependency imports. The verified runtime is Spark 3.5.9, Scala 2.12.18, and Java 17.0.16. Cloning this repository does not install Java or the Scala kernel. If a kernel is not already installed, consult the [Almond installation guide](https://almond.sh/docs/quick-start-install) and select a release compatible with Scala 2.12.18; the guide's latest-version defaults may differ from this project's runtime. The kernel must launch with Java 17 and Spark-compatible JVM options.

## Open and run

1. Clone the repository if you do not already have a local checkout:

   ```bash
   git clone https://github.com/darylshy/history-aware-popularity-calibration.git
   cd history-aware-popularity-calibration
   ```

2. Open the repository folder in your notebook editor.
3. Open [`notebooks/als-baseline.ipynb`](notebooks/als-baseline.ipynb).
4. Select a Scala 2.12.18 kernel using Java 17. The existing course kernel is named **CS 7265 - Scala 2.12.18**.
5. Populate `data/ml-1m/` as described below. Run the code cells in order to load the libraries, start Spark, define the record and parser, load the ratings, inspect the DataFrame, run the data-quality checks, construct and verify the training/test split, train ALS, inspect the learned profiles, and generate, validate, and export recommendations. Run the final shutdown cell when Spark work is complete. The kernel working directory may be the repository root or its `notebooks` folder.

The environment check uses three synthetic rows and does not require MovieLens. The subsequent loading section requires the local dataset. Successful environment-check output includes:

```text
Spark: 3.5.9
Scala: 2.12.18
Java: 17.0.16
Environment check passed: 3 synthetic rows.
```

The Java patch version may differ within Java 17. Spark runs locally with two worker threads and uses loopback addresses for the driver. The final cell stops Spark; rerun the session cell before performing additional Spark work. Restart the notebook kernel and run the cells in order to repeat the check from a fresh session.

Saved outputs record the environment check, schema and five-row preview, and full-dataset checks of counts, rating values, identifier ranges, duplicate pairs, minimum user history, and missing or nonpositive timestamps. The earliest and latest timestamps are displayed in UTC. Outputs also record split checks, ALS training completion, profile counts, previews of learned factors, recommendation checks, and sample top-10 lists.

## Dataset

The recommender uses [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/). Obtain it from GroupLens under the dataset's usage terms and retain the original files locally. Raw MovieLens data is not included in this repository.

Place the extracted dataset files in `data/ml-1m/` at the repository root, retaining the dataset's `README` alongside the data:

```text
data/
└── ml-1m/
    ├── README
    ├── ratings.dat
    ├── movies.dat
    └── users.dat
```

The `data/` directory is excluded from version control and must be populated locally after cloning the repository.

`ratings.dat` contains records in this format:

```text
UserID::MovieID::Rating::Timestamp
```

The notebook reads `ratings.dat` from this local directory. `movies.dat`, `users.dat`, and the dataset README are retained alongside it. The separate splitting section creates training and test DataFrames; no validation partition is allocated.

## Baseline split design

The notebook implements a pseudo-random 80/20 split within each user's rating history, using seed `42`. For a user with `n` ratings, `floor(0.8 * n)` records go to training and the remainder go to test. Rounding makes the overall proportions approximate. With MovieLens 1M's minimum of 20 ratings per user, this allocation leaves every user represented in both sets.

The per-user choice is based on the [LensKit splitting guide](https://lenskit.org/stable/guide/splitting#user-based-splitting), which describes user-based holdouts as a way to control experimental conditions for individual users. The 80/20 proportion comes from the project proposal; seed `42` and the rounding rule are this project's implementation choices.
The shuffle key is `xxhash64(splitSeed, userId, movieId)`, with the split seed supplied as a hash input. Within each user, ratings are ordered by that key and then movie ID to break ties. For the same input records and types, ordering is independent of input row order and Spark partitions. The notebook verifies that training and test do not overlap, reconstruct the original dataset exactly, and assign each user the intended number of training ratings.

## ALS baseline configuration

The notebook trains Spark's explicit-feedback ALS on `trainingRatings`, using `userId`, `movieId`, and `rating`. The proposal's fixed baseline settings are rank `50`, `maxIter = 10`, and `regParam = 0.1`. The named values `splitSeed = 42L` and `alsSeed = 42L` separately control the data split and ALS initialization. These settings have not been tuned on validation data.

The fitted model exposes user and movie factor DataFrames. The notebook checks profile counts and dimensions and previews the first five values of selected 50-factor profiles. The model is held in the Spark session and is not exported to disk; after restarting the kernel, rerun the notebook to train it again.

## Baseline recommendations

Recommendation generation scores every user–movie pair with learned profiles except pairs in the training history. Held-out test pairs remain eligible. For each user, the notebook keeps the 100 highest predictions, breaking ties by ascending movie ID. Predictions are retained and also min–max scaled to 0–1 within that user's candidate pool. If all 100 predictions are equal, scaled scores are set to 1.0. The first 10 candidates form the unadjusted baseline list.

Validation covers user coverage, exact list sizes, distinct movies and ranks, training-history exclusions, ranking and cutoff correctness, finite scores, scaling endpoints, and exact baseline membership. These checks do not measure recommendation quality. Ordered sample lists and aggregate counts are saved in notebook outputs. The full tables are exported locally as described below.

Scoring processes approximately 21.5 million eligible pairs for the current split. Spark temporarily persists full scores on local disk and the smaller candidate pool in memory with disk fallback to avoid repeated computation during validation. Allow local disk space for Spark's temporary data and sorting. The final cell releases these caches and stops Spark. Rerun the notebook from a fresh kernel to recreate the model and recommendation DataFrames.

## Exported recommendation tables

Section 8 writes a new timestamped folder under `outputs/als-baseline/` on each run. Earlier exports are preserved. Both Parquet tables retain `userId` (integer), `movieId` (integer), `prediction` (float), `recommendationRank` (integer), and `relevanceScore` (double):

| Table directory         | Contents                                                                |
| ----------------------- | ----------------------------------------------------------------------- |
| `top-100-candidates/` | 100 candidates per user: 604,000 rows for the current dataset and split |
| `top-10-baseline/`    | 10 baseline recommendations per user: 60,400 rows                       |

The notebook reads both exports back and verifies exact row equality with the generated DataFrames. Each table directory contains one Parquet data part and Spark metadata. Share the complete timestamped folder. Generated outputs remain local and are excluded from Git; cloning the repository does not download these tables.

Read a table in Scala/Spark by supplying its directory path:

```scala
val candidates = spark.read.parquet("/path/to/run/top-100-candidates")
val baseline = spark.read.parquet("/path/to/run/top-10-baseline")
baseline.orderBy("userId", "recommendationRank").show(10, truncate = false)
```

## UPD evaluation

Open [`notebooks/upd-evaluation.ipynb`](notebooks/upd-evaluation.ipynb) with the same Scala kernel and run its cells in order. It starts its own Spark session, validates the individual Jensen–Shannon calculation, verifies segment aggregation, runs a complete synthetic evaluation, checks invalid-input rejection, and stops Spark. It does not require ALS retraining or the baseline notebook's live session. All saved evaluation results are synthetic, not measured baseline or calibration results.

UPD compares each user's historical Head/Mid/Tail mix with the category proportions in their recommendations. Historical proportions are weighted by observed training-rating values; recommendations use movie counts. The implementation uses base-2 logarithms and reports Jensen–Shannon divergence (not its square root), giving individual scores a theoretical range of 0–1. Lower scores indicate closer popularity alignment. Overall UPD is the equally weighted mean of the three user-segment means, following [the paper's Section 3.1.2](https://arxiv.org/html/2103.06364#S3.SS1.SSS2). Relevance is evaluated separately.

### Required inputs for real evaluation

`evaluateUPD(profiles, movieGroups, recommendations, segments, expectedUsers)` accepts Spark DataFrames with the following columns and types:

| Input | Required columns |
|---|---|
| Historical profiles | `userId: Int`, `head: Double`, `mid: Double`, `tail: Double` |
| Movie popularity groups | `movieId: Int`, `popularityGroup: String` containing `H`, `M`, or `T` |
| Final recommendation lists | `userId: Int`, `movieId: Int`; exactly 10 distinct movies per user by default |
| User segments | `userId: Int`, `segment: String` containing `blockbuster-focused`, `diverse`, or `niche-focused` |
| Expected evaluation users | `userId: Int`, one row per intended user, specified independently of recommendation outputs |

Historical profiles and movie popularity groups must be derived from training data only. All inputs must use the same identifiers and split as the recommendation run. The baseline cohort is all training users; it can be obtained from `trainingRatings.select("userId").distinct()` in the baseline session and saved for handoff. The evaluator cannot determine split provenance from table contents, so retain the split configuration/run identity with the files. It does not create popularity groups, historical profiles, or user segments.

The evaluator checks schemas, nulls, valid proportions and labels, unique keys, exact user coverage, complete movie-label coverage, and list sizes before aggregation. Profiles, segments, and recommendation users must exactly match the expected cohort; a larger movie-category catalog is allowed. Every segment must be represented. A different list size may be passed explicitly as `listSize`; a top-100 candidate pool must not be substituted for a top-10 list.

Load the existing `top-10-baseline/` Parquet directory as the recommendation input; its additional prediction, rank, and relevance columns are ignored. Section 9 of the UPD notebook shows the loading and evaluation call with placeholder paths. Adapt other file layouts explicitly to the documented columns. Prepared profile, category, segment, and cohort files are not bundled; until supplied, only the synthetic validation is available.

The call returns `(userScores, segmentScores, overallScore)`. Per-user results retain both mixes, recommendation counts, segment labels, and mismatch scores. Segment results include counts and mean UPD; the overall table has one `overallUPD` value. For comparisons, run once per method and split while keeping profiles, categories, segments, cohort, and list size fixed. Save generated results under ignored `outputs/` with their method and split identity. No real-data results are generated automatically.

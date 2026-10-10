# History-Aware Popularity Calibration in Recommender Systems

This CS 7265 project investigates whether popularity calibration in movie recommendations should depend on the amount of rating history available for each user. It uses MovieLens 1M and Apache Spark ALS in Scala to compare unadjusted recommendations, fixed popularity calibration, and history-aware calibration.

## Repository contents

| Path                             | Purpose                                                                                        |
| -------------------------------- | ---------------------------------------------------------------------------------------------- |
| `notebooks/als-baseline.ipynb` | Scala notebook for Spark setup, MovieLens rating loading, splitting, and ALS baseline training |
| `data/ml-1m/`                  | Local MovieLens 1M dataset; excluded from version control                                      |
| `.gitignore`                   | Excludes local data, generated models, and runtime files from version control                  |
| `.gitattributes`               | Keeps text-file line endings consistent                                                        |

The notebook implements the environment check, local MovieLens rating loading with a serializable parser, a DataFrame schema and row preview, and checks of dataset counts, rating values, identifier ranges, duplicate user–movie pairs, per-user history counts, basic timestamp validity, and a reproducible per-user training/test split. It trains an explicit-feedback ALS baseline and inspects its learned user and movie profiles. Recommendation generation, calibration, and evaluation are not implemented yet.

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

   Access to the private repository is required.
2. Open the repository folder in your notebook editor.
3. Open [`notebooks/als-baseline.ipynb`](notebooks/als-baseline.ipynb).
4. Select a Scala 2.12.18 kernel using Java 17. The existing course kernel is named **CS 7265 - Scala 2.12.18**.
5. Populate `data/ml-1m/` as described below. Run the code cells in order to load the libraries, start Spark, define the record and parser, load the ratings, inspect the DataFrame, run the data-quality checks, construct and verify the training/test split, train ALS, and inspect the learned profiles. Run the final shutdown cell when Spark work is complete. The kernel working directory may be the repository root or its `notebooks` folder.

The environment check uses three synthetic rows and does not require MovieLens. The subsequent loading section requires the local dataset. Successful environment-check output includes:

```text
Spark: 3.5.9
Scala: 2.12.18
Java: 17.0.16
Environment check passed: 3 synthetic rows.
```

The Java patch version may differ within Java 17. Spark runs locally with two worker threads and uses loopback addresses for the driver. The final cell stops Spark; rerun the session cell before performing additional Spark work. Restart the notebook kernel and run the cells in order to repeat the check from a fresh session.

Saved outputs record the environment check, schema and five-row preview, and full-dataset checks of counts, rating values, identifier ranges, duplicate pairs, minimum user history, and missing or nonpositive timestamps. The earliest and latest timestamps are displayed in UTC. Outputs also record split checks, ALS training completion, profile counts, and previews of learned factors.

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

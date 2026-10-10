# History-Aware Popularity Calibration in Recommender Systems

This CS 7265 project investigates whether popularity calibration in movie recommendations should depend on the amount of rating history available for each user. It uses MovieLens 1M and Apache Spark ALS in Scala to compare unadjusted recommendations, fixed popularity calibration, and history-aware calibration.

## Repository contents

| Path                             | Purpose                                                                       |
| -------------------------------- | ----------------------------------------------------------------------------- |
| `notebooks/als-baseline.ipynb` | Scala notebook for the local Spark environment check and planned ALS baseline |
| `.gitignore`                   | Excludes local data, generated models, and runtime files from version control |
| `.gitattributes`               | Keeps text-file line endings consistent                                       |

The notebook currently implements the environment check only. Data loading, ALS training, recommendation generation, calibration, and evaluation are not implemented yet.

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
5. Run the three code cells in order: load the libraries, start Spark and check the runtime, then stop the check session.

The check uses three synthetic rows and does not require MovieLens. Successful output includes:

```text
Spark: 3.5.9
Scala: 2.12.18
Java: 17.0.16
Environment check passed: 3 synthetic rows.
Environment-check session stopped.
```

The Java patch version may differ within Java 17. Spark runs locally with two worker threads and uses loopback addresses for the driver. The final cell stops Spark; rerun the session cell before performing additional Spark work. Restart the notebook kernel and run the cells in order to repeat the check from a fresh session.

Saved outputs show a previous successful environment check. They do not represent a trained recommender or experimental results.

## Dataset

The planned recommender uses [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/). Obtain it from GroupLens under the dataset's usage terms and retain the original files locally. Raw MovieLens data is not included in this repository.

`ratings.dat` contains records in this format:

```text
UserID::MovieID::Rating::Timestamp
```

Dataset paths and training inputs are not connected to the notebook yet. The environment check can run independently of them.

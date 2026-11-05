# Titanic Survival Prediction

Northwestern University MLDS — Introduction to Data Engineering, Homework 3.

This repository provides separate Python and R Docker environments for exploring the Titanic passenger dataset and fitting logistic regression models to predict survival. Both programs print their preprocessing steps, training accuracy, and sample test predictions to the terminal.

## Prerequisites

- Git, or GitHub Desktop, to clone the repository.
- Docker Desktop installed and running.
- Internet access for the initial image builds and dataset download.
- A Kaggle account to download the Titanic competition data; accept the competition rules if prompted.

Python and R do not need to be installed locally when using Docker. A local virtual environment is also unnecessary for these commands.

The commands below use macOS/Linux shell syntax. On Windows, use a WSL terminal with Docker integration enabled.

## 1. Clone the repository

```bash
git clone https://github.com/boxiongdu/mlds400_titanic-disaster.git
cd mlds400_titanic-disaster
```

Alternatively, clone with GitHub Desktop and open a terminal in the cloned repository folder.

**Run all commands below from the repository root**, the folder containing this README and `src`.

## 2. Download and place the data

1. Visit the [Titanic competition data page](https://www.kaggle.com/competitions/titanic/data).
2. Sign in and download the competition data.
3. Extract the downloaded ZIP file.
4. Create the local data folder:

   ```bash
   mkdir -p src/data
   ```

5. Copy `train.csv` and `test.csv` directly into `src/data/`. Do not leave them inside another nested folder or ZIP file.

The required local file paths are:

| File | Purpose |
| --- | --- |
| `src/data/train.csv` | Passenger features and known survival outcomes for training |
| `src/data/test.csv` | Passenger features for prediction |

`gender_submission.csv` is not needed. It contains example predictions, not the true test labels.

The dataset is intentionally excluded from Git and the Docker build context. Each user must download it separately. The `src/data/` folder may not exist immediately after cloning because Git does not track empty folders.

Check that both files are present:

```bash
ls src/data/train.csv src/data/test.csv
```

## 3. Build and run Python

Build the image:

```bash
docker build -f src/python/Dockerfile -t titanic-python .
```

Run the analysis:

```bash
docker run --rm \
  -v "$(pwd)/src/data:/app/src/data:ro" \
  titanic-python
```

The build installs the package versions listed in `src/python/requirements.txt`. The container runs `src/python/main.py`.

## 4. Build and run R

Build the image:

```bash
docker build -f src/r/Dockerfile -t titanic-r .
```

Run the analysis:

```bash
docker run --rm \
  -v "$(pwd)/src/data:/app/src/data:ro" \
  titanic-r
```

The build runs `src/r/install_packages.R`. This implementation uses base R and the included `stats` package, so no additional R packages are required. The container runs `src/r/main.R`.

## Expected output and modeling approach

Both programs print:

- Dataset dimensions, sample rows, summary statistics, and missing-value counts.
- Overall survival rate and survival rates grouped by sex and passenger class.
- Descriptions of preprocessing and categorical encoding.
- Logistic regression training accuracy.
- The first 10 test predictions and the total number of predictions generated.
- An explanation of why accuracy cannot be calculated for the unlabeled Kaggle test data.

The models use `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, and `Fare`. Missing age and fare values are filled with medians calculated from the training data. These same medians are used for the test data. Sex and passenger class are encoded numerically for modeling.

Python uses a standardization and logistic regression pipeline. R uses `glm()` with a binomial family and logit link. Python's default logistic regression is regularized, while R's `glm()` is unregularized; their results need not be identical.

Predicted survival is coded as `0` for died and `1` for survived. The scripts print predictions to the terminal; they do not export prediction CSV files.

### Accuracy limitation

Training accuracy is calculated on the same observations used to fit the model; it is not an independent estimate of performance on unseen data.

The official Kaggle `test.csv` does not contain `Survived`. Therefore, its accuracy cannot be measured locally without the true labels. For the assignment's test-accuracy requirement, the scripts report this limitation rather than treating example predictions as ground truth. Kaggle evaluation would require exporting and submitting a prediction file separately; these scripts do not perform that submission.

## How the Docker commands work

- `-f` specifies which Dockerfile to use.
- `-t` assigns a name to the image.
- The final `.` uses the repository root as the build context.
- `-v` exposes the local `src/data/` folder inside the container at `/app/src/data/`.
- `ro` makes the data mount read-only.
- `--rm` removes the container after execution, preserving the image and local CSV files.

The Docker working directory is `/app`. Keep the R data path as `file.path("src", "data")` for this setup.

## Troubleshooting

| Problem | Resolution |
| --- | --- |
| Cannot connect to the Docker daemon | Open Docker Desktop and wait until it is running. |
| `docker: command not found` | Install Docker Desktop and reopen the terminal. |
| CSV file not found | Confirm both CSV files are directly in `src/data/`, run from the repository root, and include the volume mount. |
| Dockerfile or requirements file not found | Check the current directory and exact filenames. Use `Dockerfile` with a capital D. |
| Old output after editing a script | Save the file, rebuild the corresponding image, and run a new container. A Git push is not required. |
| Missing Python package | Add the dependency with an appropriate pinned version to `src/python/requirements.txt`, then rebuild. |
| Data folder absent after cloning | Create it with `mkdir -p src/data` and add the downloaded CSVs. |

The root ignore files should be named exactly `.gitignore` and `.dockerignore`. Both should exclude `src/data/`; the local `.venv/` should also be excluded. Adding an ignore rule does not automatically untrack previously committed files.

## Repository files

| Path | Purpose |
| --- | --- |
| `src/python/main.py` | Python analysis and predictions |
| `src/python/requirements.txt` | Pinned Python dependencies |
| `src/python/dockerfile` | Python container definition |
| `src/r/main.R` | R analysis and predictions |
| `src/r/install_packages.R` | R package setup |
| `src/r/dockerfile` | R container definition |
| `.gitignore` | Excludes local data and environment files from Git |
| `.dockerignore` | Excludes local data and unnecessary files from Docker builds |

Development changes are reviewed through feature-branch pull requests into `dev`, then from `dev` into `main`. The completed assignment is submitted as the public GitHub repository link, without uploading the dataset.

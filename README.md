# Online News Popularity — Predicting Article Shares

## Project Description

Given statistics for news articles from an online news website, this project builds a
regression pipeline to predict **`shares`** — how many times an article is shared on social
media (a proxy for popularity). At least three model families are trained, tuned, and compared,
and the most suitable one is selected with a documented, reproducible procedure.

**Key challenge.** `shares` is extremely right-skewed (skew ≈ 34; median ≈ 1,400 but max ≈
843,000) and only weakly correlated with any single feature. This is an inherently hard target,
so the pipeline models **`log1p(shares)`** and reports metrics on both the log and original
scales — selecting on log-scale R², because raw-scale R² is dominated by a handful of viral
outliers.

---

## Project Structure

```
.
├── data/
│   └── data.csv                # raw article statistics
├── src/
│   ├── config.yaml             # all pipeline settings (no hard-coded values)
│   ├── data_preparation.py     # config/data loaders, cleaning, preprocessor
│   └── model_training.py       # model factory, tuning, selection, evaluation, artifacts
├── tests/                      # pytest suite (cleaning, preprocessing, training, integration)
├── scripts/                    # check_env.py (env gate), compare_runs.py (run comparison)
├── reports/                    # persisted EDA summaries (correlations, outliers, VIF)
├── eda.ipynb                   # exploratory data analysis
├── main.py                     # end-to-end pipeline entry point (train / predict)
├── requirements.txt            # pinned dependencies
├── pyproject.toml              # Ruff lint/format config
└── README.md
```

---

## Prerequisites & Installation

* Python 3.11

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Dependencies are version-pinned in `requirements.txt` for reproducibility.

---

## Usage

```bash
# Train, compare, select the best model, evaluate on test, and save artifacts:
python main.py                       # (train is the default command)
python main.py train --log-level DEBUG

# Score new articles (same raw schema) with the saved model:
python main.py predict --input data/new_articles.csv --output predictions.csv

# Tooling:
python -m pytest tests/ -q           # run the test suite
python scripts/check_env.py          # verify installed versions match requirements.txt
python scripts/compare_runs.py       # summarise all saved runs into one table
ruff check .                         # lint (config in pyproject.toml)
```

Training validates the config, loads and cleans the data, trains + tunes every configured model
on the log target, selects the best on validation (deterministic tie-break), refits it on
train+validation, evaluates once on the held-out test set, and writes artifacts to `./artifacts/`.
Every run also appends to `artifacts/run.log`.

### Loading the saved model for inference

```python
import joblib, numpy as np, pandas as pd
from src.data_preparation import DataPreparation, load_config

config = load_config("./src/config.yaml")
model = joblib.load("artifacts/best_model.joblib")           # fitted Pipeline
new = pd.read_csv("data/new_articles.csv")                   # raw schema
X = DataPreparation(config).clean_data(new).drop(columns=[config["target_column"]], errors="ignore")
pred_log = model.predict(X)                                  # model outputs log1p(shares)
shares = np.clip(np.expm1(pred_log), 0, None)                # inverse-transform to shares
```

(The `python main.py predict` command does exactly this end to end.)

### Artifacts written per run

`best_model.joblib` (stable) + `best_model_<model>_<runid>.joblib` (versioned), the standalone
`preprocessor.joblib`, `model_comparison.csv` (with fold-level CV mean/std), `test_predictions.csv`,
`split_indices.csv` (exact train/val/test row assignment), `environment.txt` (package versions),
and `run_manifest.json` (best model, params, metrics, seed, config snapshot). Run ids combine a
timestamp with a short random suffix to stay collision-resistant.

All behaviour is driven by `./src/config.yaml`: seed, target transform, feature lists,
imputation strategies, split ratios, the models and their grids, the selection metric/tie-break,
an optional feature-selection stage, and the artifacts location. The config is validated on load.

---

## Data Dictionary (selected)

| Column | Description | Handling |
| ------ | ----------- | -------- |
| `ID`, `URL` | Article identifiers | Dropped |
| `timedelta` | Days between publication and data acquisition | **Dropped** (not known at publish time) |
| `weekday` | Day of week published | One-hot |
| `data_channel` | Article topic (world, tech, …); 15% missing | One-hot; missing → `unknown` |
| `n_tokens_title/content` | Word counts | Numeric |
| `n_unique_tokens`, `n_non_stop_words`, `n_non_stop_unique_tokens` | Rate features in [0,1] | Range-clipped, numeric |
| `num_hrefs/self_hrefs/imgs/videos` | Link/media counts (`num_videos` 47% missing) | Median-imputed |
| `n_comments` | Reader comments (accrue after publication) | **Dropped** (leakage — see **Limitations**) |
| `kw_*` | Keyword share statistics (min/max/avg families) | Numeric (collinear) |
| `self_reference_*_shares` | Shares of referenced articles | Numeric |
| `shares` | **Target** — times the article was shared | `log1p` transformed |

---

## Pipeline Flow

1. **Config load + validation** and **data load + schema check**.
2. **Cleaning** — drop `ID`/`URL`, remove exact duplicates, NaN-coerce any rate value outside [0,1].
3. **Preprocessing** (inside a `ColumnTransformer`/`Pipeline`, fit on training data only):
   median-impute + scale numerics; constant-impute (`unknown`) + one-hot encode categoricals.
4. **Split** 80/10/10 train/validation/test, seeded.
5. **Train + tune** each model on `log1p(shares)` (grid search where a grid is given).
6. **Select** the best by validation log-scale R² (tie-break: log-scale RMSE, then name).
7. **Refit** the winner on train+val, **evaluate once** on the test set.
8. **Persist** the model, a comparison table (with fold-level CV mean/std), predictions, and a
   run manifest.

---

## Key EDA Findings

See `eda.ipynb`. In short: the target needs a log transform; `num_videos` (47%) is missing at
random (uninformative → median-impute) while missing `data_channel` is informative (kept as
`unknown`); weekend articles are shared more; every predictor is weak (the strongest correlate,
`n_comments` r≈0.33, is excluded as a post-publication leak — among pre-publication features
`kw_avg_avg` r≈0.22 leads); the `kw_*` and `self_reference_*` families are strongly collinear.

---

## Models & Comparison

Five model families plus a baseline are compared on identical preprocessing:

| Model | Role |
| ----- | ---- |
| DummyRegressor | Predict-the-mean baseline (frames how much signal real models add) |
| Linear / Ridge / Lasso | Interpretable linear benchmarks (Ridge/Lasso regularise the collinear features) |
| RandomForest | Non-linear ensemble |
| HistGradientBoosting | Fast gradient boosting for large data |

**How they are compared.** All models are scored on the validation set on both scales, and via
3-fold cross-validation on the log target (mean ± std, saved to `artifacts/model_comparison.csv`
for a stability read). Selection is on **validation log-scale R²**.

### Results

This is a **pre-publication** model: `n_comments` and `timedelta` are excluded as
post-publication signals (see **Limitations**). Only the selected model is scored on the
held-out test set; the others show cross-validation and validation figures.

| Model | CV R²(log) | Val R²(log) | Val MAE (shares) |
| ----- | ---------- | ----------- | ---------------- |
| **HistGradientBoosting** ⭐ | **0.158 ± 0.007** | **0.154** | **~2,248** |
| RandomForest | 0.153 ± 0.004 | 0.145 | ~2,262 |
| Ridge / Lasso / Linear | 0.119 ± 0.002 | 0.099 | ~2,298 |
| Dummy (baseline) | 0.000 | 0.000 | ~2,367 |

**Selected model on the held-out test set:** HistGradientBoosting — **R²(log) 0.16, MAE ~2,296
shares**.

**Which model is most suitable, and why.** **HistGradientBoosting** is selected. It has the best
and most *stable* cross-validated log-R² (0.158, std 0.007), edges out RandomForest, and beats
the linear models (0.099) and the Dummy baseline (0.000). The gap from linear to tree models
shows the (weak) signal is non-linear; the ~0 Dummy R² confirms the task is genuinely hard, so a
~0.16 log-R² from *pre-publication* features alone is a reasonable result. HistGradientBoosting
is also the practical choice — it trains far faster than RandomForest at equal accuracy.
Raw-scale R²/RMSE are unstable (viral outliers), so **MAE (~2,296 shares)** is the interpretable
original-scale figure.

> For context, including `n_comments` lifts the best test R²(log) to ~0.46 — but that leaks
> post-publication engagement, so it is deliberately excluded here (see Limitations).

---

## Limitations & Next Steps

**Limitations**

* **This is a pre-publication forecast.** `n_comments` (reader comments accrue *after*
  publication, like shares) and `timedelta` (the article's age at scrape time) are both excluded,
  because neither is knowable when an article is published — including them leaks post-publication
  information. Including `n_comments` would raise test R²(log) from ~0.16 to ~0.46, which
  quantifies exactly how much of the apparent accuracy was leakage.
* Most of the variance in `shares` is genuinely unexplained — popularity is driven substantially
  by external factors (timing, promotion, luck) absent from the data, so a modest R² is expected.
* Raw-scale error metrics are dominated by a few viral articles; the log scale is the reliable lens.

**Next steps**

* Try target/impact encoding for `data_channel`, and engineer an explicit `is_weekend` feature.
* Add SHAP-based feature importance to explain the selected model.
* Consider reframing as a "viral vs not" classification target, which is often more tractable for
  this dataset than point-predicting a heavy-tailed count.

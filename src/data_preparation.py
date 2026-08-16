"""
data_preparation.py
===================

Deterministic loading, validation, cleaning, and preprocessing for the Online
News Popularity regression task (predicting article ``shares``).

Public entry points
-------------------
* ``load_config(path)``           -> validated config dict
* ``load_data(config)``           -> raw DataFrame (schema-checked)
* ``DataPreparation.clean_data``  -> cleaned DataFrame
* ``DataPreparation.preprocessor``-> sklearn ColumnTransformer (impute/scale/encode)
"""

# Standard library imports
import logging
from pathlib import Path
from typing import Any, Dict

# Related third-party imports
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REQUIRED_CONFIG_KEYS = [
    "file_path", "target_column", "random_seed", "log_transform_target",
    "drop_columns", "rate_features", "numerical_features", "categorical_features",
    "numeric_imputer_strategy", "categorical_imputer_strategy", "categorical_fill_value",
    "val_test_size", "val_size", "selection_metric", "selection_mode",
    "cv", "scoring", "models", "artifacts_dir",
]


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate the YAML config, failing early with a clear message.

    Raises
    ------
    FileNotFoundError : the config file does not exist.
    ValueError        : the YAML is malformed or a required key is missing.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path.resolve()}")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ValueError(f"Could not parse config YAML at {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"Config at {path} did not parse to a mapping.")
    missing = [k for k in REQUIRED_CONFIG_KEYS if k not in config]
    if missing:
        raise ValueError(f"Config is missing required key(s): {missing}")
    _validate_config_schema(config)
    logging.info("Configuration loaded and validated (%d models).", len(config["models"]))
    return config


# Lightweight schema: (key -> (expected python type(s), optional validator, message)).
_CONFIG_SCHEMA = {
    "file_path": (str, None, ""),
    "target_column": (str, None, ""),
    "random_seed": (int, lambda v: v >= 0, "must be a non-negative integer"),
    "log_transform_target": (bool, None, ""),
    "drop_columns": (list, None, ""),
    "rate_features": (list, None, ""),
    "numerical_features": (list, lambda v: len(v) > 0, "must be a non-empty list"),
    "categorical_features": (list, None, ""),
    "numeric_imputer_strategy": (str, None, ""),
    "categorical_imputer_strategy": (str, None, ""),
    "categorical_fill_value": (str, None, ""),
    "val_test_size": ((int, float), lambda v: 0 < v < 1, "must be a fraction in (0, 1)"),
    "val_size": ((int, float), lambda v: 0 < v < 1, "must be a fraction in (0, 1)"),
    "selection_metric": (str, None, ""),
    "selection_mode": (str, lambda v: v in ("max", "min"), "must be 'max' or 'min'"),
    "cv": (int, lambda v: v >= 2, "must be an integer >= 2"),
    "scoring": (str, None, ""),
    "artifacts_dir": (str, None, ""),
}


def _validate_config_schema(config: Dict[str, Any]) -> None:
    """Validate types, ranges, and the nested model definitions — so a malformed
    value fails immediately with a precise message rather than deep in the run."""
    for key, (expected_type, validator, message) in _CONFIG_SCHEMA.items():
        value = config[key]
        # bool is a subclass of int; guard int fields from accepting True/False.
        if expected_type is int and isinstance(value, bool):
            raise ValueError(f"config['{key}'] must be an integer, not a bool.")
        if not isinstance(value, expected_type):
            raise ValueError(
                f"config['{key}'] must be {expected_type}, got {type(value).__name__}."
            )
        if validator is not None and not validator(value):
            raise ValueError(f"config['{key}'] {message} (got {value!r}).")

    if not isinstance(config["models"], dict) or not config["models"]:
        raise ValueError("config['models'] must be a non-empty mapping of models.")
    for name, spec in config["models"].items():
        if not isinstance(spec, dict) or "estimator" not in spec:
            raise ValueError(f"model '{name}' must be a mapping with an 'estimator' key.")
        if not isinstance(spec["estimator"], str):
            raise ValueError(f"model '{name}'.estimator must be a string.")
        grid = spec.get("grid", {})
        if grid is not None and not isinstance(grid, dict):
            raise ValueError(f"model '{name}'.grid must be a mapping (or empty).")


def load_data(config: Dict[str, Any]) -> pd.DataFrame:
    """Load the raw CSV declared in config, verifying it exists and carries the
    target plus the configured feature columns."""
    path = Path(config["file_path"])
    if not path.is_file():
        raise FileNotFoundError(f"Data file not found: {path.resolve()}")
    df = pd.read_csv(path)
    required = set(config["numerical_features"]) | set(config["categorical_features"])
    required.add(config["target_column"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Raw data is missing expected column(s): {sorted(missing)}")
    if not pd.api.types.is_numeric_dtype(df[config["target_column"]]):
        raise ValueError(f"Target '{config['target_column']}' must be numeric.")
    mem_mb = df.memory_usage(deep=True).sum() / 1024 ** 2
    logging.info("Loaded raw data: %d rows x %d columns (%.1f MB).", *df.shape, mem_mb)
    # Schema diagnostics: dtype mix, columns with the most missingness, and any
    # non-finite values in numeric columns (inf/-inf) that would break scaling.
    dtype_counts = df.dtypes.value_counts().to_dict()
    logging.info("Column dtypes: %s", {str(k): int(v) for k, v in dtype_counts.items()})
    top_missing = df.isna().sum()
    top_missing = top_missing[top_missing > 0].sort_values(ascending=False).head(5)
    if not top_missing.empty:
        logging.info("Top missingness: %s",
                     {c: int(n) for c, n in top_missing.items()})
    numeric = df.select_dtypes("number")
    non_finite = int(np.isinf(numeric.to_numpy(dtype="float64", na_value=np.nan)).sum())
    if non_finite:
        logging.warning("Found %d non-finite (inf) values in numeric columns.", non_finite)
    return df


class DataPreparation:
    """Clean the news dataset and build the preprocessing ColumnTransformer.

    Attributes
    ----------
    config : dict
        Parameters for cleaning and preprocessing.
    preprocessor : sklearn.compose.ColumnTransformer
        Imputes + scales numeric features and imputes + one-hot-encodes
        categoricals; fit inside the model pipeline on training data only.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.preprocessor = self._create_preprocessor()

    def clean_data(self, df: pd.DataFrame, deduplicate: bool = True) -> pd.DataFrame:
        """Deterministic, documented cleaning:

        1. Drop identifier columns (``ID``, ``URL``).
        2. Drop any exact duplicate rows (``deduplicate=True``). This is a
           *training* concern (avoid biasing the fit); at inference time pass
           ``deduplicate=False`` so every input row gets exactly one prediction.
        3. Coerce out-of-range rate features (bounded in [0, 1]) to NaN so a
           corrupt value is imputed rather than distorting scaling and the models.
        """
        df = df.copy()
        drop_cols = [c for c in self.config.get("drop_columns", []) if c in df.columns]
        df = df.drop(columns=drop_cols)

        if deduplicate:
            df = df.drop_duplicates()

        coerced = 0
        for col in self.config.get("rate_features", []):
            if col in df.columns:
                out_of_range = ~df[col].between(0.0, 1.0) & df[col].notna()
                coerced += int(out_of_range.sum())
                df[col] = df[col].where(df[col].between(0.0, 1.0), np.nan)
        if coerced:
            logging.info(
                "Cleaning coerced %d out-of-range rate value(s) to NaN (for imputation).",
                coerced,
            )

        df = df.reset_index(drop=True)
        logging.info("Cleaned data: %d rows x %d columns.", *df.shape)
        return df

    def _create_preprocessor(self) -> ColumnTransformer:
        num_strategy = self.config.get("numeric_imputer_strategy", "median")
        cat_strategy = self.config.get("categorical_imputer_strategy", "constant")
        fill_value = self.config.get("categorical_fill_value", "unknown")

        numerical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy=num_strategy)),
                ("scaler", StandardScaler()),
            ]
        )
        cat_imputer = (
            SimpleImputer(strategy="constant", fill_value=fill_value)
            if cat_strategy == "constant"
            else SimpleImputer(strategy=cat_strategy)
        )
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", cat_imputer),
                # Dense output so gradient-boosting estimators (which require dense
                # input) work; the one-hot width here is small.
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        return ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, self.config["numerical_features"]),
                ("cat", categorical_transformer, self.config["categorical_features"]),
            ],
            remainder="drop",
        )

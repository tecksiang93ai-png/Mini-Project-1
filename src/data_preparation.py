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
    if config["selection_mode"] not in ("max", "min"):
        raise ValueError("selection_mode must be 'max' or 'min'.")
    if not isinstance(config["models"], dict) or not config["models"]:
        raise ValueError("config['models'] must be a non-empty mapping of models.")
    logging.info("Configuration loaded and validated (%d models).", len(config["models"]))
    return config


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

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Deterministic, documented cleaning:

        1. Drop identifier columns (``ID``, ``URL``).
        2. Drop any exact duplicate rows.
        3. Coerce out-of-range rate features (bounded in [0, 1]) to NaN so a
           corrupt value (e.g. the known n_non_stop_words spike) is imputed
           rather than distorting scaling and the models.
        """
        df = df.copy()
        drop_cols = [c for c in self.config.get("drop_columns", []) if c in df.columns]
        df = df.drop(columns=drop_cols)

        df = df.drop_duplicates()

        for col in self.config.get("rate_features", []):
            if col in df.columns:
                df[col] = df[col].where(df[col].between(0.0, 1.0), np.nan)

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

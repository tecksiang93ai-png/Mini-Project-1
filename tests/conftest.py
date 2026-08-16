"""Shared pytest fixtures for the news-popularity test suite."""
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_preparation import load_config  # noqa: E402

CONFIG_PATH = ROOT / "src" / "config.yaml"


@pytest.fixture(scope="session")
def config():
    return load_config(str(CONFIG_PATH))


@pytest.fixture
def fast_config(config):
    """Config restricted to fast models, for training smoke tests."""
    cfg = copy.deepcopy(config)
    cfg["models"] = {
        "dummy": {"estimator": "DummyRegressor", "grid": {}},
        "linear_regression": {"estimator": "LinearRegression", "grid": {}},
        "ridge": {"estimator": "Ridge", "grid": {"regressor__alpha": [1.0, 10.0]}},
    }
    cfg["cv"] = 2
    return cfg


@pytest.fixture
def raw_df(config):
    """Synthetic frame mirroring the raw schema, with deliberate quality issues:
    a duplicate row, an out-of-range rate value, and missing num_videos/data_channel."""
    rng = np.random.default_rng(0)
    n = 60
    data = {"ID": np.arange(1, n + 1),
            "URL": [f"http://example.com/{i}" for i in range(n)],
            "weekday": rng.choice(
                ["monday", "tuesday", "saturday", "sunday"], n),
            "data_channel": rng.choice(
                ["world", "technology", "business", "social_media"], n),
            "shares": rng.integers(100, 5000, n).astype(float)}
    rate = {"n_unique_tokens", "n_non_stop_words", "n_non_stop_unique_tokens"}
    for col in config["numerical_features"]:
        if col in rate:
            data[col] = rng.uniform(0, 1, n)
        else:
            data[col] = rng.uniform(0, 100, n)
    df = pd.DataFrame(data)
    # Inject issues:
    df.loc[0, "n_non_stop_words"] = 5.0          # out of [0,1] -> must become NaN
    df.loc[1, "num_videos"] = np.nan             # missing numeric
    df.loc[2, "data_channel"] = np.nan           # missing category
    df = pd.concat([df, df.iloc[[5]]], ignore_index=True)   # exact duplicate
    return df

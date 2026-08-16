"""Tests for config/data loaders, cleaning, and the preprocessor."""
import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_preparation import (
    DataPreparation,
    _validate_config_schema,
    load_config,
    load_data,
)


def _branch(preprocessor, name):
    for n, transformer, _cols in preprocessor.transformers:
        if n == name:
            return transformer
    raise KeyError(name)


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def test_clean_drops_identifier_columns(config, raw_df):
    clean = DataPreparation(config).clean_data(raw_df.copy())
    assert "ID" not in clean.columns and "URL" not in clean.columns


def test_clean_drops_exact_duplicates(config, raw_df):
    # raw_df has 60 unique rows + 1 appended duplicate.
    clean = DataPreparation(config).clean_data(raw_df.copy())
    assert len(clean) == 60


def test_clean_clips_out_of_range_rate_feature(config, raw_df):
    # Row 0 has n_non_stop_words = 5.0, which is outside [0, 1] -> NaN.
    clean = DataPreparation(config).clean_data(raw_df.copy())
    assert clean["n_non_stop_words"].max() <= 1.0
    assert clean["n_non_stop_words"].isna().sum() >= 1


def test_clean_keeps_target(config, raw_df):
    clean = DataPreparation(config).clean_data(raw_df.copy())
    assert config["target_column"] in clean.columns


def test_clean_data_without_dedup_keeps_all_rows(config, raw_df):
    # raw_df has an appended exact duplicate; deduplicate=False keeps every row
    # (the inference path relies on this so predictions stay 1:1 with input).
    clean = DataPreparation(config).clean_data(raw_df.copy(), deduplicate=False)
    assert len(clean) == len(raw_df)


# --------------------------------------------------------------------------- #
# Preprocessor
# --------------------------------------------------------------------------- #
def test_numeric_branch_imputes_and_scales(config):
    pre = DataPreparation(config).preprocessor
    num = _branch(pre, "num")
    assert isinstance(num.named_steps["imputer"], SimpleImputer)
    assert num.named_steps["imputer"].strategy == config["numeric_imputer_strategy"]
    assert isinstance(num.named_steps["scaler"], StandardScaler)


def test_categorical_branch_constant_impute_and_dense_onehot(config):
    pre = DataPreparation(config).preprocessor
    cat = _branch(pre, "cat")
    imp = cat.named_steps["imputer"]
    onehot = cat.named_steps["onehot"]
    assert isinstance(imp, SimpleImputer) and imp.strategy == "constant"
    assert imp.fill_value == config["categorical_fill_value"]
    assert onehot.handle_unknown == "ignore" and onehot.sparse_output is False


def test_preprocessor_imputes_all_nans(config, raw_df):
    prep = DataPreparation(config)
    clean = prep.clean_data(raw_df.copy())
    X = clean.drop(columns=[config["target_column"]])
    out = prep.preprocessor.fit_transform(X)
    assert not np.isnan(out).any()


def test_missing_data_channel_becomes_unknown_category(config, raw_df):
    """The constant imputer routes a missing data_channel to its own category, so
    the model can learn that 'missing channel' is distinct."""
    prep = DataPreparation(config)
    clean = prep.clean_data(raw_df.copy())
    X = clean.drop(columns=[config["target_column"]])
    prep.preprocessor.fit(X)
    # After fitting, the fitted sub-transformers live in named_transformers_.
    cat_onehot = prep.preprocessor.named_transformers_["cat"].named_steps["onehot"]
    channel_cats = cat_onehot.categories_[1]  # 2nd categorical = data_channel
    assert config["categorical_fill_value"] in channel_cats


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def test_load_config_valid(config):
    assert isinstance(config["models"], dict) and config["models"]
    assert config["log_transform_target"] is True


def test_load_config_missing_file_raises():
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        load_config("./src/nope.yaml")


def test_load_config_missing_key_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("target_column: shares\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required key"):
        load_config(str(bad))


def test_config_schema_rejects_bad_type(config):
    bad = dict(config, cv=2.5)  # float where an int is required
    with pytest.raises(ValueError, match="cv"):
        _validate_config_schema(bad)


def test_config_schema_rejects_out_of_range(config):
    bad = dict(config, val_test_size=1.5)  # must be a fraction in (0, 1)
    with pytest.raises(ValueError, match="val_test_size"):
        _validate_config_schema(bad)


def test_config_schema_rejects_malformed_model(config):
    bad = dict(config, models={"oops": {"grid": {}}})  # missing 'estimator'
    with pytest.raises(ValueError, match="estimator"):
        _validate_config_schema(bad)


def test_load_data_missing_column_raises(config, tmp_path):
    incomplete = tmp_path / "data.csv"
    pd.DataFrame({"shares": [1.0, 2.0]}).to_csv(incomplete, index=False)
    cfg = dict(config, file_path=str(incomplete))
    with pytest.raises(ValueError, match="missing expected column"):
        load_data(cfg)


def test_load_data_non_numeric_target_raises(config, tmp_path):
    bad = tmp_path / "data.csv"
    frame = {c: [0.0, 1.0] for c in config["numerical_features"]}
    for c in config["categorical_features"]:
        frame[c] = ["x", "y"]
    frame[config["target_column"]] = ["a", "b"]  # non-numeric target
    pd.DataFrame(frame).to_csv(bad, index=False)
    cfg = dict(config, file_path=str(bad))
    with pytest.raises(ValueError, match="must be numeric"):
        load_data(cfg)

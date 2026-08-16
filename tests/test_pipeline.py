"""End-to-end integration tests through the top-level orchestration (main.py)."""
import copy
from pathlib import Path

import pandas as pd
import pytest
import yaml

import main as main_module


def _write_fast_project(config, raw_df, tmp_path):
    """Write a tiny self-contained project (data + fast config) into tmp_path."""
    data_csv = tmp_path / "data.csv"
    raw_df.to_csv(data_csv, index=False)
    cfg = copy.deepcopy(config)
    cfg["models"] = {
        "dummy": {"estimator": "DummyRegressor", "grid": {}},
        "ridge": {"estimator": "Ridge", "grid": {"regressor__alpha": [1.0]}},
    }
    cfg["cv"] = 2
    cfg["file_path"] = str(data_csv)
    cfg["artifacts_dir"] = str(tmp_path / "artifacts")
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return cfg, cfg_path, data_csv


def test_train_end_to_end(config, raw_df, tmp_path):
    cfg, cfg_path, _ = _write_fast_project(config, raw_df, tmp_path)
    best = main_module.train(str(cfg_path))
    assert best in cfg["models"]

    art = Path(cfg["artifacts_dir"])
    for fname in ("best_model.joblib", "preprocessor.joblib", "model_comparison.csv",
                  "split_indices.csv", "run_manifest.json", "run.log"):
        assert (art / fname).is_file(), f"missing artifact: {fname}"

    # Metric-value expectation: the mean-predicting baseline sits at ~0 log-R2.
    comp = pd.read_csv(art / "model_comparison.csv").set_index("model")
    assert comp.loc["dummy", "val_R2_log"] <= 0.01


def test_predict_end_to_end(config, raw_df, tmp_path):
    cfg, cfg_path, data_csv = _write_fast_project(config, raw_df, tmp_path)
    main_module.train(str(cfg_path))

    out = tmp_path / "preds.csv"
    result = main_module.predict(str(cfg_path), str(data_csv), str(out))
    assert out.is_file()
    pred_col = "predicted_" + cfg["target_column"]
    assert pred_col in result.columns
    assert len(result) == len(raw_df)
    assert (result[pred_col] >= 0).all()   # shares predictions are non-negative


def test_predict_without_trained_model_raises(config, raw_df, tmp_path):
    cfg, cfg_path, data_csv = _write_fast_project(config, raw_df, tmp_path)
    # No training has run, so no saved model exists.
    with pytest.raises(FileNotFoundError, match="Trained model not found"):
        main_module.predict(str(cfg_path), str(data_csv), str(tmp_path / "out.csv"))


def test_predict_missing_columns_raises(config, raw_df, tmp_path):
    cfg, cfg_path, data_csv = _write_fast_project(config, raw_df, tmp_path)
    main_module.train(str(cfg_path))
    bad_input = tmp_path / "bad_input.csv"
    pd.DataFrame({"weekday": ["monday", "tuesday"]}).to_csv(bad_input, index=False)
    with pytest.raises(ValueError, match="missing required column"):
        main_module.predict(str(cfg_path), str(bad_input), str(tmp_path / "out.csv"))


def test_train_missing_data_file_raises(config, tmp_path):
    cfg = copy.deepcopy(config)
    cfg["file_path"] = str(tmp_path / "does_not_exist.csv")
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="Data file not found"):
        main_module.train(str(cfg_path))

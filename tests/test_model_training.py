"""Tests for the model-training layer: log-target, factory, leakage, selection, artifacts."""
import copy

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from src.data_preparation import DataPreparation
from src.model_training import ModelTraining


@pytest.fixture
def trainer_and_data(fast_config, raw_df):
    prep = DataPreparation(fast_config)
    clean = prep.clean_data(raw_df.copy())
    trainer = ModelTraining(fast_config, prep.preprocessor)
    return trainer, trainer.split_data(clean)


@pytest.fixture
def trained(trainer_and_data):
    trainer, split = trainer_and_data
    X_train, X_val, X_test, y_train, y_val, y_test = split
    results = trainer.train_and_tune(X_train, y_train, X_val, y_val)
    return trainer, split, results


def test_make_pipeline_unknown_estimator_raises(fast_config):
    trainer = ModelTraining(fast_config, DataPreparation(fast_config).preprocessor)
    with pytest.raises(ValueError):
        trainer._make_pipeline("NotAModel")


def test_make_pipeline_seeds_stochastic_model(fast_config):
    trainer = ModelTraining(fast_config, DataPreparation(fast_config).preprocessor)
    rf = trainer._make_pipeline("RandomForestRegressor").named_steps["regressor"]
    assert isinstance(rf, RandomForestRegressor)
    assert rf.random_state == fast_config["random_seed"]


def test_log_target_roundtrip(fast_config):
    """_to_shares must invert _to_model_space (log1p <-> expm1)."""
    trainer = ModelTraining(fast_config, DataPreparation(fast_config).preprocessor)
    y = np.array([4.0, 100.0, 5000.0, 843300.0])
    back = trainer._to_shares(trainer._to_model_space(y))
    assert np.allclose(back, y, rtol=1e-9)


def test_split_has_no_target_leakage(fast_config, trainer_and_data):
    _, (X_train, X_val, X_test, *_ ) = trainer_and_data
    for X in (X_train, X_val, X_test):
        assert fast_config["target_column"] not in X.columns


def test_train_and_tune_returns_dual_scale_metrics(trained, fast_config):
    trainer, split, results = trained
    assert set(results) == set(fast_config["models"])
    for r in results.values():
        assert {"MAE", "RMSE", "R2", "MAE_log", "RMSE_log", "R2_log"} <= set(r["val_metrics"])
        assert "cv_mean" in r and "cv_std" in r


def test_select_best_uses_log_metric(trained, fast_config):
    trainer, split, results = trained
    best = trainer.select_best(results)
    assert best in fast_config["models"]
    assert results[best]["val_metrics"]["R2_log"] >= results["dummy"]["val_metrics"]["R2_log"]


def test_select_best_deterministic_tiebreak(fast_config):
    trainer = ModelTraining(fast_config, DataPreparation(fast_config).preprocessor)
    results = {
        "b": {"val_metrics": {"R2_log": 0.5, "RMSE_log": 0.9}},
        "a": {"val_metrics": {"R2_log": 0.5, "RMSE_log": 0.7}},
    }
    assert trainer.select_best(results) == "a"   # lower RMSE_log wins the tie


def test_refit_on_train_val_predicts_nonnegative_shares(trained):
    trainer, (X_train, X_val, X_test, y_train, y_val, y_test), results = trained
    best = trainer.select_best(results)
    final = trainer.refit_on_train_val(results[best]["pipeline"], X_train, X_val, y_train, y_val)
    preds = trainer._to_shares(final.predict(X_test))
    assert len(preds) == len(X_test)
    assert (preds >= 0).all()   # shares can never be negative


def test_save_artifacts_writes_expected_files(fast_config, raw_df, tmp_path):
    cfg = copy.deepcopy(fast_config)
    cfg["artifacts_dir"] = str(tmp_path)
    prep = DataPreparation(cfg)
    clean = prep.clean_data(raw_df.copy())
    trainer = ModelTraining(cfg, prep.preprocessor)
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.split_data(clean)
    results = trainer.train_and_tune(X_train, y_train, X_val, y_val)
    best = trainer.select_best(results)
    final = trainer.refit_on_train_val(results[best]["pipeline"], X_train, X_val, y_train, y_val)
    metrics = trainer.evaluate(final, X_test, y_test, "test")
    trainer.save_artifacts(best, final, results, metrics, X_test, y_test)
    for fname in ("best_model.joblib", "model_comparison.csv", "test_predictions.csv",
                  "run_manifest.json", "environment.txt"):
        assert (tmp_path / fname).is_file(), f"missing artifact: {fname}"

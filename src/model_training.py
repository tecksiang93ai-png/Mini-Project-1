"""
model_training.py
=================

Training, tuning, selection, evaluation, and artifact persistence for the Online
News Popularity regression task.

The set of models, their grids, the selection policy, an optional feature-selection
stage, and the output location all come from ``config``. Models are trained on the
(log-)target and scored on both the original shares scale and the log scale.

Public API
----------
* ``ModelTraining.split_data``        -> train/val/test split
* ``ModelTraining.train_and_tune``    -> per-model fitted pipeline + metrics
* ``ModelTraining.select_best``       -> winning model name (deterministic)
* ``ModelTraining.refit_on_train_val``-> winner refit on train+val
* ``ModelTraining.evaluate``          -> dual-scale metric dict
* ``ModelTraining.save_artifacts``    -> persist model, preprocessor, split, manifest
"""

# Standard library imports
import json
import logging
import platform
import shutil
import uuid
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict

# Related third-party imports
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_regression
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

# Registry mapping the estimator names used in config.yaml to their classes.
ESTIMATORS = {
    "DummyRegressor": DummyRegressor,
    "LinearRegression": LinearRegression,
    "Ridge": Ridge,
    "Lasso": Lasso,
    "RandomForestRegressor": RandomForestRegressor,
    "GradientBoostingRegressor": GradientBoostingRegressor,
    "HistGradientBoostingRegressor": HistGradientBoostingRegressor,
}
_SEEDED = {
    "RandomForestRegressor",
    "GradientBoostingRegressor",
    "HistGradientBoostingRegressor",
}
_TRACKED_PACKAGES = [
    "pandas", "numpy", "scikit-learn", "scipy",
    "matplotlib", "seaborn", "pyyaml", "joblib",
]


def _safe_version(pkg: str) -> str:
    try:
        return version(pkg)
    except PackageNotFoundError:
        return "not installed"


class ModelTraining:
    """Train, tune, select, and evaluate regression models for news `shares`.

    When ``log_transform_target`` is set, models are trained on ``log1p(shares)``
    and predictions are inverse-transformed for original-scale reporting. Every
    model is scored on BOTH the original shares scale (MAE/MSE/RMSE/R2) and the
    log scale (…_log); selection uses whichever metric the config names.
    """

    def __init__(self, config: Dict[str, Any], preprocessor: ColumnTransformer):
        self.config = config
        self.preprocessor = preprocessor
        self.seed = int(config.get("random_seed", 42))
        self.log = bool(config.get("log_transform_target", False))

    # ------------------------------------------------------------------ #
    # Target transform helpers
    # ------------------------------------------------------------------ #
    def _to_model_space(self, y):
        """Map the target into the space the models are trained in."""
        return np.log1p(y) if self.log else y

    def _to_shares(self, pred_model_space):
        """Map a model-space prediction back to non-negative shares."""
        if self.log:
            return np.clip(np.expm1(pred_model_space), 0, None)
        return pred_model_space

    # ------------------------------------------------------------------ #
    def split_data(self, df: pd.DataFrame):
        """Split into 80% train / 10% validation / 10% test using the config seed."""
        logging.info("Splitting data into train/validation/test.")
        X = df.drop(columns=self.config["target_column"])
        y = df[self.config["target_column"]]
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=self.config["val_test_size"], random_state=self.seed
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=self.config["val_size"], random_state=self.seed
        )
        logging.info(
            "Split sizes -> train=%d, val=%d, test=%d", len(X_train), len(X_val), len(X_test)
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def _feature_selector(self):
        """Optional feature-selection step, configured from config (off by default)."""
        fs = self.config.get("feature_selection") or {}
        if not fs.get("enabled"):
            return None
        method = fs.get("method", "variance_threshold")
        if method == "variance_threshold":
            return VarianceThreshold(threshold=fs.get("threshold", 0.0))
        if method == "select_k_best":
            return SelectKBest(score_func=f_regression, k=fs.get("k", "all"))
        raise ValueError(f"Unknown feature_selection.method: {method!r}")

    def _make_pipeline(self, estimator_name: str) -> Pipeline:
        """Build a preprocess -> (optional select) -> regressor pipeline."""
        if estimator_name not in ESTIMATORS:
            raise ValueError(
                f"Unknown estimator '{estimator_name}'. Known: {sorted(ESTIMATORS)}"
            )
        kwargs: Dict[str, Any] = {}
        if estimator_name in _SEEDED:
            kwargs["random_state"] = self.seed
        model = ESTIMATORS[estimator_name](**kwargs)
        steps = [("preprocessor", self.preprocessor)]
        selector = self._feature_selector()
        if selector is not None:
            steps.append(("selector", selector))
        steps.append(("regressor", model))
        return Pipeline(steps=steps)

    def assert_no_leakage(self, X_train, y_train) -> None:
        """Runtime leakage guard (fail-fast), run on every training run so the
        safeguard is visible in logs — not only enforced by architecture/tests.

        1. Fails if any configured feature is on the excluded post-publication list.
        2. Warns if any numeric feature is suspiciously correlated with the target.
        """
        leak = self.config.get("leakage") or {}
        excluded = set(leak.get("excluded_features", []))
        configured = set(self.config["numerical_features"]) | set(self.config["categorical_features"])
        reintroduced = sorted(excluded & configured)
        if reintroduced:
            raise ValueError(
                "Leakage guard: excluded post-publication feature(s) present in the "
                f"model feature lists: {reintroduced}. Remove them from config."
            )
        threshold = leak.get("correlation_warn_threshold", 0.9)
        y_m = self._to_model_space(y_train)
        numeric = [c for c in self.config["numerical_features"] if c in X_train.columns]
        for col in numeric:
            r = X_train[col].corr(y_m)
            if pd.notna(r) and abs(r) > threshold:
                logging.warning(
                    "Leakage guard: feature '%s' has |corr|=%.2f with the target (> %.2f) "
                    "— verify it is knowable at prediction time.", col, abs(r), threshold,
                )
        logging.info(
            "Leakage guard passed: %d excluded feature(s) absent from %d configured "
            "features; scanned %d numeric features for suspicious correlation.",
            len(excluded), len(configured), len(numeric),
        )

    def _fit_one(self, name, spec, X_train, y_train_m, cv, scoring):
        """Fit (and optionally tune) a single model; returns fitted pipeline, best
        params, and the CV mean/std on the log target."""
        grid = spec.get("grid") or {}
        pipeline = self._make_pipeline(spec["estimator"])
        if grid:
            logging.info("Tuning %s over %d-fold CV ...", name, cv)
            search = GridSearchCV(pipeline, grid, cv=cv, scoring=scoring, n_jobs=-1)
            search.fit(X_train, y_train_m)
            bi = search.best_index_
            return (search.best_estimator_, search.best_params_,
                    float(search.cv_results_["mean_test_score"][bi]),
                    float(search.cv_results_["std_test_score"][bi]))
        logging.info("Fitting %s (no tuning) ...", name)
        scores = cross_val_score(pipeline, X_train, y_train_m, cv=cv, scoring=scoring, n_jobs=-1)
        pipeline.fit(X_train, y_train_m)
        return pipeline, {}, float(scores.mean()), float(scores.std())

    def train_and_tune(self, X_train, y_train, X_val, y_val) -> Dict[str, Dict[str, Any]]:
        """Fit every configured model on the (log-)target. Models with a grid are
        tuned by cross-validated search; the rest are fit with defaults but still
        cross-validated for fold-level reporting. Each is evaluated on validation.

        Each model is isolated: if one fails (e.g. a bad grid), it is logged and
        skipped so the remaining candidates are still evaluated."""
        cv = self.config.get("cv", 5)
        scoring = self.config.get("scoring", "r2")
        y_train_m = self._to_model_space(y_train)
        logging.info(
            "Preprocessing is fit on the TRAINING split only (inside each pipeline); "
            "validation/test are transform-only."
        )
        results: Dict[str, Dict[str, Any]] = {}
        for name, spec in self.config["models"].items():
            try:
                fitted, best_params, cv_mean, cv_std = self._fit_one(
                    name, spec, X_train, y_train_m, cv, scoring)
            except Exception as exc:  # noqa: BLE001 - isolate a bad model, keep the rest
                logging.error("Model '%s' failed and is skipped: %s", name, exc)
                continue
            logging.info("  %s CV %s (log target) = %.4f +/- %.4f", name, scoring, cv_mean, cv_std)
            results[name] = {
                "pipeline": fitted,
                "val_metrics": self.evaluate(fitted, X_val, y_val, f"{name} (val)"),
                "best_params": best_params,
                "cv_mean": cv_mean,
                "cv_std": cv_std,
            }
        if not results:
            raise RuntimeError("All configured models failed during training.")
        return results

    def select_best(self, results: Dict[str, Dict[str, Any]]) -> str:
        """Pick the winner by the configured validation metric, breaking ties
        deterministically with a secondary metric then the model name."""
        metric = self.config["selection_metric"]
        mode = self.config["selection_mode"]
        tiebreak = self.config.get("selection_tiebreak", "RMSE_log")
        primary_sign = -1.0 if mode == "max" else 1.0

        def sort_key(name):
            m = results[name]["val_metrics"]
            return (primary_sign * m[metric], m[tiebreak], name)

        best = min(results, key=sort_key)
        logging.info("Best model on validation (%s %s, tie-break %s): %s",
                     mode, metric, tiebreak, best)
        return best

    def refit_on_train_val(self, fitted_pipeline, X_train, X_val, y_train, y_val):
        """Refit the winning model on train+val (on the log-target) before the
        single final test evaluation."""
        X = pd.concat([X_train, X_val])
        y = pd.concat([y_train, y_val])
        model = clone(fitted_pipeline)
        model.fit(X, self._to_model_space(y))
        logging.info("Refit best model on train+val (%d rows).", len(X))
        return model

    def evaluate(self, model, X, y, label: str) -> Dict[str, float]:
        """Metrics on the original shares scale and on the log-target scale.

        Raises ValueError on empty input or fewer than two samples, since R² is
        undefined without variance across at least two observations.
        """
        if len(X) < 2:
            raise ValueError(
                f"evaluate('{label}') needs >= 2 samples for R2; got {len(X)}."
            )
        pred_m = model.predict(X)
        y_m = self._to_model_space(y)
        pred = self._to_shares(pred_m)
        metrics = {
            "MAE": float(mean_absolute_error(y, pred)),
            "MSE": float(mean_squared_error(y, pred)),
            "RMSE": float(root_mean_squared_error(y, pred)),
            "R2": float(r2_score(y, pred)),
            "MAE_log": float(mean_absolute_error(y_m, pred_m)),
            "RMSE_log": float(root_mean_squared_error(y_m, pred_m)),
            "R2_log": float(r2_score(y_m, pred_m)),
        }
        logging.info(
            "%s -> R2_log=%.4f RMSE_log=%.3f | shares MAE=%.0f R2=%.4f",
            label, metrics["R2_log"], metrics["RMSE_log"], metrics["MAE"], metrics["R2"],
        )
        return metrics

    # ------------------------------------------------------------------ #
    # Artifact persistence (split into focused helpers)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _new_run_id() -> str:
        """Timestamp + short random suffix, so two runs in the same second don't
        collide on artifact filenames."""
        return f"{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"

    def _package_versions(self) -> Dict[str, str]:
        env = {pkg: _safe_version(pkg) for pkg in _TRACKED_PACKAGES}
        env["python"] = platform.python_version()
        return env

    def _save_models(self, out, best_name, final_model, run_id) -> None:
        # Serialize the pipeline once, then copy to the versioned name (cheaper than
        # dumping twice). Also save the standalone fitted preprocessor.
        stable = out / "best_model.joblib"
        joblib.dump(final_model, stable)
        shutil.copyfile(stable, out / f"best_model_{best_name}_{run_id}.joblib")
        joblib.dump(final_model.named_steps["preprocessor"], out / "preprocessor.joblib")

    def _save_comparison(self, out, results, run_id) -> None:
        """Write the per-model metric table (with fold-level CV mean/std), stable + versioned."""
        rows = []
        for name, r in results.items():
            row = {"model": name}
            row.update({f"val_{k}": v for k, v in r["val_metrics"].items()})
            row["cv_mean"] = r["cv_mean"]
            row["cv_std"] = r["cv_std"]
            rows.append(row)
        comparison = pd.DataFrame(rows).sort_values("val_R2_log", ascending=False)
        comparison.to_csv(out / "model_comparison.csv", index=False)
        comparison.to_csv(out / f"model_comparison_{run_id}.csv", index=False)

    def _save_predictions(self, out, final_model, X_test, y_test) -> None:
        """Write the held-out test set's true vs predicted shares."""
        pd.DataFrame(
            {"y_true": y_test.to_numpy(), "y_pred": self._to_shares(final_model.predict(X_test))}
        ).to_csv(out / "test_predictions.csv", index=False)

    def _save_split_indices(self, out, X_train, X_val, X_test) -> None:
        # Record the exact row assignment so the partition can be reproduced/inspected.
        frames = [pd.DataFrame({"row_index": idx.to_numpy(), "split": name})
                  for idx, name in ((X_train.index, "train"),
                                    (X_val.index, "validation"),
                                    (X_test.index, "test"))]
        pd.concat(frames, ignore_index=True).to_csv(out / "split_indices.csv", index=False)

    def _save_manifest(self, out, best_name, results, final_metrics, run_id, env) -> None:
        """Write the run manifest (best model, params, metrics, seed, env, config), stable + versioned."""
        manifest = {
            "run_id": run_id,
            "best_model": best_name,
            "best_params": results[best_name]["best_params"],
            "log_transform_target": self.log,
            "selection_metric": self.config["selection_metric"],
            "selection_mode": self.config["selection_mode"],
            "best_cv_mean": results[best_name]["cv_mean"],
            "best_cv_std": results[best_name]["cv_std"],
            "final_test_metrics": final_metrics,
            "random_seed": self.seed,
            "package_versions": env,
            "config_snapshot": self.config,
        }
        payload = json.dumps(manifest, indent=2, default=str)
        for fname in ("run_manifest.json", f"run_manifest_{run_id}.json"):
            (out / fname).write_text(payload, encoding="utf-8")

    def save_artifacts(self, best_name, final_model, results, final_metrics,
                       X_train, X_val, X_test, y_test) -> str:
        """Persist all run artifacts and return the run id. Writes: the fitted
        pipeline (stable + versioned) and standalone preprocessor, the model
        comparison table (with fold-level CV mean/std), test predictions, the exact
        split row indices, the runtime package versions, and a run manifest."""
        run_id = self._new_run_id()
        out = Path(self.config["artifacts_dir"])
        try:
            out.mkdir(parents=True, exist_ok=True)
            env = self._package_versions()
            self._save_models(out, best_name, final_model, run_id)
            self._save_comparison(out, results, run_id)
            self._save_predictions(out, final_model, X_test, y_test)
            self._save_split_indices(out, X_train, X_val, X_test)
            (out / "environment.txt").write_text(
                "\n".join(f"{k}=={v}" for k, v in env.items()) + "\n", encoding="utf-8"
            )
            self._save_manifest(out, best_name, results, final_metrics, run_id, env)
        except OSError as exc:
            logging.error("Failed to write artifacts to %s: %s", out, exc)
            raise
        logging.info("Artifacts (run %s) written to %s", run_id, out.resolve())
        return run_id

"""
main.py — end-to-end entry point for the Online News Popularity regression pipeline.

Predicts the number of `shares` a news article receives.

Train (default):
    python main.py
    python main.py train --config ./src/config.yaml --log-level DEBUG

Predict on new articles with the saved model:
    python main.py predict --input data/new_articles.csv --output predictions.csv

Every run appends to ``<artifacts_dir>/run.log`` in addition to the console.
"""

# Standard library imports
import argparse
import logging
import random
from pathlib import Path

# Third-party imports
import joblib
import numpy as np
import pandas as pd

# Local application imports
from src.data_preparation import DataPreparation, load_config, load_data

DEFAULT_CONFIG_PATH = "./src/config.yaml"


def set_global_seed(seed: int) -> None:
    """Seed Python and NumPy so a run is reproducible end to end."""
    random.seed(seed)
    np.random.seed(seed)


def _add_file_logging(artifacts_dir: str) -> None:
    """Persist logs alongside the run artifacts (in addition to the console)."""
    out = Path(artifacts_dir)
    out.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(out / "run.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(handler)


def train(config_path: str) -> str:
    """Run the full training pipeline and return the name of the selected model.

    Steps: load+validate config -> seed -> load+validate data -> clean -> split ->
    train+tune every model (on log-shares) -> select the best on validation ->
    refit it on train+val -> evaluate once on test -> persist artifacts.
    """
    # Deferred import so `predict` mode doesn't pay the training-module import cost.
    from src.model_training import ModelTraining

    config = load_config(config_path)
    _add_file_logging(config["artifacts_dir"])
    set_global_seed(config["random_seed"])
    logging.info(
        "Run settings -> seed=%s, log_transform_target=%s, selection=%s (%s), cv=%s",
        config["random_seed"], config["log_transform_target"],
        config["selection_metric"], config["selection_mode"], config["cv"],
    )

    df = load_data(config)

    data_prep = DataPreparation(config)
    cleaned_df = data_prep.clean_data(df)

    trainer = ModelTraining(config, data_prep.preprocessor)
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.split_data(cleaned_df)

    results = trainer.train_and_tune(X_train, y_train, X_val, y_val)
    best_name = trainer.select_best(results)

    final_model = trainer.refit_on_train_val(
        results[best_name]["pipeline"], X_train, X_val, y_train, y_val
    )
    final_metrics = trainer.evaluate(final_model, X_test, y_test, f"{best_name} (final test)")

    trainer.save_artifacts(
        best_name, final_model, results, final_metrics, X_train, X_val, X_test, y_test
    )
    logging.info("Pipeline complete. Best model: %s", best_name)
    return best_name


def predict(config_path: str, input_csv: str, output_csv: str) -> pd.DataFrame:
    """Score new articles with the saved model.

    Loads ``<artifacts_dir>/best_model.joblib``, validates + cleans the input with
    the same logic as training, predicts (inverse-transforming from log-shares when
    the target transform is on), and writes a CSV with a ``predicted_shares`` column.
    """
    config = load_config(config_path)
    _add_file_logging(config["artifacts_dir"])

    model_path = Path(config["artifacts_dir"]) / "best_model.joblib"
    if not model_path.is_file():
        raise FileNotFoundError(f"Trained model not found at {model_path}. Run training first.")
    if not Path(input_csv).is_file():
        raise FileNotFoundError(f"Input CSV not found: {Path(input_csv).resolve()}")

    raw = pd.read_csv(input_csv)
    required = set(config["numerical_features"]) | set(config["categorical_features"])
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required column(s): {sorted(missing)}")

    model = joblib.load(model_path)
    # deduplicate=False so every input row gets exactly one aligned prediction.
    cleaned = DataPreparation(config).clean_data(raw.copy(), deduplicate=False)
    cleaned = cleaned.drop(columns=[config["target_column"]], errors="ignore")

    pred = model.predict(cleaned)
    if config["log_transform_target"]:
        pred = np.clip(np.expm1(pred), 0, None)
    result = raw.copy()
    result["predicted_" + config["target_column"]] = pred

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    logging.info("Wrote %d predictions to %s", len(result), Path(output_csv).resolve())
    return result


def parse_args() -> argparse.Namespace:
    # Shared options so `--log-level`/`--config` work at the top level and on each
    # subcommand (e.g. `python main.py train --log-level DEBUG`).
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--log-level", default="INFO",
                        help="Console/file logging level (DEBUG, INFO, WARNING, ERROR).")
    common.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to config.yaml.")

    parser = argparse.ArgumentParser(
        parents=[common],
        description="News-popularity (shares) regression pipeline: train models or "
                    "predict on new articles. With no subcommand, trains.")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("train", parents=[common],
                   help="Train, compare, select, and save the best model.")

    p_pred = sub.add_parser("predict", parents=[common],
                            help="Score a CSV of new articles with the saved model.")
    p_pred.add_argument("--input", required=True, help="Input CSV of new articles (raw schema).")
    p_pred.add_argument("--output", default="predictions.csv", help="Where to write predictions.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        if args.command == "predict":
            predict(args.config, args.input, args.output)
        else:  # default to training when no subcommand is given
            config = getattr(args, "config", DEFAULT_CONFIG_PATH)
            train(config)
    except (FileNotFoundError, ValueError) as exc:
        logging.error("Pipeline aborted: %s", exc)
        raise
    except Exception:  # noqa: BLE001 - log unexpected failures before surfacing
        logging.exception("Unexpected error during run")
        raise


if __name__ == "__main__":
    main()

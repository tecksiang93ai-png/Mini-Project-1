"""
main.py — end-to-end entry point for the Online News Popularity regression pipeline.

Predicts the number of `shares` a news article receives.

    python main.py
    python main.py --config ./src/config.yaml --log-level DEBUG
"""

# Standard library imports
import argparse
import logging
import random

# Third-party imports
import numpy as np

# Local application imports
from src.data_preparation import DataPreparation, load_config, load_data
from src.model_training import ModelTraining

DEFAULT_CONFIG_PATH = "./src/config.yaml"


def set_global_seed(seed: int) -> None:
    """Seed Python and NumPy so a run is reproducible end to end."""
    random.seed(seed)
    np.random.seed(seed)


def train(config_path: str) -> str:
    """Run the full training pipeline and return the name of the selected model.

    Steps: load+validate config -> seed -> load+validate data -> clean ->
    split -> train+tune every model (on log-shares) -> select the best on
    validation -> refit it on train+val -> evaluate once on test -> persist
    artifacts.
    """
    config = load_config(config_path)
    set_global_seed(config["random_seed"])

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

    trainer.save_artifacts(best_name, final_model, results, final_metrics, X_test, y_test)
    logging.info("Pipeline complete. Best model: %s", best_name)
    return best_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="News-popularity (shares) regression pipeline.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to config.yaml.")
    parser.add_argument("--log-level", default="INFO",
                        help="Logging level: DEBUG, INFO, WARNING, ERROR.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        train(args.config)
    except (FileNotFoundError, ValueError) as exc:
        logging.error("Pipeline aborted: %s", exc)
        raise


if __name__ == "__main__":
    main()

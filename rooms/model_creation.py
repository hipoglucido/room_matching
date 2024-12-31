import time

import pandas as pd
import optuna.integration.lightgbm as lgb_tuner

from mlflow import MlflowClient
import numpy as np
from rooms.constants_config import (
    ColumnNames,
    MLFlowConfig,
    ModelConfig,
    SyntheticDataConfig,
    SEED,
)
from rooms.data_processing import RoomMatchingPipeline
import lightgbm as lgb
from rooms.data_processing import split_data
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
)
from rooms.synthetic_data import generate_synthetic_dataset
import mlflow
import mlflow.pyfunc
from loguru import logger
from typing import Dict, Any, Tuple


class RoomMatchingModel(mlflow.pyfunc.PythonModel):
    def __init__(
        self,
        model: lgb.LGBMClassifier,
        pipeline: RoomMatchingPipeline,
        threshold: float,
    ):
        """
        Initialize the RoomMatchingModel with a LightGBM model and a data processing pipeline.

        Args:
            model (lgb.LGBMClassifier): The trained LightGBM model.
            pipeline (RoomMatchingPipeline): The data processing pipeline.
        """
        self.model = model
        self.threshold = threshold
        self.pipeline = pipeline

    def predict(self, context: Any, model_input):
        """
        Predict the room matching using the model and pipeline.

        Args:
            context (Any): The context in which the model is being used.
            model_input (pd.DataFrame): The input data for the model.

        Returns:
            pd.Series: The predictions made by the model.
        """
        model_input = run_model(self.pipeline, self.model, self.threshold, model_input)
        return model_input[ColumnNames.DECISION]


def find_threshold_for_min_precision(y_true, y_prob, min_precision):
    """
    Determines the threshold on probability output to achieve a minimum precision.

    Args:
      y_true: True binary labels (0 or 1).
      y_prob: Predicted probabilities for the positive class.
      min_precision: The desired minimum precision.

    Returns:
      float: The threshold that achieves the minimum precision.
             Returns None if no threshold meets the criteria.
    """
    thresholds = np.arange(0, 1.01, 0.01)  # Generate potential thresholds
    best_threshold = None
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        precision = precision_score(y_true, y_pred)
        if precision >= min_precision:
            best_threshold = float(threshold)
            break  # Stop once the minimum precision is reached
    logger.info(f"{min_precision=} {best_threshold=}")
    return best_threshold


def take_decision(df, threshold):
    return df[ColumnNames.PROBA] >= threshold


def run_model(pipeline, model, threshold, df):
    if len(df) == 0:
        return pd.DataFrame(
            columns=list(df.columns) + [ColumnNames.PROBA, ColumnNames.DECISION]
        )
    X = pipeline.preprocess_data(df, is_training=False)
    df[ColumnNames.PROBA] = model.predict_proba(X[ColumnNames.FEATURES])[:, 1]
    df[ColumnNames.DECISION] = take_decision(df, threshold)
    return df


def create_synthetic_data_and_train_model(
    n_rows, match_ratio
) -> Tuple[lgb.LGBMClassifier, RoomMatchingPipeline, Dict[str, float]]:
    """
    Create synthetic data and train the room matching model.

    Returns:
        pd.DataFrame: The DataFrame containing the synthetic data.
    """
    df = generate_synthetic_dataset(
        n_rows=n_rows, match_ratio=match_ratio
    ).drop_duplicates()
    logger.info(f"Matching examples:\n{df[df['match']].sample(20)}")
    logger.info(f"Non-matching examples:\n{df[~df['match']].sample(20)}")

    model, pipeline, metrics = get_trained_model_obj(df)
    return model, pipeline, metrics


def get_best_params() -> dict:
    """
    Run LightGBM step-wise hyperparameter optimization with Optuna
    """
    params = {
        "n_rows": SyntheticDataConfig.N_ROWS,
        "match_ratio": SyntheticDataConfig.MATCH_RATIO,
    }
    df = generate_synthetic_dataset(**params).drop_duplicates()

    train_df, val_df, _ = split_data(
        df, ModelConfig.SPLIT_TRAIN_PCT, ModelConfig.SPLIT_VAL_PCT
    )
    pipeline = RoomMatchingPipeline()
    train_prep = pipeline.preprocess_data(train_df, is_training=True)
    val_prep = pipeline.preprocess_data(val_df, is_training=False)

    train_data = lgb.Dataset(
        train_prep[ColumnNames.FEATURES], label=train_prep[ColumnNames.TARGET]
    )
    val_data = lgb.Dataset(
        val_prep[ColumnNames.FEATURES], label=val_prep[ColumnNames.TARGET]
    )

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "verbose": 1,
        "random_state": SEED,
    }

    tuner = lgb_tuner.LightGBMTuner(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=800,
    )
    t0 = time.time()
    logger.info("Running tuner")
    tuner.run()

    best_params = tuner.best_params
    logger.info(f"Running tuner took {time.time() - t0} seconds:\n{best_params}")

    return best_params


def create_and_deploy_model() -> None:
    """
    Create and deploy the room matching model using synthetic data.
    """
    params = {
        "n_rows": SyntheticDataConfig.N_ROWS,
        "match_ratio": SyntheticDataConfig.MATCH_RATIO,
    }

    model, pipeline, metrics = create_synthetic_data_and_train_model(**params)
    mlflow.set_tracking_uri(MLFlowConfig.TRACKING_URI)
    with mlflow.start_run():
        logger.info(f"{mlflow.active_run().info.run_id=}")
        logger.info(f"{mlflow.get_tracking_uri()=}")
        mlflow.pyfunc.log_model(
            artifact_path="room_matching_model",
            python_model=RoomMatchingModel(model, pipeline, metrics["threshold"]),
            input_example=generate_synthetic_dataset(n_rows=1, match_ratio=1)[
                ["A", "B"]
            ],
            registered_model_name=MLFlowConfig.MODEL_NAME,
        )
        mlflow.log_metrics(metrics)
        mlflow.log_params({**params, **ModelConfig.LGBM_PARAMS})
    logger.info("Model saved")


def load_model() -> mlflow.pyfunc.PyFuncModel:
    """
    Load the latest version of the room matching model from MLflow.

    Returns:
        mlflow.pyfunc.PyFuncModel: The loaded model.
    """
    mlflow.set_tracking_uri(MLFlowConfig.TRACKING_URI)
    client = MlflowClient()
    logger.info(f"{client.tracking_uri=}")
    latest_version = client.get_registered_model(
        MLFlowConfig.MODEL_NAME
    ).latest_versions[0]
    model_version = latest_version.version
    model_uri = f"models:/{MLFlowConfig.MODEL_NAME}/{model_version}"
    logger.info(f"loading from {model_uri=}")
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    return loaded_model


def evaluate_predictions(df: pd.DataFrame) -> Dict[str, float]:
    """
    Evaluate the predictions of the model using average precision score and ROC AUC score.

    Args:
        df (pd.DataFrame): The DataFrame containing the true labels and predicted probabilities.

    Returns:
        Dict[str, float]: A dictionary containing the evaluation metrics.
    """
    metrics = {
        "average_precision_score": float(
            average_precision_score(df[ColumnNames.TARGET], df[ColumnNames.PROBA])
        ),
        "roc_auc_score": float(
            roc_auc_score(df[ColumnNames.TARGET], df[ColumnNames.PROBA])
        ),
        "accuracy_score": float(
            accuracy_score(df[ColumnNames.TARGET], df[ColumnNames.DECISION])
        ),
        "precision_score": float(
            precision_score(df[ColumnNames.TARGET], df[ColumnNames.DECISION])
        ),
        "recall_score": float(
            recall_score(df[ColumnNames.TARGET], df[ColumnNames.DECISION])
        ),
        **df[ColumnNames.PROBA]
        .quantile([0.25, 0.5, 0.75])
        .add_prefix(f"test_q")
        .to_dict(),
    }
    actual_auc_pr = metrics["average_precision_score"]
    logger.info(f"{metrics=}")
    tns = df[~df[ColumnNames.DECISION] & ~df[ColumnNames.TARGET]][["A", "B"]]
    tps = df[df[ColumnNames.DECISION] & df[ColumnNames.TARGET]][["A", "B"]]
    logger.info(f"Some TPs:\n{tps[:20]}")
    logger.info(f"Some TNs:\n{tns[:20]}")
    if actual_auc_pr < ModelConfig.MIN_AUC_PR_ON_TEST:
        logger.warning(
            f"Model performs poorly ({actual_auc_pr=}, {ModelConfig.MIN_AUC_PR_ON_TEST=})"
        )

    return metrics


def get_trained_model_obj(
    df: pd.DataFrame,
) -> Tuple[lgb.LGBMClassifier, RoomMatchingPipeline, Dict[str, float]]:
    """
    Train the room matching model and return the model, pipeline, and evaluation metrics.

    Args:
        df (pd.DataFrame): The DataFrame containing the training data.

    Returns:
        Tuple[lgb.LGBMClassifier, RoomMatchingPipeline, Dict[str, float]]: The trained model, pipeline, and evaluation metrics.
    """
    train_df, val_df, test_df = split_data(
        df, ModelConfig.SPLIT_TRAIN_PCT, ModelConfig.SPLIT_VAL_PCT
    )
    pipeline = RoomMatchingPipeline()
    train_prep = pipeline.preprocess_data(train_df, is_training=True)
    val_prep = pipeline.preprocess_data(val_df, is_training=False)
    test_prep = pipeline.preprocess_data(test_df, is_training=False)
    model = get_trained_model(train_prep, val_prep)

    val_prep[ColumnNames.PROBA] = model.predict_proba(val_prep[ColumnNames.FEATURES])[
        :, 1
    ]
    threshold = find_threshold_for_min_precision(
        y_true=val_prep[ColumnNames.TARGET],
        y_prob=val_prep[ColumnNames.PROBA],
        min_precision=ModelConfig.MIN_PRECISION,
    )
    test_prep[ColumnNames.PROBA] = model.predict_proba(test_prep[ColumnNames.FEATURES])[
        :, 1
    ]

    test_prep[ColumnNames.DECISION] = take_decision(test_prep, threshold)

    metrics = evaluate_predictions(test_prep)
    metrics["threshold"] = threshold
    return model, pipeline, metrics


def get_lgbm_feature_importance(model):
    """
    Calculates and returns LightGBM feature importance as a sorted DataFrame.

    Args:
      model: Trained LightGBM model.
      X_train: Training data (pandas DataFrame) used to train the model.

    Returns:
      pandas.DataFrame: DataFrame with feature names and their importance scores, sorted in descending order.
    """
    # Get feature importances
    importances = model.feature_importances_

    # Create a DataFrame
    importance_df = pd.DataFrame(
        {"Feature": ColumnNames.FEATURES, "Importance": importances}
    )

    # Sort the DataFrame by importance in descending order
    importance_df = importance_df.sort_values(by="Importance", ascending=False)

    return importance_df


def get_trained_model(
    train_prep: pd.DataFrame, val_prep: pd.DataFrame
) -> lgb.LGBMClassifier:
    """
    Train the LightGBM model using the training and validation data.

    Args:
        train_prep (pd.DataFrame): The preprocessed training data.
        val_prep (pd.DataFrame): The preprocessed validation data.

    Returns:
        lgb.LGBMClassifier: The trained LightGBM model.
    """

    logger.info(f"{ModelConfig.LGBM_PARAMS=}")
    model = lgb.LGBMClassifier(**ModelConfig.LGBM_PARAMS)
    model.fit(
        train_prep[ColumnNames.FEATURES],
        train_prep[ColumnNames.TARGET],
        eval_set=[(val_prep[ColumnNames.FEATURES], val_prep[ColumnNames.TARGET])],
    )
    feature_importances = get_lgbm_feature_importance(model)
    logger.info(f"Feature importances:\n{feature_importances}")
    return model


def get_dummy_prediction_from_mlflow() -> None:
    """
    Generate dummy predictions using the loaded model and log the results.
    """
    model = load_model()
    df = generate_synthetic_dataset(n_rows=5, match_ratio=1)
    df["pred"] = model.predict(df[["A", "B"]])
    logger.info(f"Match test:\n{df=}")
    df = generate_synthetic_dataset(n_rows=5, match_ratio=0)
    df["pred"] = model.predict(df[["A", "B"]])
    logger.info(f"Match test:\n{df=}")


if __name__ == "__main__":
    create_and_deploy_model()
    get_dummy_prediction_from_mlflow()


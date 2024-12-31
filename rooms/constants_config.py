from loguru import logger
import os
import pandas as pd

pd.set_option("display.max_columns", 20)  # Show all columns
pd.set_option("display.max_rows", 20)  # Show all rows (optional)
pd.set_option("display.width", 500)
pd.set_option("display.max_colwidth", 1000)

SEED = 7


class ColumnNames:
    FEATURES = [
        "cosine_similarity",
        "levenshtein_distance",
        "jaro_winkler_similarity",
        "embedding_cosine_similarity",
    ]
    TARGET = "match"
    PROBA = "proba"
    DECISION = "decision"


class ModelConfig:
    LGBM_PARAMS = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "feature_pre_filter": False,
        "lambda_l1": 1.833543386089636e-08,
        "lambda_l2": 1.2657730280659832e-08,
        "num_leaves": 2,
        "feature_fraction": 1.0,
        "bagging_fraction": 0.9421740075414173,
        "bagging_freq": 3,
        "min_child_samples": 20,
        "n_estimators": 100,
        "verbose": -1,
        "random_state": SEED,
    }
    TRANSFORMER_NAME = "all-MiniLM-L6-v2"
    MIN_AUC_PR_ON_TEST = 0.98
    MIN_PRECISION = 0.8
    SPLIT_TRAIN_PCT = 0.6
    SPLIT_VAL_PCT = 0.3
    # SPLIT_TEST_PCT is derived from the previous two


class SyntheticDataConfig:
    N_ROWS = 2000
    MATCH_RATIO = 0.5


DATA_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid warning related to transformers
LOCALHOST = "http://127.0.0.1"


class FlaskConfig:
    PORT = 8080
    ROUTE = "room_matching"
    APP_PATH = "./app.py"


class MLFlowConfig:
    MODEL_NAME = "RoomMatchingModel"
    TRACKING_URI = os.path.join(DATA_FOLDER, "mlruns")


LOGS_FILE = os.path.join(DATA_FOLDER, "rooms.log")

logger.add(
    LOGS_FILE,
    rotation="10 MB",
    retention="10 days",
    level="DEBUG",
    format="{time} {level} {module} - {message}",
)

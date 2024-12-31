import pandas as pd
import pytest

from rooms.constants_config import SyntheticDataConfig
from rooms.model_creation import (
    create_synthetic_data_and_train_model,
    run_model,
    find_threshold_for_min_precision,
)

from rooms.model_creation import get_best_params


@pytest.fixture(scope="module")
def trained_objects():
    params = {
        "n_rows": SyntheticDataConfig.N_ROWS,
        "match_ratio": SyntheticDataConfig.MATCH_RATIO,
    }

    model, pipeline, metrics = create_synthetic_data_and_train_model(**params)
    return model, pipeline, metrics


@pytest.mark.parametrize(
    argnames=["df", "expected"],
    argvalues=[
        (
            pd.DataFrame(
                [
                    {
                        "A": "room with a Veranda",
                        "B": "big room with balcony",
                    },
                    {"A": "small room", "B": "huge room"},
                ]
            ),
            pd.DataFrame(
                [
                    {
                        "A": "room with a Veranda",
                        "B": "big room with balcony",
                        "decision": True,
                    },
                    {"A": "small room", "B": "huge room", "decision": False},
                ]
            ),
        ),
        (
            pd.DataFrame(columns=["A", "B"]),
            pd.DataFrame(columns=["A", "B", "decision"]),
        ),
        (
            pd.DataFrame([{"A": "", "B": None}]),
            pd.DataFrame([{"A": "", "B": None, "decision": False}]),
        ),
        (
            pd.DataFrame([{"A": None, "B": ""}]),
            pd.DataFrame([{"A": None, "B": "", "decision": False}]),
        ),
    ],
)
def test_create_synthetic_data_and_train_model(trained_objects, df, expected):
    model, pipeline, metrics = trained_objects
    df = run_model(pipeline, model, metrics["threshold"], df)
    actual = df[["A", "B", "decision"]]
    pd.testing.assert_frame_equal(actual, expected)


def test_find_threshold_for_min_precision():
    y_true = [0, 1, 0, 1, 1, 0, 1, 0]
    y_prob = [0.2, 0.7, 0.4, 0.9, 0.3, 0.6, 0.8, 0.1]
    min_precision = 0.8

    actual = find_threshold_for_min_precision(y_true, y_prob, min_precision)
    expected = 0.61
    assert actual == expected

def test_get_best_params_dry_run():
    """Just do a dry run"""
    best_params = get_best_params()
    assert best_params
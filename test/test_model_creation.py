import pandas as pd


from rooms.common import SyntheticDataConfig, ColumnNames
from rooms.model_creation import create_synthetic_data_and_train_model, run_model

from rooms.model_creation import get_lgbm_feature_importance
def test_create_synthetic_data_and_train_model():
    params = {"n_rows": SyntheticDataConfig.N_ROWS, "match_ratio": SyntheticDataConfig.MATCH_RATIO}

    model, pipeline, metrics = create_synthetic_data_and_train_model(**params)

    df = pd.DataFrame([
        {"Big room with balcony", "room with a veranda"},

        {'huge room', 'small room'},

    ], columns=['A', 'B'])
    df = run_model(pipeline, model, df)

    actual = df[['A','B', 'decision']]
    expected = pd.DataFrame([{'A': 'room with a veranda', 'B': 'big room with balcony', 'decision': True},
     {'A': 'huge room', 'B': 'small room', 'decision': False}])
    pd.testing.assert_frame_equal(actual, expected)
    assert True

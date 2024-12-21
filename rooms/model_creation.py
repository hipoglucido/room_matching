from rooms.common import FEATURE_NAMES, TARGET_NAME, PROBA_NAME, MIN_AUC_PR_ON_TEST
from rooms.data_processing import RoomMatchingPipeline
import lightgbm as lgb
from rooms.data_processing import split_data
from sklearn.metrics import average_precision_score
from rooms.synthetic_data import generate_synthetic_dataset


def create_and_deploy_model():
    df = generate_synthetic_dataset(n_rows=1000, match_ratio=.2).drop_duplicates()
    train_df, val_df, test_df = split_data(df, .6, .2)

    pipeline = RoomMatchingPipeline()
    train_prep = pipeline.preprocess_data(train_df, is_training=True)
    val_prep = pipeline.preprocess_data(val_df, is_training=False)
    test_prep = pipeline.preprocess_data(test_df, is_training=False)

    model = get_trained_model(train_prep, val_prep)
    test_prep[PROBA_NAME] = model.predict_proba(test_prep[FEATURE_NAMES])[:, 1]
    score = average_precision_score(test_prep[TARGET_NAME], test_prep[PROBA_NAME])
    assert score > MIN_AUC_PR_ON_TEST


def get_trained_model(train_prep, val_prep):
    from sklearn.model_selection import train_test_split

    # Define LightGBM parameters (you can adjust these)
    params = {
        'objective': 'binary',  # Assuming binary classification
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'n_estimators': 100,
    }

    model = lgb.LGBMClassifier(
        objective='binary',
        metric='binary_logloss',
        boosting_type='gbdt',
        num_leaves=31,
        learning_rate=0.1,
        n_estimators=100,
        early_stopping_rounds=10,
        verbose=-1  # Print evaluation metrics during training
    )

    # Train the model
    model.fit(
        train_prep[FEATURE_NAMES],
        train_prep[TARGET_NAME],
        eval_set=[(val_prep[FEATURE_NAMES], val_prep[TARGET_NAME])],

    )
    return model


if __name__ == '__main__':
    create_and_deploy_model()

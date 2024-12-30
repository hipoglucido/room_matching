import itertools
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from jellyfish import levenshtein_distance, jaro_winkler_similarity
from rooms.common import ColumnNames, ModelConfig
from loguru import logger
from typing import Tuple, List


def split_data(
    df: pd.DataFrame, train_pct: float, val_pct: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits a DataFrame into train, validation, and test sets.
    The test set percentage is calculated automatically.
    Keeps the target variable within the DataFrame splits.

    Args:
      df: Pandas DataFrame to be split.
      train_pct: Percentage of data for the training set (float between 0 and 1).
      val_pct: Percentage of data for the validation set (float between 0 and 1).

    Returns:
      A tuple of DataFrames: (train_df, val_df, test_df)
    """
    logger.info(f"{df.shape=}, {train_pct=}, {val_pct=}")

    if train_pct + val_pct >= 1.0:
        raise ValueError(
            "Train and validation percentages must add up to less than 1.0"
        )

    test_pct = 1.0 - train_pct - val_pct  # Calculate test percentage

    # First split into train and remaining (val + test)
    train_df, remain_df = train_test_split(
        df, test_size=val_pct + test_pct, random_state=42  # Use the entire DataFrame
    )

    # Calculate the new test size for the remaining data
    remaining_test_pct = test_pct / (val_pct + test_pct)

    # Split the remaining data into validation and test
    val_df, test_df = train_test_split(
        remain_df, test_size=remaining_test_pct, random_state=42
    )

    return train_df, val_df, test_df


def normalize_string_column(series: pd.Series) -> pd.Series:
    return series.fillna("").str.lower().str.replace("[^a-zA-Z0-9\s]", "", regex=True)


class RoomMatchingPipeline:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer()
        self.std_scaler = MinMaxScaler()
        self.sentence_transformer = SentenceTransformer(ModelConfig.TRANSFORMER_NAME)
        self.scaler = StandardScaler()

    def preprocess_data(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """
        Preprocesses the DataFrame for the room matching task.

        Args:
            df: Pandas DataFrame with columns 'A', 'B', and 'match'.
            is_training: Boolean indicating if this is for training (True) or prediction (False).

        Returns:
            pd.DataFrame: DataFrame with processed features.
        """

        # 1. Text Normalization (optional, but recommended)
        df["A_norm"] = normalize_string_column(df["A"])
        df["B_norm"] = normalize_string_column(df["B"])

        # 2. Feature Engineering
        df = self.create_features(df, is_training)

        return df

    def create_features(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """
        Creates semantic, string distance, and sentence embedding features.

        Args:
            df: Pandas DataFrame with columns 'A' and 'B'.
            is_training: Boolean indicating if this is for training (True) or prediction (False).

        Returns:
            pd.DataFrame: DataFrame with added features.
        """
        logger.info(f"{len(df)=} {is_training=} {ColumnNames.FEATURES=}")
        # 1. Semantic features using TF-IDF

        if is_training:
            self.tfidf_vectorizer.fit(pd.concat([df["A_norm"], df["B_norm"]]))

        tfidf_A = self.tfidf_vectorizer.transform(df["A_norm"])
        tfidf_B = self.tfidf_vectorizer.transform(df["B_norm"])
        df["cosine_similarity"] = cosine_similarity(tfidf_A, tfidf_B).diagonal()

        # 2. String distance features

        df["levenshtein_distance"] = df.apply(
            lambda row: levenshtein_distance(row["A_norm"], row["B_norm"]), axis=1
        )

        df["jaro_winkler_similarity"] = df.apply(
            lambda row: jaro_winkler_similarity(row["A_norm"], row["B_norm"]), axis=1
        )

        embeddings_A = self.sentence_transformer.encode(
            df["A_norm"].tolist(), convert_to_tensor=True
        )
        embeddings_B = self.sentence_transformer.encode(
            df["B_norm"].tolist(), convert_to_tensor=True
        )

        embeddings_A = embeddings_A.cpu()  # Move to CPU
        embeddings_B = embeddings_B.cpu()  # Move to CPU

        df["embedding_cosine_similarity"] = cosine_similarity(
            embeddings_A, embeddings_B
        ).diagonal()

        assert all(
            [f in df for f in ColumnNames.FEATURES]
        ), f"Missing features {ColumnNames.FEATURES=} {df.columns=}"
        df.drop(columns=["A_norm", "B_norm"], inplace=True)
        return df


def prepare_match_candidate_pairs(
    reference_catalog: List[str], supplier_catalog: List[str]
) -> pd.DataFrame:
    """
    Prepare a dataframe with all possible pairs of reference and supplier rooms.

    Args:
        reference_catalog: List of reference room names.
        supplier_catalog: List of supplier room names.

    Returns:
        pd.DataFrame: DataFrame containing all possible pairs of reference and supplier rooms.
    """
    df = pd.DataFrame(
        list(itertools.product(reference_catalog, supplier_catalog)), columns=["A", "B"]
    )
    df_pos = pd.DataFrame(
        list(
            itertools.product(
                range(len(reference_catalog)), range(len(supplier_catalog))
            )
        ),
        columns=["A_pos", "B_pos"],
    )
    return pd.concat([df, df_pos], axis=1, ignore_index=False)


def is_room_valid(rooms: pd.Series) -> pd.Series:
    """
    Check if the room name is valid. Assume that a valid room has more than one character.

    Args:
        rooms: The room names to check.

    Returns:
        pd.Series: A boolean series indicating if the room name is valid.
    """
    return rooms.str.strip().str.len() > 1


def remove_invalid_rooms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes rows where either room name is either null, empty, or contains only whitespace.

    Args:
        df: DataFrame containing room names.

    Returns:
        pd.DataFrame: DataFrame with invalid room names removed.
    """
    return df[is_room_valid(df["A"]) & is_room_valid(df["B"])]

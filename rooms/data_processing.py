import pandas as pd
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from jellyfish import levenshtein_distance, jaro_winkler_similarity

from rooms.common import FEATURE_NAMES


def split_data(df, train_pct, val_pct):
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

  if train_pct + val_pct >= 1.0:
    raise ValueError("Train and validation percentages must add up to less than 1.0")

  test_pct = 1.0 - train_pct - val_pct  # Calculate test percentage

  # First split into train and remaining (val + test)
  train_df, remain_df = train_test_split(
      df,  # Use the entire DataFrame
      test_size=val_pct + test_pct,
      random_state=42
  )

  # Calculate the new test size for the remaining data
  remaining_test_pct = test_pct / (val_pct + test_pct)

  # Split the remaining data into validation and test
  val_df, test_df = train_test_split(
      remain_df,
      test_size=remaining_test_pct,
      random_state=42
  )

  return train_df, val_df, test_df



class RoomMatchingPipeline:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer()
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        self.scaler = StandardScaler()

    def preprocess_data(self, df, is_training=True):
        """
        Preprocesses the DataFrame for the room matching task.

        Args:
            df: Pandas DataFrame with columns 'A', 'B', and 'match'.
            is_training: Boolean indicating if this is for training (True) or prediction (False).

        Returns:
            A tuple containing:
                - X: DataFrame with processed features.
                - y: Series with the target variable ('match') if is_training is True, otherwise None.
        """

        # 1. Text Normalization (optional, but recommended)
        df['A'] = df['A'].str.lower().str.replace('[^a-zA-Z0-9\s]', '', regex=True)
        df['B'] = df['B'].str.lower().str.replace('[^a-zA-Z0-9\s]', '', regex=True)

        # 2. Feature Engineering
        df = self.create_features(df, is_training)

        return df

    def create_features(self, df, is_training):
        """
        Creates semantic, string distance, and sentence embedding features.

        Args:
            df: Pandas DataFrame with columns 'A' and 'B'.
            is_training: Boolean indicating if this is for training (True) or prediction (False).

        Returns:
            Pandas DataFrame with added features.
        """

        # 1. Semantic features using TF-IDF
        if is_training:
            tfidf_A = self.tfidf_vectorizer.fit_transform(df['A'])
        else:
            tfidf_A = self.tfidf_vectorizer.transform(df['A'])
        tfidf_B = self.tfidf_vectorizer.transform(df['B'])
        df['cosine_similarity'] = cosine_similarity(tfidf_A, tfidf_B).diagonal()

        # 2. String distance features
        df['levenshtein_distance'] = df.apply(lambda row: levenshtein_distance(row['A'], row['B']), axis=1)
        df['jaro_winkler_similarity'] = df.apply(lambda row: jaro_winkler_similarity(row['A'], row['B']), axis=1)

        # 3. Sentence embedding features
        embeddings_A = self.sentence_transformer.encode(df['A'].tolist(), convert_to_tensor=True)
        embeddings_B = self.sentence_transformer.encode(df['B'].tolist(), convert_to_tensor=True)

        embeddings_A = embeddings_A.cpu()  # Move to CPU
        embeddings_B = embeddings_B.cpu()  # Move to CPU

        df['embedding_cosine_similarity'] = cosine_similarity(embeddings_A, embeddings_B).diagonal()
        df[FEATURE_NAMES]
        return df

if __name__ == '__main__':
    # Code to be executed only when the script is run directly
    from rooms.synthetic_data import generate_synthetic_dataset

    df = generate_synthetic_dataset(n_rows=100, match_ratio=.2)
    # Create and use the pipeline
    pipeline = RoomMatchingPipeline()

    # For training:
    p = pipeline.preprocess_data(df, is_training=True)

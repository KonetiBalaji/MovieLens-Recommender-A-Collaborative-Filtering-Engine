import pandas as pd
import os

def load_movielens_100k(data_path: str = 'data/raw/u.data') -> pd.DataFrame:
    """
    Load the MovieLens 100k dataset.

    Args:
        data_path (str): Path to the 'u.data' file.

    Returns:
        pd.DataFrame: Loaded dataset with columns ['user_id', 'item_id', 'rating', 'timestamp'].
    """
    column_names = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv(data_path, sep='\t', names=column_names)
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the ratings DataFrame by:
      - Dropping the 'timestamp' column
      - Reindexing user and item IDs to be 0-based and sequential

    Args:
        df (pd.DataFrame): Raw ratings DataFrame.

    Returns:
        pd.DataFrame: Cleaned and reindexed DataFrame.
    """
    # Drop timestamp as it's not used
    df = df.drop(columns=['timestamp'])

    # Reindex user and item IDs for matrix operations
    user_mapping = {original: new for new, original in enumerate(df['user_id'].unique())}
    item_mapping = {original: new for new, original in enumerate(df['item_id'].unique())}

    df['user_id'] = df['user_id'].map(user_mapping)
    df['item_id'] = df['item_id'].map(item_mapping)

    return df


def save_processed_data(df: pd.DataFrame, output_path: str = 'data/processed/ratings.csv') -> None:
    """
    Save the preprocessed DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): DataFrame to save.
        output_path (str): File path for saving the processed data.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

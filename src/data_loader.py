import pandas as pd
import os

def load_movielens_100k(data_path: str = 'data/raw/u.data') -> pd.DataFrame:
    """
    Load the MovieLens 100k dataset.

    Args:
        data_path (str): Path to the u.data file.

    Returns:
        pd.DataFrame: Loaded ratings DataFrame.
    """
    column_names = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv(data_path, sep='\t', names=column_names)
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic preprocessing: Drop timestamp, reindex users/items if needed.

    Args:
        df (pd.DataFrame): Raw DataFrame.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df.drop(columns=['timestamp'], inplace=True)

    # Optionally reindex users/items (optional step, keeps things neat)
    user_mapping = {id_: idx for idx, id_ in enumerate(df['user_id'].unique())}
    item_mapping = {id_: idx for idx, id_ in enumerate(df['item_id'].unique())}

    df['user_id'] = df['user_id'].map(user_mapping)
    df['item_id'] = df['item_id'].map(item_mapping)

    return df

def save_processed_data(df: pd.DataFrame, output_path: str = 'data/processed/ratings.csv'):
    """
    Save the preprocessed ratings to disk.

    Args:
        df (pd.DataFrame): DataFrame to save.
        output_path (str): Output CSV path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

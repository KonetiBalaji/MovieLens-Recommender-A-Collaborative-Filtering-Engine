import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

def evaluate_model(df: pd.DataFrame, user_item_matrix: pd.DataFrame,
                   similarity_df: pd.DataFrame, k: int = 5, sample_size: int = 1000) -> float:
    """
    Evaluate the recommendation model using RMSE.

    Args:
        df (pd.DataFrame): Original ratings DataFrame.
        user_item_matrix (pd.DataFrame): User-item matrix.
        similarity_df (pd.DataFrame): User similarity matrix.
        k (int): Top-k similar users to consider.
        sample_size (int): Number of samples for evaluation.

    Returns:
        float: RMSE of predicted ratings.
    """
    from src.recommender import predict_rating

    sample_df = df.sample(n=sample_size, random_state=42)
    actuals, preds = [], []

    for _, row in sample_df.iterrows():
        uid, iid, actual = row['user_id'], row['item_id'], row['rating']
        pred = predict_rating(uid, iid, user_item_matrix, similarity_df, k)
        if not np.isnan(pred):
            actuals.append(actual)
            preds.append(pred)

    mse = np.mean((np.array(actuals) - np.array(preds)) ** 2)
    rmse = np.sqrt(mse)

    return rmse

def evaluate_mf_model(df: pd.DataFrame, predicted_ratings: pd.DataFrame, sample_size: int = 500) -> float:
    """
    Evaluate MF model using RMSE on a sample of actual vs predicted ratings.

    Args:
        df (pd.DataFrame): Original ratings data.
        predicted_ratings (pd.DataFrame): Full predicted rating matrix.
        sample_size (int): Sample size for evaluation.

    Returns:
        float: RMSE score.
    """
    sample_df = df.sample(n=sample_size, random_state=42)
    actuals, preds = [], []

    for _, row in sample_df.iterrows():
        user, item, actual = row['user_id'], row['item_id'], row['rating']
        try:
            pred = predicted_ratings.loc[user, item]
            if not np.isnan(pred):
                actuals.append(actual)
                preds.append(pred)
        except KeyError:
            continue  # Skip users/items not in matrix

    if not preds:
        return float('nan')

    mse = np.mean((np.array(actuals) - np.array(preds)) ** 2)
    return np.sqrt(mse)


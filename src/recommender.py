import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

def create_user_item_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a user-item matrix from ratings data.

    Args:
        df (pd.DataFrame): Processed DataFrame with user_id, item_id, rating.

    Returns:
        pd.DataFrame: User-Item matrix (rows: users, columns: items).
    """
    return df.pivot_table(index='user_id', columns='item_id', values='rating')

def calculate_user_similarity(user_item_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Computes cosine similarity between users.

    Args:
        user_item_matrix (pd.DataFrame): User-Item rating matrix.

    Returns:
        pd.DataFrame: User-user similarity matrix.
    """
    matrix_filled = user_item_matrix.fillna(0)
    sparse_matrix = csr_matrix(matrix_filled.values)
    similarity = cosine_similarity(sparse_matrix)
    similarity_df = pd.DataFrame(similarity, index=user_item_matrix.index, columns=user_item_matrix.index)
    return similarity_df


def predict_rating(user_id: int, item_id: int, user_item_matrix: pd.DataFrame,
                   similarity_df: pd.DataFrame, k: int = 5) -> float:
    """
    Predict a user's rating for a given item using user-based CF.

    Args:
        user_id (int): ID of the target user.
        item_id (int): ID of the item to predict.
        user_item_matrix (pd.DataFrame): Ratings matrix.
        similarity_df (pd.DataFrame): User similarity matrix.
        k (int): Number of top similar users to consider.

    Returns:
        float: Predicted rating.
    """
    if item_id not in user_item_matrix.columns:
        return np.nan  # Item not in matrix

    # Users who rated the item
    item_ratings = user_item_matrix[item_id].dropna()
    if user_id not in similarity_df.index:
        return np.nan  # User not in similarity matrix

    # Similarities with target user
    similarities = similarity_df.loc[user_id, item_ratings.index]
    top_k_users = similarities.sort_values(ascending=False).head(k)

    top_k_ratings = item_ratings.loc[top_k_users.index]
    top_k_similarities = top_k_users

    if top_k_similarities.sum() == 0:
        return np.nan  # Avoid division by zero

    predicted_rating = np.dot(top_k_ratings, top_k_similarities) / top_k_similarities.sum()
    return predicted_rating

def recommend_top_n(user_id: int, user_item_matrix: pd.DataFrame,
                    similarity_df: pd.DataFrame, n: int = 5, k: int = 5) -> pd.DataFrame:
    """
    Generate top-N item recommendations for a user.

    Args:
        user_id (int): Target user ID.
        user_item_matrix (pd.DataFrame): Ratings matrix.
        similarity_df (pd.DataFrame): User similarity matrix.
        n (int): Number of recommendations to return.
        k (int): Number of similar users to consider for prediction.

    Returns:
        pd.DataFrame: Top-N recommended items and their predicted ratings.
    """
    user_ratings = user_item_matrix.loc[user_id]
    unrated_items = user_ratings[user_ratings.isna()].index

    predictions = []
    for item_id in unrated_items:
        pred = predict_rating(user_id, item_id, user_item_matrix, similarity_df, k)
        if not np.isnan(pred):
            predictions.append((item_id, pred))

    top_n = sorted(predictions, key=lambda x: x[1], reverse=True)[:n]
    return pd.DataFrame(top_n, columns=['item_id', 'predicted_rating'])

def calculate_item_similarity(user_item_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Computes cosine similarity between items based on user ratings.

    Args:
        user_item_matrix (pd.DataFrame): User-Item rating matrix.

    Returns:
        pd.DataFrame: Item-item similarity matrix.
    """
    matrix_filled = user_item_matrix.fillna(0)
    item_matrix = matrix_filled.T  # Transpose to make items as rows
    similarity = cosine_similarity(item_matrix)
    similarity_df = pd.DataFrame(similarity, index=item_matrix.index, columns=item_matrix.index)
    return similarity_df

def predict_item_based(user_id: int, item_id: int, user_item_matrix: pd.DataFrame,
                       item_similarity_df: pd.DataFrame, k: int = 5) -> float:
    """
    Predict a user's rating for an item using item-based CF.

    Args:
        user_id (int): ID of the user.
        item_id (int): ID of the item to predict.
        user_item_matrix (pd.DataFrame): Ratings matrix.
        item_similarity_df (pd.DataFrame): Item similarity matrix.
        k (int): Top-k similar items to consider.

    Returns:
        float: Predicted rating.
    """
    if item_id not in item_similarity_df.columns:
        return np.nan

    user_ratings = user_item_matrix.loc[user_id].dropna()

    if user_ratings.empty:
        return np.nan

    # Get similarity scores for the target item
    sim_items = item_similarity_df[item_id].drop(index=item_id)
    sim_items = sim_items.loc[user_ratings.index]

    if sim_items.empty:
        return np.nan

    # Top-k similar items the user rated
    top_k_items = sim_items.sort_values(ascending=False).head(k)
    top_k_ratings = user_ratings.loc[top_k_items.index]

    if top_k_items.sum() == 0:
        return np.nan

    predicted_rating = np.dot(top_k_ratings, top_k_items) / top_k_items.sum()
    return predicted_rating

def recommend_top_n_item_based(user_id: int, user_item_matrix: pd.DataFrame,
                               item_similarity_df: pd.DataFrame, n: int = 5, k: int = 5) -> pd.DataFrame:
    """
    Generate top-N item recommendations for a user using item-based CF.

    Args:
        user_id (int): Target user ID.
        user_item_matrix (pd.DataFrame): Ratings matrix.
        item_similarity_df (pd.DataFrame): Item similarity matrix.
        n (int): Number of recommendations to return.
        k (int): Number of similar items to consider.

    Returns:
        pd.DataFrame: Top-N recommended items with predicted ratings.
    """
    user_ratings = user_item_matrix.loc[user_id]
    unrated_items = user_ratings[user_ratings.isna()].index

    predictions = []
    for item_id in unrated_items:
        pred = predict_item_based(user_id, item_id, user_item_matrix, item_similarity_df, k)
        if not np.isnan(pred):
            predictions.append((item_id, pred))

    top_n = sorted(predictions, key=lambda x: x[1], reverse=True)[:n]
    return pd.DataFrame(top_n, columns=['item_id', 'predicted_rating'])


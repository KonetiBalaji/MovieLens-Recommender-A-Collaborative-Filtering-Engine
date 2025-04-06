import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

# --------------------------------------------------------------------
# 1. Build User-Item Matrix
# --------------------------------------------------------------------

def create_user_item_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert ratings DataFrame into a user-item matrix.

    Args:
        df (pd.DataFrame): Processed DataFrame with ['user_id', 'item_id', 'rating'].

    Returns:
        pd.DataFrame: Matrix with users as rows and items as columns.
    """
    return df.pivot_table(index='user_id', columns='item_id', values='rating')


# --------------------------------------------------------------------
# 2. Similarity Computation
# --------------------------------------------------------------------

def calculate_user_similarity(user_item_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate cosine similarity between users.

    Args:
        user_item_matrix (pd.DataFrame): Ratings matrix.

    Returns:
        pd.DataFrame: User-user similarity matrix.
    """
    matrix_filled = user_item_matrix.fillna(0)
    sparse_matrix = csr_matrix(matrix_filled.values)
    similarity = cosine_similarity(sparse_matrix)
    return pd.DataFrame(similarity, index=user_item_matrix.index, columns=user_item_matrix.index)


def calculate_item_similarity(user_item_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate cosine similarity between items.

    Args:
        user_item_matrix (pd.DataFrame): Ratings matrix.

    Returns:
        pd.DataFrame: Item-item similarity matrix.
    """
    matrix_filled = user_item_matrix.fillna(0)
    item_matrix = matrix_filled.T  # Transpose so items are rows
    similarity = cosine_similarity(item_matrix)
    return pd.DataFrame(similarity, index=item_matrix.index, columns=item_matrix.index)


# --------------------------------------------------------------------
# 3. User-Based CF Prediction
# --------------------------------------------------------------------

def predict_rating(user_id: int, item_id: int, user_item_matrix: pd.DataFrame,
                   similarity_df: pd.DataFrame, k: int = 5) -> float:
    """
    Predict rating using User-Based Collaborative Filtering.

    Args:
        user_id (int): Target user.
        item_id (int): Target item.
        user_item_matrix (pd.DataFrame): Ratings matrix.
        similarity_df (pd.DataFrame): User-user similarity.
        k (int): Top-k neighbors.

    Returns:
        float: Predicted rating.
    """
    if item_id not in user_item_matrix.columns:
        return np.nan

    item_ratings = user_item_matrix[item_id].dropna()
    if user_id not in similarity_df.index:
        return np.nan

    similarities = similarity_df.loc[user_id, item_ratings.index]
    top_k = similarities.sort_values(ascending=False).head(k)
    top_k_ratings = item_ratings.loc[top_k.index]

    if top_k.sum() == 0:
        return np.nan

    return np.dot(top_k_ratings, top_k) / top_k.sum()


def recommend_top_n(user_id: int, user_item_matrix: pd.DataFrame,
                    similarity_df: pd.DataFrame, n: int = 5, k: int = 5) -> pd.DataFrame:
    """
    Recommend Top-N items using User-Based CF.

    Args:
        user_id (int): Target user.
        user_item_matrix (pd.DataFrame): Ratings matrix.
        similarity_df (pd.DataFrame): User-user similarity matrix.
        n (int): Number of recommendations.
        k (int): Top-k neighbors.

    Returns:
        pd.DataFrame: Top-N recommended item IDs and predicted ratings.
    """
    unrated = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id].isna()].index
    predictions = []

    for item_id in unrated:
        pred = predict_rating(user_id, item_id, user_item_matrix, similarity_df, k)
        if not np.isnan(pred):
            predictions.append((item_id, pred))

    top_n = sorted(predictions, key=lambda x: x[1], reverse=True)[:n]
    return pd.DataFrame(top_n, columns=['item_id', 'predicted_rating'])


# --------------------------------------------------------------------
# 4. Item-Based CF Prediction
# --------------------------------------------------------------------

def predict_item_based(user_id: int, item_id: int, user_item_matrix: pd.DataFrame,
                       item_similarity_df: pd.DataFrame, k: int = 5) -> float:
    """
    Predict rating using Item-Based Collaborative Filtering.

    Args:
        user_id (int): Target user.
        item_id (int): Target item.
        user_item_matrix (pd.DataFrame): Ratings matrix.
        item_similarity_df (pd.DataFrame): Item-item similarity.
        k (int): Top-k similar items.

    Returns:
        float: Predicted rating.
    """
    if item_id not in item_similarity_df.columns:
        return np.nan

    user_ratings = user_item_matrix.loc[user_id].dropna()
    if user_ratings.empty:
        return np.nan

    sim_items = item_similarity_df[item_id].drop(index=item_id)
    sim_items = sim_items.loc[user_ratings.index]

    if sim_items.empty:
        return np.nan

    top_k_items = sim_items.sort_values(ascending=False).head(k)
    top_k_ratings = user_ratings.loc[top_k_items.index]

    if top_k_items.sum() == 0:
        return np.nan

    return np.dot(top_k_ratings, top_k_items) / top_k_items.sum()


def recommend_top_n_item_based(user_id: int, user_item_matrix: pd.DataFrame,
                               item_similarity_df: pd.DataFrame, n: int = 5, k: int = 5) -> pd.DataFrame:
    """
    Recommend Top-N items using Item-Based CF.

    Args:
        user_id (int): Target user.
        user_item_matrix (pd.DataFrame): Ratings matrix.
        item_similarity_df (pd.DataFrame): Item-item similarity.
        n (int): Number of recommendations.
        k (int): Top-k similar items.

    Returns:
        pd.DataFrame: Top-N recommended item IDs and predicted ratings.
    """
    unrated = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id].isna()].index
    predictions = []

    for item_id in unrated:
        pred = predict_item_based(user_id, item_id, user_item_matrix, item_similarity_df, k)
        if not np.isnan(pred):
            predictions.append((item_id, pred))

    top_n = sorted(predictions, key=lambda x: x[1], reverse=True)[:n]
    return pd.DataFrame(top_n, columns=['item_id', 'predicted_rating'])
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD

def train_svd_model(user_item_matrix: pd.DataFrame, n_components: int = 20):
    """
    Train a truncated SVD model on the user-item matrix.

    Args:
        user_item_matrix (pd.DataFrame): User-item matrix with NaNs filled as 0.
        n_components (int): Number of latent features.

    Returns:
        tuple: (U matrix, V matrix, user IDs, item IDs)
    """
    matrix_filled = user_item_matrix.fillna(0)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    U = svd.fit_transform(matrix_filled)
    V = svd.components_

    return U, V, user_item_matrix.index, user_item_matrix.columns

def predict_ratings_from_svd(U, V, user_ids, item_ids) -> pd.DataFrame:
    """
    Reconstruct the full predicted user-item rating matrix using SVD.

    Args:
        U (ndarray): User latent matrix (users x features).
        V (ndarray): Item latent matrix (features x items).
        user_ids (Index): User ID index from original matrix.
        item_ids (Index): Item ID index from original matrix.

    Returns:
        pd.DataFrame: Predicted ratings DataFrame (users x items).
    """
    predicted_matrix = np.dot(U, V)
    predicted_df = pd.DataFrame(predicted_matrix, index=user_ids, columns=item_ids)
    return predicted_df

def recommend_top_n_from_mf(user_id: int, user_item_matrix: pd.DataFrame,
                            predicted_ratings: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Recommend top-N items to a user using MF predicted ratings.

    Args:
        user_id (int): Target user ID.
        user_item_matrix (pd.DataFrame): Original user-item matrix (to exclude already rated items).
        predicted_ratings (pd.DataFrame): Predicted rating matrix from MF.
        n (int): Number of recommendations.

    Returns:
        pd.DataFrame: Top-N item recommendations with predicted scores.
    """
    user_pred_ratings = predicted_ratings.loc[user_id]
    already_rated = user_item_matrix.loc[user_id].dropna().index

    # Exclude already rated items
    recommendations = user_pred_ratings.drop(index=already_rated)

    top_n = recommendations.sort_values(ascending=False).head(n)
    return pd.DataFrame({'item_id': top_n.index, 'predicted_rating': top_n.values})

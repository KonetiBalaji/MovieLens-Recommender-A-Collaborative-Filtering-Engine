import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD

# --------------------------------------------------------------------
# Train Matrix Factorization Model (SVD)
# --------------------------------------------------------------------

def train_svd_model(user_item_matrix: pd.DataFrame, n_components: int = 20):
    """
    Train a Truncated SVD model to factorize the user-item matrix.

    Args:
        user_item_matrix (pd.DataFrame): User-Item matrix (users as rows, items as columns).
        n_components (int): Number of latent features to extract.

    Returns:
        tuple: (U, V, user_ids, item_ids)
            - U: User latent feature matrix
            - V: Item latent feature matrix
            - user_ids: Index of users
            - item_ids: Index of items
    """
    # Fill missing values with 0 for SVD
    matrix_filled = user_item_matrix.fillna(0)

    # Apply Truncated SVD
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    U = svd.fit_transform(matrix_filled)       # shape: (n_users x n_components)
    V = svd.components_                        # shape: (n_components x n_items)

    return U, V, user_item_matrix.index, user_item_matrix.columns


# --------------------------------------------------------------------
# Predict Full Rating Matrix
# --------------------------------------------------------------------

def predict_ratings_from_svd(U: np.ndarray, V: np.ndarray,
                              user_ids, item_ids) -> pd.DataFrame:
    """
    Predict the full user-item rating matrix using dot product of U and V.

    Args:
        U (np.ndarray): User latent matrix (n_users x n_features).
        V (np.ndarray): Item latent matrix (n_features x n_items).
        user_ids (Index): Index corresponding to user IDs.
        item_ids (Index): Index corresponding to item IDs.

    Returns:
        pd.DataFrame: Predicted ratings matrix (users x items).
    """
    predicted_matrix = np.dot(U, V)  # shape: (n_users x n_items)
    return pd.DataFrame(predicted_matrix, index=user_ids, columns=item_ids)


# --------------------------------------------------------------------
# Recommend Top-N Items Using MF Predictions
# --------------------------------------------------------------------

def recommend_top_n_from_mf(user_id: int, user_item_matrix: pd.DataFrame,
                            predicted_ratings: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Generate Top-N recommendations for a user using MF predictions.

    Args:
        user_id (int): Target user.
        user_item_matrix (pd.DataFrame): Original matrix to exclude already rated items.
        predicted_ratings (pd.DataFrame): Full predicted rating matrix.
        n (int): Number of top items to recommend.

    Returns:
        pd.DataFrame: DataFrame with 'item_id' and 'predicted_rating'.
    """
    # Get all predicted ratings for user
    user_pred_ratings = predicted_ratings.loc[user_id]

    # Exclude items the user already rated
    already_rated = user_item_matrix.loc[user_id].dropna().index
    recommendations = user_pred_ratings.drop(index=already_rated)

    # Sort by predicted score
    top_n = recommendations.sort_values(ascending=False).head(n)

    return pd.DataFrame({
        'item_id': top_n.index,
        'predicted_rating': top_n.values
    })

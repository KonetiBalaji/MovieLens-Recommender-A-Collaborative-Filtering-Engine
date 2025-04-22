import streamlit as st
import pandas as pd

from src.metadata_loader import load_movie_metadata, enrich_recommendations
from src.recommender import (
    create_user_item_matrix,
    calculate_user_similarity,
    calculate_item_similarity,
    predict_rating,
    predict_item_based,
    recommend_top_n,
    recommend_top_n_item_based
)
from src.matrix_factorization import (
    train_svd_model,
    predict_ratings_from_svd,
    recommend_top_n_from_mf
)
from src.data_loader import load_movielens_100k, preprocess_data


@st.cache_data
def load_data():
    raw_df = load_movielens_100k()
    df = preprocess_data(raw_df)
    metadata_df = load_movie_metadata()
    return df, metadata_df


@st.cache_data
def prepare_models(df):
    user_item_matrix = create_user_item_matrix(df)
    user_similarity_df = calculate_user_similarity(user_item_matrix)
    item_similarity_df = calculate_item_similarity(user_item_matrix)
    U, V, user_ids, item_ids = train_svd_model(user_item_matrix, n_components=20)
    mf_predictions = predict_ratings_from_svd(U, V, user_ids, item_ids)
    return user_item_matrix, user_similarity_df, item_similarity_df, mf_predictions


def run_user_cf(user_id, user_item_matrix, user_similarity_df, metadata_df):
    top_n = recommend_top_n(user_id, user_item_matrix, user_similarity_df, n=5, k=5)
    enriched = enrich_recommendations(top_n, metadata_df)
    return enriched


def run_item_cf(user_id, user_item_matrix, item_similarity_df, metadata_df):
    top_n = recommend_top_n_item_based(user_id, user_item_matrix, item_similarity_df, n=5, k=5)
    enriched = enrich_recommendations(top_n, metadata_df)
    return enriched


def run_mf(user_id, user_item_matrix, mf_predictions, metadata_df):
    top_n = recommend_top_n_from_mf(user_id, user_item_matrix, mf_predictions, n=5)
    enriched = enrich_recommendations(top_n, metadata_df)
    return enriched


# Streamlit UI
st.set_page_config(page_title="Movie Recommender", layout="centered")
st.title("🎬 Collaborative Filtering Movie Recommender")

with st.expander("🔍 What are these recommendation methods?"):
    st.markdown("""
**User-Based Collaborative Filtering**  
Recommends movies based on ratings from users similar to you.  
*Example: "Users like you rated these movies highly."*

**Item-Based Collaborative Filtering**  
Recommends movies similar to the ones you've already rated highly.  
*Example: "You liked The Matrix? Then try Inception."*

**Matrix Factorization (SVD)**  
Learns your hidden preferences from ratings and recommends movies that match those patterns.  
*Example: "You seem to like sci-fi thrillers with a strong lead — here are some you might enjoy."*
    """)

df, metadata_df = load_data()
user_item_matrix, user_similarity_df, item_similarity_df, mf_predictions = prepare_models(df)

user_ids = user_item_matrix.index.tolist()
user_id = st.selectbox("Select a User ID", user_ids)

recommender_type = st.radio(
    "Choose Recommendation Algorithm",
    ("User-Based CF", "Item-Based CF", "Matrix Factorization")
)

if st.button("Get Recommendations"):
    if recommender_type == "User-Based CF":
        results = run_user_cf(user_id, user_item_matrix, user_similarity_df, metadata_df)
    elif recommender_type == "Item-Based CF":
        results = run_item_cf(user_id, user_item_matrix, item_similarity_df, metadata_df)
    else:
        results = run_mf(user_id, user_item_matrix, mf_predictions, metadata_df)

    st.subheader("Top 5 Recommended Movies")
    st.dataframe(results.reset_index(drop=True))

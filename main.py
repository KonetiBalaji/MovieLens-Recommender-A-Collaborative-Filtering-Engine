# main.py
import pandas as pd

from src.matrix_factorization import train_svd_model
from src.matrix_factorization import predict_ratings_from_svd
from src.matrix_factorization import recommend_top_n_from_mf
from src.evaluation import evaluate_mf_model



from src.data_loader import load_movielens_100k, preprocess_data, save_processed_data
from src.recommender import (
    create_user_item_matrix,
    calculate_user_similarity,
    predict_rating  
)

from src.recommender import (
    create_user_item_matrix,
    calculate_user_similarity,
    calculate_item_similarity, 
    predict_rating,
    predict_item_based,
    recommend_top_n,
    recommend_top_n_item_based  
     
)

from src.evaluation import evaluate_model



def main():
    print("Loading data...")
    raw_df = load_movielens_100k()

    print("Preprocessing data...")
    processed_df = preprocess_data(raw_df)

    print("Saving processed data...")
    save_processed_data(processed_df)

    print("Step 1 completed: Processed data saved to 'data/processed/ratings.csv'.")

from src.recommender import create_user_item_matrix, calculate_user_similarity

def run_step_2():
    print("Loading processed data...")
    df = pd.read_csv("data/processed/ratings.csv")

    print("Creating user-item matrix...")
    user_item_matrix = create_user_item_matrix(df)

    print("Calculating user similarity matrix...")
    similarity_df = calculate_user_similarity(user_item_matrix)

    print("Sample similarity matrix:")
    print(similarity_df.head())

def run_step_3():
    print("Loading processed data...")
    df = pd.read_csv("data/processed/ratings.csv")
    user_item_matrix = create_user_item_matrix(df)
    similarity_df = calculate_user_similarity(user_item_matrix)

    test_user = 0
    test_item = 50
    predicted = predict_rating(test_user, test_item, user_item_matrix, similarity_df, k=5)

    print(f"Predicted rating for user {test_user} on item {test_item}: {predicted:.2f}")

    print(f"\nTop 5 recommendations for user {test_user}:")
    recommendations = recommend_top_n(test_user, user_item_matrix, similarity_df, n=5, k=5)
    print(recommendations)

    print("\nEvaluating model with RMSE...")
    rmse_score = evaluate_model(df, user_item_matrix, similarity_df, k=5, sample_size=500)
    print(f"RMSE on sample of 500 ratings: {rmse_score:.4f}")

    output_path = "outputs/recommendations.csv"
    recommendations.to_csv(output_path, index=False)
    print(f"\nTop-N recommendations saved to: {output_path}")


def run_item_cf_step():
    print("\n[Item-Based CF] Loading processed data...")
    df = pd.read_csv("data/processed/ratings.csv")

    print("[Item-Based CF] Creating user-item matrix...")
    user_item_matrix = create_user_item_matrix(df)

    print("[Item-Based CF] Calculating item similarity matrix...")
    item_similarity_df = calculate_item_similarity(user_item_matrix)

    print("[Item-Based CF] Sample item-item similarity matrix:")
    print(item_similarity_df.iloc[:5, :5])

    print("\n[Item-Based CF] Predicting a rating...")
    test_user = 0
    test_item = 50
    predicted_rating = predict_item_based(test_user, test_item, user_item_matrix, item_similarity_df, k=5)
    print(f"Predicted rating for user {test_user} on item {test_item}: {predicted_rating:.2f}")

    print(f"\n[Item-Based CF] Top 5 recommendations for user {test_user}:")
    top_recommendations = recommend_top_n_item_based(test_user, user_item_matrix, item_similarity_df, n=5, k=5)
    print(top_recommendations)

    # Save to output file
    top_recommendations.to_csv("outputs/item_cf_recommendations.csv", index=False)
    print("\n[Item-Based CF] Recommendations saved to 'outputs/item_cf_recommendations.csv'")

def run_matrix_factorization_step():
    print("\n[MF] Loading processed data...")
    df = pd.read_csv("data/processed/ratings.csv")

    print("[MF] Creating user-item matrix...")
    user_item_matrix = create_user_item_matrix(df)

    print("[MF] Training SVD model...")
    U, V, user_ids, item_ids = train_svd_model(user_item_matrix, n_components=20)

    print("[MF] Latent matrices shape:")
    print(f"U shape (users x features): {U.shape}")
    print(f"V shape (features x items): {V.shape}")

    print("[MF] Predicting full rating matrix...")
    predicted_ratings = predict_ratings_from_svd(U, V, user_ids, item_ids)

    # Show a sample of predictions
    print("[MF] Sample predictions:")
    print(predicted_ratings.iloc[:5, :5])

    # Save to file (optional)
    predicted_ratings.to_csv("outputs/mf_predicted_ratings.csv")
    print("[MF] Full predicted rating matrix saved to 'outputs/mf_predicted_ratings.csv'")

    print("\n[MF] Generating Top-5 recommendations for user 0...")
    top_mf_recs = recommend_top_n_from_mf(user_id=0,
                                          user_item_matrix=user_item_matrix,
                                          predicted_ratings=predicted_ratings,
                                          n=5)

    print(top_mf_recs)
    top_mf_recs.to_csv("outputs/mf_top_recommendations.csv", index=False)
    print("[MF] Recommendations saved to 'outputs/mf_top_recommendations.csv'")

    print("\n[MF] Evaluating Matrix Factorization with RMSE...")
    rmse = evaluate_mf_model(df, predicted_ratings, sample_size=500)
    print(f"[MF] RMSE on 500 samples: {rmse:.4f}")



if __name__ == "__main__":
    main()
    run_step_2()
    run_step_3()
    run_item_cf_step()
    run_matrix_factorization_step()


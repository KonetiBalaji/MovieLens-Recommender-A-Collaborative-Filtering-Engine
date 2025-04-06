Collaborative Filtering Recommender System

This project implements a modular, script-based recommender system using the MovieLens 100k dataset. It includes User-Based Collaborative Filtering, Item-Based Collaborative Filtering, and Matrix Factorization (SVD) techniques, with evaluation metrics and output recommendations.

The code is designed to run as Python scripts on a local machine (not notebooks), and follows a clean, maintainable project structure.

Project Structure

CollaborativeFiltering_Recommender/
│
├── data/
│   ├── raw/                  # Place the original 'u.data' file here
│   └── processed/            # Preprocessed ratings stored as CSV
│
├── outputs/                  # All recommendation and evaluation outputs
│   ├── user_cf_recommendations.csv
│   ├── item_cf_recommendations.csv
│   ├── mf_predicted_ratings.csv
│   ├── mf_top_recommendations.csv
│
├── src/                      # Source code modules
│   ├── data_loader.py              # Data loading and preprocessing
│   ├── recommender.py              # User-based and item-based CF logic
│   ├── matrix_factorization.py     # SVD-based recommendation
│   ├── evaluation.py               # Evaluation using RMSE
│
├── main.py                  # Main pipeline driver (script-based)
├── README.md                # Project documentation
├── requirements.txt         # Python dependencies

Download and Place Dataset

1. Download the MovieLens 100k dataset from:
   https://grouplens.org/datasets/movielens/100k/

2. Extract it and place the 'u.data' file in:
   data/raw/u.data

The file should be a tab-separated file with the following columns:
user_id   item_id   rating   timestamp

The script will preprocess and reindex the data automatically.

Run the Full Pipeline

Make sure you're in the root directory of the project.

Step 1: Install Dependencies
    pip install -r requirements.txt

Step 2: Run the Main Pipeline
    python main.py

What Will Happen:
1. Data will be loaded, cleaned, and saved.
2. User-Based Collaborative Filtering:
   - Predicts sample ratings
   - Generates Top-N recommendations
   - Saves to outputs/user_cf_recommendations.csv
3. Item-Based Collaborative Filtering:
   - Computes item-item similarities
   - Recommends Top-N items
   - Saves to outputs/item_cf_recommendations.csv
4. Matrix Factorization (SVD):
   - Trains SVD model on user-item matrix
   - Generates full prediction matrix
   - Recommends Top-N items
   - Saves to outputs/mf_predicted_ratings.csv and outputs/mf_top_recommendations.csv
5. Evaluation using RMSE

Authors
Balaji Koneti
Graduate student in Computer Science
Passionate about AI, data engineering, and building real-world machine learning systems.
LinkedIn: https://www.linkedin.com/in/balajikoneti

License
This project is licensed under the MIT License.
You are free to use, modify, and distribute it with attribution.


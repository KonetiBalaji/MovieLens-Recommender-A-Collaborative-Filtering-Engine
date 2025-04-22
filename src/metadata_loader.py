import pandas as pd

def load_movie_metadata(path: str = "data/raw/u.item") -> pd.DataFrame:
    """
    Load movie metadata from MovieLens 'u.item' file.

    Args:
        path (str): Path to u.item file.

    Returns:
        pd.DataFrame: DataFrame with item_id, title, and genre tags.
    """
    column_names = ['item_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL',
                    'unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy',
                    'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                    'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

    df = pd.read_csv(path, sep='|', encoding='latin-1', names=column_names, usecols=range(24))

    # Generate a list of genres for each movie
    genre_cols = column_names[5:]
    df['genres'] = df[genre_cols].apply(lambda row: [genre for genre, v in row.items() if v == 1], axis=1)

    return df[['item_id', 'title', 'genres']]
def enrich_recommendations(recommendation_df: pd.DataFrame, metadata_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join movie metadata (title, genres) to the recommendation DataFrame.

    Args:
        recommendation_df (pd.DataFrame): Must contain 'item_id' and optionally 'predicted_rating'.
        metadata_df (pd.DataFrame): Metadata containing 'item_id', 'title', and 'genres'.

    Returns:
        pd.DataFrame: Enriched DataFrame with titles and genres.
    """
    enriched_df = pd.merge(recommendation_df, metadata_df, on='item_id', how='left')
    return enriched_df[['item_id', 'title', 'genres', 'predicted_rating']] if 'predicted_rating' in recommendation_df.columns else enriched_df

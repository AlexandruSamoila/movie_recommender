import pandas as pd
import ast
import re


def process_movies(df_movies, vote_count_min):
    """
    Processes the movies DataFrame to filter and clean data for recommendations.

    Args:
        df_movies (pd.DataFrame): The original movies DataFrame containing raw data.
        vote_count_min (int): The minimum number of votes required for a movie to be included.

    Returns:
        pd.DataFrame: A cleaned and filtered DataFrame of movies with relevant information.
    """

    # Filter movies with vote count greater than the specified minimum
    movies_processed = df_movies[(df_movies["vote_count"] > vote_count_min)][
        ["id", "title", "overview", "genres", "vote_average", "vote_count"]
    ]

    # Drop rows with missing values in critical columns
    movies_processed = movies_processed.dropna(subset=["title", "overview", "genres"])

    # Exclude movies with empty genres
    movies_processed = movies_processed[movies_processed["genres"] != "[]"]

    # Remove movies with placeholder or invalid overviews
    movies_processed = movies_processed[
        ~movies_processed["overview"].str.lower().str.contains("no overview")
    ]

    # Apply text cleaning to the overview column
    movies_processed["overview"] = movies_processed["overview"].apply(text_cleaning)

    # Convert genres column from a stringified list of dictionaries to a list of genre names
    movies_processed["genres"] = movies_processed["genres"].apply(ast.literal_eval)
    movies_processed["genres"] = movies_processed["genres"].apply(
        lambda x: [i["name"] for i in x]
    )

    # Ensure the 'id' column is of integer type for consistency
    movies_processed.loc[:, "id"] = movies_processed["id"].astype(int)

    return movies_processed


def process_ratings(df_ratings):
    """
    Processes the ratings DataFrame to filter and clean data for recommendations.

    Args:
        df_ratings (pd.DataFrame): The original ratings DataFrame containing user ratings.

    Returns:
        pd.DataFrame: A cleaned and processed DataFrame of ratings with normalized scores.
    """
    # Count the number of ratings per user
    user_count = df_ratings["userId"].value_counts()

    # Identify users who have provided at least 20 ratings
    valid_users = user_count[user_count >= 20].index

    # Filter ratings to keep only ratings from valid users
    ratings_processed = df_ratings[df_ratings["userId"].isin(valid_users)]

    # Convert id columns to int
    ratings_processed.loc[:, "movieId"] = ratings_processed["movieId"].astype(int)

    # Find the minimum and maximum rating values
    min_rating = ratings_processed["rating"].min()
    max_rating = ratings_processed["rating"].max()

    # Normalize the ratings to a 0-1 scale
    ratings_processed["rating_normalized"] = (
        ratings_processed["rating"] - min_rating
    ) / (max_rating - min_rating)

    return ratings_processed


def create_train_test_set(df_ratings, k=5):
    """
     Creates a train-test split for ratings data. The test set contains k random ratings per user.

    Args:
        df_ratings (pd.DataFrame): Original ratings DataFrame.
        k (int): Number of ratings to sample per user for the test set.

    Returns:
        tuple: A tuple containing the training set (pd.DataFrame) and test set (pd.DataFrame).
    """
    # Sample k random ratings for each user to create the test set
    test_set = df_ratings.groupby("userId").sample(n=k)

    # Create the training set by excluding the test set entries
    train_set = df_ratings.drop(test_set.index)

    return train_set, test_set


def process_credits(df_credits):
    """
    Processes the movie credits DataFrame by extracting and cleaning relevant information
    (director and main cast members).

    Args:
        df_credits (pd.DataFrame): The original credits DataFrame.

    Returns:
        pd.DataFrame: Processed credits DataFrame with cleaned director and cast information.
    """
    # Convert the 'id' column to integer type for consistency
    df_credits.loc[:, "id"] = df_credits["id"].astype(int)

    # Parse 'crew' and 'cast' columns from JSON-like strings to Python objects
    df_credits["crew"] = df_credits["crew"].apply(ast.literal_eval)
    df_credits["cast"] = df_credits["cast"].apply(ast.literal_eval)

    # Drop rows with missing values in critical columns
    df_credits = df_credits.dropna(subset=["crew", "cast"])

    # Extract the director's name from the crew column
    # Lowercase and remove spaces for consistency
    df_credits["crew"] = df_credits["crew"].apply(
        lambda x: [
            i["name"].lower().replace(" ", "") for i in x if i["job"] == "Director"
        ][
            :1
        ]  # Keep only the first director if multiple exist
    )

    # Extract the names of the top 3 cast members
    df_credits["cast"] = df_credits["cast"].apply(
        lambda x: [i["name"].lower().replace(" ", "") for i in x[:3]]
    )
    return df_credits


def process_keywords(df_keywords):
    """
    Processes the movie keywords DataFrame by parsing and cleaning the keywords column.

    Args:
        df_keywords (pd.DataFrame): The original keywords DataFrame.

    Returns:
        pd.DataFrame: Processed keywords DataFrame with cleaned keywords.
    """
    # Convert the 'id' column to integer type for consistency
    df_keywords.loc[:, "id"] = df_keywords["id"].astype(int)

    # Parse 'keywords' column from JSON-like strings to Python objects
    df_keywords["keywords"] = df_keywords["keywords"].apply(ast.literal_eval)

    # Extract and convert to lowercase the keyword names
    df_keywords["keywords"] = df_keywords["keywords"].apply(
        lambda x: [i["name"].lower() for i in x]
    )

    # Drop rows with missing values in the 'keywords' column
    df_keywords = df_keywords.dropna(subset=["keywords"])
    return df_keywords


def text_cleaning(text):
    """
    Cleans text data by lowercasing, removing non-alphanumeric characters, and extra spaces.

    Args:
        text (str): The input text to clean.

    Returns:
        str: The cleaned text.
    """
    if isinstance(text, str):
        text = text.lower()
        # Remove all characters except alphanumeric and spaces
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        return text
    # Return an empty string if the input is not a valid string
    return ""


if __name__ == "__main__":

    # Load the dataset
    df_movies = pd.read_csv("archive/movies_metadata.csv", low_memory=False)
    df_ratings = pd.read_csv("archive/ratings_small.csv", low_memory=False)
    df_keywords = pd.read_csv("archive/keywords.csv", low_memory=False)
    df_credits = pd.read_csv("archive/credits.csv", low_memory=False)

    print("Process dataset ...")
    content_movies = process_movies(df_movies, 100)
    collaborative_movies = process_movies(df_movies, 0)

    # Filter ratings for valid movies
    df_ratings = df_ratings[df_ratings["movieId"].isin(collaborative_movies["id"])]

    # Filter credits and keywords for valid movies
    credits_processed = df_credits[df_credits["id"].isin(content_movies["id"])]
    keywords_processed = df_keywords[df_keywords["id"].isin(content_movies["id"])]

    ratings_processed = process_ratings(df_ratings)
    credits_processed = process_credits(df_credits)
    keywords_processed = process_keywords(df_keywords)

    # Create train-test split for the ratings subset
    ratings_train, ratings_test = create_train_test_set(ratings_processed, 5)

    # Add the keywords and credits to the movies dataframe
    content_movies = content_movies.merge(keywords_processed, on="id", how="left")
    content_movies = content_movies.merge(credits_processed, on="id", how="left")

    # Save the processed datasets to CSV
    content_movies.to_csv("data/content_movies.csv", index=False)
    collaborative_movies.to_csv("data/collaborative_movies.csv", index=False)
    ratings_train.to_csv("data/ratings_train.csv", index=False)
    ratings_test.to_csv("data/ratings_test.csv", index=False)

    print("Done!")

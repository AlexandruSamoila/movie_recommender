import pandas as pd
from models.collaborative_recommender import CollaborativeRecommender
from models.content_recommender import ContentRecommender
from utils import load_model

if __name__ == "__main__":

    print("\nWelcome to the Movie Recommendation System!")
    print("We have two recommendation use cases:")
    print("-" * 80)
    print("1. Finding movies with similar themes and genres (Content-Based).")
    print(
        "2. Personalized movie recommendations based on user preferences (Collaborative Filtering)."
    )
    print("-" * 80)

    # Get the user's choice
    while True:
        try:
            choice = int(
                input("Enter the number corresponding to your choice (1 or 2): ")
            )
            if choice in [1, 2]:
                break
            else:
                print("Invalid choice. Please enter 1 or 2.")
        except ValueError:
            print("Invalid input. Please enter a number (1 or 2).")

    if choice == 1:
        print(
            "\nYou selected: Finding movies with similar themes and genres (Content-Based)."
        )
        print("Loading data ...")
        df_movies_processed = pd.read_csv("data/content_movies.csv", low_memory=False)

        print("Initializing the Content-Based Recommender...")
        content_recommender = ContentRecommender(df_movies_processed)
        content_recommender.run()  # Run the content-based recommendation system

    elif choice == 2:
        print(
            "\nYou selected: Personalized movie recommendations (Collaborative Filtering)."
        )
        print("Loading data ...")
        df_all_movies = pd.read_csv("data/collaborative_movies.csv", low_memory=False)
        df_ratings = pd.read_csv("data/ratings_train.csv", low_memory=False)
        df_links = pd.read_csv("archive/links_small.csv", low_memory=False)

        print("Initializing the Collaborative Filtering Recommender...")

        # Load the pre-trained model from the saved path
        model_path = "checkpoints/20250126_2340_NCF_h128-64-32-16/model.pth"
        model, user_id_map, movie_id_map = load_model(model_path)

        # Initialize the CollaborativeRecommender with necessary data and model
        collaborative_recommender = CollaborativeRecommender(
            model, df_all_movies, df_ratings, user_id_map, movie_id_map, df_links
        )
        collaborative_recommender.run()  # Run the collaborative filtering recommendation system

import torch


class CollaborativeRecommender:
    def __init__(
        self, model, df_movies, df_ratings, user_id_map, movie_id_map, df_links
    ):
        """
        A collaborative recommender system based on the Neural Collaborative Filtering (NCF) model.
        This class provides movie recommendations for a given user based on their historical ratings
        and predicted ratings for unrated movies.

        Args:
            model (NCF): the model to use for recommendations
            df_movies (pd.DataFrame): the movies dataframe
            df_ratings (pd.DataFrame): the ratings dataframe
            user_id_map (dict): the user id map
            movie_id_map (dict): the movie id map
            df_links (pd.DataFrame): The DataFrame linking movie IDs to external movie IDs.
        """
        self.model = model
        self.df_movies = df_movies
        self.df_ratings = df_ratings
        self.user_id_map = user_id_map
        self.movie_id_map = movie_id_map
        self.df_links = df_links

    def run(self):
        """
        Runs the recommendation system interactively.

        Continuously prompts the user for a user ID and the number of recommendations they want.
        Displays the user's rating history and the top K recommendations based on predicted ratings.
        """
        while True:
            try:
                user_id = int(input("\nEnter user ID (or -1 to quit): "))
                if user_id == -1:
                    break

                # Check if user exists in the dataset
                if user_id not in self.user_id_map:
                    print(f"User ID {user_id} not found in training data.")
                    continue

                max_movies_history = 10  # Define the maximum number of movies to show in the user's history
                # Show user's rating history
                self.show_user_history(
                    user_id,
                    self.df_ratings,
                    self.df_movies,
                    self.df_links,
                    max_movies_history,
                )

                k = int(input("\nHow many recommendations do you want? "))

                print(f"\nGenerating top {k} recommendations for user {user_id}...")

                # Get recommendations
                recommendations = self.get_top_k_recommendations(
                    self.model,
                    user_id,
                    self.df_movies,
                    self.df_ratings,
                    self.df_links,
                    self.user_id_map,
                    self.movie_id_map,
                    k,
                )
                # Display recommendations
                print("\nTop Recommendations:")
                print("-" * 80)
                for i, rec in enumerate(recommendations, 1):
                    print(f"{i}. {rec['title']}")
                    print(f"   Genres: {rec['genres']}")
                    print(f"   Predicted Rating: {rec['predicted_rating']:.2f}/5.0")
                    print()

            except ValueError as ve:
                print(f"Error: {ve}")
            except Exception as e:
                print(f"An error occurred: {e}")

    def get_top_k_recommendations(
        self,
        model,
        user_id,
        movies_df,
        ratings_df,
        links_df,
        user_id_map,
        movie_id_map,
        k=10,
    ):
        """
        Gets the top K movie recommendations for a given user based on predicted ratings.

        Args:
            model (NCF): The model used for generating recommendations.
            user_id (int): The user ID for which to generate recommendations.
            movies_df (pd.DataFrame): The movie metadata DataFrame.
            ratings_df (pd.DataFrame): The ratings DataFrame.
            links_df (pd.DataFrame): The DataFrame containing movie links.
            user_id_map (dict): A mapping of external user IDs to internal IDs.
            movie_id_map (dict): A mapping of external movie IDs to internal IDs.
            k (int): The number of recommendations to generate.

        Returns:
            List[dict]: A list of recommended movies with titles, genres, and predicted ratings.
        """
        # Reverse the movie_id_map for looking up original IDs
        reverse_movie_map = {v: k for k, v in movie_id_map.items()}

        # Map the external user ID to internal ID
        if user_id not in user_id_map:
            raise ValueError(f"User ID {user_id} not found in training data")
        internal_user_id = user_id_map[user_id]

        # Get movies already rated by the user (using original movie IDs)
        rated_movies = set(
            ratings_df[ratings_df["userId"] == user_id]["movieId"].values
        )

        # Create tensor of all movie IDs (excluding rated movies)
        all_movie_ids = set(links_df["tmdbId"].values)
        unrated_movie_ids = list(all_movie_ids - rated_movies)
        unrated_movie_ids = [
            movie_id_map[mid] for mid in unrated_movie_ids if mid in movie_id_map
        ]

        movie_ids = torch.tensor(unrated_movie_ids)
        user_ids = torch.full_like(movie_ids, internal_user_id)

        # Get predictions
        with torch.no_grad():
            predictions = model(user_ids, movie_ids).squeeze()

        # Get top k movie indices
        top_k_indices = torch.topk(predictions, k).indices.numpy()
        top_k_scores = torch.topk(predictions, k).values.numpy()

        # Get movie information
        recommendations = []
        for idx, score in zip(top_k_indices, top_k_scores):
            internal_movie_id = unrated_movie_ids[idx]
            original_movie_id = reverse_movie_map[internal_movie_id]

            # Ensure there is a matching movie in movies_df
            matched_links = links_df[links_df["movieId"] == original_movie_id]
            matched_movie = movies_df[
                movies_df["id"] == matched_links["tmdbId"].values[0]
            ]
            if matched_movie.empty:
                print(f"Movie ID {original_movie_id} not found in movies_df.")
                continue  # Skip this recommendation if no match is found

            movie_info = matched_movie.iloc[0]
            recommendations.append(
                {
                    "title": movie_info["title"],
                    "genres": movie_info["genres"],
                    "predicted_rating": score * 5,  # Denormalize to 5-star scale
                }
            )

        return recommendations

    def show_user_history(
        self, user_id, ratings_df, movies_df, links_df, max_movies_history
    ):
        """
        Displays the rating history of a user.

        Args:
            user_id (int): The user ID whose rating history is to be displayed.
            ratings_df (pd.DataFrame): The ratings DataFrame.
            movies_df (pd.DataFrame): The movie metadata DataFrame.
            links_df (pd.DataFrame): The DataFrame containing movie links.
            max_movies_history (int): The maximum number of movies to display from the user's history.
        """
        user_ratings = ratings_df[ratings_df["userId"] == user_id]
        if len(user_ratings) == 0:
            print(f"\nUser {user_id} has no rating history.")
            return

        print(f"\nUser {user_id}'s rating history:")
        print("-" * 80)

        movies_counter = 0  # Initialize a counter

        for _, row in user_ratings.iterrows():
            # Stop after reaching the maximum number of movies to process
            if movies_counter >= max_movies_history:
                break

            # Retrieve the TMDB ID corresponding to the movieId from the links dataset
            movie_id = links_df[links_df["movieId"] == row["movieId"]]["tmdbId"].values[
                0
            ]

            # Find the movie details in the movies dataset using the TMDB ID
            movie = movies_df[movies_df["id"] == movie_id]

            # If the movie is not found, print a message and skip this iteration
            if movie.empty:
                print(f"Movie ID {movie_id} not found in movies_df. Skipping...")
                continue

            # Extract the first row (since filtering results in a DataFrame)
            movie_info = movie.iloc[0]

            # Print the movie title along with the user's rating
            print(f"{movie_info['title']}: {row['rating']}/5.0")

            # Increment the counter to track the number of processed movies
            movies_counter += 1

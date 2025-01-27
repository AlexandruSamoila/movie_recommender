import pandas as pd

import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
from sentence_transformers import SentenceTransformer


class ContentRecommender:
    def __init__(self, df_movies):
        """
        A content-based recommender system that suggests similar movies based on movie overview and genres.

        The system utilizes the `sentence-transformers` model for encoding movie overviews and `MultiLabelBinarizer`
        for encoding movie genres into a format suitable for similarity comparison. The combined embeddings from these
        two features are used to calculate cosine similarities between movies and suggest
        the most similar ones.

        Args:
            df_movies (pd.DataFrame): DataFrame containing movie information with columns
            such as 'title', 'overview', and 'genres'.
        """
        self.df_movies = df_movies

        # MultiLabelBinarizer to convert genre lists into binary arrays
        self.mlb = MultiLabelBinarizer()

        # Set up device (GPU if available, else CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(
            "sentence-transformers/static-retrieval-mrl-en-v1",
            device=device,
        )

    def find_similar_movies(self, movie_title, movies_df, combined_embeddings, top_k):
        """
        Finds and returns similar movies to the given movie title based on computed
        embeddings and cosine similarity.

        Args:
            movie_title (str): The title of the movie to find similarities for.
            movies_df (pd.DataFrame): The DataFrame containing movie information.
            combined_embeddings (torch.Tensor): The tensor containing movie embeddings (text + genre).
            top_k (int): The number of similar movies to return.

        Returns:
            List[str]: List of movie titles similar to the input movie.
        """
        try:
            # Find the index of the movie title in the dataframe
            movie_index = movies_df[
                movies_df["title"].str.lower() == movie_title.lower()
            ].index[0]

        except:
            print(f"Movie '{movie_title}' not found!")
            return []

        # Compute similarity matrix
        similarities = cosine_similarity(
            combined_embeddings.cpu().numpy(), combined_embeddings.cpu().numpy()
        )

        # Retrieve all the similarities based on the required movie
        sim_scores = list(enumerate(similarities[movie_index]))

        # Reverse sorting the movies based on the similarity score
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Remove the movie itself from the list
        sim_scores = [score for score in sim_scores if score[0] != movie_index]

        # Keep only the first 10 recommendations
        sim_scores = sim_scores[0:top_k]

        # Return the movie titles
        movie_indices = [i[0] for i in sim_scores]
        similar_movies = movies_df.iloc[movie_indices]["title"].tolist()
        return similar_movies

    def run(self):
        """
        Encodes the movies and finds similar ones interactively based on user input.

        Continuously prompts the user for a movie title and number of recommendations.
        Displays the top K most similar movies based on the title provided.
        """
        # Encode genres into binary format using MultiLabelBinarizer
        encoded_genres = pd.DataFrame(
            self.mlb.fit_transform(self.df_movies["genres"]),
            columns=self.mlb.classes_,
            index=self.df_movies.index,
        )

        print("Encoding movies...")

        # Encode movie overviews into embeddings using the SentenceTransformer
        movie_text_embeddings = self.model.encode(
            self.df_movies["overview"].tolist(), convert_to_tensor=True
        )

        # Genre Encoding
        genre_embeddings = torch.tensor(encoded_genres.values, dtype=torch.float32)

        # Combined Embeddings
        combined_embeddings = torch.cat(
            (
                movie_text_embeddings,
                genre_embeddings,
            ),
            dim=1,
        )

        print("Finding similar movies...")

        while True:
            try:
                target_movie = input("\nEnter a movie title (or 'exit' to quit): ")
                if target_movie.lower() == "exit":
                    break

                top_k = int(input("\nHow many recommendations do you want? "))

                # Get the similar movies
                similar_movies = self.find_similar_movies(
                    target_movie, self.df_movies, combined_embeddings, top_k
                )

                # Print the recommendations
                if similar_movies:
                    print(f"Movies similar to '{target_movie}':")
                    for i, movie in enumerate(similar_movies):
                        print(f"{i+1}. {movie}")
            except ValueError as ve:
                print(f"Error: {ve}")
            except Exception as e:
                print(f"An error occurred: {e}")

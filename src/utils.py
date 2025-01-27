import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
from models.ncf import NCF


def create_user_item_maps(df):
    """
    Creates mappings for user IDs and movie IDs from their original values to new indices for consistency.

    Args:
        df (pd.DataFrame): DataFrame containing 'userId' and 'movieId' columns.

    Returns:
        tuple: Two dictionaries:
            - user_id_map: Maps original user IDs to new indices.
            - movie_id_map: Maps original movie IDs to new indices.
    """
    user_ids = df["userId"].unique()
    movie_ids = df["movieId"].unique()

    # Create mappings: old IDs to new indices
    user_id_map = {old_id: new_id for new_id, old_id in enumerate(user_ids)}
    movie_id_map = {old_id: new_id for new_id, old_id in enumerate(movie_ids)}

    return user_id_map, movie_id_map


def load_data(df):
    """
    Processes the input ratings DataFrame to map user and movie IDs to sequential indices
    and extracts relevant information for model training/testing.

    Args:
        df (pd.DataFrame): DataFrame containing 'userId', 'movieId', and 'rating_normalized' columns.

    Returns:
        tuple: Contains user IDs, movie IDs, normalized ratings, total number of users,
               total number of movies, user ID mapping, and movie ID mapping.
    """
    # Generate mappings for user and movie IDs
    user_id_map, movie_id_map = create_user_item_maps(df)

    # Replace old IDs with new indices
    df["userId"] = df["userId"].apply(lambda id: user_id_map[id])
    df["movieId"] = df["movieId"].apply(lambda id: movie_id_map[id])

    # Get the total number of users and movies
    num_users = len(user_id_map)
    num_movies = len(movie_id_map)

    # Extract user IDs, movie IDs, and ratings
    user_ids = df["userId"].values
    movie_ids = df["movieId"].values
    ratings = df["rating_normalized"].values
    return (
        user_ids,
        movie_ids,
        ratings,
        num_users,
        num_movies,
        user_id_map,
        movie_id_map,
    )


def load_model(model_path):
    """
    Loads a pre-trained model along with its hyperparameters and ID mappings.

    Args:
        model_path (str): Path to the saved model checkpoint.

    Returns:
        tuple: Contains the loaded model, user ID mapping, and movie ID mapping.
    """

    # Load the saved model and hyperparameters
    checkpoint = torch.load(model_path)

    # Get the ID mappings
    movie_id_map = checkpoint["movie_id_map"]
    user_id_map = checkpoint["user_id_map"]

    # Extract hyperparameters
    hyperparameters = checkpoint["hyperparameters"]
    hidden_units = hyperparameters["hidden_units"]
    embedding_dim_mf = hyperparameters["embedding_dim_mf"]
    embedding_dim_mlp = hyperparameters["embedding_dim_mlp"]

    # Get the total number of users and movies
    num_users = len(user_id_map)
    num_movies = len(movie_id_map)

    # Initialize the model using the extracted hyperparameters
    model = NCF(
        num_users,
        num_movies,
        embedding_mf_dim=embedding_dim_mf,
        embedding_mlp_dim=embedding_dim_mlp,
        hidden_units=hidden_units,
    )

    # Load the state dict
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, user_id_map, movie_id_map


def load_dataloader(data_path):
    """
    Loads a DataLoader for test data from a CSV file containing ratings.

    Args:
        data_path (str): Path to the test data CSV file.

    Returns:
        DataLoader: DataLoader object containing test data.
    """
    df_ratings_test = pd.read_csv(data_path)

    # Process the test data to extract user IDs, movie IDs, and ratings
    (
        user_ids_test,
        movie_ids_test,
        ratings_test,
        _,
        _,
        _,
        _,
    ) = load_data(df_ratings_test)

    # Create a test dataset from the extracted data
    test_dataset = MovieRatingDataset(user_ids_test, movie_ids_test, ratings_test)

    # Initialize a DataLoader for the test dataset
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
    )
    return test_loader


def create_data_loaders(
    user_ids_train,
    movie_ids_train,
    ratings_train,
    user_ids_val,
    movie_ids_val,
    ratings_val,
    batch_size=32,
):
    """
    Create weighted train loader and regular validation loader

    Arguments:
        user_ids_train, movie_ids_train, ratings_train: Training data.
        user_ids_val, movie_ids_val, ratings_val: Validation data.
        batch_size: Number of samples in each batch (default 32).

    Returns:
        train_loader, val_loader: DataLoader instances for training and validation.
    """
    # Create datasets
    train_dataset = MovieRatingDataset(user_ids_train, movie_ids_train, ratings_train)
    val_dataset = MovieRatingDataset(user_ids_val, movie_ids_val, ratings_val)

    # Create samplers
    train_sampler = WeightedRandomSampler(
        weights=train_dataset.weights, num_samples=len(train_dataset), replacement=True
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )

    return train_loader, val_loader


class MovieRatingDataset(Dataset):
    def __init__(self, user_ids, movie_ids, ratings):
        """
        Custom Dataset for movie ratings.

        Arguments:
            user_ids: List of user IDs.
            movie_ids: List of movie IDs.
            ratings: List of ratings.
        """
        self.user_ids = torch.LongTensor(user_ids)
        self.movie_ids = torch.LongTensor(movie_ids)
        self.ratings = torch.FloatTensor(ratings)

        # Calculate inverse frequency weights
        unique_ratings, counts = np.unique(ratings, return_counts=True)
        rating_to_weight = {}
        max_count = max(counts)

        # Create weight dictionary - less frequent ratings get higher weights
        for rating, count in zip(unique_ratings, counts):
            # Using inverse frequency with smoothing
            rating_to_weight[rating] = max_count / count

        # Assign weights to each sample
        self.weights = np.array([rating_to_weight[r] for r in ratings])
        # Normalize weights
        self.weights = self.weights / self.weights.sum()

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.user_ids[idx], self.movie_ids[idx], self.ratings[idx]

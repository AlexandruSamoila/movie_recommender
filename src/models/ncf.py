import torch
import torch.nn as nn


class NCF(nn.Module):

    def __init__(
        self,
        num_users,
        num_movies,
        embedding_mf_dim,
        embedding_mlp_dim,
        hidden_units,
        dropout_rate=0.2,
    ):
        """
        Neural Collaborative Filtering (NCF) model based on the paper: https://arxiv.org/abs/1708.05031
        The model is a combination of a Matrix Factorization (MF) and a Multi-Layer Perceptron (MLP)

        Args:
            num_users: int, number of users
            num_movies: int, number of movies
            embedding_mf_dim: int, embedding dimension for MF
            embedding_mlp_dim: int, embedding dimension for MLP
            hidden_units: list[int], hidden units for MLP
            dropout_rate: float, dropout rate
        """
        super().__init__()

        # MF embeddings
        self.mf_user_embedding = nn.Embedding(num_users, embedding_mf_dim)
        self.mf_movie_embedding = nn.Embedding(num_movies, embedding_mf_dim)

        # MLP embeddings
        self.mlp_user_embedding = nn.Embedding(num_users, embedding_mlp_dim)
        self.mlp_movie_embedding = nn.Embedding(num_movies, embedding_mlp_dim)

        # MLP layers
        self.mlp = nn.Sequential()
        input_size_mlp = 2 * embedding_mlp_dim
        input_size_mf = embedding_mf_dim
        for units in hidden_units:
            self.mlp.append(nn.Linear(input_size_mlp, units))
            self.mlp.append(nn.BatchNorm1d(units))
            self.mlp.append(nn.ReLU())
            self.mlp.append(nn.Dropout(dropout_rate))
            input_size_mlp = units

        self.final = nn.Sequential(
            nn.Linear(input_size_mlp + input_size_mf, 1), nn.Sigmoid()
        )

    def forward(self, user_ids, movie_ids):
        # MLP embeddings
        mlp_user_embed = self.mlp_user_embedding(user_ids)
        mlp_movie_embed = self.mlp_movie_embedding(movie_ids)
        mlp_embed = torch.cat(
            [mlp_user_embed, mlp_movie_embed], dim=1
        )  # Concatenate along feature dimension

        # MF embeddings
        mf_user_embed = self.mf_user_embedding(user_ids)
        mf_movie_embed = self.mf_movie_embedding(movie_ids)
        mf_embed = torch.mul(
            mf_user_embed, mf_movie_embed
        )  # Element-wise multiplication

        # Process through MLP
        mlp_output = self.mlp(mlp_embed)

        # Combine MF and MLP outputs
        combined = torch.cat([mf_embed, mlp_output], dim=1)
        x = self.final(combined)
        return x

import torch
import pandas as pd
import numpy as np
from models.ncf import NCF
from utils import (
    load_data,
    create_data_loaders,
)
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from datetime import datetime


def one_epoch(model, data_loader, loss_fn, opt=None, scheduler=None):
    """
    Runs one epoch of training or validation.

    Args:
        model: The model to be trained or evaluated.
        data_loader: DataLoader object that loads the batch of data.
        loss_fn: The loss function to optimize.
        opt: Optimizer for training phase (None during evaluation).

    Returns:
        avg_loss: Average loss over the epoch.
    """
    # Get the device where the model is located (CPU or GPU)
    device = next(model.parameters()).device

    # Set the model to training or evaluation mode based on the optimizer
    train = False if opt is None else True
    model.train() if train else model.eval()

    losses = []
    for user_ids, movie_ids, ratings in data_loader:
        # Move inputs and targets to the same device as the model
        user_ids, movie_ids, ratings = (
            user_ids.to(device),
            movie_ids.to(device),
            ratings.to(device),
        )
        # Enable gradient computation only during training
        with torch.set_grad_enabled(train):
            logits = model(user_ids, movie_ids)
        loss = loss_fn(logits.squeeze(), ratings)

        if train:
            opt.zero_grad()  # Reset gradients
            loss.backward()  # Backpropagate gradients
            opt.step()  # Update model parameters
            if scheduler:
                scheduler.step()

        losses.append(loss.item())
    avg_loss = np.mean(losses)
    return avg_loss


def custom_loss(predictions, targets):
    """
    Custom loss function combining Mean Squared Error (MSE), variance matching, and range penalty.

    Args:
        predictions: Model's predicted ratings.
        targets: Ground truth ratings.

    Returns:
        total_loss: The total custom loss.
    """
    # Standard MSE loss
    mse_loss = F.mse_loss(predictions, targets)

    # Encourage prediction variance to match target variance
    pred_var = torch.var(predictions)
    target_var = torch.var(targets)
    variance_loss = F.mse_loss(pred_var, target_var)

    # Encourage the model to use the full range of predictions (between 0 and 1)
    range_loss = torch.abs(torch.min(predictions)) + torch.abs(
        1 - torch.max(predictions)
    )

    # Total loss is a weighted combination of these components
    total_loss = mse_loss + 0.2 * variance_loss + 0.1 * range_loss

    return total_loss


def train(
    model,
    train_loader,
    val_loader,
    lr=1e-3,
    max_epochs=30,
    weight_decay=0.01,
    patience=3,
):
    """
    Trains the model using the provided training and validation data loaders.

    Args:
        model: The neural network model to train.
        train_loader: DataLoader object for the training set.
        val_loader: DataLoader object for the validation set.
        lr: Learning rate for the optimizer. Default is 1e-3.
        max_epochs: Maximum number of epochs to train. Default is 30.
        weight_decay: Weight decay for L2 regularization. Default is 0.01.
        patience: Number of epochs with no improvement after which training will stop. Default is 3.
        custom_loss: Custom loss function (if needed). Default is None.

    Returns:
        train_losses: List of training losses per epoch.
        valid_losses: List of validation losses per epoch.
    """

    # Initialize the optimizer
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=lr,
        epochs=max_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,  # Spend 30% of time warming up
        div_factor=25.0,  # Start with lr/25
        final_div_factor=1e4,  # End with lr/10000
    )

    loss_fn = custom_loss
    best_valid_loss = float("inf")  # Initialize best validation loss
    patience_counter = 0  # Counter for early stopping
    train_losses, valid_losses = [], []  # Track losses for plotting/analysis

    # Training loop
    t = tqdm(range(max_epochs))
    for epoch in t:
        # Training phase
        train_loss = one_epoch(model, train_loader, loss_fn, opt, scheduler)
        # Validation phase
        valid_loss = one_epoch(model, val_loader, loss_fn)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        # Early stopping
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            patience_counter = 0  # Reset counter if validation loss improves
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

        t.set_description(f"train loss: {train_loss:.4f}, val loss: {valid_loss:.4f}")

    return train_losses, valid_losses


def plot_history(train_losses, valid_losses):
    """
    Plots the training and validation loss history over epochs.

    Args:
        train_losses: List of training losses for each epoch.
        valid_losses: List of validation losses for each epoch.
    """
    plt.figure(figsize=(7, 3))
    plt.subplot(1, 2, 1)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(train_losses, label="train")
    plt.plot(valid_losses, label="valid")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


def distribution_analysis(ratings):
    """
    Analyzes the distribution of ratings.

    Args:
        ratings: Array or list of ratings to analyze.
    """
    print("\nRating Distribution Analysis:")
    # Find unique ratings and their counts
    unique_ratings, counts = np.unique(ratings, return_counts=True)

    # Print the distribution of each rating
    for rating, count in zip(unique_ratings, counts):
        print(f"Rating {rating:.3f}: {count} samples ({count/len(ratings)*100:.1f}%)")

    # Create a histogram
    plt.figure(figsize=(10, 5))
    plt.hist(ratings, bins=20, edgecolor="black")
    plt.title("Distribution of Ratings")
    plt.xlabel("Rating Value")
    plt.ylabel("Count")
    plt.show()


# Model hyperparameters
EMBEDDING_DIM_MF = 16  # Matrix Factorization embedding dimension
EMBEDDING_DIM_MLP = 64  # MLP embedding dimension
HIDDEN_UNITS = [128, 64, 32, 16]  # Hidden units for MLP layers
DROPOUT = 0.2

# Training parameters
BATCH_SIZE = 64  # Batch size for training
LEARNING_RATE = 1e-3  # Learning rate for the optimizer
EPOCHS = 50  # Number of epochs
WEIGHT_DECAY = 0.01  # L2 regularization term
PATIENCE = 3

# Data split parameters
TRAIN_SIZE = 0.8  # Percentage of data for training

# Dataset visualization
SHOW_DATASET_DISTRIBUTION = True  # Whether to show dataset distribution

if __name__ == "__main__":

    # Load ratings dataset from CSV
    df_ratings = pd.read_csv("data/ratings_train.csv")

    # Extract data
    user_ids, movie_ids, ratings, num_users, num_movies, user_id_map, movie_id_map = (
        load_data(df_ratings)
    )

    # Optionally display dataset distribution for analysis
    if SHOW_DATASET_DISTRIBUTION:
        distribution_analysis(ratings)

    # Split the data into training and validation sets
    (
        user_ids_train,
        user_ids_val,
        movie_ids_train,
        movie_ids_val,
        ratings_train,
        ratings_val,
    ) = train_test_split(
        user_ids, movie_ids, ratings, train_size=TRAIN_SIZE, random_state=42
    )

    # Create data loaders for training and validation sets
    train_loader, val_loader = create_data_loaders(
        user_ids_train,
        movie_ids_train,
        ratings_train,
        user_ids_val,
        movie_ids_val,
        ratings_val,
        BATCH_SIZE,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the NCF model with specified hyperparameters and move it to the device
    model = NCF(
        num_users,
        num_movies,
        EMBEDDING_DIM_MF,
        EMBEDDING_DIM_MLP,
        HIDDEN_UNITS,
        DROPOUT,
    ).to(device)

    # Start the training process and plot the training history
    print("Begin training ...")
    plot_history(
        *train(
            model,
            train_loader,
            val_loader,
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            max_epochs=EPOCHS,
            patience=PATIENCE,
        )
    )

    # Save the trained model and its hyperparameters
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    # Folder name to save the trained model
    models_dir = "checkpoints"
    os.makedirs(models_dir, exist_ok=True)

    # Create model name with hyperparameters
    model_name = "NCF_" + f"h{'-'.join(map(str, HIDDEN_UNITS))}"

    # Construct the path for saving the model
    save_dir = os.path.join(models_dir, f"{timestamp}_{model_name}")
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "model.pth")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "hyperparameters": {
                "embedding_dim_mf": EMBEDDING_DIM_MF,
                "embedding_dim_mlp": EMBEDDING_DIM_MLP,
                "hidden_units": HIDDEN_UNITS,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
            },
            "user_id_map": user_id_map,
            "movie_id_map": movie_id_map,
        },
        model_path,
    )

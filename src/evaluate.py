import torch
import numpy as np
from sklearn.metrics import (
    ndcg_score,
)
from utils import load_dataloader, load_model


def evaluate_model(model, data_loader, k=10):
    """
    Evaluates the performance of the model on a given dataset using several evaluation metrics.
    This includes calculating the nDCG@k, Precision@k, and Recall@k.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        data_loader (torch.utils.data.DataLoader): A data loader that provides batches of user IDs, movie IDs, and true ratings.
        k (int): The value of k for calculating nDCG@k, Precision@k, and Recall@k (default is 10).

    Returns:
        dict: A dictionary containing the evaluation metrics: nDCG, Precision, and Recall at k.
    """
    device = next(
        model.parameters()
    ).device  # Get the device (CPU or GPU) the model is using
    model.eval()

    all_predictions = []
    all_true_ratings = []

    with torch.no_grad():
        for user_ids, movie_ids, ratings in data_loader:
            # Move tensors to the appropriate device (CPU/GPU)
            user_ids, movie_ids, ratings = (
                user_ids.to(device),
                movie_ids.to(device),
                ratings.to(device),
            )
            predictions = model(user_ids, movie_ids).squeeze()

            # Store predictions and true ratings for evaluation
            all_predictions.extend(predictions.cpu().numpy())
            all_true_ratings.extend(ratings.cpu().numpy())

    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_true_ratings = np.array(all_true_ratings)

    # Print diagnostic statistics for predictions and true ratings
    print("\nOverall Statistics:")
    print(
        f"Predictions - Mean: {all_predictions.mean():.3f}, Std: {all_predictions.std():.3f}"
    )
    print(
        f"Predictions - Min: {all_predictions.min():.3f}, Max: {all_predictions.max():.3f}"
    )
    print(
        f"True Ratings - Mean: {all_true_ratings.mean():.3f}, Std: {all_true_ratings.std():.3f}"
    )
    print(
        f"True Ratings - Min: {all_true_ratings.min():.3f}, Max: {all_true_ratings.max():.3f}"
    )

    # Calculate nDCG@k (Normalized Discounted Cumulative Gain at k)
    predictions_array = all_predictions.reshape(
        1, -1
    )  # Reshape for compatibility with ndcg_score function
    true_ratings_array = all_true_ratings.reshape(1, -1)
    ndcg = ndcg_score(
        true_ratings_array, predictions_array, k=min(k, len(all_predictions))
    )

    # Calculate Precision@k and Recall@k with a relevance threshold
    relevant_threshold = 0.6
    true_relevant = all_true_ratings >= relevant_threshold
    predicted_relevant = all_predictions >= relevant_threshold

    # Calculate Precision@k
    denominator_p = np.sum(predicted_relevant)
    numerator_p = np.sum(true_relevant & predicted_relevant)

    if denominator_p > 0:
        precision_k = numerator_p / denominator_p
    else:
        precision_k = 0.0

    # Calculate Recall@k
    denominator_r = np.sum(true_relevant)
    numerator_r = np.sum(true_relevant & predicted_relevant)
    if denominator_r > 0:
        recall_k = numerator_r / denominator_r
    else:
        recall_k = 0.0

    print(f"\nEvaluation metrics at k={k}:")
    print(f"nDCG@{k}: {ndcg:.3f}")
    print(f"Precision@{k}: {precision_k:.3f}")
    print(f"Recall@{k}: {recall_k:.3f}")

    return {"nDCG": ndcg, "precision": precision_k, "recall": recall_k}


if __name__ == "__main__":

    k = 5
    data_path = "data/ratings_test.csv"
    model_path = "path/to/model"
    test_loader = load_dataloader(data_path)
    model, _, _ = load_model(model_path)
    result = evaluate_model(model, test_loader, k=k)

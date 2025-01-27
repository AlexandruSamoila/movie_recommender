# Exploring Movie Recommendation Systems Using the Movies Dataset

## Objective:

Develop an AI model to meet the requirements of two distinct use cases for movie recommendation systems:

1. Finding movies with similar themes and genres (Content-Based Recommender)
2. Personalized movie recommendations based on user preferences (Collaborative Filtering)

## Introduction:

This project delves into the world of movie recommendation systems using the **Movies Dataset** from Kaggle. The dataset consists of valuable information about movies, such as titles, genres, release dates, and user ratings, which is used to build AI models capable of recommending movies based on themes, genres, and user preferences.

## Dataset:

The dataset used in this project is available [here on Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset/).

The dataset consists of two main files:

- **movies_metadata.csv**: Contains information about movies, including titles, genres, release dates, etc.
- **ratings.csv**: Contains user ratings for movies, including user IDs, movie IDs, and ratings.

## Use Cases:

### 1. Finding Movies with Similar Themes and Genres (Content-Based Recommender):

In this use case, the system identifies movies with similar themes and genres to a given movie. For example, if a user has rated **"Inception"** with five stars, the system will recommend other movies with similar themes and genres, such as **"Interstellar"** and **"The Matrix"**. The goal is to help users discover movies that align with their tastes, based on shared attributes like genres and themes.

### 2. Personalized Movie Recommendations Based on User Preferences (Collaborative Filtering):

This use case provides personalized movie recommendations by analyzing the preferences of users. The system identifies users with similar movie preferences and recommends movies based on what similar users have liked. The collaborative filtering approach is used to suggest movies that a user might enjoy, based on the collective ratings and preferences of similar users.

## System Architecture:

The system utilizes two recommendation models:

- **Content-Based Recommender**: Uses the genres and themes of movies to recommend similar ones based on user input.
- **Collaborative Filtering Recommender**: Leverages past user ratings and interactions to recommend personalized movies.

### Collaborative Filtering:

For collaborative filtering, the model is implemented using **Neural Collaborative Filtering (NCF)**. The model is trained on user ratings and movie interactions, learning to predict ratings for unseen movies. The model architecture uses embeddings for users and movies, followed by multi-layer perceptrons (MLP) to predict ratings.

## Features:

- **Content-Based Recommendations**: Get recommendations for movies similar to a given movie based on shared genres and themes.
- **Collaborative Filtering Recommendations**: Get personalized movie recommendations based on a user’s previous ratings and preferences.
- **Model Training**: Train the collaborative filtering model using the **Neural Collaborative Filtering (NCF)** approach.
- **Evaluation Metrics**: Evaluate the model performance using metrics such as nDCG, Precision@k, and Recall@k.

## Setup and Installation:

### Prerequisites:

- Python 3.x
- [Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset/) (Download and place the `archive/` folder in the project folder)

### Installation Steps:

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

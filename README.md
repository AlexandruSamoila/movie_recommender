# Exploring Movie Recommendation Systems Using the Movies Dataset

## Introduction:

This project delves into the world of movie recommendation systems using the **Movies Dataset** from Kaggle. The dataset consists of valuable information about movies, such as titles, genres, release dates, and user ratings, which is used to build AI models capable of recommending movies based on themes, genres, and user preferences.

## Dataset:

The dataset used in this project is available [here on Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset/).

## Use Cases:

### 1. Finding Movies with Similar Themes and Genres (Content-Based Recommender):

In this use case, the system identifies movies with similar themes and genres to a given movie. The goal is to help users discover movies that align with their tastes, based on shared attributes like genres and themes. The system utilizes the sentence-transformers model for encoding movie overviews and MultiLabelBinarizer for encoding movie genres. The combined embeddings are used to calculate cosine similarities between movies and suggest the most similar ones.

### 2. Personalized Movie Recommendations Based on User Preferences (Collaborative Filtering):

This use case provides personalized movie recommendations by analyzing the preferences of users. The system identifies users with similar movie preferences and recommends movies based on what similar users have liked. The collaborative filtering approach is used to suggest movies that a user might enjoy, based on the collective ratings and preferences of similar users. The model is implemented using **Neural Collaborative Filtering (NCF)**. The model is trained on user ratings, learning to predict ratings for unseen movies. The model is a combination of a Matrix Factorization (MF) and a Multi-Layer Perceptron (MLP) and is based on the paper: [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031)

## Setup and Installation:

### Prerequisites:

- Python 3.x
- [Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset/)

### Installation Steps:

1. Clone the repository:

   ```
   git clone https://github.com/AlexandruSamoila/movie_recommender.git
   ```

2. Create and activate a virtual environment:

   ```
   python3 -m venv /path/to/new/venv
   source /path/to/new/venv/bin/activate  # On Windows, use venv\\Scripts\\activate
   ```

3. Install the required packages:

   ```
   pip install -r requirements.txt
   ```

4. Download the [Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset/) and place the `archive/` folder in the project folder

## Running the system

Before running the system, the dataset must be processed. To process the dataset, run the following command:

```
python3 src/dataset_process.py
```

This will generate a `data/` folder containing all the processed CSV files that the system will use.

### Collaborative Filtering Recommender

The collaborative filtering recommender requires training. To do this:

1. Modify the hyperparameters in `src/train.py` according to your preferences

2. Run the following command to train the model:

```
   python3 src/train.py
```

This will train the model using the training data and save the trained model to a checkpoint.

To evaluate the model, you need to:

1. Modify the k parameter and the model path in `src/evaluate.py` to match your desired evaluation settings.

2. Run the following command to evaluate the trained model:

```
   python3 src/evaluate.py
```

This will compute evaluation metrics nDCG@k, Precision@k, and Recall@k based on the test dataset.

Once everything is set up, you can run the system and interact with the movie recommendation models:

```
python3 src/main.py
```

### Content-Based Recommender:

To run the content-based recommender, select option 1 when prompted. This will load the content_movies.csv dataset and recommend movies with similar themes and genres based on a given movie.

### Collaborative Filtering Recommender:

To run the collaborative filtering recommender, select option 2 when prompted. The system will load the movie and ratings data, initialize the NCF model, and provide personalized movie recommendations based on user preferences.

## Results

After training and evaluating the collaborative filtering model, the following evaluation metrics were obtained for k=50:

- nDCG@50: 0.727
- Precision@50: 0.713
- Recall@50: 0.424

While the precision and nDCG scores indicate that the model is performing relatively well at ranking movies, the recall suggests that there are opportunities for further improvement, as it captures a smaller proportion of relevant movies.

For the content-based recommendation task, the system successfully identified movies with similar themes and genres. For example, when searching for movies similar to The Godfather, the following movies were recommended:

1. The Godfather: Part II
2. The Godfather: Part III
3. Black Mass
4. The Replacement Killers
5. Find Me Guilty

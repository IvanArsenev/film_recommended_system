import pandas as pd
import numpy as np
import pickle
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MovieRecommender:
    def __init__(self, ratings_path, movies_path, k=70, norm='user', fill='mean'):
        self.ratings = pd.read_csv(ratings_path, sep="::", names=["userId", "movieId", "rating", "timestamp"],
                                   engine="python")
        self.movies = pd.read_csv(movies_path, sep="::", names=["movieId", "title", "genres"], engine="python",
                                  encoding="latin-1")

        self.k = k
        self.norm = norm
        self.fill = fill
        self.user_movie_matrix = self.ratings.pivot(index="userId", columns="movieId", values="rating")

        # Fill missing values in user_movie_matrix depending on the fill parameter
        if self.fill == 'mean':
            self.user_movie_matrix = self.user_movie_matrix.fillna(self.user_movie_matrix.mean())
        else:
            self.user_movie_matrix = self.user_movie_matrix.fillna(0)

        self.R = self.user_movie_matrix.values
        self.predicted_ratings_df = None

    def train(self, test_ratings_path='./data/ratings_test.dat'):
        # Normalize data (depending on the norm parameter)
        if self.norm == 'user':
            user_movie_matrix_mean = self.user_movie_matrix.mean(axis=1)
            R_centered = self.R - user_movie_matrix_mean.values.reshape(-1, 1)
        else:
            R_centered = self.R

        # Perform SVD on the normalized data
        U, sigma, Vt = svds(R_centered, k=self.k)
        sigma = np.diag(sigma)

        # Reconstruct predicted ratings
        predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_movie_matrix_mean.values.reshape(-1, 1) if self.norm == 'user' else np.dot(np.dot(U, sigma), Vt)

        self.predicted_ratings_df = pd.DataFrame(predicted_ratings, index=self.user_movie_matrix.index,
                                                 columns=self.user_movie_matrix.columns)

        # Save the model
        with open("svd_model.pkl", "wb") as f:
            pickle.dump(self.predicted_ratings_df, f)

        logger.info("Training completed! Model saved to svd_model.pkl")

        # Evaluate the model using test data
        test_ratings = pd.read_csv(test_ratings_path, sep="::", names=["userId", "movieId", "rating", "timestamp"],
                                   engine="python")

        # Predictions for the test data
        predictions = []
        true_ratings = []

        for _, row in test_ratings.iterrows():
            user_id = row['userId']
            movie_id = row['movieId']
            true_rating = row['rating']

            # Check if a prediction exists for this user and movie
            if user_id in self.predicted_ratings_df.index and movie_id in self.predicted_ratings_df.columns:
                predicted_rating = self.predicted_ratings_df.loc[user_id, movie_id]
            else:
                # If there's no prediction, use the average rating for the movie
                predicted_rating = self.predicted_ratings_df[movie_id].mean()

            predictions.append(predicted_rating)
            true_ratings.append(true_rating)

        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(true_ratings, predictions))
        logger.info(f"RMSE on test data: {rmse}")
        return rmse

    def load_model(self, model_path="svd_model.pkl"):
        with open(model_path, "rb") as f:
            self.predicted_ratings_df = pickle.load(f)

    def recommend_movies(self, user_id, N=10):
        if self.predicted_ratings_df is None:
            raise ValueError("Model is not loaded or trained")

        if user_id not in self.predicted_ratings_df.index:
            return "No data for the user"

        user_ratings = self.predicted_ratings_df.loc[user_id]

        watched_movies = self.ratings[self.ratings["userId"] == user_id]["movieId"].values
        recommendations = user_ratings.drop(index=watched_movies, errors="ignore")

        top_movies = recommendations.sort_values(ascending=False).head(N)

        top_movies = pd.merge(top_movies.reset_index(), self.movies, on="movieId")[["movieId", "title"]]

        return top_movies

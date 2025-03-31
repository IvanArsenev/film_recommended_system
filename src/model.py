import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.neighbors import NearestNeighbors
from typing import List, Tuple
import numpy as np
import pickle
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error
import logging
from sklearn.preprocessing import StandardScaler
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix

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

        # Заполнение пропущенных значений в user_movie_matrix в зависимости от параметра fill
        if self.fill == 'mean':
            self.user_movie_matrix = self.user_movie_matrix.fillna(self.user_movie_matrix.mean())
        else:
            self.user_movie_matrix = self.user_movie_matrix.fillna(0)

        self.R = self.user_movie_matrix.values
        self.predicted_ratings_df = None

    def train(self, test_ratings_path='./data/ratings_test.dat'):
        # Нормализация данных (в зависимости от параметра norm)
        if self.norm == 'user':
            user_movie_matrix_mean = self.user_movie_matrix.mean(axis=1)
            R_centered = self.R - user_movie_matrix_mean.values.reshape(-1, 1)
        else:
            R_centered = self.R

        # Выполнение SVD на нормализованных данных
        U, sigma, Vt = svds(R_centered, k=self.k)
        sigma = np.diag(sigma)

        # Восстановление предсказанных оценок
        predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_movie_matrix_mean.values.reshape(-1, 1) if self.norm == 'user' else np.dot(np.dot(U, sigma), Vt)

        self.predicted_ratings_df = pd.DataFrame(predicted_ratings, index=self.user_movie_matrix.index,
                                                 columns=self.user_movie_matrix.columns)

        # Сохранение модели
        with open("svd_model.pkl", "wb") as f:
            pickle.dump(self.predicted_ratings_df, f)

        logger.info("Обучение завершено! Модель сохранена в svd_model.pkl")

        # Оценка модели с использованием тестовых данных
        test_ratings = pd.read_csv(test_ratings_path, sep="::", names=["userId", "movieId", "rating", "timestamp"],
                                   engine="python")

        # Предсказания для тестовых данных
        predictions = []
        true_ratings = []

        for _, row in test_ratings.iterrows():
            user_id = row['userId']
            movie_id = row['movieId']
            true_rating = row['rating']

            # Проверка, есть ли в предсказаниях оценка для данного пользователя и фильма
            if user_id in self.predicted_ratings_df.index and movie_id in self.predicted_ratings_df.columns:
                predicted_rating = self.predicted_ratings_df.loc[user_id, movie_id]
            else:
                # Если предсказание отсутствует, ставим среднее значение для фильма
                predicted_rating = self.predicted_ratings_df[movie_id].mean()

            predictions.append(predicted_rating)
            true_ratings.append(true_rating)

        # Рассчитываем RMSE
        rmse = np.sqrt(mean_squared_error(true_ratings, predictions))
        logger.info(f"RMSE на тестовых данных: {rmse}")
        return rmse

    def load_model(self, model_path="svd_model.pkl"):
        with open(model_path, "rb") as f:
            self.predicted_ratings_df = pickle.load(f)

    def recommend_movies(self, user_id, N=10):
        if self.predicted_ratings_df is None:
            raise ValueError("Модель не загружена или не обучена")

        if user_id not in self.predicted_ratings_df.index:
            return "Нет данных для пользователя"

        user_ratings = self.predicted_ratings_df.loc[user_id]

        watched_movies = self.ratings[self.ratings["userId"] == user_id]["movieId"].values
        recommendations = user_ratings.drop(index=watched_movies, errors="ignore")

        top_movies = recommendations.sort_values(ascending=False).head(N)

        top_movies = pd.merge(top_movies.reset_index(), self.movies, on="movieId")[["movieId", "title"]]

        return top_movies

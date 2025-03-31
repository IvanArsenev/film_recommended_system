import argparse
import os
import sys
import yaml
from fastapi import FastAPI
import uvicorn
from model import MovieRecommender
from typing import Optional, List, Tuple
import logging

# Настройка логгера
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Инициализация приложения FastAPI
app = FastAPI()

# Инициализация моделей
recommender_by_user = MovieRecommender("./data/ratings_train.dat", "./data/movies.dat")


@app.get("/recommended_list/{film_id}")
async def make_prediction(
    film_id: int, count: Optional[int] = 10
) -> List[Tuple[int, str]]:
    """
    Получает список рекомендованных фильмов на основе заданного фильма.

    :param film_id: ID фильма, на основе которого будут сделаны рекомендации.
    :param count: Количество рекомендованных фильмов (по умолчанию 10).
    :return: Список кортежей (ID фильма, название фильма), представляющий рекомендации.
    """
    logger.info(f"Получение рекомендаций для фильма с ID {film_id}, количество: {count}")
    result = recommended_model.get_similar_movies_nn(movie_id=film_id, top_n=count)

    return_data = [(int(f_id), title) for f_id, title in result]

    logger.info(f"Рекомендации для фильма {film_id}: {return_data}")

    return return_data


@app.get("/collaborative_filtering/{user_id}")
async def make_prediction_for_user(
    user_id: int, count: Optional[int] = 10
) -> List[Tuple[int, str]]:
    """
    Получает список рекомендованных фильмов для пользователя на основе его предпочтений.

    :param user_id: ID пользователя, для которого будут сделаны рекомендации.
    :param count: Количество рекомендованных фильмов (по умолчанию 10).
    :return: Список кортежей (ID фильма, название фильма), представляющий рекомендации для пользователя.
    """
    recommendations = recommender_by_user.recommend_movies(user_id, count)
    logger.info(f"Рекомендации для пользователя {user_id}: {recommendations}")
    return_data = list(recommendations.itertuples(index=False, name=None))
    return return_data


@app.get("/retrain_model")
async def retrain_model():
    """
    Переобучает модель рекомендаций.

    Эта функция запускает процесс переобучения модели рекомендаций, используя текущие данные.

    :return: Сообщение об успехе и точность модели.
    """
    rmse_score = recommender_by_user.train()
    logger.info(f"Модель была переобучена. RMSE: {rmse_score}")
    return f"Модель была переобучена. RMSE: {rmse_score}"


def start() -> None:
    """
    Запускает сервис с конфигурацией, указанной в конфигурационном файле.

    Загружает параметры из конфигурационного файла и запускает сервер FastAPI с использованием
    указанного хоста и порта.
    """
    ap = argparse.ArgumentParser(description="Запуск сервиса с конфигурационным файлом.")
    ap.add_argument(
        '-c', '--config', type=str, required=True, help='Путь к конфигурационному файлу.'
    )

    options, _ = ap.parse_known_args(sys.argv[1:])

    if not os.path.isfile(options.config):
        logger.error(f"Конфигурационный файл {options.config} не существует.")
        sys.exit(1)

    with open(options.config, 'r') as file:
        config = yaml.safe_load(file)

    logger.info(f"Конфигурация загружена из {options.config}.")

    uvicorn.run(app, host=config["service"]["api_host"], port=config["service"]["port"])
    logger.info(f"Запуск сервера на {config['service']['api_host']}:{config['service']['port']}.")


if __name__ == '__main__':
    start()

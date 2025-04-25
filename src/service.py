import argparse
import os
import sys
import yaml
from fastapi import FastAPI
import uvicorn
from model import MovieRecommender
from typing import Optional, List, Tuple
import logging

# Logger setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Initialize models
recommender_by_user = MovieRecommender("./data/ratings_train.dat", "./data/movies.dat")


@app.get("/collaborative_filtering/{user_id}")
async def make_prediction_for_user(
    user_id: int, count: Optional[int] = 10
) -> List[Tuple[int, str]]:
    """
    Retrieves a list of recommended movies for a user based on their preferences.

    :param user_id: ID of the user for whom the recommendations will be generated.
    :param count: Number of recommended movies (default is 10).
    :return: A list of tuples (movie ID, movie title) representing the recommendations for the user.
    """
    recommendations = recommender_by_user.recommend_movies(user_id, count)
    logger.info(f"Recommendations for user {user_id}: {recommendations}")
    return_data = list(recommendations.itertuples(index=False, name=None))
    return return_data


@app.post("/retrain_model")
async def retrain_model():
    """
    Retrains the recommendation model.

    This function starts the retraining process using the current data.

    :return: A success message and the model's RMSE score.
    """
    rmse_score = recommender_by_user.train()
    logger.info(f"Model retrained. RMSE: {rmse_score}")
    return f"Model retrained. RMSE: {rmse_score}"


def start() -> None:
    """
    Starts the service using configuration specified in the config file.

    Loads parameters from the configuration file and launches the FastAPI server using
    the specified host and port.
    """
    ap = argparse.ArgumentParser(description="Run the service with a configuration file.")
    ap.add_argument(
        '-c', '--config', type=str, required=True, help='Path to the configuration file.'
    )

    options, _ = ap.parse_known_args(sys.argv[1:])

    if not os.path.isfile(options.config):
        logger.error(f"Configuration file {options.config} does not exist.")
        sys.exit(1)

    with open(options.config, 'r') as file:
        config = yaml.safe_load(file)

    logger.info(f"Configuration loaded from {options.config}.")

    uvicorn.run(app, host=config["service"]["api_host"], port=config["service"]["port"])
    logger.info(f"Server running on {config['service']['api_host']}:{config['service']['port']}.")


if __name__ == '__main__':
    start()

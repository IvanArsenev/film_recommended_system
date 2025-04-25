# 🎬 Movie Recommender System using SVD

This project is a movie recommendation system built with Python. It uses Singular Value Decomposition (SVD) to predict user ratings and recommend unseen movies based on collaborative filtering.

## 🚀 Features
- Loads and processes the MovieLens dataset

- Applies matrix factorization (SVD) for rating prediction

- Supports normalization (user-level) and customizable missing value handling (mean or zero)

- Computes RMSE on test data for evaluation

- Saves and loads trained models (.pkl format)

- Provides personalized top-N movie recommendations for a given user

## 🛠️ Installation
1. Clone the repository:

```
git clone https://github.com/IvanArsenev/film_recommended_system.git
cd movie-film_recommended_system-svd
```
2. Install dependencies:

```
pip install -r requirements.txt
```
3. Place your ratings.dat and movies.dat files (from MovieLens 1M dataset) in the ./data directory.

## 📦 Files
- movie_recommender.py: Core class with training, evaluation, and recommendation logic.

- svd_model.pkl: Saved model after training.

- README.md: Project overview and usage guide.

- data/ratings.dat: User ratings file.

- data/movies.dat: Movie metadata file.

## 📈 Usage

### Start script
```
cd src
set PYTHONPATH=$PWD (Linux: export PYTHONPATH=$PWD)
python service.py -c config/default.yaml
```
### Training the Model

Send request
```
POST http://127.0.0.1:8092/retrain_model
```

### Use model

Send request
```
GET http://127.0.0.1:8092/collaborative_filtering/{user_id}?count={num}
{user_id} = ID of the user for whom the recommendations will be generated
{num} = Number of recommended movies (default is 10)
```

## 📊 Example Output
```
[
    [
        3881,
        "Bittersweet Motel (2000)"
    ],
    [
        3607,
        "One Little Indian (1973)"
    ],
    ...
]
```

## ⚙️ Parameters of model in class MovieRecommender
```
k: Number of latent factors in SVD (default = 70)

norm: Normalization method ('user' or None)

fill: Missing value strategy ('mean' or 0)
```

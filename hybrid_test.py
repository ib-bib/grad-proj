import numpy as np
import pandas as pd
import pickle
from joblib import Parallel, delayed


ratings_df = pd.read_csv('./data/ratings.csv')
cf_model = None
cbf_model = None

with open("./collaborative_filter/cf_model.pkl", "rb") as f:
    cf_model = pickle.load(f)

with open("./content-based_filter/cbf_model.pkl", "rb") as f:
    cbf_model = pickle.load(f)


if cf_model:
    print("Collaborative Filtering model loaded!")
if cbf_model:
    print("Content-based Filtering model loaded!")


num_users = 610
num_movies_per_user = 2698
k_per_model = 5  # from each model
k_total = k_per_model * 2

OPTIMISTIC = True

def precision(pred, actual):
    if not pred:
        return 0.0
    return len(set(pred) & set(actual)) / len(pred)

def recall(pred, actual):
    if not actual:
        return 0.0
    return len(set(pred) & set(actual)) / len(actual)


def get_cbf_recs(movie_title, k):
    idx = cbf_model["title_idx"].get(movie_title)
    movie_vec = cbf_model['matrix'][idx]

    if isinstance(movie_vec, (np.ndarray)):
        movie_vec = movie_vec.reshape(1,-1)

    if idx is None:
        return []
    _, indices = cbf_model["knn"].kneighbors(movie_vec, n_neighbors=k+1)
    indices = indices.flatten()[1:]  # exclude the movie itself
    return [cbf_model["idx_id"][i] for i in indices if i in cbf_model["idx_id"]]


def get_cf_recs(movie_id, k):
    idx = cf_model["id_idx"].get(movie_id)
    movie_vec = cf_model['matrix'][idx]

    if isinstance(movie_vec, (np.ndarray)):
        movie_vec = movie_vec.reshape(1, -1)

    if idx is None:
        return []
    _, indices = cf_model["knn"].kneighbors(movie_vec, n_neighbors=k+1)
    indices = indices.flatten()[1:]  # exclude the movie itself
    return [cf_model["idx_id"][i] for i in indices if i in cf_model["idx_id"]]


def process_user(user_id):
    user_ratings = ratings_df[ratings_df['userId'] == user_id]
    rated_movie_ids = user_ratings['movieId'].tolist()
    top_movies = user_ratings[user_ratings['rating'] >= 3].head(num_movies_per_user)['movieId'].tolist()

    recommended = set()

    for movie_id in top_movies:
        movie_title = cbf_model["id_title"].get(movie_id)
        if movie_title:
            cbf_recs = get_cbf_recs(movie_title, k_per_model)
            recommended.update(cbf_recs)
        cf_recs = get_cf_recs(movie_id, k_per_model)
        recommended.update(cf_recs)

    if OPTIMISTIC:
        recommended = {movie_id for movie_id in recommended if movie_id in rated_movie_ids}

    relevant = user_ratings[user_ratings['rating'] >= 3]['movieId'].tolist()

    prec = precision(list(recommended), relevant)
    rec = recall(list(recommended), relevant)
    return prec, rec


def evaluate_hybrid_model():
    user_rating_counts = ratings_df.groupby('userId').size().sort_values(ascending=False)
    top_users = user_rating_counts.index[:num_users]

    results = Parallel(n_jobs=-1)(delayed(process_user)(user_id) for user_id in top_users)
    precision_values, recall_values = zip(*results)

    mean_precision = np.mean(precision_values)
    mean_recall = np.mean(recall_values)
    f1 = 2 * (mean_precision * mean_recall) / (mean_precision + mean_recall) if (mean_precision + mean_recall) else 0.0

    print(f"Hybrid Model Evaluation (Top {num_users} Users)")
    print(f"Precision@{k_total}: {mean_precision:.4f}")
    print(f"Recall@{k_total}: {mean_recall:.4f}")
    print(f"F1-Score: {f1:.4f}")


evaluate_hybrid_model()

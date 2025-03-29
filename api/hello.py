from flask import Flask
from fuzzywuzzy import process
# from markupsafe import escape
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

movies = pd.read_csv('../data/movies.csv')
all_titles = movies['title'].tolist()

cf_model = None
cbf_model = None

cf_weight = 6
cbf_weight = 12 - cf_weight

with open('../collaborative_filter/collaborative_filtering_model.pkl', 'rb') as f:
    cf_model = pickle.load(f)

with open('../content-based_filter/content_based_filtering_model.pkl', 'rb') as f:
    cbf_model = pickle.load(f)

def fuzzy_wuzzy_search(title):
    closest_match = process.extractOne(title, all_titles)
    actual_title = closest_match[0]
    movie_id = cbf_model['movie_ids'][actual_title]
    return [actual_title, movie_id]

@app.route("/")
def hello_world():
    return {"hello": "world"}

@app.route("/search/<movie>")
def search(movie):
    res = fuzzy_wuzzy_search(movie)
    return {"movie_name": res[0], "movie_id": res[1]}

@app.route("/cf/<movie>")
def get_cf_recs(movie):
    recommendations = []
    title, id = fuzzy_wuzzy_search(movie)
    idx = cf_model['movie_mapper'][id]
    vec = cf_model['M_comp_mtrx'][idx]
    if isinstance(vec, (np.ndarray)):
        vec = vec.reshape(1,-1)
    neighbors = cf_model['knn'].kneighbors(vec, return_distance=False, n_neighbors=cf_weight)
    for i in range(1, cf_weight):
        rec_idx = neighbors.item(i)
        rec_id = cf_model['inv_movie_mapper'][rec_idx]
        movie = cbf_model['movie_titles'][rec_id]
        recommendations.append(movie)

    return {"movie": title, "recommendations": recommendations}

@app.route("/cbf/<movie>")
def get_cbf_recs(movie):
    recommendations = []
    title = fuzzy_wuzzy_search(movie)[0]
    idx = cbf_model['movie_idx'][title]
    vec = cbf_model['sparse_features'][idx]
    if isinstance(vec, (np.ndarray)):
        vec = vec.reshape(1,-1)
    neighbors = cbf_model['knn_model'].kneighbors(vec, return_distance=False, n_neighbors=cbf_weight)
    for i in range(cbf_weight):
        rec_idx = neighbors.item(i)
        if idx == rec_idx:
            continue
        movie = cbf_model['movie_idx_to_title'][rec_idx]
        recommendations.append(movie)

    return {"movie": title, "recommendations": recommendations}

@app.route("/recommend/<movie>")
def get_hybrid_recs(movie):
    cf_recs = get_cf_recs(movie)
    cbf_recs = get_cbf_recs(movie)['recommendations']
    return {"movie": cf_recs['movie'], "cf": cf_recs['recommendations'], "cbf": cbf_recs}

@app.route("/like/<model>/<int:movie_id>")
def like_model_recommendation(model, movie_id: int):
    movie = str(movies[movies['movieId'] == movie_id]['title'][1])
    if model == "cf":
        return {"model": "Collaborative Filtering", "movie": movie}
    elif model == "cbf":
        return {"model": "Content-based Filtering", "movie": movie}
    else:
        return {"Error": "Tried to tamper with API request"}
    
# @app.route("/dislike")
# @app.route("/like/<model>/")
# def like_model(model):
#     if model == "cf":
#         return {"model": "Collaborative Filtering"}
#     elif model == "cbf":
#         return {"model": "Content-based Filtering"}
#     else:
#         return {"Error": "Tried to tamper with API request"}


# @app.route("/like/<int:movie_id>")
# def like_recommendation(movie_id: int):
#     movie = movies[movies['movieId'] == movie_id]['title']
#     return {"movie": str(movie)}
import math
from flask import Flask
from fuzzywuzzy import process
# from markupsafe import escape
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

cf_model = None
cbf_model = None

cf_weight = 6.0
cbf_weight = 12.0 - cf_weight

hybrid_recs = set()

with open('../collaborative_filter/cf_model.pkl', 'rb') as f:
    cf_model = pickle.load(f)

with open('../content-based_filter/cbf_model.pkl', 'rb') as f:
    cbf_model = pickle.load(f)

all_titles = list(cbf_model['title_idx'].keys())

@app.route("/")
def hello_world():
    return {"hello": "world"}

@app.route("/search/<movie>")
def search(movie) -> dict[str, int]:
    closest_match = process.extractOne(movie, all_titles)
    actual_title = closest_match[0]
    movie_id = int(cbf_model['title_id'][actual_title])
    return {"movie_title": actual_title, "movie_id": movie_id}

@app.route("/cf/<movie>")
def get_cf_recs(movie):
    recommendations = []
    actual_movie = search(movie)
    title = actual_movie['movie_title']
    id = actual_movie['movie_id']
    idx = cf_model['id_idx'][id]
    vec = cf_model['matrix'][idx]
    if isinstance(vec, (np.ndarray)):
        vec = vec.reshape(1,-1)
    neighbors = cf_model['knn'].kneighbors(vec, return_distance=False, n_neighbors=math.ceil(cf_weight))
    for i in range(1, math.ceil(cf_weight)):
        rec_idx = neighbors.item(i)
        rec_id = int(cf_model['idx_id'][rec_idx])
        rec_title = cbf_model['id_title'][rec_id]
        recommendations.append({"rec_title": rec_title, "rec_id": rec_id})

    return {"movie": title, "recommendations": recommendations, "model": "cf"}

@app.route("/cbf/<movie>")
def get_cbf_recs(movie) -> dict[str, list]:
    recommendations = []
    actual_movie = search(movie)
    title = actual_movie['movie_title']
    idx = cbf_model['title_idx'][title]
    vec = cbf_model['matrix'][idx]
    if isinstance(vec, (np.ndarray)):
        vec = vec.reshape(1,-1)
    neighbors = cbf_model['knn'].kneighbors(vec, return_distance=False, n_neighbors=math.ceil(cbf_weight))
    for i in range(math.ceil(cbf_weight)):
        rec_idx = neighbors.item(i)
        if idx == rec_idx:
            continue
        rec_title = cbf_model['idx_title'][rec_idx]
        rec_id = int(cbf_model['title_id'][rec_title])
        recommendations.append({"rec_title": rec_title, "rec_id": rec_id})

    return {"movie": title, "recommendations": recommendations, "model": "cbf"}

@app.route("/recommend/<movie>")
def get_hybrid_recs(movie):
    global hybrid_recs
    cf_recs = get_cf_recs(movie)
    cbf_recs = get_cbf_recs(movie)
    hybrid_recs = set(cf_recs['recommendations']) | set(cbf_recs['recommendations'])
    return {"movie": cf_recs['movie'], "cf": cf_recs['recommendations'], "cbf": cbf_recs['recommendations']}

@app.route("/like/<model>/<int:movie_id>")
def like_model_recommendation(model, movie_id: int):
    global cf_weight, cbf_weight
    title = cbf_model['id_title'][movie_id]
    if model == "cf":
        if cf_weight < 10:
            cf_weight = cf_weight + 0.2
    elif model == "cbf":
        if cf_weight > 2:
            cf_weight = cf_weight - 0.2
    else:
        return {"Error": "Tried to tamper with API request"}
    cbf_weight = 12 - cf_weight
    return {
        "model": "Collaborative Filtering" if model == "cf" else "Content-based Filtering",
        "movie": title,
        "cf_weight": cf_weight,
        "cbf_weight": cbf_weight
        }
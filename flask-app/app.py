import math
import pickle
import os
from flask import Flask, jsonify
from io import BytesIO
from fuzzywuzzy import process
from supabase import create_client



SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_API_KEY = os.getenv("SUPABASE_API_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_API_KEY)

bucket_name = "models"

def load_model_from_supabase(file_path: str):
    response = supabase.storage.from_(bucket_name).download(file_path)
    if response:
        print(f"Successfully fetched {file_path}")
        model = pickle.load(BytesIO(response))
        return model
    else:
        print(f"Failed to fetch {file_path}")
        return None


cf_model = load_model_from_supabase("cf_model.pkl")
cbf_model = load_model_from_supabase("cbf_model.pkl")


if cf_model:
    print("Collaborative Filtering model loaded!")
if cbf_model:
    print("Content-based Filtering model loaded!")


app = Flask(__name__)

cf_weight = 6.0
cbf_weight = 12.0 - cf_weight

latest_cf_recs = []
latest_cbf_recs = []
liked = []

all_titles = list(cbf_model['title_idx'].keys())

@app.route("/", methods=['GET'])
def hello_world():
    return jsonify({"hello": "world"})

@app.route("/search/<movie>", methods=['GET'])
def search(movie) -> dict[str, int]:
    closest_match = process.extractOne(movie, all_titles)
    actual_title = closest_match[0]
    movie_id = int(cbf_model['title_id'][actual_title])
    return jsonify({"movie_title": actual_title, "movie_id": movie_id})

@app.route("/cf/<movie>", methods=['GET'])
def get_cf_recs(movie):
    global latest_cf_recs
    recommendations = []
    actual_movie = search(movie).get_json()
    title = actual_movie['movie_title']
    id = actual_movie['movie_id']
    idx = cf_model['id_idx'][id]
    vec = cf_model['matrix'][idx]
    vec = vec.reshape(1,-1)
    neighbors = cf_model['knn'].kneighbors(vec, return_distance=False, n_neighbors=math.ceil(cf_weight))
    for i in range(1, math.ceil(cf_weight)):
        rec_idx = neighbors.item(i)
        rec_id = int(cf_model['idx_id'][rec_idx])
        rec_title = cbf_model['id_title'][rec_id]
        recommendations.append({"rec_title": rec_title, "rec_id": rec_id})
        latest_cf_recs.append(rec_id)

    return jsonify({"movie": title, "recommendations": recommendations, "model": "cf"})

@app.route("/cbf/<movie>", methods=['GET'])
def get_cbf_recs(movie) -> dict[str, list]:
    global latest_cbf_recs
    recommendations = []
    actual_movie = search(movie).get_json()
    title = actual_movie['movie_title']
    idx = cbf_model['title_idx'][title]
    vec = cbf_model['matrix'][idx]
    vec = vec.reshape(1,-1)
    neighbors = cbf_model['knn'].kneighbors(vec, return_distance=False, n_neighbors=math.ceil(cbf_weight))
    for i in range(math.ceil(cbf_weight)):
        rec_idx = neighbors.item(i)
        if idx == rec_idx:
            continue
        rec_title = cbf_model['idx_title'][rec_idx]
        rec_id = int(cbf_model['title_id'][rec_title])
        recommendations.append({"rec_title": rec_title, "rec_id": rec_id})
        latest_cbf_recs.append(rec_id)

    return jsonify({"movie": title, "recommendations": recommendations, "model": "cbf"})

@app.route("/recommend/<movie>", methods=['GET'])
def get_hybrid_recs(movie):
    cf_recs = get_cf_recs(movie).get_json()
    cbf_recs = get_cbf_recs(movie).get_json()
    return jsonify({
        "movie": cf_recs['movie'], 
        "cf": cf_recs['recommendations'], 
        "cbf": cbf_recs['recommendations']
        })

@app.route("/like/<model>/<int:movie_id>", methods=['GET'])
def like_model_recommendation(model, movie_id: int):
    global cf_weight, cbf_weight
    title = cbf_model['id_title'][movie_id]
    if model == "cf":
        if cf_weight < 10 and movie_id not in latest_cbf_recs:
            cf_weight = cf_weight + 0.2
    elif model == "cbf":
        if cf_weight > 2 and movie_id not in latest_cf_recs:
            cf_weight = cf_weight - 0.2
    else:
        return jsonify({"Error": "Tried to tamper with API request"})
    cbf_weight = 12 - cf_weight
    return jsonify({
        "model": "Collaborative Filtering" if model == "cf" else "Content-based Filtering",
        "movie": title,
        "cf_weight": cf_weight,
        "cbf_weight": cbf_weight
        })

if __name__ == "__main__":
    app.run(debug=True)
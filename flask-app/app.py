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

# cf_model = None
# cbf_model = None

# with open("cf_model.pkl", "rb") as f:
#     cf_model = pickle.load(f)

# with open("cbf_model.pkl", "rb") as f:
#     cbf_model = pickle.load(f)


if cf_model:
    print("Collaborative Filtering model loaded!")
if cbf_model:
    print("Content-based Filtering model loaded!")


app = Flask(__name__)

all_titles = list(cbf_model['title_idx'].keys())

def balanced_weights(cf_raw: float):
    cbf_raw = 12 - cf_raw
    if cf_raw < cbf_raw:
        return math.ceil(cf_raw), math.trunc(cbf_raw)
    else:
        return math.trunc(cf_raw), math.ceil(cbf_raw)
    

@app.route("/", methods=['GET'])
def hello_world():
    return jsonify({"hello": "world"})

@app.route("/search/<movie>", methods=['GET'])
def search(movie) -> dict[str, int]:
    closest_match = process.extractOne(movie, all_titles)
    actual_title = closest_match[0]
    movie_id = int(cbf_model['title_id'][actual_title])
    return jsonify({"movie_title": actual_title, "movie_id": movie_id})

@app.route("/get_title/<movie_id>")
def get_title(movie_id):
    movie_id = int(movie_id)
    return jsonify({"movie_title": cbf_model['id_title'], "movie_id": movie_id})

@app.route("/cf/<movie_id>/<cf_weight>", methods=['GET'])
def get_cf_recs(movie_id, cf_weight):
    movie_id = int(movie_id)
    cf_weight = int(cf_weight)
    recommendations = []
    idx = cf_model['id_idx'][movie_id]
    vec = cf_model['matrix'][idx]
    vec = vec.reshape(1,-1)
    neighbors = cf_model['knn'].kneighbors(vec, return_distance=False, n_neighbors=cf_weight)
    for i in range(cf_weight):
        rec_idx = neighbors.item(i)
        if idx == rec_idx:
            continue
        rec_id = int(cf_model['idx_id'][rec_idx])
        rec_title = cbf_model['id_title'][rec_id]
        recommendations.append({"rec_title": rec_title, "rec_movie_id": rec_id})

    return jsonify({"recommendations": recommendations, "model": "cf"})

@app.route("/cbf/<movie>/<cbf_weight>", methods=['GET'])
def get_cbf_recs(movie_id, cbf_weight) -> dict[str, list]:
    movie_id = int(movie_id)
    cbf_weight = int(cbf_weight)
    recommendations = []
    idx = cbf_model['id_idx'][movie_id]
    vec = cbf_model['matrix'][idx]
    vec = vec.reshape(1,-1)
    neighbors = cbf_model['knn'].kneighbors(vec, return_distance=False, n_neighbors=cbf_weight)
    for i in range(cbf_weight):
        rec_idx = neighbors.item(i)
        if idx == rec_idx:
            continue
        rec_title = cbf_model['idx_title'][rec_idx]
        rec_id = int(cbf_model['title_id'][rec_title])
        recommendations.append({"rec_title": rec_title, "rec_movie_id": rec_id})

    return jsonify({"recommendations": recommendations, "model": "cbf"})

@app.route("/recommend/<movie_id>/<cf_weight>", methods=['GET'])
def get_hybrid_recs(movie_id, cf_weight):
    cf_weight = float(cf_weight)
    cf_weight, cbf_weight = balanced_weights(float(cf_weight))
    movie_id = int(movie_id)
    cf_recs = get_cf_recs(movie_id, cf_weight).get_json()
    cbf_recs = get_cbf_recs(movie_id, cbf_weight).get_json()
    return jsonify({
        "movie_title": cbf_model['id_title'][movie_id],
        "cf": cf_recs['recommendations'], 
        "cbf": cbf_recs['recommendations']
        })


if __name__ == "__main__":
    app.run(debug=True)
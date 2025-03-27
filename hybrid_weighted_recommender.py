import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import process

# Load models
with open("./collaborative_filter/cf.pkl", "rb") as f:
    cf_model = pickle.load(f)

with open("./content-based_filter/cbf.pkl", "rb") as f:
    cbf_model = pickle.load(f)

# Load movies dataset
movies_df = pd.read_csv('./data/movies.csv')
ratings_df = pd.read_csv('./data/ratings.csv')

# Collaborative Filtering functions
def cf_find_similar_movies(movie_id, matrix, movie_mapper, inv_movie_mapper, k=5, metric='cosine'):
    X_matrix = matrix.T
    neighborIDs = []
    movie_ind = movie_mapper[movie_id]
    movie_vec = X_matrix[movie_ind]
    if isinstance(movie_vec, (np.ndarray)):
        movie_vec = movie_vec.reshape(1, -1)
    kNN = NearestNeighbors(n_neighbors=k + 1, algorithm="brute", metric=metric)
    kNN.fit(X_matrix)
    neighbor = kNN.kneighbors(movie_vec, return_distance=False)
    for i in range(1, k + 1):
        n = neighbor.item(i)
        neighborIDs.append(inv_movie_mapper[n])
    return neighborIDs

# Content-Based Filtering functions
def cbf_get_content_based_recommendations(title_string, n_recommendations=5):
    title = movie_finder(title_string)
    idx = cbf_model["movie_idx"][title]
    movie_vector = cbf_model["knn_model"]._fit_X[idx]  # Access the movie's vector from the fitted data
    _, indices = cbf_model["knn_model"].kneighbors(movie_vector.reshape(1, -1), n_neighbors=n_recommendations + 1)
    similar_movies = movies_df['title'].iloc[indices.flatten()[1:]]
    return similar_movies.tolist()

def movie_finder(title):
    all_titles = movies_df['title'].tolist()
    closest_match = process.extractOne(title, all_titles)
    return closest_match[0]

# Hybrid Recommendation Function
def hybrid_recommendations(movie_title, cf_weight=0.5, cbf_weight=0.5):
    movie_id = movies_df[movies_df['title'] == movie_title]['movieId'].iloc[0]
    cf_recommendations = cf_find_similar_movies(movie_id, cf_model["M_comp_mtrx"].T, cf_model["movie_mapper"], cf_model["inv_movie_mapper"], k=5)
    cbf_recommendations = cbf_get_content_based_recommendations(movie_title, n_recommendations=5)

    cf_titles = [movies_df[movies_df['movieId'] == movie_id]['title'].iloc[0] for movie_id in cf_recommendations]

    return cf_titles, cbf_recommendations

# Feedback and Weight Adjustment
def adjust_weights(feedback, cf_weight, cbf_weight):
    if feedback == "cf":
        cf_weight += 0.1
        cbf_weight -= 0.1
    elif feedback == "cbf":
        cf_weight -= 0.1
        cbf_weight += 0.1

    cf_weight = max(0.1, min(0.9, cf_weight))
    cbf_weight = max(0.1, min(0.9, cbf_weight))

    return cf_weight, cbf_weight

# Example Usage
movie_title = "Harry Potter"
cf_weight = 0.5
cbf_weight = 0.5

cf_recs, cbf_recs = hybrid_recommendations(movie_title, cf_weight, cbf_weight)

print("Collaborative Filtering Recommendations:")
for rec in cf_recs:
    print(rec)

print("\nContent-Based Filtering Recommendations:")
for rec in cbf_recs:
    print(rec)

feedback = input("\nWhich set of recommendations did you prefer? (cf/cbf): ").lower()

cf_weight, cbf_weight = adjust_weights(feedback, cf_weight, cbf_weight)

print(f"\nUpdated Weights: CF={cf_weight:.2f}, CBF={cbf_weight:.2f}")

# Example of updated recommendations with new weights
cf_recs, cbf_recs = hybrid_recommendations(movie_title, cf_weight, cbf_weight)

print("\nUpdated Collaborative Filtering Recommendations:")
for rec in cf_recs:
    print(rec)

print("\nUpdated Content-Based Filtering Recommendations:")
for rec in cbf_recs:
    print(rec)
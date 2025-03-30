import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.special import expit  # Sigmoid function
from fuzzywuzzy import process
from kneed import KneeLocator
import pickle


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# fuzzywuzzy search function
def movie_finder(title):
    all_titles = movies_df['title'].tolist()
    closest_match = process.extractOne(title, all_titles)
    print(f'Query Result: {closest_match[0]}')
    return closest_match[0]

# unsupervised nearest neighbor retrieval algorithm from scikit-learn
def find_similar_movies(movie_id, matrix, movie_mapper, inv_movie_mapper, k=5):
    neighborIDs = []
    movie_ind = movie_mapper[movie_id] # get index of movie in matrix
    movie_vec = matrix[movie_ind] # single vector; one row from the matrix; one movie

    if isinstance(movie_vec, (np.ndarray)):
        movie_vec = movie_vec.reshape(1, -1) # convert to 2D to comply with sklearn function

    # we want the k next results (k+1) since the output will include the query vector
    neighbors = kNN.kneighbors(movie_vec, return_distance=False, n_neighbors=k+1)

    for i in range(1, k + 1):
        idx = neighbors.item(i)
        neighborIDs.append(inv_movie_mapper[idx])

    return neighborIDs

# accuracy metrics
# MAE
def mean_absolute_error(actual, predictions):
    actual = np.array(actual)
    predictions = np.array(predictions)
    
    if actual.shape != predictions.shape:
        raise ValueError("Shapes of actual and predicted ratings must match.")
    
    return np.mean(np.abs(actual - predictions))

# RMSE
def root_mean_square_error(actual, predictions):
    if len(actual) != len(predictions):
        raise ValueError('The length of the actual ratings must be equal to the length of the predictions')
    
    n = len(actual)
    total_error = 0

    for i in range(n):
        total_error += (actual[i] - predictions[i]) ** 2

    rmse = math.sqrt(total_error / n)
    return rmse

# Recommendation Quality metrics
# Precision
def precision(recommended_items, relevant_items):
    true_positives = len(set(recommended_items).intersection(set(relevant_items)))
    total_recommended_items = len(recommended_items)

    precision_value = true_positives / total_recommended_items if total_recommended_items > 0 else 0
    return precision_value

# Recall
def recall(recommended_items, relevant_items):
    true_positives = len(set(recommended_items).intersection(set(relevant_items)))
    total_relevant_items = len(relevant_items)

    recall_value = true_positives / total_relevant_items if total_relevant_items > 0 else 0
    return recall_value

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# load dataset
movies_df = pd.read_csv('../data/movies.csv')
ratings_df = pd.read_csv('../data/ratings.csv')

# number of users
U = ratings_df['userId'].nunique()
# number of movies
M = ratings_df['movieId'].nunique()

# mapper from userId to index in our upcoming utility matrix
user_mapper = dict(zip(np.unique(ratings_df['userId']), list(range(U))))
# mapper from movieId to index
movie_mapper = dict(zip(np.unique(ratings_df['movieId']), list(range(M))))

# inverse mappers (from index to id)
inv_user_mapper = dict(zip(list(range(U)), np.unique(ratings_df['userId'])))
inv_movie_mapper = dict(zip(list(range(M)), np.unique(ratings_df['movieId'])))

# indices
user_index = [user_mapper[i] for i in ratings_df['userId']]
movie_index = [movie_mapper[i] for i in ratings_df['movieId']]

# compressed sparse row matrix (utility matrix)
X = csr_matrix((ratings_df['rating'], (user_index, movie_index)), shape=(U, M))

# calculating bayesian average rating for each movie
movie_stats = ratings_df.groupby('movieId')['rating'].agg(['count', 'mean'])
C = movie_stats['count'].mean()
m = movie_stats['mean'].mean()

def bayesian_avg(ratings):
    bayesian_avg = (C*m+ratings.sum())/(C+ratings.count())
    return round(bayesian_avg, 2)

bayesian_avg_ratings = ratings_df.groupby('movieId')['rating'].agg(bayesian_avg).reset_index()
bayesian_avg_ratings.columns = ['movieId', 'bayesian_avg']
movie_stats = movie_stats.merge(bayesian_avg_ratings, on='movieId')
movie_stats = movie_stats.merge(movies_df[['movieId', 'title']])

# movie_stats.to_csv('movie_stats.csv')

# preparing for training-test split
X_arr = X.toarray()
test_data_coords = []
test_ratings = []

for i, user in enumerate(X_arr):
    nonzero_coords = []
    nnz = np.count_nonzero(user) # number of ratings the user submitted
    twenty_percent_of_nnz = math.ceil(0.2 * nnz) # taking 20 percent of each user's ratings only for testing
    for j, rating in enumerate(user):
        if rating > 0:
            nonzero_coords.append([i, j])
            movie_id = inv_movie_mapper[j]
            movie_bayesian_avg = movie_stats[movie_stats['movieId'] == movie_id]['bayesian_avg']
            X[i, j] = movie_bayesian_avg # masking value with bayesian average rating of that movie
            # this increases MAE and RMSE (reconstruction error) but higher precision and recall than masking with 0
            test_ratings.append(rating)
        if len(nonzero_coords) == twenty_percent_of_nnz:
            break
    test_data_coords.append(nonzero_coords)

X_arr = None
# Optimize n_components using Frobenius norm
# errors = []
# components_range = range(5, 100, 5) # Test components from 5 to 100
# for n in components_range:
#     svd = TruncatedSVD(n_components=n, random_state=42, n_iter=10)
#     H = svd.fit_transform(X.T)
#     W = svd.components_
#     reconstructed_X = np.dot(W.T, H.T)
#     error = np.linalg.norm(X - reconstructed_X, ord='fro')  # Frobenius norm of reconstruction error
#     errors.append(error)

# # Find elbow point automatically
# knee_locator = KneeLocator(components_range, errors, curve='convex', direction='decreasing')
# optimal_n = knee_locator.knee
# print(f'Optimal number of components: {optimal_n}') # output was 35

# # Plot the elbow curve
# plt.figure(figsize=(8, 5))
# plt.plot(components_range, errors, marker='o')
# # Highlight the elbow point
# plt.scatter(optimal_n, knee_locator.knee_y, color='red', s=150, edgecolors='black', label=f'Elbow at n={optimal_n}', zorder=3)
# # Dashed line at elbow
# plt.axvline(optimal_n, color='r', linestyle='--', alpha=0.6)
# plt.ylabel('Frobenius Norm')
# plt.xlabel('Components')
# plt.title('Elbow graph for optimal number of latent')
# plt.grid()
# plt.show()

# Matrix Factorization using the optimal n of components
svd = TruncatedSVD(n_components=35, random_state=42, n_iter=10) # hard-coded 35 components (elbow point)
M_comp_mtrx = svd.fit_transform(X.T) # movies x latent features (9274 movies, 26 components)
U_comp_mtrx = svd.components_ # latent features (of the movies) x users (26 x 610)

# nearest neighbors
kNN = NearestNeighbors(algorithm="brute", metric='cosine')
kNN.fit(M_comp_mtrx)

# reconstructing the matrix to create our predictions
ndarr_reconstruct_X = np.dot(U_comp_mtrx.T, M_comp_mtrx.T) # dot product function returns numpy ndarray
# sigmoid transformation (upper bound=5, lower bound=0.5)
sigmoid_reconstruct_X = expit(ndarr_reconstruct_X) * 4.5 + 0.5
reconstructed_X = csr_matrix(sigmoid_reconstruct_X)

# predictions array to be used in evaluation
predictions = []
for i in range(len(test_data_coords)):
    for j in range(len(test_data_coords[i])):
        x = test_data_coords[i][j][0]
        y = test_data_coords[i][j][1]
        predictions.append(reconstructed_X[x, y])

# Mean Absolute Error
print(f'MAE: {mean_absolute_error(test_ratings, predictions):.2f}')

# Root Mean Square Error
print(f'RMSE: {root_mean_square_error(test_ratings, predictions):.2f}')

# Precision@k for the top power users:
# Number of users to evaluate
num_users = 10
# Number of movies whose rating is >= 3.5 which we get to generate recommendations
num_movies_per_user = 10
# Number of recommendations we generate
k = 20
# Find the top users who have rated the most movies (descending from highest rating downwards)
user_rating_counts = ratings_df.groupby('userId').size().sort_values(ascending=False)
top_users = user_rating_counts.index[:num_users]  # Get top N users with most ratings

precision_values = []
recall_values = []
# map movie titles to movie IDs
movie_titles = dict(zip(movies_df['movieId'], movies_df['title']))

# Iterate through each top user
for user_id in top_users:
    # Get all movies rated by this user
    user_ratings = ratings_df[ratings_df['userId'] == user_id]
    top_movies = user_ratings[user_ratings['rating'] >= 3.5].head(num_movies_per_user)['movieId'].tolist()  # First n movies

    # Get recommended movies
    recommended_movies = []

    for movie_id in top_movies:
        similar_movies = find_similar_movies(movie_id, M_comp_mtrx, movie_mapper, inv_movie_mapper, k=k)
        for similar_movie_id in similar_movies:
            recommended_movies.append(similar_movie_id)  # Add recommendations to list

    # Get relevant movies (rated 3.5 or above by the user)
    relevant_movies = user_ratings[user_ratings['rating'] >= 3.5]['movieId'].to_list()

    # Compute precision
    prec = precision(recommended_movies, relevant_movies)
    rec = recall(recommended_movies, relevant_movies)
    precision_values.append(prec)
    recall_values.append(rec)

mean_precision = np.mean(precision_values)
print(f"Mean average Precision@{k} of the top {num_users} power users is {mean_precision:.4f}")

mean_recall = np.mean(recall_values)
print(f"Mean average Recall@{k} of the top {num_users} power users is {mean_recall:.4f}")

f1_score = 2 * (mean_precision * mean_recall) / (mean_precision + mean_recall)
print(f"F1-Score {f1_score:.4f}")

# Save the trained model, feature matrix and mappings
model_data = {
    "knn": kNN,
    "id_idx": movie_mapper,
    "idx_id": inv_movie_mapper,
    "matrix": M_comp_mtrx
}

# Save to disk
with open("cf_model.pkl", "wb") as f:
    pickle.dump(model_data, f)

print("Model saved successfully!")
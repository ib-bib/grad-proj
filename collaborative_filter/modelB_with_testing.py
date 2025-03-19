import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from scipy.sparse import csr_matrix
from scipy.special import expit  # Sigmoid function
from fuzzywuzzy import process
from kneed import KneeLocator

# ~~~~~~~~~~~~~~~~~~~~~~~~~~#

# fuzzywuzzy search function
def movie_finder(title):
    all_titles = movies_df['title'].tolist()
    closest_match = process.extractOne(title, all_titles)
    print(f'Query Result: {closest_match[0]}')
    return closest_match[0]

# unsupervised nearest neighbor retrieval algorithm from scikit-learn
def find_similar_movies(movie_id, matrix, movie_mapper, inv_movie_mapper, k=10, metric='cosine'):
    X_matrix = matrix.T
    neighborIDs = []
    movie_ind = movie_mapper[movie_id] # get index of movie in matrix
    movie_vec = X_matrix[movie_ind] # single vector; one row from the matrix; one movie
    if isinstance(movie_vec, (np.ndarray)):
        movie_vec = movie_vec.reshape(1, -1) # convert to 2D to comply with sklearn function

    # we want the k next results (k+1) since the output will include the query vector
    kNN = NearestNeighbors(n_neighbors=k+1, algorithm="brute", metric=metric)
    kNN.fit(X_matrix)
    neighbor = kNN.kneighbors(movie_vec, return_distance=False)

    for i in range(1, k + 1):
        n = neighbor.item(i)
        neighborIDs.append(inv_movie_mapper[n])

    return neighborIDs

# accuracy metrics
# MAE
def mean_absolute_error(actual, predictions):
    if len(actual) != len(predictions):
        raise ValueError('The length of the actual ratings must be equal to the length of the predictions')
    
    n = len(actual)
    total_error = 0

    for i in range(n):
        total_error += abs(actual[i] - predictions[i])

    mae = total_error / n
    return mae

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

# def f1_score

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# load dataset
movies_df = pd.read_csv('movies.csv')
ratings_df = pd.read_csv('ratings.csv')

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

X_arr = X.toarray()

test_data_coords = []
test_ratings = []

for i, user in enumerate(X_arr):
    nonzero_coords = []
    nnz = np.count_nonzero(user)
    for j, rating in enumerate(user):
        if rating > 0:
            nonzero_coords.append([i, j])
            test_ratings.append(rating)
        if len(nonzero_coords) == nnz:
            break
    test_data_coords.append(nonzero_coords)

X_arr = None

# print(len(test_ratings))
# print(f'Shape of data: {X.shape}') # (610 users, 9724 movies)

# Optimize n_components using RMSE
# errors = []
# components_range = range(5, 100, 5)
# for n in components_range:
#     svd = TruncatedSVD(n_components=n, random_state=42, n_iter=10)
#     H = svd.fit_transform(X.T)
#     W = svd.components_
#     reconstructed_X = np.dot(W.T, H.T)
#     flat_test_data_coords = [coord for sublist in test_data_coords for coord in sublist]
#     predictions = [reconstructed_X[i, j] for i, j in flat_test_data_coords]
#     errors.append(root_mean_square_error(test_ratings, predictions))

# # Find optimal components using RMSE knee point
# knee_locator = KneeLocator(components_range, errors, curve='convex', direction='decreasing')
# optimal_n = knee_locator.knee
# print(f'Optimal number of components: {optimal_n}')

# # Plot RMSE curve
# plt.figure(figsize=(8, 5))
# plt.plot(components_range, errors, marker='o', label='RMSE')
# plt.scatter(optimal_n, knee_locator.knee_y, color='red', s=150, edgecolors='black', label=f'Optimal n={optimal_n}')
# plt.axvline(optimal_n, color='r', linestyle='--', alpha=0.6)
# plt.xlabel('Number of Components')
# plt.ylabel('RMSE')
# plt.title('RMSE as a function of the number of components')
# plt.legend()
# plt.grid()
# plt.show()

# Matrix Factorization using the optimal n of components
svd = TruncatedSVD(n_components=30, random_state=42, n_iter=10)
M_comp_mtrx = svd.fit_transform(X.T) # movies x latent features (9274 movies, 26 components)
U_comp_mtrx = svd.components_ # latent features (of the movies) x users (26 x 610)

# reconstructing the matrix to create our predictions
ndarr_reconstruct_X = np.dot(U_comp_mtrx.T, M_comp_mtrx.T) # dot product function returns numpy ndarray
# sigmoid transformation (upper bound=5, lower bound=0.5)
sigmoid_reconstruct_X = expit(ndarr_reconstruct_X) * 4.5 + 0.5
reconstructed_X = csr_matrix(sigmoid_reconstruct_X)

# predictions array for those 20% masked values
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
# Number of movies whose rating is >= 3.5 which we get
num_movies_per_user = 10
k = 30
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
        similar_movies = find_similar_movies(movie_id, M_comp_mtrx.T, movie_mapper, inv_movie_mapper, k=k, metric='cosine')
        for similar_movie_id in similar_movies:
            recommended_movies.append(similar_movie_id)  # Add recommendations to list

    # Get relevant movies (rated 3.5 or above by the user)
    relevant_movies = user_ratings[user_ratings['rating'] >= 3.5]['movieId']

    # Compute precision
    prec = precision(recommended_movies, relevant_movies)
    rec = recall(recommended_movies, relevant_movies)
    precision_values.append(prec)
    recall_values.append(rec)

mean_average_precision = np.mean(precision_values)
print(f"Mean average Precision@{k} of the top {num_users} power users is {mean_average_precision:.4f}")

mean_average_recall = np.mean(recall_values)
print(f"Mean average Recall@{k} of the top {num_users} power users is {mean_average_recall:.4f}")

f1_score = 2 * (mean_average_precision * mean_average_recall) / (mean_average_precision + mean_average_recall)
print(f"F1-Score {f1_score}")

'''
TO DO:
1. Save model to be retrieved from file system easily
2. Build pipeline to incorporate new ratings of movies
3. Retrain model with new ratings
4. When combined with content-based, use bayesian average rating to sort recommendations
5. Pipeline : Split (80/20) => TruncatedSVD => NN (unsupervised) => Test => Classification Report
6. Experiment with Surprise
'''
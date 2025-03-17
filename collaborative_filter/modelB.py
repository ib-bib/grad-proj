import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from kneed import KneeLocator
from fuzzywuzzy import process

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

# inverse mappers
inv_user_mapper = dict(zip(list(range(U)), np.unique(ratings_df['userId'])))
inv_movie_mapper = dict(zip(list(range(M)), np.unique(ratings_df['movieId'])))

# indices
user_index = [user_mapper[i] for i in ratings_df['userId']]
movie_index = [movie_mapper[i] for i in ratings_df['movieId']]

# compressed sparse row matrix (utility matrix)
X = csr_matrix((ratings_df['rating'], (user_index, movie_index)), shape=(U, M))

# shape of data
print(X.shape)

# sparsity of matrix
n_total = X.shape[0] * X.shape[1] # rows * columns
n_ratings = X.nnz # number of non-zero values
sparsity = n_ratings / n_total
print(f"Matrix sparsity: {round(sparsity * 100, 2)}%")

errors = []
components_range = range(5, 100, 5) # Test components from 5 to 100
for n in components_range:
    svd = TruncatedSVD(n_components=n, random_state=42, n_iter=10)
    H = svd.fit_transform(X.T)
    W = svd.components_
    reconstructed_X = np.dot(W.T, H.T)
    error = np.linalg.norm(X.toarray() - reconstructed_X, ord='fro')  # Frobenius norm
    errors.append(error)

# Find elbow point automatically
knee_locator = KneeLocator(components_range, errors, curve='convex', direction='decreasing')
optimal_n = knee_locator.knee
print(f'Optimal number of components: {optimal_n}') # output was 35

# Plot the elbow curve
plt.figure(figsize=(8, 5))
plt.plot(components_range, errors, marker='o')
# Highlight the elbow point
plt.scatter(optimal_n, knee_locator.knee_y, color='red', s=150, edgecolors='black', label=f'Elbow at n={optimal_n}', zorder=3)
# Dashed line at elbow
plt.axvline(optimal_n, color='r', linestyle='--', alpha=0.6)
plt.xlabel('Number of Components')
plt.xlabel('Error')
plt.title('Frobenius error plot as a function of the number of components')
plt.grid()
plt.show()

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
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#  

# Matrix Factorization using the optimal n of components
svd = TruncatedSVD(n_components=35, random_state=42, n_iter=10) # hard-coded 35 components
Q = svd.fit_transform(X.T) # orthogonal matrix to our original; T puts movies in rows => M X F (features)
print(Q.shape)

# Testing out the model's recommendations
movie_title = "Toy Story"
title = movie_finder(movie_title)
movie_id_dict = dict(zip(movies_df['title'], movies_df['movieId']))
movie_id = movie_id_dict[title]
similar_movies = find_similar_movies(movie_id, Q.T, movie_mapper, inv_movie_mapper, k=10, metric='cosine') # transpose because function expects U X M matrix

# map movie titles to movie IDs
movie_titles = dict(zip(movies_df['movieId'], movies_df['title']))

for i in similar_movies:
    print(movie_titles[i])
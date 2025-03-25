import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process
from scipy.sparse import csr_matrix

# Load the data
ratings = pd.read_csv('./data/ratings.csv')
movies = pd.read_csv('./data/movies.csv')

# Create a user-item utility matrix using the ratings
def createMatrix(df: pd.DataFrame):
    """
    Generate a user-item matrix from the ratings dataframe

    Arguments:
        df: a pandas dataframe object representing all the ratings in a 3-column format (userId, movieId, rating)

    Returns:
        X: a compressed sparse row user-item matrix
        movie_mapper: a dictionary that maps the movie IDs to movie indices
        movie_inv_mapper: (invserse of the above) a dictionary that maps movie indices to movie IDs
    """

    U = df['userId'].nunique() # number of unique users
    M = df['movieId'].nunique() # number of unique movies

    # using the zip function, we create tuples of the form of (key, value) pairs
    # the numpy unique function returns the sorted unique elements of the array
    user_mapper = dict(zip(np.unique(df['userId']), list(range(U))))
    movie_mapper = dict(zip(np.unique(df['movieId']), list(range(M))))

    movie_inv_mapper = dict(zip(list(range(M)), np.unique(df['movieId'])))

    user_index = [user_mapper[i] for i in df['userId']]
    item_index = [movie_mapper[i] for i in df['movieId']]

    X = csr_matrix((df['rating'], (user_index, item_index)), shape=(U,M))

    return X, movie_mapper, movie_inv_mapper

utilityMatrix, movie_mapper, movie_inv_mapper = createMatrix(ratings)

# Apply Non-negative Matrix Factorization (NMF)
nmf = NMF(n_components=30, max_iter=160, random_state=42)
user_factors = nmf.fit_transform(utilityMatrix)
item_factors = nmf.components_

# Compute cosine similarity for item factors
movie_similarity = cosine_similarity(item_factors.T)

def get_movie_id_from_title(movie_title: str):
    """
    Find the closest matching movie title using fuzzy matching.
    
    Arguments:
    - movie_title: The name of the movie input by the user.
    
    Returns:
    - movie_id (int): The ID of the closest matching movie.
    """
    choices = movies['title'].tolist()
    closest_match = process.extractOne(movie_title, choices)
    if closest_match:
        match_title = closest_match[0]
        movie_id = movies[movies['title'] == match_title]['movieId'].values[0]
        print(f"Matched input to: '{match_title}' (Movie ID: {movie_id})")
        return movie_id
    else:
        raise ValueError("No close match found for the movie title.")

def recommend_movies_by_title(movie_title: str, top_n: int):
    """
    Recommend movies based on the title of a given movie, excluding the input movie itself.
    
    Arguments:
    - movie_title (str): Title of the movie to base recommendations on.
    - top_n (int): Number of recommendations to return.
    
    Returns:
    - List of recommended movie titles.
    """
    movie_id = get_movie_id_from_title(movie_title)

    if movie_id not in movie_mapper:
        raise ValueError("Movie ID not found in matrix.")
    
    movie_index = movie_mapper[movie_id]
    
    similarity_scores = list(enumerate(movie_similarity[movie_index]))
    sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True) # Descending (highest to lowest)
    
    # Filter out the matched movie's own index
    recommended_movie_indices = [i[0] for i in sorted_scores if i[0] != movie_index][:top_n]
    recommended_movie_ids = [movie_inv_mapper[idx] for idx in recommended_movie_indices]
    
    recommended_movies = movies[movies['movieId'].isin(recommended_movie_ids)]['title'].tolist()
    return recommended_movies

# Prompt the user for input
user_input = input("Enter the name of a movie you like: ")
recommendations = recommend_movies_by_title(user_input, top_n=10)

# Display recommendations
print("\nRecommendations based on your input:")
for idx, title in enumerate(recommendations, start=1):
    print(f"{idx}. {title}")

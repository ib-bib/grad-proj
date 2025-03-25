import pandas as pd
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process

# load the dataset, since we're only concerned with movie content, this is all we care about
movies = pd.read_csv('./data/movies.csv')

# dictionary to map movie titles to their IDs, will be used later in the code
movie_idx = dict(zip(movies['title'], list(movies.index)))

# the main content we are concerned with for the movies is the list of genres in each movie
genres = set(g for G in movies['genres'] for g in G)
for g in genres:
    movies[g] = movies.genres.transform(lambda x: int(g in x))
# create an array of just the genres of each movie
movie_genres = movies.drop(columns=['movieId', 'title','genres'])
cosine_sim = cosine_similarity(movie_genres, movie_genres)


def movie_finder(title):
    all_titles = movies['title'].tolist()
    closest_match = process.extractOne(title, all_titles)
    return closest_match[0]


def get_content_based_recommendations(title_string, n_recommendations=10):
    title = movie_finder(title_string)
    idx = movie_idx[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:(n_recommendations+1)]
    similar_movies = [i[0] for i in sim_scores]
    print(f"Because you watched {title}:")
    print(movies['title'].iloc[similar_movies])
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process

# Load data
movies = pd.read_csv('movies.csv')
tags = pd.read_csv('tags.csv')

# preprocess the genres
movies['genres'] = movies['genres'].str.split('|')

# one-hot encoding of genres
all_genres = set(g for genre_list in movies['genres'] for g in genre_list)
# print(f'Number of genres in our data: {len(all_genres)}')
for genre in all_genres:
    movies[genre] = movies['genres'].apply(lambda x: 1 if genre in x else 0)
genre_features = movies[list(all_genres)]

# combine tags left by each user for each movie
tags['tag'] = tags['tag'].str.lower()
tags_grouped_by_movie_id = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()

# merge tags data with movie data
movies_with_tags = movies.merge(tags_grouped_by_movie_id, on='movieId', how='left')
movies_with_tags['tag'] = movies_with_tags['tag'].fillna('')

# Vectorize tags using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tag_features_raw = tfidf_vectorizer.fit_transform(movies_with_tags['tag'])
tag_feature_names = tfidf_vectorizer.get_feature_names_out()
# print(f'Number of different tags in our data: {len(tag_feature_names)}')
tag_features = pd.DataFrame(tag_features_raw.toarray(), columns=tag_feature_names, index=movies.index)

# Combine genres and tag features
combined_features = pd.concat([pd.DataFrame(genre_features), pd.DataFrame(tag_features)], axis=1)

# Dimensionality reduction (Truncated SVD for sparse data)
svd = TruncatedSVD(n_components=60)
reduced_features = svd.fit_transform(combined_features)

# Cosine similarity
cosine_sim = cosine_similarity(reduced_features, reduced_features)

# a movie index
movie_idx = dict(zip(movies['title'], list(movies.index)))

# a utility function to query the data and find a movie
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

# Prompt the user for input
user_input = input("Enter the name of a movie you like: ")
recommendations = get_content_based_recommendations(user_input, 10)

import re
import pickle
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from fuzzywuzzy import process
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, hstack
from joblib import Parallel, delayed

# ~~~~~~~~~~~~~~~~~~~ #
# fuzzywuzzy search function
def movie_finder(title):
    all_titles = movies['title'].tolist()
    closest_match = process.extractOne(title, all_titles)
    return closest_match[0]

# retrieve similar movies
def get_content_based_recommendations(title_string, n_recommendations=10):
    title = movie_finder(title_string)
    idx = movie_titles_idx[title]
    movie_vec = sparse_features[idx]

    if isinstance(movie_vec, (np.ndarray)):
        movie_vec = movie_vec.reshape(1,-1)

    _, indices = knn_model.kneighbors(movie_vec, n_neighbors=n_recommendations + 1)
    similar_movies = indices.flatten()[1:]  # Exclude itself (index 0)
    return movies['title'].iloc[similar_movies]

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

# ~~~~~~~~~~~~~~~~~~~ #

# first: load dataset
movies = pd.read_csv('../data/movies.csv')
tags = pd.read_csv('../data/tags.csv')
ratings = pd.read_csv('../data/ratings.csv') # will be used in testing

# movie title => index
movie_titles_idx = dict(zip(movies['title'], list(movies.index)))
# movie index => title
movie_idx_titles = dict(zip(list(movies.index), movies['title'],))
# map movie IDs to movie titles
movie_titles = dict(zip(movies['movieId'], movies['title']))
# map movie titles to movie IDs
movie_ids = dict(zip(movies['title'], movies['movieId']))
# movie id to idx
movie_ids_idx = dict(zip(movies['movieId'], list(movies.index)))
# movie idx to id
movie_idx_ids = dict(zip(list(movies.index), movies['movieId']))

# second: extract the genres from the movies and one-hot encode genres
movies['genres'] = movies['genres'].str.split('|')

all_genres = set(genre for genre_list in movies['genres'] for genre in genre_list)

for genre in all_genres:
    movies[genre] = movies['genres'].apply(lambda x: 1 if genre in x else 0)

genre_features = movies[list(all_genres)] # one-hot encoded matrix
# Save one-hot encoded genre matrix to CSV
genre_features.to_csv("one_hot_encoded_genres.csv", index=False)

# third: process the tags
# remove stopwords
# stemming and lemmatization
stop_words = set(stopwords.words('english'))
custom_stopwords = {"movie", "film", "like", "scene", "story", "character", "good", "bad", "something",
                    "is", "are", "was", "were", "it", "its", "it's", "the", "a", "to", "of", "in", "on"}
stop_words.update(custom_stopwords)

# Initialize Lemmatizer
lemmatizer = WordNetLemmatizer()
# we aren't using a stemmer as well because stemming aggressively cuts words down to their root
# sometimes leading to confusing/incorrect mappings like "boxing" => "box" {boxing day, music box, boxing gloves}

# Preprocessing function for tags
def clean_tag(tag):
    tag = tag.lower()  # Lowercase
    tag = re.sub(r'[^a-zA-Z\s-]', '', tag)  # Remove special characters (except dashes)
    words = tag.split()  # Tokenize on white space
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Lemmatization & stopword removal
    return " ".join(words)  # Keep phrases (multiple words) after lemmatization and stemming

# applying lemmatization on tags
tags['tag'] = tags['tag'].apply(clean_tag)
tags_grouped_by_movie_id = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()

# custom dataframe applying lemmatization on movie titles
processed_titles_grouped_by_id = pd.DataFrame({
    'movieId': movies['movieId'],
    'processed_title': movies['title'].apply(clean_tag)
})

# Merge preprocessed tags with movies
movies_with_tags = movies.merge(tags_grouped_by_movie_id, on='movieId', how='left')

# Ensure 'tag' exists before merging
movies_with_tags['tag'] = movies_with_tags['tag'].fillna('')

# Merge the processed titles
movies_with_tags = movies_with_tags.merge(processed_titles_grouped_by_id, on='movieId', how='left')

# Ensure 'processed_title' exists and has no NaN values
movies_with_tags['processed_title'] = movies_with_tags['processed_title'].fillna('')

# Combine tags and processed titles for TF-IDF processing
movies_with_tags['combined_text'] = movies_with_tags['tag'] + ' ' + movies_with_tags['processed_title']

# TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(
    analyzer='word',
    ngram_range=(1,2),   # Min n-grams and Max n-grams
    # 1, 2 is sufficient to cover a decent range of title combos and tags left by users
    max_df=0.95,          # Ignore very frequent words
    dtype=np.float32,     # Reduce memory usage
    sublinear_tf=True     # Smooth term frequency scaling
    ) 
processed_features_raw = tfidf_vectorizer.fit_transform(movies_with_tags['combined_text'])
processed_features = pd.DataFrame(processed_features_raw.toarray(), index=movies.index)

print(processed_features.shape)

# Extract feature names (words/phrases from TF-IDF)
feature_names = tfidf_vectorizer.get_feature_names_out()
# Convert TF-IDF feature matrix to DataFrame
tfidf_df = pd.DataFrame(processed_features_raw.toarray(), columns=feature_names)
# Save the extracted TF-IDF features as CSV
# tfidf_df.to_csv("tfidf_features.csv", index=False)

# Sum occurrences of each feature across all movies
feature_counts = tfidf_df.sum(axis=0).sort_values(ascending=False)
# Convert to DataFrame for easier analysis
feature_counts_df = pd.DataFrame({'Feature': feature_counts.index, 'Count': feature_counts.values})
# Save feature occurrences to CSV
feature_counts_df.to_csv("tfidf_feature_counts.csv", index=False)

# combine features
# combined_features = pd.concat([genre_features, tag_features], axis=1)
sparse_combined_features = hstack([csr_matrix(genre_features), csr_matrix(processed_features)])
# cosine_sim = cosine_similarity(combined_features, combined_features)
sparse_features = csr_matrix(sparse_combined_features)
# Fit Nearest Neighbors model (faster than cosine_similarity on full matrix)
knn_model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20)
knn_model.fit(sparse_features)


# Precision@k for the top power users:
# Number of users to evaluate
num_users = 10
# Number of movies whose rating is >= 3.5 which we get to generate recommendations
num_movies_per_user = 10
# Number of recommendations we generate
k = 100
# Find the top users who have rated the most movies (descending from highest rating downwards)
user_rating_counts = ratings.groupby('userId').size().sort_values(ascending=False)
top_users = user_rating_counts.index[:num_users]  # Get top N users with most ratings

precision_values = []
recall_values = []

# Iterate through each top user
# for user_id in top_users:
#     # Get all movies rated by this user
#     user_ratings = ratings[ratings['userId'] == user_id]
#     top_movies = user_ratings[user_ratings['rating'] >= 3.5].head(num_movies_per_user)['movieId'].tolist()  # First n movies

#     # Get recommended movies
#     recommended_movies_ids = []

#     for movie_id in top_movies:
#         similar_movies = get_content_based_recommendations(movie_titles[movie_id], k)
#         for similar_movie in similar_movies:
#             recommended_movies_ids.append(movie_ids[similar_movie])  # Add recommendations to list

#     # Get relevant movies (rated 3.5 or above by the user)
#     relevant_movies_ids = user_ratings[user_ratings['rating'] >= 3.5]['movieId']

#     # Compute precision
#     prec = precision(recommended_movies_ids, relevant_movies_ids)
#     rec = recall(recommended_movies_ids, relevant_movies_ids)
#     precision_values.append(prec)
#     recall_values.append(rec)
# ratings_dict = ratings.groupby('userId').apply(lambda x: x.set_index('movieId')['rating'].to_dict()).to_dict()

def process_user(user_id):
    # Get all movies rated by this user
    user_ratings = ratings[ratings['userId'] == user_id]
    rated_movie_ids = user_ratings['movieId'].tolist()

    # Select positively rated movies as seeds
    top_movies = user_ratings[user_ratings['rating'] >= 3.5].head(num_movies_per_user)['movieId'].tolist()

    recommended_movies_ids = []
    for movie_id in top_movies:
        similar_movies = get_content_based_recommendations(movie_titles[movie_id], k)
        recommended_movies_ids.extend([movie_ids[similar_movie] for similar_movie in similar_movies])

    # Filter recommendations: only keep those that the user has rated
    rated_recommended_movies = [movie_id for movie_id in recommended_movies_ids if movie_id in rated_movie_ids]

    # Define relevant items: movies rated >= 3.5 by the user
    relevant_movies_ids = user_ratings[user_ratings['rating'] >= 3.5]['movieId'].tolist()

    return precision(rated_recommended_movies, relevant_movies_ids), recall(rated_recommended_movies, relevant_movies_ids)

# Run in parallel
results = Parallel(n_jobs=-1)(delayed(process_user)(user_id) for user_id in top_users)

# Unpack results
precision_values, recall_values = zip(*results)

mean_precision = np.mean(precision_values)
print(f"Mean average Precision@{k} of the top {num_users} power users is {mean_precision:.4f}")

mean_recall = np.mean(recall_values)
print(f"Mean average Recall@{k} of the top {num_users} power users is {mean_recall:.4f}")

f1_score = 2 * (mean_precision * mean_recall) / (mean_precision + mean_recall)
print(f"F1-Score {f1_score:.4f}")

# Save the trained model, feature matrix and mappings
model_data = {
    "knn": knn_model,
    "title_idx": movie_titles_idx,
    "idx_title": movie_idx_titles,
    "id_title": movie_titles,
    "title_id": movie_ids,
    "id_idx": movie_ids_idx,
    "idx_id": movie_idx_ids,
    "matrix": sparse_features
}

# Save to disk
with open("cbf_model.pkl", "wb") as f:
    pickle.dump(model_data, f)

print("Model saved successfully!")
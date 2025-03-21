import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from fuzzywuzzy import process
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# ~~~~~~~~~~~~~~~~~~~ #
# fuzzywuzzy search function
def movie_finder(title):
    all_titles = movies['title'].tolist()
    closest_match = process.extractOne(title, all_titles)
    return closest_match[0]

# retrieve similar movies
def get_content_based_recommendations(title_string, n_recommendations=10):
    title = movie_finder(title_string)
    idx = movie_idx[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:(n_recommendations+1)]
    similar_movies = [i[0] for i in sim_scores]
    return (movies['title'].iloc[similar_movies])

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
movies = pd.read_csv('movies.csv')
tags = pd.read_csv('tags.csv')
ratings = pd.read_csv('ratings.csv') # will be used in testing

# second: extract the genres from the movies and one-hot encode genres
movies['genres'] = movies['genres'].str.split('|')

all_genres = set(genre for genre_list in movies['genres'] for genre in genre_list)

for genre in all_genres:
    movies[genre] = movies['genres'].apply(lambda x: 1 if genre in x else 0)

genre_features = movies[list(all_genres)] # one-hot encoded matrix

# third: process the tags
# remove stopwords
# stemming and lemmatization
stop_words = set(stopwords.words('english'))
custom_stopwords = {"movie", "film", "like", "scene", "story", "character", "good", "bad", "something",
                    "is", "are", "was", "were", "it", "its", "it's", "the", "a", "to", "of", "in", "on"}
stop_words.update(custom_stopwords)

# Initialize Lemmatizer
lemmatizer = WordNetLemmatizer() # preserves meaning

# Preprocessing function for tags
def clean_tag(tag):
    tag = tag.lower()  # Lowercase
    tag = re.sub(r'[^a-zA-Z\s-]', '', tag)  # Remove special characters (except dashes)
    words = tag.split()  # Tokenize on white space
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Lemmatization & stopword removal
    return " ".join(words)  # Keep phrases (multiple words) after lemmatization and stemming

# print(tags[tags['movieId'] == 60756].head())
tags['tag'] = tags['tag'].apply(clean_tag)
# print(tags[tags['movieId'] == 60756].head())
tags_grouped_by_movie_id = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
# print(tags_grouped_by_movie_id[tags_grouped_by_movie_id['movieId'] == 60756].head()) # faces some issues with people's names like will ferrell will just tbe categorized as ferrell

# Merge preprocessed tags with movies
movies_with_tags = movies.merge(tags_grouped_by_movie_id, on='movieId', how='left')
movies_with_tags['tag'] = movies_with_tags['tag'].fillna('') # ensure shape and order of features data aligns with movies

# TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(3, 3))  # Min n-grams and Max n-grams
tag_features_raw = tfidf_vectorizer.fit_transform(movies_with_tags['tag'])
tag_features = pd.DataFrame(tag_features_raw.toarray(), index=movies.index)

# combine features
combined_features = pd.concat([genre_features, tag_features], axis=1)
cosine_sim = cosine_similarity(combined_features, combined_features)

# movie index => title:index in dataframe
movie_idx = dict(zip(movies['title'], list(movies.index)))


# Precision@k for the top power users:
# Number of users to evaluate
num_users = 10
# Number of movies whose rating is >= 3.5 which we get to generate recommendations
num_movies_per_user = 10
# Number of recommendations we generate
k = 20
# Find the top users who have rated the most movies (descending from highest rating downwards)
user_rating_counts = ratings.groupby('userId').size().sort_values(ascending=False)
top_users = user_rating_counts.index[:num_users]  # Get top N users with most ratings

precision_values = []
recall_values = []
# map movie titles to movie IDs
movie_titles = dict(zip(movies['movieId'], movies['title']))
# map movie IDs to movie titles
movie_ids_dict = dict(zip(movies['title'], movies['movieId']))

# Iterate through each top user
for user_id in top_users:
    # Get all movies rated by this user
    user_ratings = ratings[ratings['userId'] == user_id]
    top_movies = user_ratings[user_ratings['rating'] >= 3.5].head(num_movies_per_user)['movieId'].tolist()  # First n movies

    # Get recommended movies
    recommended_movies_ids = []

    for movie_id in top_movies:
        similar_movies = get_content_based_recommendations(movie_titles[movie_id], k)
        for similar_movie in similar_movies:
            recommended_movies_ids.append(movie_ids_dict[similar_movie])  # Add recommendations to list

    # Get relevant movies (rated 3.5 or above by the user)
    relevant_movies_ids = user_ratings[user_ratings['rating'] >= 3.5]['movieId']

    # Compute precision
    prec = precision(recommended_movies_ids, relevant_movies_ids)
    rec = recall(recommended_movies_ids, relevant_movies_ids)
    precision_values.append(prec)
    recall_values.append(rec)

mean_precision = np.mean(precision_values)
print(f"Mean average Precision@{k} of the top {num_users} power users is {mean_precision:.4f}")

mean_recall = np.mean(recall_values)
print(f"Mean average Recall@{k} of the top {num_users} power users is {mean_recall:.4f}")

f1_score = 2 * (mean_precision * mean_recall) / (mean_precision + mean_recall)
print(f"F1-Score {f1_score:.4f}")
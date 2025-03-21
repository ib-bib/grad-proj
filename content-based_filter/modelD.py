import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process

# ---------------- LOAD DATA ----------------
movies = pd.read_csv('movies.csv')
tags = pd.read_csv('tags.csv')
ratings = pd.read_csv('ratings.csv')

# ---------------- GENRES PROCESSING ----------------
movies['genres'] = movies['genres'].str.split('|')
all_genres = set(g for genre_list in movies['genres'] for g in genre_list)

for genre in all_genres:
    movies[genre] = movies['genres'].apply(lambda x: 1 if genre in x else 0)
genre_features = movies[list(all_genres)]

# ---------------- TAGS PROCESSING ----------------
# Define stopwords (Pre-installed, no downloads)
stop_words = set(stopwords.words('english'))
custom_stopwords = {"movie", "film", "like", "scene", "story", "character", "good", "bad", 
                    "is", "are", "was", "were", "it", "its", "it's", "the", "a", "to", "of", "in", "on"}
stop_words.update(custom_stopwords)

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# Preprocessing function for tags
def clean_tag(tag):
    tag = tag.lower()  # Lowercase
    tag = re.sub(r'[^a-zA-Z\s-]', '', tag)  # Remove special characters (except dashes)
    words = tag.split()  # Tokenize on white space
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Lemmatization & stopword removal
    words = [stemmer.stem(word) for word in words]  # Stemming
    return " ".join(words)  # Keep phrases (multiple words) after lemmatization and stemming

# Apply preprocessing to tags
tags['tag'] = tags['tag'].apply(clean_tag)
tags_grouped_by_movie_id = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()

# Merge preprocessed tags with movies
movies_with_tags = movies.merge(tags_grouped_by_movie_id, on='movieId', how='left')
movies_with_tags['tag'] = movies_with_tags['tag'].fillna('')

# ---------------- TF-IDF Vectorization ----------------
tfidf_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 3))  # Unigrams, bigrams and trigrams
tag_features_raw = tfidf_vectorizer.fit_transform(movies_with_tags['tag'])
tag_features = pd.DataFrame(tag_features_raw.toarray(), index=movies.index)

# ---------------- COMBINE FEATURES ----------------
combined_features = pd.concat([genre_features, tag_features], axis=1)
cosine_sim = cosine_similarity(combined_features, combined_features)

# ---------------- RECOMMENDATION SYSTEM ----------------
# Movie index mapping
movie_idx = dict(zip(movies['title'], list(movies.index)))

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
    return movies['movieId'].iloc[similar_movies].tolist()  # Return movieId list

# Precision
def precision(recommended_items, relevant_items):
    true_positives = len(set(recommended_items).intersection(set(relevant_items)))
    total_recommended_items = len(recommended_items)
    return true_positives / total_recommended_items if total_recommended_items > 0 else 0

# Recall
def recall(recommended_items, relevant_items):
    true_positives = len(set(recommended_items).intersection(set(relevant_items)))
    total_relevant_items = len(relevant_items)
    return true_positives / total_relevant_items if total_relevant_items > 0 else 0


# ---------------- EVALUATION ----------------
# Find top 10 power users (users with most reviews)
top_users = ratings['userId'].value_counts().head(10).index.tolist()
ratings_top_users = ratings[ratings['userId'].isin(top_users)]

# Prepare evaluation
precision_scores = []
recall_scores = []

for user in top_users:
    user_ratings = ratings_top_users[ratings_top_users['userId'] == user].sort_values(by='timestamp').head(10)
    
    for movie_id in user_ratings['movieId']:
        recommended_movie_ids = get_content_based_recommendations(movies[movies['movieId'] == movie_id]['title'].values[0])
        relevant_movies = ratings[(ratings['userId'] == user) & (ratings['movieId'].isin(recommended_movie_ids)) & (ratings['rating'] >= 3.5)]['movieId'].tolist()
        
        precision_scores.append(precision(recommended_movie_ids, relevant_movies))
        recall_scores.append(recall(recommended_movie_ids, relevant_movies))

# Compute Mean Precision & Recall
mean_precision = np.mean(precision_scores)
mean_recall = np.mean(recall_scores)
f1_score = 2 * (mean_precision * mean_recall) / (mean_precision + mean_recall)

print(f"Mean Precision: {mean_precision:.4f}")
print(f"Mean Recall: {mean_recall:.4f}")
print(f"F1-Score: {f1_score:.4f}")
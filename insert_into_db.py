import csv
import os
import re
import psycopg2
import dotenv
from psycopg2.extras import execute_values


dotenv.load_dotenv()

# === CONFIG ===
MOVIES_CSV = "data/movies.csv"
STATS_CSV = "collaborative_filter/movie_stats.csv"
IMAGE_DIR = "images"
DB_CONFIG = {
    "host": os.getenv("PGHOST"),
    "dbname": os.getenv("PGDATABASE"),
    "user": os.getenv("PGUSER"),
    "password": os.getenv("PGPASSWORD"),
    "port": 5432,
    "sslmode": "require"
}

# === Connect to Neon Postgres ===
conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

# === Helper: Parse release year from title ===
def extract_year(title):
    match = re.search(r"\((\d{4})\)", title)
    return int(match.group(1)) if match else None

# === Step 1: Load ratings from movie_stats.csv ===
print("Loading ratings...")
ratings_lookup = {}
with open(STATS_CSV, newline='', encoding='utf-8') as stats_file:
    reader = csv.DictReader(stats_file)
    for row in reader:
        movie_id = int(row["movieId"])
        mean = float(row["mean"])
        bayesian = float(row["bayesian_avg"])
        ratings_lookup[movie_id] = (mean, bayesian)

# === Step 2: Process movies.csv and prepare data ===
print("Processing movies...")
unique_genres = set()
movies = []

with open(MOVIES_CSV, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        movie_id = int(row["movieId"])

        # Skip movies that don't have stats
        if movie_id not in ratings_lookup:
            continue

        title_raw = row["title"]
        genres = row["genres"].split("|") if row["genres"] != "(no genres listed)" else []
        release_year = extract_year(title_raw)
        title_clean = re.sub(r"\(\d{4}\)", "", title_raw).strip()

        image_filename = f"{movie_id}__{title_clean} {release_year}.jpg"
        image_path = os.path.join(IMAGE_DIR, image_filename)
        image = image_filename if os.path.exists(image_path) else None

        mean_rating, bayesian_rating = ratings_lookup[movie_id]
        # Round mean rating to 2 decimal places
        mean_rating = round(mean_rating, 2)
        bayesian_rating = round(bayesian_rating, 2)

        unique_genres.update(genres)

        movies.append({
            "movieId": movie_id,
            "title": title_clean,
            "releaseYear": release_year,
            "image": image,
            "meanRating": mean_rating,
            "bayesianRating": bayesian_rating
        })

# === Step 3: Insert genres ===
print("Inserting genres...")
genre_list = list(unique_genres)
genre_values = [(g,) for g in genre_list]
cur.executemany(
    'INSERT INTO "movie-rec_genre" (name) VALUES (%s) ON CONFLICT (name) DO NOTHING',
    genre_values
)

# === Step 4: Insert movies with ratings ===
print("Inserting movies with ratings...")
movie_values = [
    (
        m["movieId"],
        m["title"],
        m["image"],
        m["releaseYear"],
        m["meanRating"],
        m["bayesianRating"]
    ) for m in movies
]

insert_query = """
    INSERT INTO "movie-rec_movie" ("movieId", "title", "image", "releaseYear", "meanRating", "bayesianRating")
    VALUES %s
    ON CONFLICT ("movieId") DO NOTHING
"""
execute_values(cur, insert_query, movie_values)

# === Done ===
conn.commit()
cur.close()
conn.close()
print("✅ Movie + Genre insertion complete.")
import csv
import psycopg2
import dotenv
import os

dotenv.load_dotenv()

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

# Create a cursor to execute SQL queries
cur = conn.cursor()

# Open movies.csv and read data
with open('data/movies.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)

    for row in reader:
        movie_id = row['movieId']
        title = row['title']
        genres = row['genres'].split('|')

        # Check if the movieId exists in the movie table
        cur.execute('SELECT "movieId" FROM "movie-rec_movie" WHERE "movieId" = %s', (movie_id,))
        movie_result = cur.fetchone()

        # If the movieId doesn't exist, skip this movie
        if not movie_result:
            print(f"Movie {movie_id} ({title}) not found in 'movie-rec_movie'. Skipping...")
            continue  # Skip to the next movie

        for genre in genres:
            # Check if the genre exists in the genres table
            cur.execute('SELECT id FROM "movie-rec_genre" WHERE name = %s', (genre,))
            genre_result = cur.fetchone()

            if genre_result:
                genre_id = genre_result[0]
            else:
                # If the genre doesn't exist, insert it into the genres table
                cur.execute('INSERT INTO "movie-rec_genre" ("name") VALUES (%s) RETURNING id', (genre,))
                genre_id = cur.fetchone()[0]  # Get the id of the newly inserted genre

            # Insert the movie-genre relationship into the movie_genre table, avoiding duplicates
            cur.execute(
                '''
                INSERT INTO "movie-rec_movie_genre" ("movieId", "genreId")
                VALUES (%s, %s)
                ON CONFLICT ("movieId", "genreId") DO NOTHING
                ''', 
                (movie_id, genre_id)
            )

        # Commit after processing each movie
        conn.commit()

# Close the cursor and the connection
cur.close()
conn.close()
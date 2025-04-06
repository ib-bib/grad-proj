import csv
import os
import requests
import re
from dotenv import load_dotenv

# Load TMDB API key from .env
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

# File paths
CSV_FILE = "missing_posters3.csv"
OUTPUT_DIR = "tmdb_posters"
TRACKER_FILE = "last_tmdb_processed2.txt"

# TMDB API base URLs
SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

# Make sure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def sanitize_filename(filename):
    """Remove or replace invalid characters for filenames."""
    return re.sub(r'[\\/*?:"<>|]', "", filename)

def get_last_processed_id():
    """Get last processed movieId from file."""
    if os.path.exists(TRACKER_FILE):
        with open(TRACKER_FILE, "r") as f:
            return int(f.read().strip())
    return -1  # If file doesn't exist, start from beginning

def update_last_processed_id(movie_id):
    """Update the tracker file with the latest movieId."""
    with open(TRACKER_FILE, "w") as f:
        f.write(str(movie_id))

def fetch_poster(movie_id, title):
    """Query TMDB and download the poster image for the given movie."""
    params = {
        "api_key": TMDB_API_KEY,
        "query": title,
    }
    response = requests.get(SEARCH_URL, params=params)
    data = response.json()

    if data.get("results"):
        poster_path = data["results"][0].get("poster_path")
        if poster_path:
            image_url = f"{IMAGE_BASE_URL}{poster_path}"
            safe_title = sanitize_filename(f"{title}")
            filename = f"{movie_id}__{safe_title}.jpg"
            filepath = os.path.join(OUTPUT_DIR, filename)

            # Download and save image
            img_data = requests.get(image_url).content
            with open(filepath, "wb") as f:
                f.write(img_data)
            print(f"✅ Downloaded: {filename}")
            return True
        else:
            print(f"⚠️ No poster found for {title}")
    else:
        print(f"❌ No TMDB result for: {title}")
    return False

def download_missing_posters():
    """Read CSV and download posters starting from the last processed movieId."""
    last_id = get_last_processed_id()

    with open(CSV_FILE, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            movie_id = int(row["movieId"])
            if movie_id <= last_id:
                continue  # Skip already processed
            title = row["title"]
            if fetch_poster(movie_id, title):
                update_last_processed_id(movie_id)


download_missing_posters()

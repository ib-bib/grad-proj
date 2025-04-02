import os
import csv
import requests
from PIL import Image
from io import BytesIO

# OMDB API Settings
OMDB_API_KEY = os.getenv("OMDB_API_KEY")  # Use your actual API key
OMDB_URL = "http://www.omdbapi.com/"
BATCH_SIZE = 1000  # Number of posters to fetch per day

# Paths
CSV_FILE = "data/movies.csv"
OUTPUT_DIR = "images"
CHECKPOINT_FILE = "last_processed.txt"

# Create images directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Read last processed line
def get_last_processed():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return int(f.read().strip())
    return 0  # Start from the first movie

# Save last processed line
def save_last_processed(index):
    with open(CHECKPOINT_FILE, "w") as f:
        f.write(str(index))

# Fix movie titles like "Conjuring, The" → "The Conjuring"
def fix_title(title):
    if ", The" in title:
        return "The " + title.replace(", The", "")
    if ", A" in title:
        return "A " + title.replace(", A", "")
    if ", An" in title:
        return "An " + title.replace(", An", "")
    return title  # No changes needed

# Read CSV and process movies
start_index = get_last_processed()
with open(CSV_FILE, newline="", encoding="utf-8") as csvfile:
    reader = list(csv.DictReader(csvfile))  # Convert to list to allow indexing
    total_movies = len(reader)

    for i in range(start_index, min(start_index + BATCH_SIZE, total_movies)):
        row = reader[i]
        movie_id = row['movieId']
        title_plus_year = row["title"]
        
        # Fix titles that are written as "Conjuring, The"
        title = fix_title(title_plus_year.split(" (")[0])  # Remove year from title

        # Fetch poster from OMDB API
        params = {"t": title, "apikey": OMDB_API_KEY}
        response = requests.get(OMDB_URL, params=params).json()

        if "Poster" in response and response["Poster"] != "N/A":
            image_url = response["Poster"]
            print(f"📥 Downloading: {title} -> {image_url}")

            # Download the image
            img_response = requests.get(image_url, stream=True)

            # Verify if the response is an actual image
            content_type = img_response.headers.get("Content-Type", "")
            if "image" not in content_type:
                print(f"⚠️ Skipping {title}: URL did not return an image.")
                continue

            try:
                img = Image.open(BytesIO(img_response.content))

                # Save image as JPG
                clean_title = "".join(c for c in title_plus_year if c.isalnum() or c in " _-").strip()
                img_path = os.path.join(OUTPUT_DIR, f"{movie_id}__{clean_title}.jpg")
                img.convert("RGB").save(img_path, "JPEG")
                print(f"✅ Saved: {img_path}")

            except Image.UnidentifiedImageError:
                print(f"❌ Failed to download a valid image for {title} (corrupted file)")

        else:
            print(f"⚠️ No image found for {title}.")

        # Save progress after each movie
        save_last_processed(i + 1)

print("✅ Daily batch complete. Resume tomorrow!")

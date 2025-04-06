import csv
import os
import requests
import re
import dotenv

dotenv.load_dotenv()

# === CONFIG ===
OMDB_API_KEY = os.getenv("OMDB_API_KEY")
CSV_FILE = "missing_posters3.csv"
OUTPUT_DIR = "images"

# === Ensure output dir exists ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Sanitize filenames ===
def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "", name)

# === Download poster from OMDb ===
def fetch_and_save_poster(movie_id, title):
    params = {
        "t": title,
        "apikey": OMDB_API_KEY
    }
    response = requests.get("http://www.omdbapi.com/", params=params)
    data = response.json()

    if data.get("Response") == "True":
        year = data.get("Year", "Unknown")
        poster_url = data.get("Poster", "")

        if poster_url and poster_url != "N/A":
            safe_title = sanitize_filename(f"{title} {year}")
            filename = f"{movie_id}__{safe_title}.jpg"
            filepath = os.path.join(OUTPUT_DIR, filename)

            try:
                img_data = requests.get(poster_url).content
                with open(filepath, "wb") as f:
                    f.write(img_data)
                print(f"✅ Saved: {filename}")
            except Exception as e:
                print(f"❌ Failed to download image for {title}: {e}")
        else:
            print(f"⚠️ No poster found for: {title}")
    else:
        print(f"❌ OMDb failed for: {title} | Error: {data.get('Error')}")

# === Process CSV ===
with open(CSV_FILE, newline='', encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        movie_id = row["movieId"]
        title = row["title"]
        fetch_and_save_poster(movie_id, title)

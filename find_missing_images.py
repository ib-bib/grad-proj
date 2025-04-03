import os
import csv
import re

# Define file paths
movies_csv_path = "data/movies.csv"
images_dir = "images/"
missing_posters_csv_path = "missing_posters.csv"

def parse_movie_title(movie_title):
    """Extracts the title and release year from the movie title field."""
    match = re.match(r"(.+?) \((\d{4})\)$", movie_title.strip())
    if match:
        title, release_year = match.groups()
    else:
        title, release_year = movie_title.strip(), "Unknown"
    
    # Handle cases where title is in "Title, The" format
    if ", The" in title:
        title = title.replace(", The", "")
        title = "The " + title.strip()
    
    return title, release_year

def find_missing_posters():
    """Checks which movies don't have corresponding images and writes them to a CSV file."""
    missing_movies = []

    # Get list of existing poster filenames (without extensions)
    existing_posters = {filename.split("__")[0] for filename in os.listdir(images_dir) if "__" in filename}

    # Read movies.csv
    with open(movies_csv_path, mode="r", encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row

        for row in reader:
            movie_id, raw_title = row[0], row[1]
            title, release_year = parse_movie_title(raw_title)

            # Check if the movie ID is missing in the images directory
            if movie_id not in existing_posters:
                missing_movies.append([movie_id, title, release_year])

    # Write missing posters to CSV
    with open(missing_posters_csv_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["movieId", "title", "release_year"])  # Write header
        writer.writerows(missing_movies)

    print(f"Missing posters list saved to {missing_posters_csv_path}")


find_missing_posters()

from flask import Flask
from fuzzywuzzy import process
from markupsafe import escape
import pandas as pd

app = Flask(__name__)

movies = pd.read_csv('../data/movies.csv')
all_titles = movies['title'].tolist()

@app.route("/")
def hello_world():
    return {"hello": "world"}

@app.route("/search/<movie>")
def search(movie):
    title = escape(movie)
    closest_match = process.extractOne(title, all_titles)
    return {"result": closest_match[0]}
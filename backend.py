
import os
import pickle
import pandas as pd
import numpy as np
import requests

from fastapi import FastAPI, HTTPException, Query
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Movie Recommendation API")

# Load models once when the server starts
with open("model/movies_dict.pkl", "rb") as file:
    movies = pd.DataFrame(pickle.load(file))

with open("model/similarity.pkl", "rb") as file:
    similarity = np.asarray(pickle.load(file))

TMDB_API_KEY = os.getenv("TMDB_API_KEY")


def fetch_poster(movie_id):
    if not TMDB_API_KEY:
        return None

    url = f"https://api.themoviedb.org/3/movie/{movie_id}"

    try:
        response = requests.get(
            url,
            params={"api_key": TMDB_API_KEY, "language": "en-US"},
            timeout=5
        )
        response.raise_for_status()

        poster_path = response.json().get("poster_path")

        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"

    except requests.RequestException:
        pass

    return None


def recommend(movie_title, top_n=5):
    matches = movies.index[
        movies["title"].str.casefold() == movie_title.casefold()
    ]

    if len(matches) == 0:
        raise HTTPException(
            status_code=404,
            detail=f"Movie '{movie_title}' not found"
        )

    index = int(matches[0])

    if similarity.shape[0] != len(movies):
        raise HTTPException(
            status_code=500,
            detail="Movie data and similarity matrix do not match"
        )

    distances = sorted(
        enumerate(similarity[index]),
        key=lambda item: item[1],
        reverse=True
    )

    results = []

    for movie_index, score in distances:
        if movie_index == index:
            continue

        movie = movies.iloc[movie_index]

        results.append({
            "movie_id": int(movie["movie_id"]),
            "title": movie["title"],
            "similarity_score": round(float(score), 4),
            "poster_url": fetch_poster(movie["movie_id"])
        })

        if len(results) == top_n:
            break

    return results


@app.get("/")
def home():
    return {
        "message": "Movie Recommendation API is running",
        "docs": "/docs"
    }


@app.get("/movies")
def get_movies():
    return {
        "movies": movies["title"].tolist()
    }


@app.get("/recommend")
def get_recommendations(
    movie: str = Query(..., min_length=1),
    top_n: int = Query(default=5, ge=1, le=20)
):
    recommendations = recommend(movie, top_n)

    return {
        "selected_movie": movie,
        "recommendations": recommendations
    }

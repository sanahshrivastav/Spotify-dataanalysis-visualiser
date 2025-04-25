import os
import pandas as pd
import requests
from nltk.sentiment import SentimentIntensityAnalyzer
from bs4 import BeautifulSoup
import re

GENIUS_API_TOKEN = os.getenv("GENIUS_API_TOKEN")  # Optional: Store this as environment variable

# Caching lyrics locally to avoid repeated requests
lyrics_cache = {}

def load_data(user_path):
    """Load and combine all streaming history JSON files for a user."""
    json_files = [os.path.join(user_path, file) for file in os.listdir(user_path) if file.endswith('.json')]
    data = pd.concat([pd.read_json(file) for file in json_files], ignore_index=True)
    return data

def preprocess_data(data, user_label):
    """Preprocess data by adding user labels and extracting useful features."""
    data['endTime'] = pd.to_datetime(data['endTime'])
    data['hour'] = data['endTime'].dt.hour
    data['user'] = user_label
    return data

def save_data(data, output_path):
    """Save processed data to a CSV file."""
    data.to_csv(output_path, index=False)

def get_lyrics(title, artist):
    """Fetch song lyrics using Genius API or scraping fallback."""
    cache_key = f"{artist}_{title}"
    if cache_key in lyrics_cache:
        return lyrics_cache[cache_key]

    query = f"{title} {artist}"
    headers = {"Authorization": f"Bearer {GENIUS_API_TOKEN}"} if GENIUS_API_TOKEN else {}
    search_url = f"https://genius.com/api/search/multi?per_page=1&q={query}"
    response = requests.get(search_url, headers=headers)

    try:
        path = response.json()['response']['sections'][0]['hits'][0]['result']['path']
        song_url = f"https://genius.com{path}"
        page = requests.get(song_url)
        soup = BeautifulSoup(page.text, "html.parser")
        lyrics_div = soup.find("div", class_="lyrics") or soup.find_all("div", class_=re.compile("^Lyrics__Container"))

        if isinstance(lyrics_div, list):
            lyrics = "\n".join([div.get_text(separator="\n") for div in lyrics_div])
        else:
            lyrics = lyrics_div.get_text(separator="\n") if lyrics_div else None

        if lyrics:
            lyrics_cache[cache_key] = lyrics.strip()
            return lyrics.strip()
        return None
    except Exception:
        return None

def analyze_lyrics_sentiment(lyrics):
    """Analyze sentiment of lyrics using NLTK's VADER."""
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(lyrics)

# src/utils.py

import os
import pandas as pd
import requests
from nltk.sentiment import SentimentIntensityAnalyzer
from bs4 import BeautifulSoup
import re
import joblib

# Load your trained ML model and vectorizer
ml_model = joblib.load("../models/sentiment_model.pkl")
vectorizer = joblib.load("../models/vectorizer.pkl")

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
    """Fetch song lyrics using Genius scraping."""
    base_url = "https://genius.com"
    search_url = f"https://genius.com/api/search/multi?per_page=1&q={title} {artist}"
    response = requests.get(search_url)

    try:
        path = response.json()['response']['sections'][0]['hits'][0]['result']['path']
        song_url = base_url + path
        page = requests.get(song_url)
        soup = BeautifulSoup(page.text, "html.parser")
        lyrics_div = soup.find("div", class_="lyrics") or soup.find_all("div", class_=re.compile("^Lyrics__Container"))

        if isinstance(lyrics_div, list):
            lyrics = "\n".join([div.get_text(separator="\n") for div in lyrics_div])
        else:
            lyrics = lyrics_div.get_text(separator="\n") if lyrics_div else None

        return lyrics.strip() if lyrics else None
    except Exception:
        return None

def analyze_lyrics_sentiment_vader(lyrics):
    """Analyze sentiment of lyrics using NLTK's VADER."""
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(lyrics)
    return sentiment

def predict_lyrics_sentiment_ml(lyrics):
    """Predict sentiment using trained ML model."""
    lyrics_vector = vectorizer.transform([lyrics])
    prediction = ml_model.predict(lyrics_vector)
    return prediction[0]

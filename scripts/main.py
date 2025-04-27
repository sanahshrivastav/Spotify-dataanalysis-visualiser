# scripts/main.py

import os
import pandas as pd
from utils import load_model, preprocess_text
import matplotlib.pyplot as plt

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # One level up
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

def combine_all_user_data():
    data_frames = []
    for user in ['alvin', 'kabir', 'sanah']:
        user_dir = os.path.join(DATA_DIR, user)
        for file in os.listdir(user_dir):
            file_path = os.path.join(user_dir, file)
            df = pd.read_json(file_path)
            df['user'] = user  # track which user
            data_frames.append(df)
    combined_df = pd.concat(data_frames, ignore_index=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    combined_df.to_csv(os.path.join(OUTPUT_DIR, "combined_streaming_data.csv"), index=False)
    print("Combined data saved!")
    return combined_df

def analyze_sentiment(df):
    vectorizer = load_model(os.path.join(MODEL_DIR, "vectorizer.pkl"))
    model = load_model(os.path.join(MODEL_DIR, "sentiment_model.pkl"))
    df['text_cleaned'] = df['trackName'].apply(preprocess_text)
    X = vectorizer.transform(df['text_cleaned'])
    df['sentiment'] = model.predict(X)
    return df

def plot_sentiment_distribution(df):
    sentiment_counts = df['sentiment'].value_counts()
    sentiment_counts.plot(kind='bar', color=['skyblue', 'salmon'])
    plt.title('Sentiment Distribution in Spotify Songs')
    plt.xlabel('Sentiment')
    plt.ylabel('Number of Songs')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "sentiment_distribution.png"))
    plt.close()
    print("Sentiment distribution plot saved!")

def main():
    combined_df = combine_all_user_data()
    df_with_sentiment = analyze_sentiment(combined_df)
    plot_sentiment_distribution(df_with_sentiment)

if __name__ == "__main__":
    main()

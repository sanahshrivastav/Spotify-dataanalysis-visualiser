# src/main.py

import pandas as pd
import matplotlib.pyplot as plt
from utils import load_data, preprocess_data, save_data, get_lyrics, analyze_lyrics_sentiment_vader, predict_lyrics_sentiment_ml

# Paths to user data folders
data_paths = {
    "sanah": "../data/sanah",
    "kabir": "../data/kabir",
    "alvin": "../data/alvin"
}

# Load and preprocess data for all users
all_data = []
for user, path in data_paths.items():
    print(f"Loading data for {user}...")
    raw_data = load_data(path)
    preprocessed_data = preprocess_data(raw_data, user)

    # Fetch lyrics and perform sentiment analysis
    lyrics_list = []
    vader_sentiment_list = []
    ml_sentiment_list = []

    for idx, row in preprocessed_data.iterrows():
        title = row['trackName']
        artist = row['artistName']
        lyrics = get_lyrics(title, artist)
        if lyrics:
            lyrics_list.append(lyrics)
            vader_sentiment = analyze_lyrics_sentiment_vader(lyrics)
            vader_sentiment_list.append(vader_sentiment['compound'])
            ml_sentiment = predict_lyrics_sentiment_ml(lyrics)
            ml_sentiment_list.append(ml_sentiment)
        else:
            lyrics_list.append(None)
            vader_sentiment_list.append(None)
            ml_sentiment_list.append(None)

    preprocessed_data['lyrics'] = lyrics_list
    preprocessed_data['vader_sentiment'] = vader_sentiment_list
    preprocessed_data['ml_sentiment'] = ml_sentiment_list

    all_data.append(preprocessed_data)

# Combine all user data into a single DataFrame
combined_data = pd.concat(all_data, ignore_index=True)

# Save combined data to CSV
save_data(combined_data, "../output/combined_streaming_data_with_sentiment.csv")

# Plot listening habits by hour
listening_by_hour = combined_data.groupby(['hour', 'user']).size().unstack()
listening_by_hour.plot(kind='line', figsize=(10, 6))
plt.title('Listening Habits by Hour')
plt.xlabel('Hour of Day')
plt.ylabel('Tracks Played')
plt.legend(title='User')
plt.grid()
plt.savefig('../output/listening_habits_by_hour.png')
plt.show()

# Analyze top artists
top_artists = combined_data.groupby(['artistName', 'user']).size().unstack(fill_value=0)
top_artists['Total'] = top_artists.sum(axis=1)
top_artists = top_artists.sort_values(by='Total', ascending=False).head(10)
top_artists.drop(columns='Total').plot(kind='bar', figsize=(10, 6))
plt.title('Top 10 Artists Comparison')
plt.xlabel('Artist')
plt.ylabel('Play Count')
plt.legend(title='User')
plt.grid()
plt.savefig('../output/top_artists_comparison.png')
plt.show()

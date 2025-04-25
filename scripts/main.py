# --- main.py ---
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_data, preprocess_data, save_data, get_lyrics, analyze_lyrics_sentiment
import nltk
nltk.download('vader_lexicon')


# Set base directory relative to this script
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(BASE_DIR, "data")
output_dir = os.path.join(BASE_DIR, "output")

# Paths to user data folders
data_paths = {
    "sanah": os.path.join(data_dir, "sanah"),
    "kabir": os.path.join(data_dir, "kabir"),
    "alvin": os.path.join(data_dir, "alvin")
}

# Load and preprocess data for all users
all_data = []
for user, path in data_paths.items():
    print(f"Loading data for {user}...")
    raw_data = load_data(path)
    preprocessed_data = preprocess_data(raw_data, user)
    all_data.append(preprocessed_data)

# Combine all user data into a single DataFrame
combined_data = pd.concat(all_data, ignore_index=True)
save_data(combined_data, os.path.join(output_dir, "combined_streaming_data.csv"))

# Analyze listening habits by hour
listening_by_hour = combined_data.groupby(['hour', 'user']).size().unstack()
listening_by_hour.plot(kind='line', figsize=(10, 6))
plt.title('Listening Habits by Hour')
plt.xlabel('Hour of Day')
plt.ylabel('Tracks Played')
plt.legend(title='User')
plt.grid()
plt.savefig(os.path.join(output_dir, "listening_habits_by_hour.png"))
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
plt.savefig(os.path.join(output_dir, "top_artists_comparison.png"))
plt.show()

# ----- SENTIMENT ANALYSIS BLOCK -----
print("Running sentiment analysis based on song lyrics...")
sentiments = []
for idx, row in combined_data.iterrows():
    title = row.get('trackName')
    artist = row.get('artistName')
    lyrics = get_lyrics(title, artist)
    if lyrics:
        sentiment = analyze_lyrics_sentiment(lyrics)
        sentiments.append(sentiment['compound'])
    else:
        sentiments.append(None)

combined_data['sentiment_score'] = sentiments
combined_data.to_csv(os.path.join(output_dir, "combined_streaming_with_sentiment.csv"), index=False)

# Plot sentiment over time
combined_data['endTime'] = pd.to_datetime(combined_data['endTime'])
combined_data = combined_data.dropna(subset=['sentiment_score'])
plt.figure(figsize=(12, 6))
sns.lineplot(x='endTime', y='sentiment_score', hue='user', data=combined_data)
plt.title("Sentiment Score Over Time per User")
plt.xlabel("Time")
plt.ylabel("Sentiment Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "sentiment_over_time.png"))
plt.show()
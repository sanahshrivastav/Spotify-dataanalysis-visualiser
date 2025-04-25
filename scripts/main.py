import pandas as pd
import matplotlib.pyplot as plt
from utils import load_data, preprocess_data, save_data

# Paths to user data folders
data_paths = {
    "sanah": "../data/sanah",
    "kabir": "../data/kabir",
    "alvin": "../data/alvin"
}

# Load and preprocess data for both users
all_data = []
for user, path in data_paths.items():
    print(f"Loading data for {user}...")
    raw_data = load_data(path)
    preprocessed_data = preprocess_data(raw_data, user)
    all_data.append(preprocessed_data)

# Combine all user data into a single DataFrame
combined_data = pd.concat(all_data, ignore_index=True)

# Save combined data to CSV
save_data(combined_data, "../output/combined_streaming_data.csv")

# Analyze listening habits by hour
listening_by_hour = combined_data.groupby(['hour', 'user']).size().unstack()

# Plot listening habits by hour
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

# Plot top artists comparison
top_artists.drop(columns='Total').plot(kind='bar', figsize=(10, 6))
plt.title('Top 10 Artists Comparison')
plt.xlabel('Artist')
plt.ylabel('Play Count')
plt.legend(title='User')
plt.grid()
plt.savefig('../output/top_artists_comparison.png')
plt.show()

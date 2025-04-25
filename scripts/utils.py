import os
import pandas as pd

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

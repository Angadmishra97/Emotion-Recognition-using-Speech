import os
import pandas as pd
import numpy as np
from feature_extraction import extract_features  # Ensure this module exists

# Correct file paths
dataset_dir = r"d:\SpeechEmotion REcognition\Actor_24"  # Use raw string (r"") or replace \ with /
labels_file = r"d:\SpeechEmotion REcognition\labels.csv"  # Update with actual labels CSV file path

# Load CSV file
try:
    data = pd.read_csv(labels_file)
except FileNotFoundError:
    print(f"Error: Labels file not found at {labels_file}")
    exit()

X, y = [], []

# Extract features from each audio file
for index, row in data.iterrows():
    file_path = os.path.join(dataset_dir, row['filename'])

    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} not found. Skipping...")
        continue

    try:
        features = extract_features(file_path)
        X.append(features)
        y.append(row['label'])
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Convert lists to NumPy arrays
X = np.array(X)
y = np.array(y)

# Save processed dataset
np.savez("processed_dataset.npz", X=X, y=y)
print("Dataset prepared and saved successfully!")

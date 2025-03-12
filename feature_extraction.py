import librosa
import numpy as np

def extract_features(file_path):
    signal, sr = librosa.load(file_path, sr=22050)
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

# Test feature extraction
features = extract_features(r"d:\SpeechEmotion REcognition\Actor_24\03-01-01-01-01-01-24.wav")  # Replace with a valid audio file path
print("Extracted Features:", features)

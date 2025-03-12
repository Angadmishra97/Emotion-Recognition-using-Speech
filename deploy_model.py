from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import librosa
import os
from feature_extraction import extract_features  # Ensure this module exists

app = Flask(__name__)

# Load the trained model safely
model_path = "emotion_recognition_model.h5"

if not os.path.exists(model_path):
    print(f"⚠️ Error: Model file '{model_path}' not found! Ensure you have trained the model.")
    model = None  # Avoid breaking the app
else:
    model = load_model(model_path)
    print("✅ Model loaded successfully!")

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({"error": "Model not loaded. Train and save 'emotion_recognition_model.h5' first."}), 500

    # Ensure file is included in request
    if 'file' not in request.files:
        return jsonify({"error": "No file provided. Upload an audio file."}), 400

    file = request.files['file']
    file_path = "uploaded_audio.wav"
    file.save(file_path)

    try:
        # Extract features safely
        features = extract_features(file_path)
        if features is None or features.size == 0:
            return jsonify({"error": "Feature extraction failed. Ensure the audio file is valid."}), 500
        
        features = features.reshape(1, -1)  # Ensure correct shape
        prediction = model.predict(features)
        predicted_label = np.argmax(prediction, axis=1)[0]

        # Replace with actual emotion labels used in training
        emotions = ['Happy', 'Sad', 'Angry', 'Neutral']
        if predicted_label >= len(emotions):
            return jsonify({"error": "Invalid prediction."}), 500

        return jsonify({"emotion": emotions[predicted_label]})

    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=2000, debug=False)  # Debug mode off for production
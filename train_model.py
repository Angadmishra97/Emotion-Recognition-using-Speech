import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Check if dataset exists
dataset_path = "processed_dataset.npz"
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Error: Dataset file '{dataset_path}' not found!")

# Load dataset
data = np.load(dataset_path)
X, y = data['X'], data['y']

# Check if dataset is empty
if X.size == 0 or y.size == 0:
    raise ValueError("Error: Dataset is empty. Re-run dataset_preparation.py.")

# Convert labels to one-hot encoding
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(y)

# Updated for compatibility with sklearn 1.2+
onehot_encoder = OneHotEncoder(sparse_output=False)  
y = onehot_encoder.fit_transform(integer_encoded.reshape(-1, 1))

# Build model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(y.shape[1], activation='softmax')
])

# Compile and train model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2)

# Save model
model.save("emotion_recognition_model.h5")
print("✅ Model trained and saved successfully!")

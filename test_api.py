import requests
import os

# Set API URL and test file path
url = "http://127.0.0.1:2000/predict"
file_path = r"d:\SpeechEmotion REcognition\Actor_24\03-01-01-01-01-01-24.wav"  # Update with your actual test audio file path

# Check if file exists
if not os.path.exists(file_path):
    print(f"❌ Error: Audio file '{file_path}' not found! Please provide a valid file.")
    exit()

try:
    # Send the request with the audio file
    with open(file_path, 'rb') as f:
        response = requests.post(url, files={'file': f})

    # Print the response
    if response.status_code == 200:
        print("✅ Success! Response:", response.json())
    else:
        print(f"❌ API Error {response.status_code}: {response.text}")

except requests.exceptions.ConnectionError:
    print("❌ Error: Unable to connect to the API. Ensure Flask is running using:")
    print("   python deploy_model.py")
except Exception as e:
    print(f"❌ Unexpected Error: {e}")

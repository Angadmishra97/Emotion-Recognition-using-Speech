import librosa
import librosa.display
import matplotlib.pyplot as plt

def visualize_audio(file_path):
    try:
        # ✅ Load audio file without 'backend' argument
        signal, sr = librosa.load(file_path, sr=22050, res_type='kaiser_best')

        # ✅ Plot waveform
        plt.figure(figsize=(10, 4))
        librosa.display.waveshow(signal, sr=sr)
        plt.title("Waveform")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.show()

    except Exception as e:
        print(f"Error: {e}")

# ✅ Use a raw string to avoid path errors
file_path = r"d:\SpeechEmotion REcognition\Actor_24\03-01-01-01-01-01-24.wav"
visualize_audio(file_path)

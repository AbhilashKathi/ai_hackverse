import os
import numpy as np
import librosa
import pandas as pd


# RAVDESS emotion mapping based on filename
emotion_map = {
    "01": "Neutral", "02": "Calm", "03": "Happy", "04": "Sad",
    "05": "Angry", "06": "Fearful", "07": "Disgust", "08": "Surprised"
}

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050, duration=3, offset=0.5)  # 3 sec clips
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

data = []
audio_path = "C:\\Users\\Abhilash Kathi\\Desktop\\AuraDetect2\\ai_hackverse\\audio files"

for actor in os.listdir(audio_path):
    actor_folder = os.path.join(audio_path, actor)
    for file in os.listdir(actor_folder):
        if file.endswith(".wav"):
            file_path = os.path.join(actor_folder, file)
            emotion_label = emotion_map[file.split("-")[2]]  # Extract correct emotion
            features = extract_features(file_path)
            data.append({"Features": features, "Emotion": emotion_label})

dataset_path = "C:\\Users\\Abhilash Kathi\\Desktop\\AuraDetect2\\ai_hackverse\\audio files"
actors = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
print("Actors found:", actors)  # Should list all 24 actor folders
df = pd.DataFrame(data)
df.to_pickle("audio_features.pkl")
print("✅ Features extracted and saved.")

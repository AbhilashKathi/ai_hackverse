import numpy as np
import librosa
import keras
import pickle

# Load model
model = keras.models.load_model('emotion_model.h5')

# Load label encoder
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050, duration=3, offset=0.5)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

file_path = "your_test_audio.wav"
features = extract_features(file_path).reshape(1, -1)
prediction = model.predict(features)
predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])

print(f"🎵 Predicted Emotion: {predicted_label[0]}")

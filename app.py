import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import random
import webbrowser

# Load trained model
model = tf.keras.models.load_model("emotion_model.h5")

# Load label encoder
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Function to extract features
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050, duration=3, offset=0.5)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

# Function to log emotion with user entry
def log_emotion(user, emotion, journal_entry=""):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = pd.DataFrame([[timestamp, user, emotion, journal_entry]], columns=["Timestamp", "User", "Emotion", "Journal"])
    try:
        log_entry.to_csv("journal.csv", mode="a", header=False, index=False)
    except:
        st.error("Error saving emotion log.")

# Function to show emotion history
def show_emotion_history(user):
    try:
        df = pd.read_csv("journal.csv", names=["Timestamp", "User", "Emotion", "Journal"])
        user_df = df[df["User"] == user]
        if user_df.empty:
            st.warning("No records found for this user.")
        else:
            st.write("### Emotion History for", user)
            st.dataframe(user_df)

            # Plot mood graph
            st.write("### Mood Over Time")
            plt.figure(figsize=(10, 4))
            emotion_counts = user_df["Emotion"].value_counts()
            plt.bar(emotion_counts.index, emotion_counts.values, color="skyblue")
            plt.xlabel("Emotion")
            plt.ylabel("Occurrences")
            plt.title("Emotion Frequency")
            st.pyplot(plt)
    except FileNotFoundError:
        st.warning("No emotion history found.")

# Function for activity recommendations
import random

# Function for self-care activity recommendations
def get_recommendation(emotion):
    activities = {
        "Happy": ["Go for a walk outside 🌿", "Watch a comedy show 🎭", "Write about your happiness 📖"],
        "Sad": ["Listen to uplifting music 🎶", "Talk to a friend ☎️", "Write down what’s bothering you 📝"],
        "Angry": ["Try deep breathing exercises 🧘", "Do a short workout 🏋️", "Listen to calm music 🎼"],
        "Fearful": ["Practice mindfulness 🧠", "Watch motivational videos 🎥", "Write about your fear 💭"],
        "Disgusted": ["Engage in an activity you love 🎨", "Take a break 🛌", "Watch a feel-good movie 🍿"],
        "Surprised": ["Write about why you feel surprised ✍️", "Embrace new experiences 🚀"],
        "Neutral": ["Stay productive 📚", "Try something creative 🎨"]
    }
    return random.choice(activities.get(emotion, ["Take a moment to reflect."]))

# Function for music & video recommendations
def get_music_video_recommendation(emotion):
    recommendations = {
        "Happy": ["https://www.youtube.com/watch?v=d-diB65scQU", "https://www.youtube.com/watch?v=8Z5EjAmZS1o"],
        "Sad": ["https://www.youtube.com/watch?v=KkGVmN68ByU", "https://www.youtube.com/watch?v=RgKAFK5djSk"],
        "Angry": ["https://www.youtube.com/watch?v=6Ejga4kJUts", "https://www.youtube.com/watch?v=09R8_2nJtjg"],
        "Fearful": ["https://www.youtube.com/watch?v=Jk7LPpY8pXM", "https://www.youtube.com/watch?v=l9QpjPHEG_s"],
        "Disgusted": ["https://www.youtube.com/watch?v=2Vv-BfVoq4g", "https://www.youtube.com/watch?v=8SbUC-UaAxE"],
        "Surprised": ["https://www.youtube.com/watch?v=VbfpW0pbvaU", "https://www.youtube.com/watch?v=LHCob76kigA"],
        "Neutral": ["https://www.youtube.com/watch?v=VYOjWnS4cMY", "https://www.youtube.com/watch?v=fRh_vgS2dFE"]
    }
    return random.choice(recommendations.get(emotion, ["https://www.youtube.com/watch?v=5qap5aO4i9A"]))  # Lo-fi chill music


# Streamlit UI
st.title("Emotion Detection & Well-being Companion 🎤")

# User input for name
user_name = st.text_input("Enter your name to track your emotions:")

# Upload audio file
uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])

if uploaded_file is not None and user_name.strip():
    file_path = "temp_audio.wav"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    # Extract features
    features = extract_features(file_path)
    features = np.expand_dims(features, axis=0)

    # Get model prediction
    predictions = model.predict(features)
    predicted_index = np.argmax(predictions)
    predicted_emotion = label_encoder.inverse_transform([predicted_index])[0]

    # Log emotion
    journal_entry = st.text_area("Write about how you're feeling (Optional)")
    if st.button("Save Emotion & Journal Entry"):
        log_emotion(user_name, predicted_emotion, journal_entry)
        st.success("Emotion & journal entry saved!")

    # Show detected emotion
    st.subheader(f"Predicted Emotion: {predicted_emotion}")

    # Show recommendation
    activity = get_recommendation(predicted_emotion)
    st.write("💡 **Activity Suggestion:**", activity)

    # Show music/video recommendation
    video_url = get_music_video_recommendation(predicted_emotion)
    if st.button("🎵 Play a Mood-Based Video"):
        webbrowser.open(video_url)

# Show user-specific emotion history
if st.button("Show My Emotion History") and user_name.strip():
    show_emotion_history(user_name)

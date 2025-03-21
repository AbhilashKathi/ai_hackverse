import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

print("🚀 Training started...")

# Load new features
df = pd.read_pickle('audio_features.pkl')
X = np.array(df['Features'].tolist())
y = np.array(df['Emotion'].tolist())

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Save the label encoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Neural network
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(40,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),  # Added new layer
    layers.Dense(len(encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test))

# Save
model.save('emotion_model.h5')
print("✅ Model retrained and saved with 100 epochs & extra layer!")

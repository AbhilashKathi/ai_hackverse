import pickle
from sklearn.preprocessing import LabelEncoder

# Load the label encoder
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Check if it's actually a LabelEncoder instance
if isinstance(label_encoder, LabelEncoder):
    print("✅ Label encoder is correctly saved.")
    print("Classes:", label_encoder.classes_)  # Print the classes to verify
else:
    print("❌ The loaded object is NOT a LabelEncoder instance!")

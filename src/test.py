import os
import joblib
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
import numpy as np

# Correct absolute path to the saved model
model_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "model", "savedmodel.pth")
)

print("Loading model from:", model_path)

model_data = joblib.load(model_path)
clf = model_data["model"]

# Load dataset
data = fetch_olivetti_faces()
X = data.images
y = data.target

# Flatten
X = X.reshape((X.shape[0], -1))

# Test split
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Evaluate
accuracy = clf.score(X_test, y_test)

print(f"Test Accuracy: {accuracy:.4f}")

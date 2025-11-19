import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib
import os

# Load dataset
data = fetch_olivetti_faces()
X = data.images
y = data.target

# Flatten images
X = X.reshape((X.shape[0], -1))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train SVM model
clf = SVC(kernel="linear", probability=True)
clf.fit(X_train, y_train)

# Save model
model_dir = os.path.join("..", "model")
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "savedmodel.pth")
joblib.dump({"model": clf}, model_path)

print(f"Model saved to {model_path}")

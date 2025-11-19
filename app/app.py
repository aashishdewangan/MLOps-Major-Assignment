from flask import Flask, request, render_template
from PIL import Image, ImageOps
import numpy as np
import joblib
import io
import os

app = Flask(__name__)

print(" Loading saved model for Flask app...")

# Correct absolute path to model file
model_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "model", "savedmodel.pth")
)

print("Loading model from:", model_path)

model_data = joblib.load(model_path)
clf = model_data["model"]

# Preprocessing for uploaded image
def preprocess(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("L")
    img = ImageOps.fit(img, (64, 64))
    arr = np.array(img).astype("float32") / 255.0
    return arr.reshape(1, -1)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        file = request.files["file"]
        if file:
            processed = preprocess(file.read())
            result = int(clf.predict(processed)[0])
    return render_template("upload.html", result=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

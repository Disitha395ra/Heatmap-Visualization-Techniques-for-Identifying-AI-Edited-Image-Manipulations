from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import os
import base64
from io import BytesIO

from utils import preprocess_image
from model import predict_image, generate_heatmap

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

@app.route("/")
def home():
    return "Flask Backend Running ✅"

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    # Save uploaded image
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Open image
    image = Image.open(filepath)

    # 🔹 STEP 1: Preprocess
    processed = preprocess_image(image)

    # 🔹 STEP 2: Model Prediction
    result, confidence = predict_image(processed)

    # 🔹 STEP 3: Heatmap
    heatmap_path = generate_heatmap(filepath)

    # Convert heatmap to base64 (send to React)
    with open(heatmap_path, "rb") as img_file:
        base64_string = base64.b64encode(img_file.read()).decode("utf-8")

    return jsonify({
        "prediction": result,
        "confidence": confidence,
        "heatmap": base64_string
    })


if __name__ == "__main__":
    app.run(debug=True)
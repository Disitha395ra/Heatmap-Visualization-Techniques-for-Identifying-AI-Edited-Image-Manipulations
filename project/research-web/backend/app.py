from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import base64

from utils import preprocess_image
from model import predict_image, generate_heatmap

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "Backend Running ✅"

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    import io
    from PIL import ImageOps

    file = request.files["file"]

    # Load image and perfectly handle EXIF rotation for smartphone photos
    file_bytes = file.read()
    image = Image.open(io.BytesIO(file_bytes))
    image = ImageOps.exif_transpose(image)

    # 🔹 Preprocess
    tensor = preprocess_image(image)

    # 🔹 Prediction
    result, confidence = predict_image(tensor)

    # 🔥 Passed original image for resizing CAM to original dimensions
    heatmap_path = generate_heatmap(tensor, image)

    # Convert to base64
    with open(heatmap_path, "rb") as f:
        base64_img = base64.b64encode(f.read()).decode("utf-8")

    return jsonify({
        "prediction": result,
        "confidence": confidence,
        "heatmap": base64_img
    })

if __name__ == "__main__":
    app.run(debug=True)
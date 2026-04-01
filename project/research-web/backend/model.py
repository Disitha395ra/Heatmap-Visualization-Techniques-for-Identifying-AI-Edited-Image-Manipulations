import torch
import cv2
import numpy as np
import os

# Dummy model (replace later)
def predict_image(tensor):
    # Simulated prediction
    prediction = torch.rand(1).item()

    if prediction > 0.5:
        return "Manipulated", round(prediction, 2)
    else:
        return "Real", round(1 - prediction, 2)


# Dummy heatmap generator (replace with Grad-CAM later)
def generate_heatmap(image_path):
    image = cv2.imread(image_path)

    # Create fake heatmap
    heatmap = cv2.applyColorMap(image, cv2.COLORMAP_JET)

    output_path = os.path.join("outputs", "heatmap.jpg")
    cv2.imwrite(output_path, heatmap)

    return output_path
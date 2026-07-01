import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np
import os

model = None

# 🔹 Load Model
def load_model():
    global model

    if model is None:
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 2)

        model.load_state_dict(
            torch.load("model/best_resnet50.pth", map_location=torch.device('cpu'))
        )

        model.eval()

    return model


# 🔹 Prediction
def predict_image(tensor):
    model = load_model()

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred = torch.argmax(probs).item()
        conf = probs[pred].item()

    if pred == 0:
        return "Manipulated", round(conf, 3)
    else:
        return "Real", round(conf, 3)


from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# 🔥 GRAD-CAM (FIXED VERSION)
def generate_heatmap(tensor, original_image):
    model = load_model()
    
    # Target layer for ResNet50
    target_layers = [model.layer4[-1]]
    
    # Initialize GradCAM
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # Compute CAM for the squashed 224x224 tensor
    grayscale_cam = cam(input_tensor=tensor)[0]
    
    # Convert original PIL image to OpenCV format
    orig_img_cv = cv2.cvtColor(np.array(original_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w, _ = orig_img_cv.shape
    
    # Resize the high-quality 224x224 CAM back to the original image dimensions
    cam_resized = cv2.resize(grayscale_cam, (w, h))
    
    # Convert image to float [0, 1] for show_cam_on_image
    rgb_float = cv2.cvtColor(orig_img_cv, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    
    # Overlay heatmap exactly as done in gradcam_one_image.py
    cam_overlay = show_cam_on_image(rgb_float, cam_resized, use_rgb=True)
    
    # Convert back to BGR for saving
    cam_overlay_bgr = cv2.cvtColor(cam_overlay, cv2.COLOR_RGB2BGR)
    
    # Save
    output_path = os.path.join("outputs", "heatmap.jpg")
    cv2.imwrite(output_path, cam_overlay_bgr)
    
    return output_path
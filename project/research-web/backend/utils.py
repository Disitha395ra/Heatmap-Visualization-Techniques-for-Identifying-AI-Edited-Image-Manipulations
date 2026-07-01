import torch
import cv2
import numpy as np
from torchvision import transforms

def preprocess_image(image):
    # Convert PIL Image to RGB numpy array
    rgb = np.array(image.convert("RGB"))
    
    # Use cv2.resize to perfectly match gradcam_one_image.py
    rgb_resized = cv2.resize(rgb, (224, 224))
    
    # Apply identical transforms
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    
    tensor = preprocess(rgb_resized).unsqueeze(0)
    return tensor
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# ---------------- SETTINGS ----------------
MODEL_PATH = "resnet18_baseline_v2.pth"
IMG_SIZE = 224
CLASSES = ["fake", "real"]

# 🔥 Put any image path here (start with a fake test image)
IMAGE_PATH = r"dataset\test\fake"  # folder or file

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Device:", device)

# ---------------- LOAD MODEL ----------------
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()
print("✅ Loaded model:", MODEL_PATH)

# ---------------- PICK IMAGE ----------------
if os.path.isdir(IMAGE_PATH):
    # pick first image from folder
    first = os.listdir(IMAGE_PATH)[0]
    img_path = os.path.join(IMAGE_PATH, first)
else:
    img_path = IMAGE_PATH

print("✅ Using image:", img_path)

# ---------------- READ & RESIZE ----------------
bgr = cv2.imread(img_path)
if bgr is None:
    raise ValueError("❌ Could not read image. Check IMAGE_PATH.")

rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))

# For overlay (must be 0..1 float)
rgb_float = rgb.astype(np.float32) / 255.0

# ---------------- PREPROCESS FOR MODEL ----------------
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

input_tensor = preprocess(rgb).unsqueeze(0).to(device)

# ---------------- PREDICT + CONFIDENCE ----------------
with torch.no_grad():
    logits = model(input_tensor)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred = int(np.argmax(probs))
    conf = float(probs[pred])

print(f"✅ Prediction: {CLASSES[pred]}  | Confidence: {conf:.4f}")
print(f"   Prob(fake)={probs[0]:.4f}, Prob(real)={probs[1]:.4f}")

# ---------------- GRAD-CAM ----------------
# ResNet18 last conv block
target_layers = [model.layer4[-1]]

cam = GradCAM(model=model, target_layers=target_layers)
grayscale_cam = cam(input_tensor=input_tensor)[0]  # (H,W) 0..1

# Overlay heatmap on image
cam_overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)

# ---------------- SAVE OUTPUTS ----------------
os.makedirs("outputs", exist_ok=True)

base = os.path.splitext(os.path.basename(img_path))[0]
out_overlay = os.path.join("outputs", f"{base}_gradcam_overlay.png")
out_heatmap = os.path.join("outputs", f"{base}_gradcam_heatmap.png")

# save overlay
cv2.imwrite(out_overlay, cv2.cvtColor(cam_overlay, cv2.COLOR_RGB2BGR))

# save raw heatmap as grayscale image (0..255)
heatmap_u8 = (grayscale_cam * 255).astype(np.uint8)
cv2.imwrite(out_heatmap, heatmap_u8)

print("✅ Saved:")
print("  ", out_overlay)
print("  ", out_heatmap)

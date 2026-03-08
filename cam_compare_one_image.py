import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

MODEL_PATH = "resnet18_baseline_v2.pth"
IMG_SIZE = 224
CLASSES = ["fake", "real"]

# pick a tampered test image
IMG_PATH = r"dataset\test\fake"  # folder OR file

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- model ----
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

target_layers = [model.layer4[-1]]

# ---- choose image ----
if os.path.isdir(IMG_PATH):
    img_name = os.listdir(IMG_PATH)[0]
    img_path = os.path.join(IMG_PATH, img_name)
else:
    img_path = IMG_PATH

bgr = cv2.imread(img_path)
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
rgb_float = rgb.astype(np.float32) / 255.0

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
input_tensor = preprocess(rgb).unsqueeze(0).to(device)

# prediction
with torch.no_grad():
    logits = model(input_tensor)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred = int(np.argmax(probs))

print("Image:", img_path)
print(f"Prediction: {CLASSES[pred]} | Prob(fake)={probs[0]:.3f} Prob(real)={probs[1]:.3f}")

# ---- CAM methods ----
methods = {
    "gradcam": GradCAM(model=model, target_layers=target_layers),
    "gradcampp": GradCAMPlusPlus(model=model, target_layers=target_layers),
    "scorecam": ScoreCAM(model=model, target_layers=target_layers),
}

os.makedirs("cam_compare", exist_ok=True)

for name, cam in methods.items():
    grayscale_cam = cam(input_tensor=input_tensor)[0]
    overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)

    out_path = os.path.join("cam_compare", f"{os.path.splitext(os.path.basename(img_path))[0]}_{name}.png")
    cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print("Saved:", out_path)

import os
import csv
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# ---------------- SETTINGS ----------------
DATA_DIR = "dataset"
MODEL_PATH = "resnet18_baseline_v2.pth"
OUT_DIR = "outputs_batch"
CSV_PATH = os.path.join(OUT_DIR, "predictions.csv")

IMG_SIZE = 224
BATCH_SIZE = 1
NUM_WORKERS = 0
CLASSES = ["fake", "real"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Device:", device)

# ---------------- MODEL ----------------
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()
print("✅ Loaded model:", MODEL_PATH)

# CAM target layer
target_layers = [model.layer4[-1]]
cam = GradCAM(model=model, target_layers=target_layers)

# ---------------- DATASET ----------------
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- CSV HEADER ----------------
with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "true_label", "pred_label", "confidence", "prob_fake", "prob_real", "overlay_path", "heatmap_path"])

# ---------------- RUN ----------------
for idx in range(len(test_dataset)):
    img_path, true_label = test_dataset.samples[idx]
    true_name = CLASSES[true_label]

    # read original image for overlay
    bgr = cv2.imread(img_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
    rgb_float = rgb.astype(np.float32) / 255.0

    # model input tensor
    input_tensor, _ = test_dataset[idx]
    input_tensor = input_tensor.unsqueeze(0).to(device)

    # prediction
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = int(np.argmax(probs))
        conf = float(probs[pred])
        pred_name = CLASSES[pred]

    # grad-cam
    grayscale_cam = cam(input_tensor=input_tensor)[0]
    overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)

    base = os.path.splitext(os.path.basename(img_path))[0]
    overlay_path = os.path.join(OUT_DIR, f"{base}_overlay.png")
    heatmap_path = os.path.join(OUT_DIR, f"{base}_heatmap.png")

    cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite(heatmap_path, (grayscale_cam * 255).astype(np.uint8))

    # log to csv
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([os.path.basename(img_path), true_name, pred_name, conf, probs[0], probs[1], overlay_path, heatmap_path])

    # print progress sometimes
    if (idx + 1) % 100 == 0:
        print(f"Processed {idx+1}/{len(test_dataset)}")

print("✅ Done! Saved results to:", CSV_PATH)

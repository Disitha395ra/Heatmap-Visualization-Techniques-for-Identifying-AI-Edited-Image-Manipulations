import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM

TP_TEST_DIR = r"dataset\test\fake"
GT_DIR = r"CASIA2\CASIA 2 Groundtruth"

MODEL_PATH = "best_resnet50.pth"
IMG_SIZE = 224
MASK_THRESHOLD = 127

# We'll test these thresholds (your best was around 80-100)
THRESHOLDS = [60, 80, 100, 120]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- model ----
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# ---- cams for different layers ----
cams = {
    "layer2": GradCAM(model=model, target_layers=[model.layer2[-1]]),
    "layer3": GradCAM(model=model, target_layers=[model.layer3[-1]]),
    "layer4": GradCAM(model=model, target_layers=[model.layer4[-1]]),
}

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def binarize(gray, t):
    _, b = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
    return b

def iou_score(pred_bin, gt_bin):
    pred = pred_bin > 0
    gt = gt_bin > 0
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return (inter / union) if union != 0 else 0.0

def find_gt_mask(tp_filename):
    base = os.path.splitext(tp_filename)[0]
    for f in os.listdir(GT_DIR):
        if base in f:
            return os.path.join(GT_DIR, f)
    return None

tp_files = [f for f in os.listdir(TP_TEST_DIR) if f.lower().endswith((".jpg",".jpeg",".png",".bmp"))]

pairs = []
missing = 0
for f in tp_files:
    gt_path = find_gt_mask(f)
    if gt_path is None:
        missing += 1
        continue
    pairs.append((f, os.path.join(TP_TEST_DIR, f), gt_path))

print("Matched pairs:", len(pairs), "Missing:", missing)
print("Testing thresholds:", THRESHOLDS)

best = ("", None, -1)

for layer_name, cam in cams.items():
    for thr in THRESHOLDS:
        ious = []
        for fname, img_path, gt_path in pairs:
            bgr = cv2.imread(img_path)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
            input_tensor = preprocess(rgb).unsqueeze(0).to(device)

            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            gt = cv2.resize(gt, (IMG_SIZE, IMG_SIZE))
            gt_bin = binarize(gt, MASK_THRESHOLD)

            grayscale_cam = cam(input_tensor=input_tensor)[0]   # 0..1
            hm_u8 = (grayscale_cam * 255).astype(np.uint8)
            pred_bin = binarize(hm_u8, thr)

            ious.append(iou_score(pred_bin, gt_bin))

        avg_iou = sum(ious)/len(ious) if ious else 0
        print(f"{layer_name:6s} thr={thr:3d}  Avg IoU={avg_iou:.4f}")

        if avg_iou > best[2]:
            best = (layer_name, thr, avg_iou)

print("\n✅ BEST:")
print(f"Layer={best[0]}, Threshold={best[1]}, Avg IoU={best[2]:.4f}")
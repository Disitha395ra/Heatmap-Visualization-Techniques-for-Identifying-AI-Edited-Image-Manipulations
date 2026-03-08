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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Load model ----
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

cam_l3 = GradCAM(model=model, target_layers=[model.layer3[-1]])
cam_l4 = GradCAM(model=model, target_layers=[model.layer4[-1]])

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

# Try weights (favor layer4 more strongly)
weights = [0.0, 0.2, 0.3, 0.4, 0.5]
thresholds = [80, 100, 120, 140]

print("Weights (layer3):", weights)
print("Thresholds:", thresholds)

# Pre-load GT masks & images list (for speed)
pairs = []
missing = 0
for f in tp_files:
    gt_path = find_gt_mask(f)
    if gt_path is None:
        missing += 1
        continue
    img_path = os.path.join(TP_TEST_DIR, f)
    pairs.append((f, img_path, gt_path))

print("Matched pairs:", len(pairs), "Missing:", missing)

best = (None, None, -1)  # (w, t, iou)

for w in weights:
    for t in thresholds:
        ious = []
        for fname, img_path, gt_path in pairs:
            bgr = cv2.imread(img_path)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
            input_tensor = preprocess(rgb).unsqueeze(0).to(device)

            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            gt = cv2.resize(gt, (IMG_SIZE, IMG_SIZE))
            gt_bin = binarize(gt, MASK_THRESHOLD)

            cam3 = cam_l3(input_tensor=input_tensor)[0]
            cam4 = cam_l4(input_tensor=input_tensor)[0]

            fused = w * cam3 + (1 - w) * cam4

            hm_u8 = (fused * 255).astype(np.uint8)
            pred_bin = binarize(hm_u8, t)

            ious.append(iou_score(pred_bin, gt_bin))

        avg_iou = sum(ious) / len(ious) if ious else 0
        print(f"w(layer3)={w:.1f}, thr={t}: Avg IoU={avg_iou:.4f}")

        if avg_iou > best[2]:
            best = (w, t, avg_iou)

print("\n✅ Best setting:")
print(f"w(layer3)={best[0]}, threshold={best[1]}, Avg IoU={best[2]:.4f}")
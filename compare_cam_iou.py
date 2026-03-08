import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM

TP_TEST_DIR = r"dataset\test\fake"
GT_DIR = r"CASIA2\CASIA 2 Groundtruth"

MODEL_PATH = "resnet18_baseline_v2.pth"
IMG_SIZE = 224

HEATMAP_THRESHOLD = 100
MASK_THRESHOLD = 127

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- model ----
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

target_layers = [model.layer4[-1]]

cams = {
    "GradCAM": GradCAM(model=model, target_layers=target_layers),
    "GradCAM++": GradCAMPlusPlus(model=model, target_layers=target_layers),
    "ScoreCAM": ScoreCAM(model=model, target_layers=target_layers),
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

results = {k: [] for k in cams.keys()}
matched = 0
missing = 0

for f in tp_files:
    gt_path = find_gt_mask(f)
    if gt_path is None:
        missing += 1
        continue

    img_path = os.path.join(TP_TEST_DIR, f)

    # read image & resize
    bgr = cv2.imread(img_path)
    if bgr is None:
        missing += 1
        continue
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))

    input_tensor = preprocess(rgb).unsqueeze(0).to(device)

    # load & resize mask
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    if gt is None:
        missing += 1
        continue
    gt = cv2.resize(gt, (IMG_SIZE, IMG_SIZE))
    gt_bin = binarize(gt, MASK_THRESHOLD)

    # compute each CAM IoU
    for name, cam in cams.items():
        grayscale_cam = cam(input_tensor=input_tensor)[0]   # 0..1
        hm_u8 = (grayscale_cam * 255).astype(np.uint8)
        pred_bin = binarize(hm_u8, HEATMAP_THRESHOLD)
        results[name].append(iou_score(pred_bin, gt_bin))

    matched += 1
    if matched % 50 == 0:
        print("Processed:", matched)

print("\n✅ IoU Comparison Done")
print("Matched:", matched, "Missing:", missing)
for name, vals in results.items():
    avg = sum(vals)/len(vals) if vals else 0
    print(f"{name:10s} Avg IoU: {avg:.4f}")

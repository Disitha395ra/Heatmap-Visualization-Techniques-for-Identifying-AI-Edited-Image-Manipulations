import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
import matplotlib.pyplot as plt

# -------- PATHS --------
TP_TEST_DIR = r"dataset\test\fake"
GT_DIR = r"CASIA2\CASIA 2 Groundtruth"

MODEL_PATH = "best_resnet50.pth"
IMG_SIZE = 224

# Best setting you found
HEATMAP_THRESHOLD = 80
MASK_THRESHOLD = 127

OUT_DIR = "resnet50_iou_examples"
TOP_K = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Device:", device)

# -------- LOAD MODEL --------
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()
print("✅ Loaded model:", MODEL_PATH)

cam = GradCAM(model=model, target_layers=[model.layer4[-1]])

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

# -------- COMPUTE IOU FOR ALL MATCHED --------
scores = []
tp_files = [f for f in os.listdir(TP_TEST_DIR) if f.lower().endswith((".jpg",".jpeg",".png",".bmp"))]

for f in tp_files:
    gt_path = find_gt_mask(f)
    if gt_path is None:
        continue

    img_path = os.path.join(TP_TEST_DIR, f)
    bgr = cv2.imread(img_path)
    if bgr is None:
        continue

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
    input_tensor = preprocess(rgb).unsqueeze(0).to(device)

    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    if gt is None:
        continue
    gt = cv2.resize(gt, (IMG_SIZE, IMG_SIZE))
    gt_bin = binarize(gt, MASK_THRESHOLD)

    grayscale_cam = cam(input_tensor=input_tensor)[0]  # 0..1
    hm_u8 = (grayscale_cam * 255).astype(np.uint8)
    pred_bin = binarize(hm_u8, HEATMAP_THRESHOLD)

    iou = iou_score(pred_bin, gt_bin)
    scores.append((iou, f, img_path, gt_path, hm_u8, pred_bin, gt_bin))

# sort
scores_sorted = sorted(scores, key=lambda x: x[0], reverse=True)
best = scores_sorted[:TOP_K]
worst = scores_sorted[-TOP_K:]

# -------- SAVE VISUALIZATIONS --------
best_dir = os.path.join(OUT_DIR, "top_best")
worst_dir = os.path.join(OUT_DIR, "top_worst")
os.makedirs(best_dir, exist_ok=True)
os.makedirs(worst_dir, exist_ok=True)

def save_panel(items, out_folder):
    for iou, fname, img_path, gt_path, hm_u8, pred_bin, gt_bin in items:
        bgr = cv2.imread(img_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 4, 1)
        plt.imshow(rgb)
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1, 4, 2)
        plt.imshow(hm_u8, cmap="jet")
        plt.title("Grad-CAM")
        plt.axis("off")

        plt.subplot(1, 4, 3)
        plt.imshow(gt_bin, cmap="gray")
        plt.title("Groundtruth")
        plt.axis("off")

        plt.subplot(1, 4, 4)
        # show predicted mask with GT overlap highlight
        overlay = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        # red = predicted only
        overlay[(pred_bin > 0) & (gt_bin == 0)] = [255, 0, 0]
        # green = GT only
        overlay[(gt_bin > 0) & (pred_bin == 0)] = [0, 255, 0]
        # yellow = overlap
        overlay[(pred_bin > 0) & (gt_bin > 0)] = [255, 255, 0]

        plt.imshow(overlay)
        plt.title(f"IoU={iou:.3f}")
        plt.axis("off")

        plt.tight_layout()
        out_path = os.path.join(out_folder, f"{os.path.splitext(fname)[0]}_IoU_{iou:.3f}.png")
        plt.savefig(out_path, dpi=200)
        plt.close()

save_panel(best, best_dir)
save_panel(worst, worst_dir)

print("✅ Saved examples to:", OUT_DIR)
print("Best folder :", best_dir)
print("Worst folder:", worst_dir)
print("Total matched scored:", len(scores_sorted))
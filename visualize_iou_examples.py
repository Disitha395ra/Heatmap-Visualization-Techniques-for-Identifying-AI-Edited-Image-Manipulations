import os
import cv2
import numpy as np
import csv
import matplotlib.pyplot as plt

TP_TEST_DIR = r"dataset\test\fake"
GT_DIR = r"CASIA2\CASIA 2 Groundtruth"
HEATMAP_DIR = r"outputs_batch"
IOU_CSV = r"outputs_batch\iou_results.csv"

BEST_THRESHOLD = 100
MASK_THRESHOLD = 127

def binarize(img, t):
    _, b = cv2.threshold(img, t, 255, cv2.THRESH_BINARY)
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

# Load IoU results
rows = []
with open(IOU_CSV, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        r["iou"] = float(r["iou"])
        rows.append(r)

# Sort by IoU descending
rows_sorted = sorted(rows, key=lambda x: x["iou"], reverse=True)

# Take top 5 examples
top_examples = rows_sorted[:5]

os.makedirs("iou_visualizations", exist_ok=True)

for r in top_examples:
    fname = r["tp_image"]
    iou_val = r["iou"]

    img_path = os.path.join(TP_TEST_DIR, fname)
    heatmap_path = os.path.join(HEATMAP_DIR, os.path.splitext(fname)[0] + "_heatmap.png")
    gt_path = find_gt_mask(fname)

    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    hm = cv2.imread(heatmap_path, cv2.IMREAD_GRAYSCALE)
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

    gt = cv2.resize(gt, (hm.shape[1], hm.shape[0]))

    pred_bin = binarize(hm, BEST_THRESHOLD)
    gt_bin = binarize(gt, MASK_THRESHOLD)

    plt.figure(figsize=(12,4))

    plt.subplot(1,4,1)
    plt.imshow(img_rgb)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1,4,2)
    plt.imshow(hm, cmap="jet")
    plt.title("Grad-CAM")
    plt.axis("off")

    plt.subplot(1,4,3)
    plt.imshow(gt_bin, cmap="gray")
    plt.title("Groundtruth")
    plt.axis("off")

    plt.subplot(1,4,4)
    overlay = np.zeros_like(gt_bin)
    overlay[(pred_bin>0) & (gt_bin>0)] = 255
    plt.imshow(overlay, cmap="hot")
    plt.title(f"IoU={iou_val:.3f}")
    plt.axis("off")

    plt.tight_layout()
    out_path = os.path.join("iou_visualizations", f"{os.path.splitext(fname)[0]}_viz.png")
    plt.savefig(out_path)
    plt.close()

print("✅ Saved top IoU visualization examples.")

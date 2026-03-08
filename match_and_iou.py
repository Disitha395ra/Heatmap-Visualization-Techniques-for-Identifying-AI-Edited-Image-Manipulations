import os
import csv
import cv2
import numpy as np

# -------- PATHS --------
TP_TEST_DIR = r"dataset\test\fake"
GT_DIR = r"CASIA2\CASIA 2 Groundtruth"
HEATMAP_DIR = r"outputs_batch"
OUT_CSV = os.path.join("outputs_batch", "iou_results.csv")

# -------- SETTINGS --------
HEATMAP_THRESHOLD = 200  # 0-255. raise to focus on strongest region
MASK_THRESHOLD = 127     # binarize groundtruth mask

def binarize(img_gray, thresh):
    _, binary = cv2.threshold(img_gray, thresh, 255, cv2.THRESH_BINARY)
    return binary

def iou_score(pred_bin, gt_bin):
    pred = pred_bin > 0
    gt = gt_bin > 0
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return (inter / union) if union != 0 else 0.0

def find_gt_mask(tp_filename):
    """
    CASIA masks are not always named exactly the same.
    We will try:
    1) exact match by base name
    2) contains base name search
    """
    base = os.path.splitext(tp_filename)[0]

    # 1) exact match
    for ext in [".png", ".jpg", ".bmp", ".tif"]:
        p = os.path.join(GT_DIR, base + ext)
        if os.path.exists(p):
            return p

    # 2) contains match
    for f in os.listdir(GT_DIR):
        if base in f:
            return os.path.join(GT_DIR, f)

    return None

tp_files = [f for f in os.listdir(TP_TEST_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg", ".bmp"))]

rows = []
missing = 0

for f in tp_files:
    gt_path = find_gt_mask(f)
    heatmap_path = os.path.join(HEATMAP_DIR, os.path.splitext(f)[0] + "_heatmap.png")

    if gt_path is None or not os.path.exists(heatmap_path):
        missing += 1
        continue

    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    hm = cv2.imread(heatmap_path, cv2.IMREAD_GRAYSCALE)

    if gt is None or hm is None:
        missing += 1
        continue

    # ensure same size
    gt = cv2.resize(gt, (hm.shape[1], hm.shape[0]))

    gt_bin = binarize(gt, MASK_THRESHOLD)
    hm_bin = binarize(hm, HEATMAP_THRESHOLD)

    iou = iou_score(hm_bin, gt_bin)

    rows.append([f, os.path.basename(gt_path), round(iou, 4)])

# save CSV
with open(OUT_CSV, "w", newline="", encoding="utf-8") as fp:
    writer = csv.writer(fp)
    writer.writerow(["tp_image", "gt_mask", "iou"])
    writer.writerows(rows)

ious = [r[2] for r in rows]
avg_iou = sum(ious) / len(ious) if ious else 0

print("✅ IoU evaluation done!")
print("Matched:", len(rows))
print("Missing:", missing)
print("Average IoU:", round(avg_iou, 4))
print("Saved:", OUT_CSV)

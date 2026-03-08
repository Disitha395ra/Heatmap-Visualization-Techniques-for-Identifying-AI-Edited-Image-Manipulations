import os
import csv
import shutil

IN_DIR = "outputs_batch"
CSV_PATH = os.path.join(IN_DIR, "predictions.csv")

OUT_CORRECT = os.path.join(IN_DIR, "top_correct")
OUT_WRONG = os.path.join(IN_DIR, "top_wrong")

os.makedirs(OUT_CORRECT, exist_ok=True)
os.makedirs(OUT_WRONG, exist_ok=True)

rows = []
with open(CSV_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        r["confidence"] = float(r["confidence"])
        rows.append(r)

# wrong predictions (high confidence)
wrong = [r for r in rows if r["true_label"] != r["pred_label"]]
wrong_sorted = sorted(wrong, key=lambda x: x["confidence"], reverse=True)[:20]

# correct predictions (high confidence)
correct = [r for r in rows if r["true_label"] == r["pred_label"]]
correct_sorted = sorted(correct, key=lambda x: x["confidence"], reverse=True)[:20]

def copy_set(items, out_dir):
    for r in items:
        overlay = r["overlay_path"]
        heatmap = r["heatmap_path"]
        if os.path.exists(overlay):
            shutil.copy(overlay, out_dir)
        if os.path.exists(heatmap):
            shutil.copy(heatmap, out_dir)

copy_set(wrong_sorted, OUT_WRONG)
copy_set(correct_sorted, OUT_CORRECT)

print("✅ Saved examples:")
print("  Correct:", OUT_CORRECT)
print("  Wrong  :", OUT_WRONG)

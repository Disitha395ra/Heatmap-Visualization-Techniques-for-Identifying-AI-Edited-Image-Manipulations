import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

# -------- CONFIG --------
DATA_DIR = "dataset"
MODEL_PATH = "best_resnet50.pth"
BATCH_SIZE = 16
IMG_SIZE = 224
NUM_WORKERS = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------- TRANSFORMS (NO AUGMENTATION) --------
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------- DATASET / LOADER --------
test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

class_names = test_dataset.classes  # ['fake','real']
print("Classes:", class_names)
print("Test size:", len(test_dataset))

# -------- LOAD MODEL --------
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()
print(f"✅ Loaded model: {MODEL_PATH}")

# -------- PREDICT --------
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

# -------- METRICS --------
acc = accuracy_score(all_labels, all_preds)

# pos_label=1 means "real" is positive class (since fake=0, real=1)
prec = precision_score(all_labels, all_preds, average="binary", pos_label=1)
rec = recall_score(all_labels, all_preds, average="binary", pos_label=1)

cm = confusion_matrix(all_labels, all_preds)

print("\n✅ Evaluation Results (TEST SET)")
print("Accuracy :", round(acc, 4))
print("Precision:", round(prec, 4))
print("Recall   :", round(rec, 4))

print("\nConfusion Matrix (rows=true, cols=pred):")
print(cm)

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))
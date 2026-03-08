#This code removes corrupted / broken images from your dataset before training
from PIL import Image # Pillow: The standard library for opening/manipulating images
import os

base_dir = "dataset"
removed = 0

for root, _, files in os.walk(base_dir):
    for f in files:
        path = os.path.join(root, f)
        try:
            img = Image.open(path)
            img.verify() # Check if the file is actually an image without loading the whole thing into memory
        except Exception:
            print("Removing corrupted:", path)
            os.remove(path)
            removed += 1

print("✅ Cleaning done. Removed:", removed)


# os.walk is like a recursive explorer. It doesn't just look at the top folder; it dives into train/real, train/fake, val/real, etc., automatically.
# root is the current folder path, and files is a list of all images in that folder.
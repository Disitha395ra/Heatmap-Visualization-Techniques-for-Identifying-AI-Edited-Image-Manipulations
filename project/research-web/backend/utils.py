import numpy as np
import torch

def preprocess_image(image):
    # Convert to RGB
    image = image.convert("RGB")

    # Resize
    image = image.resize((224, 224))

    # Normalize
    img_array = np.array(image) / 255.0

    # Convert to tensor
    tensor = torch.tensor(img_array).permute(2, 0, 1).unsqueeze(0).float()

    return tensor
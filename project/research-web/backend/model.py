import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np
import os

model = None

# 🔹 Load Model
def load_model():
    global model

    if model is None:
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 1)

        model.load_state_dict(
            torch.load("model/best_resnet50.pth", map_location=torch.device('cpu'))
        )

        model.eval()

    return model


# 🔹 Prediction
def predict_image(tensor):
    model = load_model()

    with torch.no_grad():
        output = model(tensor)
        prob = torch.sigmoid(output).item()

    if prob > 0.5:
        return "Manipulated", round(prob, 3)
    else:
        return "Real", round(1 - prob, 3)


# 🔥 GRAD-CAM (FIXED VERSION)
def generate_heatmap(tensor):
    model = load_model()

    gradients = []
    activations = []

    # Hook functions
    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    # ✅ Correct layer for ResNet50
    target_layer = model.layer4[-1]

    # Register hooks
    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_full_backward_hook(backward_hook)

    # Forward
    output = model(tensor)

    # Backward
    model.zero_grad()
    output.backward()

    # Get data
    grad = gradients[0].detach().cpu().numpy()[0]
    act = activations[0].detach().cpu().numpy()[0]

    # Compute weights
    weights = np.mean(grad, axis=(1, 2))

    cam = np.zeros(act.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * act[i]

    # ReLU
    cam = np.maximum(cam, 0)

    # Normalize
    cam = cv2.resize(cam, (224, 224))
    cam = cam - np.min(cam)
    cam = cam / (np.max(cam) + 1e-8)

    # Convert to heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    # Convert tensor image back to displayable image
    img = tensor.squeeze().permute(1, 2, 0).cpu().numpy()

    # UNNORMALIZE
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    img = (img * std) + mean
    img = np.clip(img, 0, 1)
    img = np.uint8(img * 255)

    # Convert RGB → BGR for OpenCV
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Overlay
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    # Save
    output_path = os.path.join("outputs", "heatmap.jpg")
    cv2.imwrite(output_path, overlay)

    # Remove hooks (VERY IMPORTANT)
    handle_fwd.remove()
    handle_bwd.remove()

    return output_path
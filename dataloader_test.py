import os #used for file operation , naviagte, join paths
import torch # core library of deep learning framework
from torch.utils.data import DataLoader # used to create batches
from torchvision import datasets, transforms
import matplotlib.pyplot as plt #used for visualization images 

DATA_DIR = "dataset" #main dataset folder
BATCH_SIZE = 32 #load 32 images at a time 
IMG_SIZE = 224 #image size for resizing ( 224 x 224 standard for models like ResNet)
NUM_WORKERS = 2  # use 2 background cpu workers to load data faster

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("✅ Device:", device)
    print("✅ CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("✅ GPU:", torch.cuda.get_device_name(0))

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transform)
    val_dataset   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"),   transform=val_test_transform)
    test_dataset  = datasets.ImageFolder(os.path.join(DATA_DIR, "test"),  transform=val_test_transform)

    print("✅ Classes:", train_dataset.classes)
    print("✅ Class to index:", train_dataset.class_to_idx)
    print("✅ Train size:", len(train_dataset))
    print("✅ Val size:", len(val_dataset))
    print("✅ Test size:", len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    images, labels = next(iter(train_loader))
    print("\n✅ Batch loaded!")
    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)

    images_gpu = images.to(device)
    labels_gpu = labels.to(device)
    print("✅ Batch moved to device successfully:", images_gpu.device)

    # Show 6 images
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    plt.figure(figsize=(12, 4))
    for i in range(6):
        img = images[i].cpu() * std + mean
        img = img.permute(1, 2, 0).clamp(0, 1)
        plt.subplot(2, 3, i + 1)
        plt.imshow(img)
        plt.title(train_dataset.classes[labels[i].item()])
        plt.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import glob
from tqdm import tqdm
import numpy as np

# Import our custom modules
from model import DRModel
from augmentation import train_transform, val_transform

# ================================
# Configuration
# ================================
DATA_DIR = 'data_processed'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')
NUM_CLASSES = 5
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-4

# ================================
# Dataset Class
# ================================
class DRDataset(Dataset):
    """
    Custom Dataset for Diabetic Retinopathy images.
    It loads images, applies augmentations, and returns the image and its label.
    """
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        # Extract labels from the folder structure (e.g., '.../train/0/image.jpg')
        self.labels = [int(os.path.basename(os.path.dirname(path))) for path in image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        # Load image with OpenCV
        image = cv2.imread(image_path)
        # Convert from BGR (OpenCV default) to RGB (Albumentations/PyTorch standard)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        label = self.labels[idx]

        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, torch.tensor(label, dtype=torch.long)

# ================================
# Main Training & Evaluation Logic
# ================================
if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading ---
    # Find all image paths in train and val directories
    train_image_paths = glob.glob(os.path.join(TRAIN_DIR, '**', '*.jpg'), recursive=True)
    val_image_paths = glob.glob(os.path.join(VAL_DIR, '**', '*.jpg'), recursive=True)

    if not train_image_paths or not val_image_paths:
        print("Error: Training or validation data not found. Make sure you have run preprocess.py")
        print(f"Searched in '{TRAIN_DIR}' and '{VAL_DIR}'")
        exit()

    # Create datasets
    train_dataset = DRDataset(train_image_paths, transform=train_transform)
    val_dataset = DRDataset(val_image_paths, transform=val_transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Found {len(train_dataset)} training images and {len(val_dataset)} validation images.")

    # --- Model, Loss, Optimizer ---
    model = DRModel(num_classes=NUM_CLASSES).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

    # --- Training Loop ---
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        print(f"\n{'='*20} Epoch {epoch+1}/{EPOCHS} {'='*20}")

        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        train_correct = 0
        
        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels.data)

        epoch_train_loss = running_loss / len(train_dataset)
        epoch_train_acc = train_correct.double() / len(train_dataset)
        print(f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f}")

        # --- Validation Phase ---
        model.eval()
        running_val_loss = 0.0
        val_correct = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validating"):
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data)

        epoch_val_loss = running_val_loss / len(val_dataset)
        epoch_val_acc = val_correct.double() / len(val_dataset)
        print(f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")
        
        # Step the scheduler
        scheduler.step(epoch_val_loss)

        # Save the best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"New best model saved with accuracy: {best_val_acc:.4f}")

    print("\n✅ Training complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")

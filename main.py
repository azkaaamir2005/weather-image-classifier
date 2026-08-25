import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pathlib import Path

# 1. Configuration & Relative Paths
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "data" / "weather_dataset"
OUTPUT_PATH = BASE_DIR / "outputs" / "confusion_matrix.png"
WEIGHTS_PATH = BASE_DIR / "weather_mobilenet.pth"

SELECTED_CLASSES = ['fogsmog', 'lightning', 'rain', 'sandstorm', 'snow']
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. CPU-Optimized Transformations (160x160 cuts processing load)
data_transforms = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. Dataset Loader
class WeatherDataset(Dataset):
    def __init__(self, root_dir, selected_classes, transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.classes = selected_classes
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(selected_classes)}
        
        for cls_name in selected_classes:
            cls_dir = os.path.join(root_dir, cls_name)
            if os.path.exists(cls_dir):
                for fname in os.listdir(cls_dir):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(cls_dir, fname))
                        self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Data Splitting
full_dataset = WeatherDataset(DATASET_PATH, SELECTED_CLASSES, transform=data_transforms)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Model Setup
model = models.mobilenet_v3_small(weights="DEFAULT")
in_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(in_features, len(SELECTED_CLASSES))
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def train_model():
    print("--- Starting MobileNetV3 CPU Training ---")
    for epoch in range(EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)
            total += labels.size(0)
            
        epoch_loss = running_loss / total
        epoch_acc = (correct.double() / total) * 100
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

def evaluate_model():
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=SELECTED_CLASSES, yticklabels=SELECTED_CLASSES)
    plt.xlabel('Predicted Weather Label')
    plt.ylabel('True Weather Label')
    plt.title('MobileNetV3 Weather Classifier - Confusion Matrix')
    plt.tight_layout()
    
    # Save chart silently to outputs/ without triggering pop-up
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Saved confusion matrix to: {OUTPUT_PATH}")

if __name__ == "__main__":
    train_model()
    evaluate_model()
    torch.save(model.state_dict(), WEIGHTS_PATH)
    print(f"[SUCCESS] Exported model weights to: {WEIGHTS_PATH}")
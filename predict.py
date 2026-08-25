import os
import time
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = BASE_DIR / "weather_mobilenet.pth"
CLASSES = ['fogsmog', 'lightning', 'rain', 'sandstorm', 'snow']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_transforms = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def run_prediction():
    root = tk.Tk()
    root.withdraw()
    
    file_path = filedialog.askopenfilename(
        title="Select Weather Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )
    if not file_path:
        print("No image selected.")
        return

    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, len(CLASSES))
    
    if not os.path.exists(WEIGHTS_PATH):
        print(f"Error: {WEIGHTS_PATH} not found. Run main.py first to train & save weights!")
        return
        
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    img = Image.open(file_path).convert('RGB')
    tensor_img = data_transforms(img).unsqueeze(0).to(DEVICE)

    start_time = time.time()
    with torch.no_grad():
        outputs = model(tensor_img)
        probabilities = torch.softmax(outputs, dim=1)[0]
    inference_time = (time.time() - start_time) * 1000

    confidences = (probabilities * 100).cpu().numpy()
    top_conf, top_idx = torch.max(probabilities, dim=0)
    predicted_class = CLASSES[top_idx.item()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    ax1.imshow(img)
    ax1.axis('off')
    ax1.set_title("Input Image", fontsize=12, fontweight='bold')

    colors = ['#1f77b4' if c != predicted_class else '#2ca02c' for c in CLASSES]
    bars = ax2.barh(CLASSES, confidences, color=colors)
    ax2.set_xlim(0, 100)
    ax2.set_xlabel("Confidence (%)")
    ax2.set_title(f"Prediction: {predicted_class.upper()} ({top_conf.item()*100:.1f}%)\nLatency: {inference_time:.1f} ms",
                  fontsize=12, fontweight='bold', color='navy')

    for bar in bars:
        width = bar.get_width()
        ax2.text(width + 1, bar.get_y() + bar.get_height()/2, f"{width:.1f}%", va='center', ha='left', fontsize=9)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_prediction()
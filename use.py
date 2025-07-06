import torch
from torchvision import models, transforms
from PIL import Image
import os
import tkinter as tk
from tkinter import filedialog

# Disable tkinter root window
root = tk.Tk()
root.withdraw()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load class names from the training folder
CLASS_NAMES = sorted(os.listdir("tomato_diseases"))

# Define the image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x if x.shape[0] == 3 else x.expand(3, -1, -1)),  # Ensure 3 channels
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load the model
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load("tomato_disease_model.pth", map_location=device))
model = model.to(device)
model.eval()

# Prediction function
def predict_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)

        predicted_class = CLASS_NAMES[predicted.item()]
        print(f"\n🧠 Predicted Class: {predicted_class}")
        return predicted_class
    except Exception as e:
        print(f"⚠️ Error: {e}")

# Main
if __name__ == "__main__":
    print("📂 Select an image to predict:")
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
    )

    if file_path:
        print(f"📸 Selected: {file_path}")
        predict_image(file_path)
    else:
        print("❌ No image selected.")

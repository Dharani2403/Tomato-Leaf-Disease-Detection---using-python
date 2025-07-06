import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import torch
from torchvision import models, transforms
import os

# ------------------ Load Model and Setup ------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = sorted(os.listdir("tomato_diseases"))

transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x if x.shape[0] == 3 else x.expand(3, -1, -1)),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load("tomato_disease_model.pth", map_location=device))
model = model.to(device)
model.eval()

# ------------------ Prediction Function ------------------

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = CLASS_NAMES[predicted.item()]
    return predicted_class

# ------------------ GUI Setup ------------------

class TomatoDiseaseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🍅 Tomato Disease Detector")

        self.label = Label(root, text="Upload an image to classify", font=("Arial", 16))
        self.label.pack(pady=10)

        self.upload_btn = Button(root, text="📂 Upload Image", command=self.upload_image, font=("Arial", 14))
        self.upload_btn.pack(pady=10)

        self.image_label = Label(root)
        self.image_label.pack(pady=10)

        self.result_label = Label(root, text="", font=("Arial", 16, "bold"), fg="green")
        self.result_label.pack(pady=10)

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if file_path:
            img = Image.open(file_path)
            img.thumbnail((300, 300))  # Resize to fit the GUI
            img = ImageTk.PhotoImage(img)

            self.image_label.configure(image=img)
            self.image_label.image = img

            # Predict
            prediction = predict_image(file_path)
            self.result_label.configure(text=f"🧠 Predicted: {prediction}")

# ------------------ Run App ------------------

if __name__ == "__main__":
    root = tk.Tk()
    app = TomatoDiseaseApp(root)
    root.mainloop()

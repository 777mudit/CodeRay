import torch
import timm
from torchvision import transforms
from PIL import Image
import os

# -------------------------------
# 1️⃣ Device
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------------
# 2️⃣ Body parts mapping
# -------------------------------
# Make sure this matches your training dataset
body_part_to_idx = {
    "ELBOW": 0,
    "WRIST": 1,
    "SHOULDER": 2,
    # Add other body parts here exactly as in training
}

idx_to_body_part = {v: k for k, v in body_part_to_idx.items()}

# -------------------------------
# 3️⃣ Define Multi-Task Model
# -------------------------------
class MultiTaskSwin(torch.nn.Module):
    def __init__(self, num_body_parts):
        super().__init__()
        self.encoder = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=0)
        in_features = self.encoder.num_features
        self.body_part_head = torch.nn.Linear(in_features, num_body_parts)
        self.fracture_head = torch.nn.Linear(in_features, 2)
    
    def forward(self, x):
        features = self.encoder(x)
        body_part_out = self.body_part_head(features)
        fracture_out = self.fracture_head(features)
        return body_part_out, fracture_out

model = MultiTaskSwin(len(body_part_to_idx))
model.load_state_dict(torch.load("best_multi_task_swin.pth", map_location=device))
model.to(device)
model.eval()

# -------------------------------
# 4️⃣ Image Transform
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

# -------------------------------
# 5️⃣ Load and Predict
# -------------------------------
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        body_out, fracture_out = model(image)
        body_pred = torch.argmax(body_out, dim=1).item()
        fracture_pred = torch.argmax(fracture_out, dim=1).item()

    body_part = idx_to_body_part[body_pred]
    fracture = "Fracture" if fracture_pred == 1 else "Normal"
    return body_part, fracture

# -------------------------------
# 6️⃣ Test Example
# -------------------------------
test_image_path = r"C:\Users\preeti\Desktop\x-rayData\test_image.png"
body_part, fracture = predict_image(test_image_path)
print(f"Predicted Body Part: {body_part}")
print(f"Predicted Fracture: {fracture}")

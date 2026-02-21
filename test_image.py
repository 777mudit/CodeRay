import torch
import timm
from torchvision import transforms
from PIL import Image
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = timm.create_model(
    'swin_tiny_patch4_window7_224',
    pretrained=False,
    num_classes=2
)

model.load_state_dict(torch.load("best_swin_model.pth", map_location=device))
model.to(device)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load test image
image = Image.open("your_test_xray.png").convert("RGB")
image = transform(image).unsqueeze(0).to(device)

# Predict
with torch.no_grad():
    output = model(image)
    _, predicted = torch.max(output, 1)

if predicted.item() == 1:
    print("Prediction: Positive")
else:
    print("Prediction: Negative")

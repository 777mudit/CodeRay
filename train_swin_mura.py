import os
import torch
import torch.nn as nn
import timm
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

# -------------------------------
# 1️⃣ DEVICE (CPU now, GPU later auto)
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------------
# 2️⃣ CUSTOM MURA DATASET
# -------------------------------


class MURADataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform

        for patient in os.listdir(root_dir):
            patient_path = os.path.join(root_dir, patient)

            if os.path.isdir(patient_path):
                for study in os.listdir(patient_path):
                    study_path = os.path.join(patient_path, study)

                    if "positive" in study:
                        label = 1
                    elif "negative" in study:
                        label = 0
                    else:
                        continue

                    for img in os.listdir(study_path):
                        if img.endswith(".png"):
                            self.samples.append(
                                (os.path.join(study_path, img), label)
                            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# -------------------------------
# 3️⃣ TRANSFORMS
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------------
# 4️⃣ LOAD DATASET
# -------------------------------
DATA_DIR = r"C:\Users\preeti\Desktop\x-rayData\muramskxrays\MURA-v1.1\MURA-v1.1\clahe_train\XR_ELBOW"

full_dataset = MURADataset(DATA_DIR, transform=transform)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

print("Total images:", len(full_dataset))
print("Train:", train_size)
print("Validation:", val_size)

# -------------------------------
# 5️⃣ LOAD SWIN TRANSFORMER
# -------------------------------
model = timm.create_model(
    'swin_tiny_patch4_window7_224',
    pretrained=True,
    num_classes=2   # ✅ THIS is the correct way
)


model = model.to(device)

# -------------------------------
# 6️⃣ LOSS + OPTIMIZER
# -------------------------------
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-5,
    weight_decay=0.05
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=10
)

# -------------------------------
# 7️⃣ TRAINING LOOP
# -------------------------------
EPOCHS = 10
best_val_acc = 0

for epoch in range(EPOCHS):

    # ---- Training ----
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = 100 * correct / total

    # ---- Validation ----
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = 100 * val_correct / val_total
    scheduler.step()

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    print(f"Train Loss: {train_loss / len(train_loader):.4f}")
    print(f"Train Acc: {train_acc:.2f}%")
    print(f"Val Acc: {val_acc:.2f}%")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_swin_model.pth")
        print("Best Model Saved!")

print("\nTraining Complete.")

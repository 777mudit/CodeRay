import os
import torch
import torch.nn as nn
import timm
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

# -------------------------------
# 1️⃣ DEVICE
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------------
# 2️⃣ CUSTOM MULTI-TASK DATASET
# -------------------------------
class XRayDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        self.body_part_to_idx = {}
        self.num_body_parts = 0

        # Scan folders
        for idx, body_part in enumerate(sorted(os.listdir(root_dir))):
            body_part_path = os.path.join(root_dir, body_part)
            if os.path.isdir(body_part_path):
                self.body_part_to_idx[body_part] = idx
                self.num_body_parts += 1

                for fracture_label in os.listdir(body_part_path):
                    label_path = os.path.join(body_part_path, fracture_label)
                    if os.path.isdir(label_path):
                        fracture = 1 if fracture_label.lower() == "fracture" else 0
                        for img_file in os.listdir(label_path):
                            if img_file.endswith(".png") or img_file.endswith(".jpg"):
                                self.samples.append(
                                    (os.path.join(label_path, img_file), idx, fracture)
                                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, body_part_label, fracture_label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, body_part_label, fracture_label

# -------------------------------
# 3️⃣ TRANSFORMS
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

# -------------------------------
# 4️⃣ LOAD DATASET
# -------------------------------
DATA_DIR = r"C:\Users\preeti\Desktop\x-rayData\muramskxrays\MURA-v1.1\MURA-v1.1\clahe_train"

full_dataset = XRayDataset(DATA_DIR, transform=transform)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

print("Total images:", len(full_dataset))
print("Train:", train_size)
print("Validation:", val_size)
print("Body parts detected:", full_dataset.num_body_parts)

# -------------------------------
# 5️⃣ MULTI-TASK MODEL
# -------------------------------
class MultiTaskSwin(nn.Module):
    def __init__(self, num_body_parts):
        super().__init__()
        self.encoder = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=0) # remove classifier
        in_features = self.encoder.num_features
        # Two heads
        self.body_part_head = nn.Linear(in_features, num_body_parts)
        self.fracture_head = nn.Linear(in_features, 2)
    
    def forward(self, x):
        features = self.encoder(x)
        body_part_out = self.body_part_head(features)
        fracture_out = self.fracture_head(features)
        return body_part_out, fracture_out

model = MultiTaskSwin(full_dataset.num_body_parts).to(device)

# -------------------------------
# 6️⃣ LOSS + OPTIMIZER
# -------------------------------
criterion_body = nn.CrossEntropyLoss()
criterion_fracture = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.05)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# -------------------------------
# 7️⃣ TRAINING LOOP
# -------------------------------
EPOCHS = 10
best_val_acc = 0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    correct_body = 0
    correct_fracture = 0
    total = 0

    for images, body_labels, fracture_labels in train_loader:
        images = images.to(device)
        body_labels = body_labels.to(device)
        fracture_labels = fracture_labels.to(device)

        optimizer.zero_grad()
        out_body, out_fracture = model(images)
        loss = criterion_body(out_body, body_labels) + criterion_fracture(out_fracture, fracture_labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, pred_body = torch.max(out_body, 1)
        _, pred_fracture = torch.max(out_fracture, 1)
        total += body_labels.size(0)
        correct_body += (pred_body == body_labels).sum().item()
        correct_fracture += (pred_fracture == fracture_labels).sum().item()

    train_acc_body = 100 * correct_body / total
    train_acc_fracture = 100 * correct_fracture / total

    # Validation
    model.eval()
    val_correct_body = 0
    val_correct_fracture = 0
    val_total = 0

    with torch.no_grad():
        for images, body_labels, fracture_labels in val_loader:
            images = images.to(device)
            body_labels = body_labels.to(device)
            fracture_labels = fracture_labels.to(device)

            out_body, out_fracture = model(images)
            _, pred_body = torch.max(out_body, 1)
            _, pred_fracture = torch.max(out_fracture, 1)
            val_total += body_labels.size(0)
            val_correct_body += (pred_body == body_labels).sum().item()
            val_correct_fracture += (pred_fracture == fracture_labels).sum().item()

    val_acc_body = 100 * val_correct_body / val_total
    val_acc_fracture = 100 * val_correct_fracture / val_total
    scheduler.step()

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {train_loss/len(train_loader):.4f}")
    print(f"Train Body Acc: {train_acc_body:.2f}%, Train Fracture Acc: {train_acc_fracture:.2f}%")
    print(f"Val Body Acc: {val_acc_body:.2f}%, Val Fracture Acc: {val_acc_fracture:.2f}%")

    # Save best model (based on fracture detection accuracy)
    if val_acc_fracture > best_val_acc:
        best_val_acc = val_acc_fracture
        torch.save(model.state_dict(), "best_multi_task_swin.pth")
        print("Best model saved!")

print("\nTraining complete.")

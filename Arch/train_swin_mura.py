# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: 'Python (ray) '
#     language: python
#     name: ray_kernel
# ---

# %%
import os
import torch
import torch.nn as nn
import timm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchmetrics.classification import BinaryCohenKappa
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
import seaborn as sns
from torch.amp import autocast, GradScaler

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# %%
DATA_DIR = "/home/Aurelius/Documents/AdoVs/coderay/rays/XR_ELBOW"
BATCH_SIZE = 16
EPOCHS = 20
LR = 2e-5
PATIENCE = 5

# %%
# ---------------- Patient-wise dataset ----------------
class MURADataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# %%
def collect_patient_samples(root_dir):
    patients = {}

    for patient in os.listdir(root_dir):
        patient_path = os.path.join(root_dir, patient)
        if not os.path.isdir(patient_path):
            continue

        patient_samples = []

        for study in os.listdir(patient_path):
            study_path = os.path.join(patient_path, study)

            if "positive" in study.lower():
                label = 1
            elif "negative" in study.lower():
                label = 0
            else:
                continue

            for img in os.listdir(study_path):
                if img.lower().endswith((".png", ".jpg", ".jpeg")):
                    patient_samples.append(
                        (os.path.join(study_path, img), label)
                    )

        if patient_samples:
            patients[patient] = patient_samples

    return patients

# %%
patients = collect_patient_samples(DATA_DIR)
patient_ids = list(patients.keys())
np.random.shuffle(patient_ids)

# %%
split = int(0.8 * len(patient_ids))
train_ids = patient_ids[:split]
val_ids = patient_ids[split:]

# %%
train_samples = [s for pid in train_ids for s in patients[pid]]
val_samples = [s for pid in val_ids for s in patients[pid]]

# %%
print("Train images:", len(train_samples))
print("Val images:", len(val_samples))

# %%
# ---------------- Transforms ----------------
train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# %%
val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# %%
train_dataset = MURADataset(train_samples, train_transform)
val_dataset = MURADataset(val_samples, val_transform)

# %%
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# %%
# ---------------- Class weights ----------------
labels = [label for _, label in train_samples]
class_counts = np.bincount(labels)
weights = 1. / class_counts
class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

# %%
# ---------------- Model ----------------
model = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=2)
model.to(device)

# %%
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
kappa_metric = BinaryCohenKappa().to(device)

# %%
scaler = GradScaler('cuda')

# %%
best_loss = float('inf')
early_counter = 0

# %%
for epoch in range(EPOCHS):

    # ---------- TRAIN ----------
    model.train()
    train_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        with autocast(device_type="cuda"):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()

    # ---------- VALIDATE ----------
    model.eval()
    val_loss = 0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            with autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            val_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)[:,1]
            _, preds = torch.max(outputs,1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())

    train_loss /= len(train_loader)
    val_loss /= len(val_loader)

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)

    kappa = kappa_metric(preds.to(device), labels.to(device)).item()
    kappa_metric.reset()

    y_true = labels.numpy()
    y_pred = preds.numpy()
    y_prob = torch.cat(all_probs).numpy()

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp/(tp+fn+1e-8)
    specificity = tn/(tn+fp+1e-8)
    precision = precision_score(y_true,y_pred)
    f1 = f1_score(y_true,y_pred)

    fpr,tpr,_ = roc_curve(y_true,y_prob)
    roc_auc = auc(fpr,tpr)

    print(f"\nEpoch {epoch+1}")
    print("Train Loss:",train_loss,"Val Loss:",val_loss,"Kappa:",kappa)
    print("Sensitivity:",sensitivity,"Specificity:",specificity,"F1:",f1,"AUC:",roc_auc)

    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(),"best_research_model.pth")
        print("Saved best model")
        early_counter = 0
    else:
        early_counter += 1

    if early_counter >= PATIENCE:
        print("Early stopping triggered")
        break

    scheduler.step()

# %%
# ---------- FINAL CONFUSION MATRIX ----------
plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Final Confusion Matrix")
plt.show()

# %%

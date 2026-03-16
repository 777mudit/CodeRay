import torch
import timm
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
from torchvision import transforms

# 1. Setup Device & Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 50 # SSL usually needs more epochs

# 2. Simple Unlabeled Dataset
class UnlabeledXrayDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = []
        for root, _, files in os.walk(root_dir):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(root, f))
        self.transform = transform

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform: img = self.transform(img)
        return img

# 3. MAE Specific Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = UnlabeledXrayDataset(
    r"C:\Users\preeti\Desktop\x-rayData\Xray\Bone -Fracture\Bone Fracture\Augmented\Comminuted Bone Fracture",
    transform=transform
)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# 4. Create the MAE Model (using a Vision Transformer backbone compatible with Swin logic)
# Note: timm has built-in MAE support for ViT which you can transfer to Swin later
model = timm.create_model('vit_base_patch16_224.mae', pretrained=False).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = torch.nn.MSELoss() # We want pixels to match

# 5. Training Loop
model.train()

for epoch in range(EPOCHS):
    total_loss = 0
    
    for batch in loader:
        images = batch.to(device)

        loss = model(images)[0].mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Reconstruction Loss: {total_loss/len(loader)}")

# 6. Save the 'Backbone' (The part that understands bones)
torch.save(model.state_dict(), "mae_bone_features.pth")



# Output fill will be mae_bone_features.pth which contains the learned weights of the MAE model after training on the unlabeled X-ray dataset. 
# This file can be loaded later to initialize a Swin Transformer model for fine-tuning on a specific task, such as classification or segmentation, 
# using the learned bone features.
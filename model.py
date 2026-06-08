import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os
import numpy as np
from tqdm import tqdm

DATA_DIR = r'D:\Python\Rail\rsdds-dataset_link\dataset_final' 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 50
BATCH_SIZE = 4 
LR = 1e-4 
IMAGE_SIZE = 256 

class RSDDS_Final_Dataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.split_dir = os.path.join(root_dir, split)
        self.img_dir = os.path.join(self.split_dir, 'images')
        self.mask_dir = os.path.join(self.split_dir, 'masks')
        self.transform = transform
        
        self.ids = [os.path.splitext(f)[0] for f in os.listdir(self.img_dir) 
                    if f.lower().endswith(('.jpg', '.png', '.bmp'))]
        
        print(f"--- [Tập {split}] Đã tìm thấy: {len(self.ids)} mẫu ---")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        file_id = self.ids[i]
        
        img_file = next(f for f in os.listdir(self.img_dir) if f.startswith(file_id))
        image = cv2.imread(os.path.join(self.img_dir, img_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask_file = next(f for f in os.listdir(self.mask_dir) if f.startswith(file_id))
        mask = cv2.imread(os.path.join(self.mask_dir, mask_file), cv2.IMREAD_GRAYSCALE)
        
        mask = (mask > 0).astype(np.float32)

        if self.transform:
            sample = self.transform(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        return image, mask.unsqueeze(0)

train_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

def train_model():
    train_dataset = RSDDS_Final_Dataset(DATA_DIR, split='train', transform=train_transform)
    val_dataset = RSDDS_Final_Dataset(DATA_DIR, split='val', transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = smp.Unet(
        encoder_name="resnet34", 
        encoder_weights="imagenet", 
        in_channels=3, 
        classes=1, 
        activation=None 
    ).to(DEVICE)

    criterion = lambda out, target: 0.5 * nn.BCEWithLogitsLoss()(out, target) + 0.5 * smp.losses.DiceLoss(mode='binary', from_logits=True)(out, target)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for imgs, masks in pbar:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                v_loss = criterion(model(imgs), masks)
                val_loss += v_loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Summary: Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {avg_val_loss:.4f}")
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), "best_rail_unet.pth")
            print(">>> Cập nhật Best Model!")

if __name__ == '__main__':
    train_model()

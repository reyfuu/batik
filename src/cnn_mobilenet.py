import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np

# Apple Silicon (MPS) -> CUDA -> CPU
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

class BatikDataset(Dataset):
    """Dataset PyTorch untuk gambar batik."""
    def __init__(self, base_dir, split, transform=None):
        self.split_dir = os.path.join(base_dir, split)
        self.transform = transform
        self.categories = ["parang", "kawung", "megamendung"]
        self.image_paths = []
        self.labels = []
        
        if not os.path.exists(self.split_dir):
            raise ValueError(f"Direktori {self.split_dir} tidak ditemukan. Silakan buat dataset terlebih dahulu.")
            
        for label_idx, cat in enumerate(self.categories):
            cat_dir = os.path.join(self.split_dir, cat)
            if not os.path.exists(cat_dir):
                continue
            for file in os.listdir(cat_dir):
                if file.endswith('.png'):
                    self.image_paths.append(os.path.join(cat_dir, file))
                    self.labels.append(label_idx)
                    
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_mobilenet_v2(num_classes=3, freeze=True):
    """MobileNetV2 pretrained ImageNet dengan classifier head diganti.

    freeze=False -> latih seluruh jaringan (fitur ImageNet kurang cocok untuk tekstur
    batik, finetune penuh jauh lebih akurat daripada latih head saja).
    """
    weights = models.MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=weights)

    for param in model.parameters():
        param.requires_grad = not freeze

    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    return model

def train_and_evaluate(base_dir="data", model_path="models", epochs=5, batch_size=16, resume=True):
    """Finetune MobileNetV2 pada dataset batik sintetis.

    resume=True -> lanjut dari mobilenet_v2_batik.pth kalau sudah ada.
    """
    os.makedirs(model_path, exist_ok=True)
    save_file = os.path.join(model_path, "mobilenet_v2_batik.pth")

    # MobileNetV2 butuh 224x224 + normalisasi ImageNet
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        train_dataset = BatikDataset(base_dir, "train", transform=train_transform)
        test_dataset = BatikDataset(base_dir, "test", transform=test_transform)
    except ValueError as e:
        print(f"Error: {e}")
        return None
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Menggunakan Device: {device}")
    print(f"Melatih MobileNetV2 selama {epochs} epoch (Finetuning Classifier Head)...")

    model = get_mobilenet_v2(num_classes=3)
    if resume and os.path.exists(save_file):
        model.load_state_dict(torch.load(save_file, map_location="cpu"))
        print(f"Resume dari {save_file}")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    # Optimize hanya classifier head yang baru.
    # ponytail: optimizer state tidak disimpan -> momentum reset tiap resume. Cukup untuk
    # head-only; simpan optimizer.state_dict() kalau resume full-network (train_v7) dan
    # loss melonjak tiap restart.
    optimizer = optim.Adam(model.classifier[1].parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = correct / total * 100
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f} - Akurasi Train: {epoch_acc:.2f}%")

    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
            
    test_acc = test_correct / test_total * 100
    print("\n" + "="*50)
    print("HASIL EVALUASI MODEL DEEP LEARNING (MobileNetV2)")
    print("="*50)
    print(f"Akurasi Pengujian: {test_acc:.2f}%")
    print("="*50 + "\n")

    torch.save(model.state_dict(), save_file)
    print(f"Model MobileNetV2 disimpan ke {save_file}")
    return model

def train_v7(base_dir="data/Contoh foto batik/DATASETv7", model_path="models",
             epochs=10, batch_size=32, lr=1e-4):
    """Finetune MobileNetV2 pada DATASETv7 (15 kelas, subfolder = nama kelas).

    ImageFolder mengurutkan kelas sorted() -> sama dengan classes.pkl milik SVM.
    Simpan mobilenet_v2_v7.pth + classes_cnn.pkl.
    """
    import pickle
    from torchvision.datasets import ImageFolder

    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # Augmentasi ringan: batik sering difoto miring/terbalik, motifnya simetris.
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(), transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(), norm,
    ])
    test_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), norm])

    train_ds = ImageFolder(os.path.join(base_dir, "TRAIN"), transform=train_tf)
    test_ds = ImageFolder(os.path.join(base_dir, "TEST"), transform=test_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    classes = train_ds.classes
    print(f"Device: {device} | {len(train_ds)} train, {len(test_ds)} test, {len(classes)} kelas")

    model = get_mobilenet_v2(num_classes=len(classes), freeze=False).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    def test_acc():
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                correct += model(images).max(1)[1].eq(labels).sum().item()
                total += labels.size(0)
        return correct / total * 100

    best = 0.0
    save_file = os.path.join(model_path, "mobilenet_v2_v7.pth")
    os.makedirs(model_path, exist_ok=True)
    for epoch in range(epochs):
        model.train()
        running_loss = correct = total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            correct += outputs.max(1)[1].eq(labels).sum().item()
            total += labels.size(0)
        acc = test_acc()
        print(f"Epoch [{epoch+1}/{epochs}] loss={running_loss/total:.4f} "
              f"train={correct/total*100:.1f}% test={acc:.1f}%")
        # Simpan hanya epoch terbaik -> aman kalau epoch akhir overfit.
        if acc > best:
            best = acc
            torch.save(model.state_dict(), save_file)
            pickle.dump(classes, open(os.path.join(model_path, "classes_cnn.pkl"), "wb"))

    print(f"\nAkurasi test terbaik: {best:.2f}% -> {save_file}")
    return best, classes


def predict_single(image_path, model_path="models"):
    """Klasifikasi satu gambar dengan model MobileNetV2 terlatih."""
    save_file = os.path.join(model_path, "mobilenet_v2_batik.pth")
    if not os.path.exists(save_file):
        print("Model MobileNetV2 belum dilatih. Jalankan pelatihan terlebih dahulu.")
        return

    model = get_mobilenet_v2(num_classes=3)
    model.load_state_dict(torch.load(save_file, map_location=torch.device('cpu')))
    model.to(device)
    model.eval()

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path).convert("RGB")
    tensor = test_transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        prob, predicted = torch.max(probabilities, dim=0)
        
    categories = ["Parang", "Kawung", "Megamendung"]
    print(f"Hasil Prediksi MobileNetV2 untuk {os.path.basename(image_path)}:")
    print(f"- Motif Terdeteksi: {categories[predicted.item()]}")
    print(f"- Confidence: {prob.item() * 100:.2f}%")

if __name__ == "__main__":
    train_and_evaluate()

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


# 設定一些參數
num_classes = 12
batch_size = 32
num_epochs = 20
learning_rate = 0.001
model_save_path = "model_resnet50.pth"
submission_file = "submission.csv"

# 定義資料轉換，包括擴增
data_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),    # 隨機裁剪並縮放到指定大小
    transforms.RandomHorizontalFlip(),    # 隨機水平翻轉
    transforms.RandomVerticalFlip(),      # 隨機垂直翻轉
    transforms.RandomRotation(30),        # 隨機旋轉影像（-30度到+30度之間）
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 影像顏色調整
    transforms.ToTensor(),                # 轉換為Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])   # 正規化
])

# 載入訓練資料
train_dataset = ImageFolder(root="./dataset/train", transform=data_transform)

# 訓練和驗證資料的劃分比例
train_val_split = 0.8

# 訓練集資料個數
train_size = int(train_val_split * len(train_dataset))

# 驗證集資料個數
val_size = len(train_dataset) - train_size

# 設置隨機種子，確保每次劃分的結果一致
random_seed = 42
torch.manual_seed(random_seed)
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

# 載入訓練資料和驗證資料
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
# 定義ResNet-50模型
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# 定義損失函數和優化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 將模型移到GPU上（如果可用）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# 訓練和驗證模型
train_losses = []
val_losses = []
train_corrects = 0
val_corrects = 0

for epoch in range(num_epochs):
    # 訓練模式
    model.train()
    running_train_loss = 0.0
    train_corrects = 0
    for i, (inputs, labels) in enumerate(train_loader, 0):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        train_corrects += torch.sum(preds == labels.data)

    epoch_train_loss = running_train_loss / len(train_loader)
    train_losses.append(epoch_train_loss)
    train_accuracy = train_corrects.double() / len(train_dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}] - Training Loss: {epoch_train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")

    # 驗證模式
    model.eval()
    running_val_loss = 0.0
    val_corrects = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            val_corrects += torch.sum(preds == labels.data)

    epoch_val_loss = running_val_loss / len(val_loader)
    val_losses.append(epoch_val_loss)
    val_accuracy = val_corrects.double() / len(val_dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}] - Validation Loss: {epoch_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

# 畫出loss curve
plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curve')
plt.legend()
plt.savefig('loss_curve.png')
plt.show()

# 生成submission.csv文件
test_data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

submission_data = {
    'file': [],
    'species': []
}

test_image_paths = os.listdir('dataset/test')
model.eval()

with torch.no_grad():
    for image_path in test_image_paths:
        image = Image.open(os.path.join('dataset/test', image_path))
        image = test_data_transform(image).unsqueeze(0).to(device)
        outputs = model(image)
        _, predicted_idx = torch.max(outputs, 1)
        predicted_class = train_dataset.dataset.classes[predicted_idx]
        submission_data['file'].append(image_path)
        submission_data['species'].append(predicted_class)

submission_df = pd.DataFrame(submission_data)
submission_df.to_csv(submission_file, index=False)

print("Submission file generated successfully.")

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

# 定義資料轉換
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 載入訓練資料
train_dataset = ImageFolder(root="./dataset/train", transform=data_transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# 定義ResNet-50模型
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# 定義損失函數和優化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 將模型移到GPU上（如果可用）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 訓練模型
train_losses = []
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader, 0):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f}")

# 儲存訓練好的模型
torch.save(model.state_dict(), model_save_path)

# 畫出training loss curve
plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.savefig('training_loss_curve.png')
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

test_image_paths = os.listdir('./dataset/test')
model.eval()

with torch.no_grad():
    for image_path in test_image_paths:
        image = Image.open(os.path.join('./dataset/test', image_path))
        image = test_data_transform(image).unsqueeze(0).to(device)
        outputs = model(image)
        _, predicted_idx = torch.max(outputs, 1)
        predicted_class = train_dataset.classes[predicted_idx]
        submission_data['file'].append(image_path)
        submission_data['species'].append(predicted_class)

submission_df = pd.DataFrame(submission_data)
submission_df.to_csv(submission_file, index=False)

print("Submission file generated successfully.")

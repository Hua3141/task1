# task_1/train.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import json
import os

# ==================== 配置 ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEATURE_DIM = 31  # 6 (mean/std) + 24 (hist) + 1 (edge) = 31
NUM_CLASSES = None
BATCH_SIZE = 32
EPOCHS = 100
LR = 0.001

# ==================== 定义 MLP 模型 ====================
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)

# ==================== 加载特征数据 ====================
print("📦 加载训练特征...")
X_train = np.load("train_features.npy")
y_train = np.load("train_labels.npy")
with open("train_class_to_idx.json", "r") as f:
    class_to_idx = json.load(f)

NUM_CLASSES = len(class_to_idx)
print(f"  特征维度: {X_train.shape[1]}, 样本数: {len(X_train)}, 类别数: {NUM_CLASSES}")

# 转为 PyTorch Tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

# 创建 DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ==================== 训练 ====================
model = SimpleMLP(FEATURE_DIM, NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

print("\n🚀 开始训练 MLP...")
best_acc = 0.0
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += y_batch.size(0)
        correct += predicted.eq(y_batch).sum().item()
    
    train_acc = 100.0 * correct / total
    avg_loss = total_loss / len(train_loader)
    
    # 验证（这里用训练集准确率，因无验证集划分）
    if train_acc > best_acc:
        best_acc = train_acc
        torch.save(model.state_dict(), "checkpoints/mlp_best.pth")
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Acc: {train_acc:.2f}%")

print(f"\n✅ 训练完成！最佳准确率: {best_acc:.2f}%")
print("模型已保存至: checkpoints/mlp_best.pth")

# 保存类别映射（供 test.py 使用）
os.makedirs("checkpoints", exist_ok=True)
with open("checkpoints/class_to_idx.json", "w") as f:
    json.dump(class_to_idx, f)
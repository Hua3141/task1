# task_1/test.py
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import json

# ==================== 模型定义（与 train.py 一致）====================
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

# ==================== 加载数据 ====================
X_test = np.load("test_features.npy")
filenames = np.load("test_filenames.npy")
with open("checkpoints/class_to_idx.json", "r") as f:
    class_to_idx = json.load(f)
idx_to_class = {v: k for k, v in class_to_idx.items()}

# ==================== 加载模型 ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleMLP(input_dim=31, num_classes=len(class_to_idx))
model.load_state_dict(torch.load("checkpoints/mlp_best.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ==================== 预测 ====================
X_test = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
with torch.no_grad():
    outputs = model(X_test)
    preds = outputs.argmax(dim=1).cpu().numpy()

# 映射为类别名
pred_classes = [idx_to_class[idx] for idx in preds]

# 保存 CSV
df = pd.DataFrame({"filename": filenames, "species": pred_classes})
df.to_csv("predictions.csv", index=False)
print("✅ 预测完成！结果已保存至: predictions.csv")
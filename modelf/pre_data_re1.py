# task_1/model/pre_data.py

from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# 自动构建数据路径（相对于当前脚本位置）
CURRENT_DIR = Path(__file__).parent  # 即 task_1/model/
DATA_DIR = CURRENT_DIR.parent / "data" / "train"  # → task_1/data/train

# 验证路径是否存在
if not DATA_DIR.exists():
    raise FileNotFoundError(f"数据目录未找到: {DATA_DIR}")

# 定义变换
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),          # 稍大旋转
    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        hue=0.1
    ),
    transforms.RandomAffine(  # 添加仿射变换（平移、缩放、剪切）
        degrees=0,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1),
        shear=5
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.489, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1))  # 随机擦除（强正则化）
])

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载完整数据集（使用训练变换，后续再划分）
full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=train_transform)

# 划分训练集和验证集（例如 80% : 20%）
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

# 注意：val_subset 仍使用 train_transform！为避免数据增强影响验证，建议重建 val_dataset
# 更严谨的做法：单独加载验证集用 val_transform
val_dataset = datasets.ImageFolder(root=DATA_DIR, transform=val_transform)
_, val_subset = random_split(val_dataset, [train_size, val_size])

# 创建 DataLoader
train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, num_workers=4)

# 可选：保存类别映射供后续推理使用
class_to_idx = full_dataset.class_to_idx
idx_to_class = {v: k for k, v in class_to_idx.items()}

if __name__ == "__main__":
    print(f"类别索引映射: {class_to_idx}")
    print(f"训练集大小: {len(train_subset)}, 验证集大小: {len(val_subset)}")
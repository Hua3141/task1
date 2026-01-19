# task_1/model/model.py

import torch
import torch.nn as nn
import torchvision.models as models
from pathlib import Path

class ImageClassifier(nn.Module):
    def __init__(self, num_classes: int, backbone: str = "resnet18", pretrained: bool = True):
        """
        图像分类模型封装
        
        Args:
            num_classes (int): 分类类别数
            backbone (str): 骨干网络名称，支持 "resnet18", "resnet34", "resnet50" 等
            pretrained (bool): 是否使用 ImageNet 预训练权重
        """
        super(ImageClassifier, self).__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone

        # 加载预训练模型
        if backbone == "resnet18":
            self.backbone = models.resnet18(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
        elif backbone == "resnet34":
            self.backbone = models.resnet34(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
        elif backbone == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
        else:
            raise ValueError(f"不支持的骨干网络: {backbone}. 请使用 'resnet18', 'resnet34' 或 'resnet50'.")

    def forward(self, x):
        return self.backbone(x)

    def save(self, save_path: str):
        """保存模型权重"""
        torch.save(self.state_dict(), save_path)
        print(f"模型已保存至: {save_path}")

    def load(self, load_path: str, map_location=None):
        """加载模型权重"""
        self.load_state_dict(torch.load(load_path, map_location=map_location))
        print(f"模型已从 {load_path} 加载")

    @staticmethod
    def get_device():
        """自动选择设备"""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================== 快速测试用例 ====================
if __name__ == "__main__":
    # 测试模型能否正常构建和前向传播
    device = ImageClassifier.get_device()
    print(f"使用设备: {device}")

    # 假设你的数据有 5 个类别（请根据实际情况修改）
    model = ImageClassifier(num_classes=5, backbone="resnet18").to(device)

    # 构造一个假输入 [batch=2, channels=3, H=224, W=224]
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    output = model(dummy_input)
    print(f"输入形状: {dummy_input.shape} → 输出形状: {output.shape}")

    # 保存测试
    test_save_path = Path(__file__).parent / "test_model.pth"
    model.save(str(test_save_path))
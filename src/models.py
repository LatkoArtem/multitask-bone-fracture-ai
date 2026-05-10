import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torchvision.models import resnet50, ResNet50_Weights


class MultiTaskClassifier(nn.Module):
    """ResNet50 backbone with two heads: bone type (multi-label) and fracture (binary)."""

    def __init__(self, num_bone_classes: int = 5):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        in_features = resnet.fc.in_features  # 2048
        self.bone_head = nn.Linear(in_features, num_bone_classes)
        self.fracture_head = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor):
        f = torch.flatten(self.backbone(x), 1)
        return self.bone_head(f), self.fracture_head(f)


class FractureSegmenter(nn.Module):
    """U-Net with pretrained ResNet50 encoder (ImageNet weights)."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__()
        self.model = smp.Unet(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=out_channels,
            activation=None,  # raw logits — sigmoid applied in loss / inference
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

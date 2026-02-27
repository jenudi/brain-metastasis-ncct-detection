import torchvision.models as models
import torch.nn as nn


def build_resnet18_binary(pretrained: bool = True, dropout_p: float = 0.3):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_p),
        nn.Linear(in_features, 1),  # logits
    )
    return model

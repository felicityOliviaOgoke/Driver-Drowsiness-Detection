from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn


def build_model(num_classes=2, freeze=True):
    model = resnet50(weights=ResNet50_Weights.DEFAULT)

    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

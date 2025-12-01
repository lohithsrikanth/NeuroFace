import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

#6,964,106 parameters
class DenseNet121(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        # Replace classifier if custom num_classes
        if num_classes != 1000:
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

#12,501,130
class DenseNet169(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.model = models.densenet169(weights=models.DenseNet169_Weights.IMAGENET1K_V1)
        if num_classes != 1000:
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

#18,112,138 parameters
class DenseNet201(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.model = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
        if num_classes != 1000:
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    densenet121 = DenseNet121(num_classes=10)
    densenet169 = DenseNet169(num_classes=10)
    densenet201 = DenseNet201(num_classes=10)

    print(f"DenseNet121 parameters: {count_parameters(densenet121)}")
    print(f"DenseNet169 parameters: {count_parameters(densenet169)}")
    print(f"DenseNet201 parameters: {count_parameters(densenet201)}")
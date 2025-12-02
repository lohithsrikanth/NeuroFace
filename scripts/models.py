import torch
import torch.nn as nn
from torchvision import models

def get_resnet(model_name: str = "resnet18", weights: str = "IMAGENET1K_V1", num_classes: int = 3) -> nn.Module:
    """
    Returns a ResNet-18 or ResNet-50 model adapted for custom classification.
    """
    if model_name == "resnet18":
        model = models.resnet18(weights=getattr(models.ResNet18_Weights, weights))
    elif model_name == "resnet50":
        model = models.resnet50(weights=getattr(models.ResNet50_Weights, weights))
    elif model_name == "resnet101":
        model = models.resnet101(weights=getattr(models.ResNet101_Weights, weights))
    elif model_name == "resnet152":
        model = models.resnet152(weights=getattr(models.ResNet152_Weights, weights))
    else:
        raise ValueError(f"Unsupported model name: {model_name}. Must be either 'resnet18', 'resnet50', 'resnet101', 'resnet152'.")

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def get_densenet(model_name: str = "densenet121", weights: str = "IMAGENET1K_V1", num_classes: int = 3) -> nn.Module:
    """
    Returns a DenseNet-121, DenseNet-169, or DenseNet-201 model adapted for custom classification.
    """
    if model_name == "densenet121":
        model = models.densenet121(weights=getattr(models.DenseNet121_Weights, weights))
    elif model_name == "densenet161":
        model = models.densenet169(weights=getattr(models.DenseNet161_Weights, weights))
    elif model_name == "densenet169":
        model = models.densenet169(weights=getattr(models.DenseNet169_Weights, weights))
    elif model_name == "densenet201":
        model = models.densenet201(weights=getattr(models.DenseNet201_Weights, weights))
    else:
        raise ValueError(f"Unsupported model name: {model_name}. Must be one of 'densenet121', 'densenet161', 'densenet169', or 'densenet201'.")

    if num_classes != 1000:
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)

    return model
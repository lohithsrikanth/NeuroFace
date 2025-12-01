import torch
import torch.nn as nn
from torchvision import models

def get_resnet(model_name:str = "resnet18", pretrained:bool = True, num_classes:int = 3) -> nn.Module:
  """
  Returns a ResNet-18 or ResNet-50 model adapted for 3-class classification
  """
  if model_name == "resnet18":
    model = models.resnet18(pretrained=pretrained)
  elif model_name == "resnet50":
    model = models.resnet50(pretrained=pretrained)
  else:
    raise ValueError("Unsupported model name: {model_name}. Must be either 'resnet18' or 'resnet50'.")
  in_features = model.fc.in_features
  model.fc = nn.Linear(in_features, num_classes)
  return model

def get_densenet(model_name: str = "densenet121", pretrained: bool = True, num_classes: int = 3) -> nn.Module:
    """
    Returns a DenseNet-121, DenseNet-169, or DenseNet-201 model adapted for custom classification.
    """
    if model_name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
    elif model_name == "densenet169":
        model = models.densenet169(weights=models.DenseNet169_Weights.IMAGENET1K_V1 if pretrained else None)
    elif model_name == "densenet201":
        model = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1 if pretrained else None)
    else:
        raise ValueError(f"Unsupported model name: {model_name}. Must be one of 'densenet121', 'densenet169', or 'densenet201'.")

    if num_classes != 1000:
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)

    return model

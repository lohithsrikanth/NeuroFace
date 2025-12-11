import torch
import torch.nn as nn
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 3

def get_resnet(model_name:str = "restnet18", pretrained:bool = True):

  if model_name == "resnet18":
    model = models.resnet18(pretrained=pretrained)
  elif model_name == "resnet50":
    model = models.resnet50(pretrained=pretrained)
  elif model_name == "resnet101":
    model = models.resnet101(pretrained=pretrained)
  elif model_name == "resnet152":
    model = models.resnet152(pretrained=pretrained)
  elif model_name == "resnet34":
    model = models.resnet34(pretrained=pretrained)
  else:
    raise ValueError("Unsupported model name: {model_name}")

  in_features = model.fc.in_features
  model.fc = nn.Linear(in_features, NUM_CLASSES)
  return model

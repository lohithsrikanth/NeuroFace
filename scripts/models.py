import torch
import torch.nn as nn
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 3

def get_resnet(model_name:str = "resnet18", pretrained:bool = True):
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
  model.fc = nn.Linear(in_features, NUM_CLASSES)
  return model

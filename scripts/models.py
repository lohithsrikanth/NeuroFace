import torch
import torch.nn as nn
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 3

def get_restnet(model_name:str = "restnet18", pretrained:bool = True):
  """
  Returns a RestNet-18 or RestNet-50 model adapted for 3-class classification
  """
  if model_name == "restnet18":
    model = models.resnet18(pretrained=pretrained)
  elif model_name == "restnet50":
    model = models.resnet50(pretrained=pretrained)
  else:
    raise ValueError("Unsupported model name: {model_name}. Must be either 'restnet18' or 'restnet50'.")
  in_features = model.fc.in_features
  model.fc = nn.Linear(in_features, NUM_CLASSES)
  return model

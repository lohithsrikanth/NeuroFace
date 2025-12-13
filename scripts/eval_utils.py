import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

def evaluate(model, test_loader, device):
  model.eval()
  all_preds = []
  all_labels = []

  with torch.no_grad():
    for images, labels in test_loader:
      images = images.to(device)
      labels = labels.to(device)

      outputs = model(images)
      preds = torch.argmax(outputs, dim=1)

      all_preds.extend(preds.cpu().tolist())
      all_labels.extend(labels.cpu().tolist())

  acc = accuracy_score(all_labels, all_preds)
  f1 = f1_score(all_labels, all_preds, average='weighted')
  return acc, f1

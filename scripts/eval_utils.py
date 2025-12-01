import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

def evaluate(model, data_loader, device, criterion=None):
    """
    Evaluate the model on the given data loader.

    Args:
        model: The model to evaluate.
        data_loader: DataLoader for the evaluation data.
        device: Device to use for evaluation (CPU or GPU).
        criterion: Loss function (optional, for calculating loss).

    Returns:
        Tuple of (loss, accuracy, f1_score).
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            if criterion:
                loss = criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)

            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    f1 = f1_score(all_labels, all_preds, average="weighted")  # Use sklearn's f1_score

    if criterion:
        avg_loss = running_loss / len(data_loader.dataset)
        return avg_loss, accuracy, f1
    else:
        return None, accuracy, f1

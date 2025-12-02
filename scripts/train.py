import torch
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm, trange
from .dataset_utils import get_dataloaders
from .models import get_resnet, get_densenet
from .eval_utils import evaluate
import os
import torch.multiprocessing as mp
import json
import random


def train_one_epoch(model,data_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(data_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(data_loader.dataset)
    return epoch_loss

def train_model(model, train_loader, val_loader, test_loader, optimizer, criterion, num_epochs, device, model_name):
    """
    Function to train and evaluate a model.

    Args:
        model: The model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        test_loader: DataLoader for test data.
        optimizer: Optimizer for training.
        criterion: Loss function.
        num_epochs: Number of epochs to train.
        device: Device to use for training (CPU or GPU).
        model_name: Name of the model (used for saving).

    Returns:
        None
    """
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "train_f1": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_f1": []
    }

    best_val_f1 = 0.0
    best_state_dict = None

    for epoch in trange(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        train_loss, train_accuracy,train_f1 = evaluate(model, train_loader, device, criterion)  # Assuming evaluate supports metric selection

        # Validation phase
        val_loss, val_accuracy, val_f1 = evaluate(model, val_loader, device, criterion=criterion)

        # Save metrics to history
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["train_f1"].append(train_f1)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        history["val_f1"].append(val_f1)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Train F1: {train_f1:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Val F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state_dict = model.state_dict().copy()

    # Load best model and evaluate on test set
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    test_loss, test_acc, test_f1 = evaluate(model, test_loader, device)
    print(f"Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), f"models/{model_name}_neuroface_best_model.pth")
    print(f"Saved best model as models/{model_name}_neuroface_best_model.pth")

    # Save training history
    os.makedirs("histories", exist_ok=True)
    with open(f"histories/{model_name}_history.json", "w") as f:
        json.dump(history, f)
    print(f"Saved training history as histories/{model_name}_history.json")

def set_seed(seed: int):
    """
    Set the random seed for reproducibility.

    Args:
        seed (int): The seed value to set.

    Returns:
        None
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    # Set seed for reproducibility
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    # Load model data
    train_loader, val_loader, test_loader, _, classes = get_dataloaders(
        data_root="processed_frames",
        batch_size=32,
        num_workers=2 # Consistent num_workers with dataset_utils.py
    )
    #model_names = ["resnet18"]
    model_names = ["resnet18", "resnet50", "resnet101", "resnet152", "densenet121", "densenet161", "densenet169", "densenet201"]

    for model_name in model_names:
        print(f"Training model: {model_name}")
        if model_name in ["resnet18", "resnet50", "resnet101", "resnet152"]:
            model = get_resnet(model_name=model_name, num_classes=len(classes))
        else:
            model = get_densenet(model_name=model_name, num_classes=len(classes))
        
        model = model.to(device)

        """
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze only the final dense layers
        if model_name in ["resnet18", "resnet50"]:
            # ResNet's final classification layer is typically 'fc'
            for param in model.fc.parameters():
                param.requires_grad = True
            print(f"Unfrozen ResNet's 'fc' layer for {model_name}")
        else: # densenets
            # DenseNet's final classification layer is typically 'classifier'
            for param in model.classifier.parameters():
                param.requires_grad = True
            print(f"Unfrozen DenseNet's 'classifier' layer for {model_name}")
        """

        criterion = nn.CrossEntropyLoss()

        # Initialize the optimizer with only the trainable parameters
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

        train_model(model, train_loader, val_loader, test_loader, optimizer, criterion, num_epochs=20, device=device, model_name=model_name)
import torch
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.notebook import tqdm, trange

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

# Load model data
train_loader, val_loader, test_loader, _, classes = get_dataloaders(
    data_root="/content/drive/MyDrive/cs6073/NeuroFace/processed_frames",
    batch_size=32,
    num_workers=2 # Consistent num_workers with dataset_utils.py
)

model_name = "mlp" # Define model_name here
#model = get_restnet(model_name=model_name, pretrained=True)
model = MultiLayerPerceptron()
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

best_val_f1 = 0.0
best_state_dict = None

for epoch in trange(3):
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_acc, val_f1 = evaluate(model, test_loader, device)

    print(f"Epoch {epoch+1}/{5}, Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_state_dict = model.state_dict().copy()

# Load best model and evaluate on test set
if best_state_dict is not None:
    model.load_state_dict(best_state_dict)
test_acc, test_f1 = evaluate(model, test_loader, device)
print(f"Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}")

torch.save(model.state_dict(), f"/content/drive/MyDrive/cs6073/NeuroFace/src/{model_name}_neuroface.pth")
print(f"Saved best model as /content/drive/MyDrive/cs6073/NeuroFace/src/{model_name}_neuroface.pth")

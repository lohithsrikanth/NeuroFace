# Create a Multi-Layer Perceptron model
import torch
import torch.nn as nn
import torch.nn.functional as F
#from torchvision import models, datasets, transforms
from tqdm.notebook import tqdm, trange

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 3 #

class MultiLayerPerceptron(nn.Module):
    def __init__(self):
        super().__init__()
        # Correct the input features for the first layer to match the flattened image size
        # Correct the output features for the last layer to match NUM_CLASSES
        self.fc1 = nn.Linear(224*224*3, 128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,NUM_CLASSES)


    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)

        return x

"""
train_loader, test_loader, test_dataset, classes = get_dataloaders(
    data_root="/content/drive/MyDrive/cs6073/NeuroFace/processed_frames",
    batch_size=32,
    num_workers=2
)
## Traing the MLP model
# instantiate the new model
model = MultiLayerPerceptron()
model = model.to(device)

# define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#Iterate through train set minibatches
for epoch in trange(3):
    for images, labels in tqdm(train_loader):
        # Zero out the gradients
        optimizer.zero_grad()

        # Forward pass
        x = images # <---- change here
        x= x.to(device)
        labels = labels.to(device)

        y = model(x)
        loss = criterion(y, labels)
        # Backward pass
        loss  .backward()
        optimizer.step()

## Testing
correct = 0
total = len(test_dataset)

with torch.no_grad():
    # Iterate through test set minibatchs
    for images, labels in tqdm(test_loader):
        # Forward pass
        x = images  # <---- change here
        x = x.to(device)
        labels = labels.to(device)
        y = model(x)

        predictions = torch.argmax(y, dim=1)
        correct += torch.sum((predictions == labels).float())

print('Test accuracy: {}'.format(correct/total))

"""

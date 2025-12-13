# Create RandomForest model for comparison with the RestNet models
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np

# Instantiate the RandomForest model
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    n_jobs=-1,
    class_weight='balanced',
    random_state=42
  )

print("\nCollecting training data for RandomForest...")

# Collect all training data
# Initialize lists to store all images and labels
all_train_images = []
all_train_labels = []

# get_dataloaders now returns 5 values: train_loader, val_loader, test_loader, test_dataset, classes
# We need train_loader for training and test_dataset for evaluation.
train_loader_rf, val_loader_rf, test_loader_rf, test_dataset_rf, test_dataset_rf_classes = get_dataloaders(
    data_root="/content/drive/MyDrive/cs6073/NeuroFace/processed_frames",
    batch_size=32,
    num_workers=4 # Consistent num_workers with dataset_utils.py
)
for images, labels in tqdm(train_loader_rf):
    # Flatten images and convert to numpy
    # Images are [batch_size, channels, height, width], flatten to [batch_size, channels*height*width]
    all_train_images.append(images.view(images.shape[0], -1).numpy())
    # Convert labels to numpy (ensure labels are tensors first, use .cpu() for robustness)
    all_train_labels.append(labels.cpu().numpy())

# Concatenate all collected data
x_train_rf = np.concatenate(all_train_images, axis=0)
y_train_rf = np.concatenate(all_train_labels, axis=0)

print("\nTraining RandomForest...")
# Train the RandomForest model on the entire dataset
rf.fit(x_train_rf, y_train_rf)

print("\nEvaluating RandomForest...")

# Collect all test data in a similar way to training data
all_test_images = []
all_test_labels = []
for images, labels in tqdm(test_loader_rf): # test_loader_rf is now correctly a DataLoader
    all_test_images.append(images.view(images.shape[0], -1).numpy())
    all_test_labels.append(labels.cpu().numpy()) # Ensure labels are tensors and moved to CPU
x_test_rf = np.concatenate(all_test_images, axis=0)
y_test_rf = np.concatenate(all_test_labels, axis=0)

# Make predictions on the test set
y_pred = rf.predict(x_test_rf)

# Print classification report
print(classification_report(y_test_rf, y_pred))

# Compute accuracy
print("\nAccuracy: {}".format(accuracy_score(y_test_rf, y_pred)))

# Print f1 score
print("\nF1 Score: {}".format(f1_score(y_test_rf, y_pred, average='weighted')))

# Print Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test_rf, y_pred))

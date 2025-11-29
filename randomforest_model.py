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

for images, labels in tqdm(train_loader):
    # Flatten images and convert to numpy
    all_train_images.append(images.view(-1, 28*28).numpy())
    # Convert labels to numpy
    all_train_labels.append(labels.numpy())

# Concatenate all collected data
x_train_rf = np.concatenate(all_train_images, axis=0)
y_train_rf = np.concatenate(all_train_labels, axis=0)

print("\nTraining RandomForest...")
# Train the RandomForest model on the entire dataset
rf.fit(x_train_rf, y_train_rf)

print("\nEvaluating RandomForest...")
# Prepare test data for evaluation
x_test_rf = mnist_test.data.view(-1, 28*28).numpy()
y_test_rf = mnist_test.targets.numpy()

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

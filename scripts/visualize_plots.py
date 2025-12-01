import matplotlib.pyplot as plt
import json
def visualize(history_path):
    """
    Function to visualize training history.

    Args:
        history_path: Path to the JSON file containing training history.

    Returns:
        None
    """


    with open(history_path, "r") as f:
        history = json.load(f)

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 4))

    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()

    # Plot validation accuracy and F1 score
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["val_acc"], label="Validation Accuracy")
    plt.plot(epochs, history["val_f1"], label="Validation F1 Score")
    plt.xlabel("Epochs")
    plt.ylabel("Metrics")
    plt.title("Validation Metrics")
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    history_path = "histories/resnet18_2_history.json"
    visualize(history_path)
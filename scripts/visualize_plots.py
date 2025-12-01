import matplotlib.pyplot as plt
import json
import os

def visualize(history_path):
    """
    Function to visualize training history and save plots.

    Args:
        history_path: Path to the JSON file containing training history.
    """
    with open(history_path, "r") as f:
        history = json.load(f)

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(16, 8))

    # Plot training and validation loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()

    # Plot training and validation accuracy
    plt.subplot(2, 2, 2)
    plt.plot(epochs, history["train_accuracy"], label="Train Accuracy")
    plt.plot(epochs, history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.legend()

    # Plot training and validation F1 score
    plt.subplot(2, 2, 3)
    plt.plot(epochs, history["train_f1"], label="Train F1 Score")
    plt.plot(epochs, history["val_f1"], label="Validation F1 Score")
    plt.xlabel("Epochs")
    plt.ylabel("F1 Score")
    plt.title("F1 Score")
    plt.legend()

    plt.tight_layout()

    # Save the plot
    os.makedirs("plots", exist_ok=True)
    model_name = os.path.basename(history_path).replace("_history.json", "")
    plot_path = f"plots/{model_name}_metrics.png"
    plt.savefig(plot_path)
    print(f"Saved plot as {plot_path}")

    plt.close()


if __name__ == "__main__":
    history_dir = "histories"
    
    # Iterate over all history json files
    for filename in os.listdir(history_dir):
        if filename.endswith("_history.json"):
            history_path = os.path.join(history_dir, filename)
            print(f"Processing {history_path}...")
            visualize(history_path)

    print("All plots generated!")

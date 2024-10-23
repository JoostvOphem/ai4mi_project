import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Define the base path and loss types
base_path = Path("results/segthor_enet")
loss_types = ["ce", "dice", "focal", "combine"]

# Function to load and process dice scores
def load_dice_scores(file_path):
    dice_scores = np.load(file_path)
    mean_dice = dice_scores.mean(axis=1)  # Average across samples
    mean_dice = mean_dice[:, 1:].mean(axis=1)  # Average across non-background classes
    return mean_dice

# Load dice scores for each loss type
dice_scores = {}
for loss_type in loss_types:
    file_path = base_path / loss_type / "dice_val.npy"
    if file_path.exists():
        dice_scores[loss_type] = load_dice_scores(file_path)
    else:
        print(f"Warning: File not found for {loss_type}")

# Plotting
plt.figure(figsize=(12, 8))
for loss_type, scores in dice_scores.items():
    epochs = range(1, len(scores) + 1)
    plt.plot(epochs, scores, label=f'{loss_type.capitalize()} Loss')

plt.xlabel('Epochs')
plt.ylabel('Mean Dice Score')
plt.title('Mean Dice Score Comparison for Different Loss Functions')
plt.legend()
plt.grid(True)

# Improve layout
plt.tight_layout()

# Save the plot
output_path = "loss_comparison.png"
plt.savefig(output_path)
print(f"Plot saved to {output_path}")

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def load_data(folder_path: Path, file_name: str) -> np.ndarray:
    file_path = folder_path / file_name
    if file_path.exists():
        return np.load(file_path)
    else:
        print(f"Warning: {file_path} not found.")
        return np.array([])

def plot_metrics(adam_folder: Path, adamw_folder: Path, output_file: Path):
    # Load data
    adam_dice = load_data(adam_folder, "dice_val.npy")
    adamw_dice = load_data(adamw_folder, "dice_val.npy")

    # Calculate average Dice score across all classes
    adam_avg_dice = adam_dice.mean(axis=(1, 2))  # Average over samples and classes
    adamw_avg_dice = adamw_dice.mean(axis=(1, 2))  # Average over samples and classes

    # Create plot
    plt.figure(figsize=(10, 6))
    
    # Plot average Dice score
    epochs = range(1, len(adam_avg_dice) + 1)
    plt.plot(epochs, adam_avg_dice, 'b-', label='Adam')
    plt.plot(epochs, adamw_avg_dice, 'r-', label='AdamW')
    
    plt.xlabel('Epochs')
    plt.ylabel('Average Dice Score')
    plt.title('Average Dice Score Comparison (All Classes)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Compare Adam and AdamW optimizer performance')
    parser.add_argument('--adam_folder', default='./results/segthor_enet_adam/ce', type=Path, help='Path to the folder containing Adam results')
    parser.add_argument('--adamw_folder', default='./results/segthor_enet_adamW/ce', type=Path, help='Path to the folder containing AdamW results')
    parser.add_argument('--output', type=Path, default='optimizer_comparison.png', help='Output file path for the plot')
    
    args = parser.parse_args()
    
    plot_metrics(args.adam_folder, args.adamw_folder, args.output)

if __name__ == "__main__":
    main()
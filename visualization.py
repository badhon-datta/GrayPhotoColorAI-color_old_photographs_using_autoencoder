import matplotlib.pyplot as plt
import numpy as np
import torch
from utils.preprocessing import postprocess_image


def visualize_batch(l_channels, ab_targets, ab_predictions, num_samples=4):
    """
    Visualize grayscale inputs, ground truth, and predictions.

    Args:
        l_channels: tensor of shape (B, 1, H, W)
        ab_targets: tensor of shape (B, 2, H, W)
        ab_predictions: tensor of shape (B, 2, H, W)
        num_samples: number of samples to display
    """
    num_samples = min(num_samples, l_channels.shape[0])

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        # Grayscale input
        grayscale = l_channels[i].cpu().numpy().squeeze()
        grayscale_denorm = (grayscale * 50.0) + 50.0  # Denormalize for display

        # Ground truth
        rgb_gt = postprocess_image(l_channels[i], ab_targets[i])

        # Prediction
        rgb_pred = postprocess_image(l_channels[i], ab_predictions[i])

        # Plot
        axes[i, 0].imshow(grayscale_denorm, cmap='gray')
        axes[i, 0].set_title('Grayscale Input')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(rgb_gt)
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(rgb_pred)
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')

    plt.tight_layout()
    return fig


def save_comparison(grayscale_path, colorized_path, l_channel, ab_prediction):
    """
    Save grayscale and colorized images side by side.

    Args:
        grayscale_path: path to save grayscale image
        colorized_path: path to save colorized image
        l_channel: tensor of shape (1, H, W)
        ab_prediction: tensor of shape (2, H, W)
    """
    # Create RGB image
    rgb_image = postprocess_image(l_channel, ab_prediction)

    # Save colorized
    plt.imsave(colorized_path, rgb_image)

    # Save grayscale
    grayscale = l_channel.cpu().numpy().squeeze()
    grayscale_denorm = (grayscale * 50.0) + 50.0
    plt.imsave(grayscale_path, grayscale_denorm, cmap='gray')


def plot_training_history(history):
    """
    Plot training and validation loss.

    Args:
        history: dict with 'train_loss' and 'val_loss' lists
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.plot(history['train_loss'], label='Train Loss', linewidth=2)
    ax.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training History', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

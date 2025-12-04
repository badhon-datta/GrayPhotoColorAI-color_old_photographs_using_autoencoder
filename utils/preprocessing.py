import torch
import numpy as np
from skimage import color
from PIL import Image
import cv2


def rgb_to_lab(rgb_image):
    """
    Convert RGB image to LAB color space.

    Args:
        rgb_image: numpy array of shape (H, W, 3) with values in [0, 255]

    Returns:
        lab_image: numpy array of shape (H, W, 3) with L in [0, 100], A and B in [-128, 127]
    """
    rgb_normalized = rgb_image.astype(np.float32) / 255.0
    lab_image = color.rgb2lab(rgb_normalized)
    return lab_image


def lab_to_rgb(lab_image):
    """
    Convert LAB image back to RGB color space.

    Args:
        lab_image: numpy array of shape (H, W, 3) with L in [0, 100], A and B in [-128, 127]

    Returns:
        rgb_image: numpy array of shape (H, W, 3) with values in [0, 255]
    """
    rgb_normalized = color.lab2rgb(lab_image)
    rgb_image = (rgb_normalized * 255.0).astype(np.uint8)
    return rgb_image


def preprocess_image(image_path, target_size=(256, 256)):
    """
    Load and preprocess an image for training.

    Args:
        image_path: path to the image file
        target_size: tuple of (height, width) for resizing

    Returns:
        l_channel: grayscale input (L channel) normalized to [-1, 1], shape (1, H, W)
        ab_channels: color target (AB channels) normalized to [-1, 1], shape (2, H, W)
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    image = image.resize(target_size, Image.BILINEAR)
    image_np = np.array(image)

    # Convert to LAB
    lab_image = rgb_to_lab(image_np)

    # Split channels
    l_channel = lab_image[:, :, 0]  # L channel [0, 100]
    ab_channels = lab_image[:, :, 1:]  # AB channels [-128, 127]

    # Normalize L channel to [-1, 1]
    l_normalized = (l_channel - 50.0) / 50.0

    # Normalize AB channels to [-1, 1]
    ab_normalized = ab_channels / 128.0

    # Convert to PyTorch tensors and change to CHW format
    l_tensor = torch.from_numpy(l_normalized).float().unsqueeze(0)  # (1, H, W)
    ab_tensor = torch.from_numpy(ab_normalized).float().permute(2, 0, 1)  # (2, H, W)

    return l_tensor, ab_tensor


def postprocess_image(l_channel, ab_channels):
    """
    Convert model output back to RGB image.

    Args:
        l_channel: tensor of shape (1, H, W) normalized to [-1, 1]
        ab_channels: tensor of shape (2, H, W) normalized to [-1, 1]

    Returns:
        rgb_image: numpy array of shape (H, W, 3) with values in [0, 255]
    """
    # Denormalize
    l_denorm = (l_channel.squeeze(0).cpu().numpy() * 50.0) + 50.0  # Back to [0, 100]
    ab_denorm = ab_channels.cpu().numpy().transpose(1, 2, 0) * 128.0  # Back to [-128, 127]

    # Combine channels
    lab_image = np.zeros((l_denorm.shape[0], l_denorm.shape[1], 3))
    lab_image[:, :, 0] = l_denorm
    lab_image[:, :, 1:] = ab_denorm

    # Convert to RGB
    rgb_image = lab_to_rgb(lab_image)

    return rgb_image


def load_grayscale_image(image_path, target_size=(256, 256)):
    """
    Load a grayscale image for inference.

    Args:
        image_path: path to the grayscale image
        target_size: tuple of (height, width)

    Returns:
        l_tensor: tensor of shape (1, 1, H, W) normalized to [-1, 1]
        original_size: tuple of original image dimensions
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    image = image.resize(target_size, Image.BILINEAR)
    image_np = np.array(image)

    # Convert to LAB and extract L channel
    lab_image = rgb_to_lab(image_np)
    l_channel = lab_image[:, :, 0]

    # Normalize to [-1, 1]
    l_normalized = (l_channel - 50.0) / 50.0

    # Convert to tensor with batch dimension
    l_tensor = torch.from_numpy(l_normalized).float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    return l_tensor, original_size, l_normalized

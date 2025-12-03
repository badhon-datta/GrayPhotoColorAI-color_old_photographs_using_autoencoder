import os
import torch
import argparse
from PIL import Image
import matplotlib.pyplot as plt

from models import get_model
from utils import load_grayscale_image, postprocess_image


def colorize_image(model, image_path, device, output_path=None, show=False):
    """
    Colorize a grayscale image using the trained model.

    Args:
        model: trained colorization model
        image_path: path to input grayscale image
        device: torch device
        output_path: path to save colorized image (optional)
        show: whether to display the result

    Returns:
        rgb_colorized: colorized image as numpy array
    """
    # Load and preprocess image
    l_tensor, original_size, l_channel_np = load_grayscale_image(image_path)
    l_tensor = l_tensor.to(device)

    # Predict AB channels
    model.eval()
    with torch.no_grad():
        ab_prediction = model(l_tensor)

    # Convert to RGB
    l_channel = l_tensor.squeeze(0)  # Remove batch dimension
    ab_channels = ab_prediction.squeeze(0)  # Remove batch dimension
    rgb_colorized = postprocess_image(l_channel, ab_channels)

    # Resize to original size if needed
    if rgb_colorized.shape[:2] != original_size[::-1]:
        rgb_colorized = Image.fromarray(rgb_colorized)
        rgb_colorized = rgb_colorized.resize(original_size, Image.BILINEAR)
        rgb_colorized = np.array(rgb_colorized)

    # Save if output path provided
    if output_path:
        Image.fromarray(rgb_colorized).save(output_path)
        print(f"Colorized image saved to {output_path}")

    # Display if requested
    if show:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Grayscale input
        grayscale_display = (l_channel_np * 50.0) + 50.0
        axes[0].imshow(grayscale_display, cmap='gray')
        axes[0].set_title('Grayscale Input', fontsize=14)
        axes[0].axis('off')

        # Colorized output
        axes[1].imshow(rgb_colorized)
        axes[1].set_title('Colorized Output', fontsize=14)
        axes[1].axis('off')

        plt.tight_layout()
        plt.show()

    return rgb_colorized


def batch_colorize(model, input_dir, output_dir, device):
    """
    Colorize all images in a directory.

    Args:
        model: trained colorization model
        input_dir: directory containing grayscale images
        output_dir: directory to save colorized images
        device: torch device
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get all image files
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_paths = [
        f for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in valid_extensions
    ]

    if len(image_paths) == 0:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(image_paths)} images to colorize")

    for image_name in image_paths:
        input_path = os.path.join(input_dir, image_name)
        output_path = os.path.join(output_dir, f"colorized_{image_name}")

        try:
            colorize_image(model, input_path, device, output_path)
            print(f"Processed: {image_name}")
        except Exception as e:
            print(f"Error processing {image_name}: {e}")

    print(f"\nAll images colorized! Results saved to {output_dir}")


def main(args):
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = get_model(args.model_type, device)

    # Load checkpoint
    if args.checkpoint.endswith('.pth'):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            model.load_state_dict(checkpoint)
    else:
        raise ValueError(f"Invalid checkpoint file: {args.checkpoint}")

    model.eval()
    print("Model loaded successfully!")

    # Single image or batch processing
    if args.input_image:
        # Single image
        print(f"\nColorizing {args.input_image}...")
        colorize_image(
            model,
            args.input_image,
            device,
            output_path=args.output_image,
            show=args.show
        )
    elif args.input_dir:
        # Batch processing
        print(f"\nBatch colorizing images from {args.input_dir}...")
        batch_colorize(model, args.input_dir, args.output_dir, device)
    else:
        print("Please provide either --input_image or --input_dir")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Colorize grayscale images')

    # Model parameters
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, default='unet', choices=['unet', 'simple'],
                        help='Model architecture')

    # Input/Output
    parser.add_argument('--input_image', type=str, default=None,
                        help='Path to input grayscale image')
    parser.add_argument('--output_image', type=str, default='colorized_output.png',
                        help='Path to save colorized image')
    parser.add_argument('--input_dir', type=str, default=None,
                        help='Directory containing grayscale images for batch processing')
    parser.add_argument('--output_dir', type=str, default='colorized_outputs',
                        help='Directory to save batch colorized images')

    # Display
    parser.add_argument('--show', action='store_true',
                        help='Display the result')

    args = parser.parse_args()

    # Import numpy here to avoid issues if not using display
    if args.show or args.input_image:
        import numpy as np

    main(args)

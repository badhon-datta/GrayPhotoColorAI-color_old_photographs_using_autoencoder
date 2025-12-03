import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

from models import get_model
from utils import preprocess_image, visualize_batch


class ColorizationDataset(Dataset):
    """Dataset for image colorization"""

    def __init__(self, data_dir, image_size=(256, 256)):
        """
        Args:
            data_dir: directory containing images
            image_size: tuple of (height, width)
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size

        # Get all image files
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        self.image_paths = [
            p for p in self.data_dir.rglob('*')
            if p.suffix.lower() in valid_extensions
        ]

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {data_dir}")

        print(f"Found {len(self.image_paths)} images in {data_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            l_channel, ab_channels = preprocess_image(str(image_path), self.image_size)
            return l_channel, ab_channels
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            # Return a random other image if this one fails
            return self.__getitem__((idx + 1) % len(self))


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')

    for l_channels, ab_targets in pbar:
        l_channels = l_channels.to(device)
        ab_targets = ab_targets.to(device)

        # Forward pass
        optimizer.zero_grad()
        ab_predictions = model(l_channels)

        # Compute loss
        loss = criterion(ab_predictions, ab_targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})

    return running_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for l_channels, ab_targets in tqdm(dataloader, desc='Validation'):
            l_channels = l_channels.to(device)
            ab_targets = ab_targets.to(device)

            ab_predictions = model(l_channels)
            loss = criterion(ab_predictions, ab_targets)
            running_loss += loss.item()

    return running_loss / len(dataloader)


def main(args):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = ColorizationDataset(args.train_dir, image_size=(args.image_size, args.image_size))
    val_dataset = ColorizationDataset(args.val_dir, image_size=(args.image_size, args.image_size))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    # Create model
    print(f"\nCreating {args.model_type} model...")
    model = get_model(args.model_type, device)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'logs'))

    # Training history
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        history['train_loss'].append(train_loss)

        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        history['val_loss'].append(val_loss)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        print(f"\nTrain Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")

        # Save visualization every N epochs
        if epoch % args.viz_freq == 0:
            model.eval()
            with torch.no_grad():
                # Get a batch from validation set
                l_batch, ab_batch = next(iter(val_loader))
                l_batch = l_batch.to(device)
                ab_batch = ab_batch.to(device)
                ab_pred = model(l_batch)

                # Visualize
                fig = visualize_batch(l_batch, ab_batch, ab_pred, num_samples=4)
                writer.add_figure('Predictions', fig, epoch)
                plt.savefig(os.path.join(args.output_dir, f'epoch_{epoch}.png'))
                plt.close()

    # Save final model
    final_path = os.path.join(args.checkpoint_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete! Final model saved to {final_path}")

    # Close writer
    writer.close()

    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
    plt.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(args.output_dir, 'training_history.png'))
    print(f"Training history saved to {args.output_dir}/training_history.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train image colorization model')

    # Data parameters
    parser.add_argument('--train_dir', type=str, default='data/train',
                        help='Directory containing training images')
    parser.add_argument('--val_dir', type=str, default='data/val',
                        help='Directory containing validation images')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory to save outputs')

    # Model parameters
    parser.add_argument('--model_type', type=str, default='unet', choices=['unet', 'simple'],
                        help='Model architecture to use')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Image size (assumes square images)')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--viz_freq', type=int, default=5,
                        help='Frequency of visualization (in epochs)')

    args = parser.parse_args()

    main(args)

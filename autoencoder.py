import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Double Convolution block: (Conv -> BatchNorm -> ReLU) x 2"""

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv with skip connections"""

    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # x2 is the skip connection from encoder
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class ColorizationUNet(nn.Module):
    """
    U-Net architecture for image colorization.

    Input: Grayscale image (L channel) - shape (B, 1, H, W)
    Output: Color channels (AB channels) - shape (B, 2, H, W)
    """

    def __init__(self, in_channels=1, out_channels=2):
        super(ColorizationUNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Encoder
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)

        # Decoder
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)

        # Output layer
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)
        self.tanh = nn.Tanh()  # Output in range [-1, 1]

    def forward(self, x):
        # Encoder with skip connections
        x1 = self.inc(x)       # 64 channels
        x2 = self.down1(x1)    # 128 channels
        x3 = self.down2(x2)    # 256 channels
        x4 = self.down3(x3)    # 512 channels
        x5 = self.down4(x4)    # 512 channels (bottleneck)

        # Decoder with skip connections
        x = self.up1(x5, x4)   # 256 channels
        x = self.up2(x, x3)    # 128 channels
        x = self.up3(x, x2)    # 64 channels
        x = self.up4(x, x1)    # 64 channels

        # Output
        x = self.outc(x)       # 2 channels
        x = self.tanh(x)       # Normalize to [-1, 1]

        return x


class SimplerColorNet(nn.Module):
    """
    Simpler encoder-decoder architecture for faster training.
    Good for initial experiments and smaller datasets.
    """

    def __init__(self, in_channels=1, out_channels=2):
        super(SimplerColorNet, self).__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Decoder
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # Encode
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)

        # Decode
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)

        return x


def get_model(model_type='unet', device='cpu'):
    """
    Factory function to get a colorization model.

    Args:
        model_type: 'unet' or 'simple'
        device: 'cpu' or 'cuda'

    Returns:
        model: PyTorch model
    """
    if model_type == 'unet':
        model = ColorizationUNet()
    elif model_type == 'simple':
        model = SimplerColorNet()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model.to(device)


if __name__ == "__main__":
    # Test the model
    model = ColorizationUNet()
    x = torch.randn(4, 1, 256, 256)  # Batch of 4 grayscale images
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

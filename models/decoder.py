import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageDecoder(nn.Module):
    """Image VAE Decoder - outputs 64x64 images."""

    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim

        # Project latent to spatial feature map
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)

        # Transposed convolutional layers to get 64x64 output
        # 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64
        self.deconv1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)  # 4x4 -> 8x8
        self.deconv2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)   # 8x8 -> 16x16
        self.deconv3 = nn.ConvTranspose2d(64, 32, 4, 2, 1)    # 16x16 -> 32x32
        self.deconv4 = nn.ConvTranspose2d(32, 1, 4, 2, 1)     # 32x32 -> 64x64

        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(32)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to reconstructed image.

        Args:
            z: Latent vector of shape (batch, latent_dim)

        Returns:
            Reconstructed image of shape (batch, 1, 64, 64)
        """
        x = self.fc(z).view(-1, 256, 4, 4)

        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))
        x = torch.sigmoid(self.deconv4(x))

        return x

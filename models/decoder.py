import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageDecoder(nn.Module):
    """Image VAE Decoder based on Table 2 architecture."""

    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim

        # Project latent to spatial feature map
        self.fc = nn.Linear(latent_dim, 64 * 4 * 4)

        # Transposed convolutional layers
        self.deconv1 = nn.ConvTranspose2d(64, 64, 4, 2)  # 4x4 -> 10x10
        self.deconv2 = nn.ConvTranspose2d(64, 64, 4, 2)  # 10x10 -> 22x22
        self.deconv3 = nn.ConvTranspose2d(64, 64, 5, 1)  # 22x22 -> 26x26
        self.deconv4 = nn.ConvTranspose2d(64, 64, 5, 2)  # 26x26 -> 55x55
        self.deconv5 = nn.ConvTranspose2d(64, 32, 5, 1)  # 55x55 -> 59x59
        self.deconv6 = nn.ConvTranspose2d(32, 32, 5, 2)  # 59x59 -> 121x121
        self.deconv7 = nn.ConvTranspose2d(32, 32, 5, 1)  # 121x121 -> 125x125
        self.out = nn.ConvTranspose2d(32, 1, 5, 1)       # 125x125 -> 129x129

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to reconstructed image.

        Args:
            z: Latent vector of shape (batch, latent_dim)

        Returns:
            Reconstructed image of shape (batch, 1, H, W)
        """
        x = self.fc(z).view(-1, 64, 4, 4)

        x = F.relu(F.instance_norm(self.deconv1(x)))
        x = F.relu(F.instance_norm(self.deconv2(x)))
        x = F.relu(F.instance_norm(self.deconv3(x)))
        x = F.relu(F.instance_norm(self.deconv4(x)))
        x = F.relu(F.instance_norm(self.deconv5(x)))
        x = F.relu(F.instance_norm(self.deconv6(x)))
        x = F.relu(F.instance_norm(self.deconv7(x)))

        x = torch.sigmoid(self.out(x))

        return x

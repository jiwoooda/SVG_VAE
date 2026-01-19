import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageEncoder(nn.Module):
    """Image VAE Encoder based on Table 1 architecture."""

    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, 5, 1)   # 64x64 -> 60x60
        self.conv2 = nn.Conv2d(32, 32, 5, 1)  # 60x60 -> 56x56
        self.conv3 = nn.Conv2d(32, 64, 5, 1)  # 56x56 -> 52x52
        self.conv4 = nn.Conv2d(64, 64, 5, 2)  # 52x52 -> 24x24
        self.conv5 = nn.Conv2d(64, 64, 4, 2)  # 24x24 -> 11x11
        self.conv6 = nn.Conv2d(64, 64, 4, 2)  # 11x11 -> 4x4

        # Latent space projection (64 channels * 4 * 4 = 1024)
        self.fc_mu = nn.Linear(64 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(64 * 4 * 4, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input image to latent distribution parameters.

        Args:
            x: Input image tensor of shape (batch, 1, 64, 64)

        Returns:
            mu: Mean of latent distribution (batch, latent_dim)
            logvar: Log variance of latent distribution (batch, latent_dim)
        """
        x = F.relu(F.instance_norm(self.conv1(x)))
        x = F.relu(F.instance_norm(self.conv2(x)))
        x = F.relu(F.instance_norm(self.conv3(x)))
        x = F.relu(F.instance_norm(self.conv4(x)))
        x = F.relu(F.instance_norm(self.conv5(x)))
        x = F.relu(F.instance_norm(self.conv6(x)))

        # Global average pooling and flatten
        x = x.view(x.size(0), -1)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar

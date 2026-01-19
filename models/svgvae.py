import torch
import torch.nn as nn

from .encoder import ImageEncoder
from .decoder import ImageDecoder
from .svg_decoder import SVGDecoder


class SVGVAE(nn.Module):
    """Full SVG-VAE Model combining Image VAE and SVG Decoder."""

    def __init__(self, latent_dim: int = 64, num_classes: int = 62):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        self.encoder = ImageEncoder(latent_dim)
        self.decoder = ImageDecoder(latent_dim)
        self.svg_decoder = SVGDecoder(latent_dim, num_classes)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for VAE.

        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution

        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode image to latent distribution parameters."""
        return self.encoder(img)

    def decode_image(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to image."""
        return self.decoder(z)

    def decode_svg(
        self,
        z: torch.Tensor,
        class_label: torch.Tensor,
        svg_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple]:
        """Decode latent vector to SVG commands."""
        seq_len = svg_tokens.size(1)
        return self.svg_decoder(
            z.unsqueeze(1).expand(-1, seq_len, -1),
            class_label.unsqueeze(1).expand(-1, seq_len, -1),
            svg_tokens,
        )

    def forward(
        self,
        img: torch.Tensor,
        class_label: torch.Tensor,
        svg_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass through SVG-VAE.

        Args:
            img: Input image (batch, 1, 64, 64)
            class_label: One-hot class label (batch, num_classes)
            svg_tokens: Previous SVG tokens for teacher forcing (batch, seq_len, 2)

        Returns:
            recon_img: Reconstructed image
            cmd_logits: Command type logits
            pi_logits: Mixture weight logits
            mu_args: Coordinate means
            log_sigma: Coordinate log stds
            mu: Latent mean
            logvar: Latent log variance
        """
        # VAE encoding
        mu, logvar = self.encoder(img)
        z = self.reparameterize(mu, logvar)

        # Image reconstruction
        recon_img = self.decoder(z)

        # SVG decoding (autoregressive with teacher forcing)
        cmd_logits, pi_logits, mu_args, log_sigma, _ = self.decode_svg(
            z, class_label, svg_tokens
        )

        return recon_img, cmd_logits, pi_logits, mu_args, log_sigma, mu, logvar

    def sample_svg(
        self,
        img: torch.Tensor,
        class_label: torch.Tensor,
        max_seq_len: int = 100,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample SVG from image.

        Args:
            img: Input image
            class_label: One-hot class label
            max_seq_len: Maximum sequence length
            temperature: Sampling temperature

        Returns:
            commands: Sampled command types
            coordinates: Sampled coordinates
        """
        mu, logvar = self.encoder(img)
        z = self.reparameterize(mu, logvar)
        return self.svg_decoder.sample(z, class_label, max_seq_len, temperature)

    def interpolate(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        num_steps: int = 10,
    ) -> torch.Tensor:
        """
        Interpolate between two images in latent space.

        Args:
            img1: First image
            img2: Second image
            num_steps: Number of interpolation steps

        Returns:
            Interpolated images (num_steps, 1, H, W)
        """
        mu1, _ = self.encoder(img1)
        mu2, _ = self.encoder(img2)

        interpolations = []
        for alpha in torch.linspace(0, 1, num_steps):
            z = (1 - alpha) * mu1 + alpha * mu2
            img = self.decoder(z)
            interpolations.append(img)

        return torch.cat(interpolations, dim=0)

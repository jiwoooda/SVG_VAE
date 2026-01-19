import torch
import torch.nn.functional as F


def reconstruction_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    reduction: str = 'sum',
) -> torch.Tensor:
    """
    Compute reconstruction loss (MSE).

    Args:
        recon_x: Reconstructed image
        x: Original image
        reduction: 'sum' or 'mean'

    Returns:
        Reconstruction loss
    """
    return F.mse_loss(recon_x, x, reduction=reduction)


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Compute KL divergence from N(mu, sigma) to N(0, 1).

    Args:
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution

    Returns:
        KL divergence loss
    """
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def vae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float = 4.68,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute full VAE loss: reconstruction + KL divergence.

    Args:
        recon_x: Reconstructed image
        x: Original image
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        kl_weight: Weight for KL divergence term (default: 4.68)

    Returns:
        total_loss: Combined loss
        recon_loss: Reconstruction loss component
        kl_loss: KL divergence component
    """
    recon_loss = reconstruction_loss(recon_x, x)
    kl_loss = kl_divergence(mu, logvar)
    total_loss = recon_loss + kl_weight * kl_loss

    return total_loss, recon_loss, kl_loss

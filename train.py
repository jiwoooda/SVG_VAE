"""Training script for SVG-VAE model."""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from models import SVGVAE
from losses import vae_loss, svg_decoder_loss


def train_epoch(
    model: SVGVAE,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    kl_weight: float = 4.68,
) -> dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_vae_loss = 0.0
    total_svg_loss = 0.0

    for batch in dataloader:
        img, class_label, svg_tokens, target_cmd, target_args = [
            x.to(device) for x in batch
        ]

        optimizer.zero_grad()

        # Forward pass
        recon_img, cmd_logits, pi_logits, mu_args, log_sigma, mu, logvar = model(
            img, class_label, svg_tokens
        )

        # Compute losses
        vae_total, recon_loss, kl_loss = vae_loss(
            recon_img, img, mu, logvar, kl_weight
        )
        svg_total, cmd_loss, mdn_loss = svg_decoder_loss(
            cmd_logits, pi_logits, mu_args, log_sigma, target_cmd, target_args
        )

        # Combined loss
        loss = vae_total + svg_total
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_vae_loss += vae_total.item()
        total_svg_loss += svg_total.item()

    num_batches = len(dataloader)
    return {
        'loss': total_loss / num_batches,
        'vae_loss': total_vae_loss / num_batches,
        'svg_loss': total_svg_loss / num_batches,
    }


def main():
    """Main training function."""
    # Hyperparameters
    latent_dim = 64
    num_classes = 62
    batch_size = 32
    learning_rate = 1e-4
    num_epochs = 100
    kl_weight = 4.68

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize model
    model = SVGVAE(latent_dim=latent_dim, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Create dummy data for demonstration
    # Replace with actual data loading
    num_samples = 1000
    seq_len = 50

    dummy_imgs = torch.randn(num_samples, 1, 64, 64)
    dummy_classes = F.one_hot(
        torch.randint(0, num_classes, (num_samples,)), num_classes
    ).float()
    dummy_svg_tokens = torch.randn(num_samples, seq_len, 2)
    dummy_target_cmd = torch.randint(0, 4, (num_samples, seq_len))
    dummy_target_args = torch.randn(num_samples, seq_len, 2)

    dataset = TensorDataset(
        dummy_imgs, dummy_classes, dummy_svg_tokens, dummy_target_cmd, dummy_target_args
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    print("Starting training...")
    for epoch in range(num_epochs):
        metrics = train_epoch(model, dataloader, optimizer, device, kl_weight)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}] "
                f"Loss: {metrics['loss']:.4f} "
                f"VAE: {metrics['vae_loss']:.4f} "
                f"SVG: {metrics['svg_loss']:.4f}"
            )

    # Save model
    torch.save(model.state_dict(), 'svgvae_model.pth')
    print("Model saved to svgvae_model.pth")


if __name__ == "__main__":
    main()

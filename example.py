"""Example usage of SVG-VAE model."""

import torch
import torch.nn.functional as F

from models import SVGVAE


def main():
    """Demonstrate SVG-VAE model usage."""
    # Model initialization
    model = SVGVAE(latent_dim=64, num_classes=62)
    print("Model initialized successfully!")

    # Dummy data
    batch_size = 4
    seq_len = 50

    img = torch.randn(batch_size, 1, 64, 64)
    class_label = F.one_hot(torch.randint(0, 62, (batch_size,)), 62).float()
    svg_tokens = torch.randn(batch_size, seq_len, 2)

    # Forward pass
    recon_img, cmd_logits, pi_logits, mu_args, log_sigma, mu, logvar = model(
        img, class_label, svg_tokens
    )

    print(f"Input image shape: {img.shape}")
    print(f"Reconstructed image shape: {recon_img.shape}")
    print(f"Command logits shape: {cmd_logits.shape}")
    print(f"Latent mu shape: {mu.shape}")
    print(f"Latent logvar shape: {logvar.shape}")

    # Latent space interpolation
    print("\n--- Latent Space Interpolation ---")
    img1 = torch.randn(1, 1, 64, 64)
    img2 = torch.randn(1, 1, 64, 64)
    interpolated = model.interpolate(img1, img2, num_steps=5)
    print(f"Interpolated images shape: {interpolated.shape}")

    # SVG sampling
    print("\n--- SVG Sampling ---")
    img_sample = torch.randn(2, 1, 64, 64)
    class_sample = F.one_hot(torch.tensor([0, 1]), 62).float()
    commands, coordinates = model.sample_svg(
        img_sample, class_sample, max_seq_len=20, temperature=0.8
    )
    print(f"Sampled commands shape: {commands.shape}")
    print(f"Sampled coordinates shape: {coordinates.shape}")

    # Model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n--- Model Statistics ---")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")


if __name__ == "__main__":
    main()

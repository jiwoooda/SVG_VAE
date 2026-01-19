"""Training script for SVG-VAE model."""

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

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
    parser = argparse.ArgumentParser(description="Train SVG-VAE model")
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to processed data file (.pkl)",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--kl-weight", type=float, default=4.68)
    parser.add_argument("--seq-len", type=int, default=50)
    parser.add_argument("--save-path", type=str, default="svgvae_model.pth")
    parser.add_argument("--save-every", type=int, default=10)

    args = parser.parse_args()

    # Hyperparameters
    latent_dim = args.latent_dim
    num_classes = 62
    batch_size = args.batch_size
    learning_rate = args.lr
    num_epochs = args.epochs
    kl_weight = args.kl_weight
    seq_len = args.seq_len

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize model
    model = SVGVAE(latent_dim=latent_dim, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Load data
    if args.data and Path(args.data).exists():
        print(f"Loading data from {args.data}")
        from data.dataset import load_processed_data, create_dataloader
        data = load_processed_data(args.data)
        dataloader = create_dataloader(
            data,
            batch_size=batch_size,
            max_seq_len=seq_len,
            require_svg=True,
        )
    else:
        print("No data file provided, using dummy data for demonstration")
        print("To use real data, run: python prepare_data.py --download")
        print()

        num_samples = 1000
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
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Batch size: {batch_size}, Learning rate: {learning_rate}")
    print()

    for epoch in range(num_epochs):
        metrics = train_epoch(model, dataloader, optimizer, device, kl_weight)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch [{epoch + 1:3d}/{num_epochs}] "
                f"Loss: {metrics['loss']:.4f} "
                f"VAE: {metrics['vae_loss']:.4f} "
                f"SVG: {metrics['svg_loss']:.4f}"
            )

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = f"checkpoint_epoch{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': metrics['loss'],
            }, checkpoint_path)

    # Save final model
    torch.save(model.state_dict(), args.save_path)
    print(f"\nModel saved to {args.save_path}")


if __name__ == "__main__":
    main()

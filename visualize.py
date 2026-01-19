"""Visualize SVG-VAE results."""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image

from models import SVGVAE
from data.dataset import load_processed_data


def save_image(tensor, path):
    """Save tensor as image."""
    img = tensor.squeeze().detach().numpy()
    img = (img * 255).astype(np.uint8)
    Image.fromarray(img, mode='L').save(path)


def main():
    # Load model
    model = SVGVAE(latent_dim=64, num_classes=62)

    if Path('svgvae_model.pth').exists():
        model.load_state_dict(torch.load('svgvae_model.pth', weights_only=True))
        print("Loaded trained model")
    else:
        print("No trained model found, using random weights")

    model.eval()

    # Load some test data
    data = load_processed_data('./processed_data.pkl')

    # Create output directory
    output_dir = Path('./output')
    output_dir.mkdir(exist_ok=True)

    # Test reconstruction
    print("\n=== Testing Reconstruction ===")
    with torch.no_grad():
        for i in range(min(5, len(data))):
            item = data[i]

            # Prepare input
            img = torch.from_numpy(item['image']).float().unsqueeze(0).unsqueeze(0)
            class_idx = item['class_idx']
            class_label = F.one_hot(torch.tensor([class_idx]), 62).float()

            # Encode
            mu, logvar = model.encode(img)
            z = model.reparameterize(mu, logvar)

            # Decode
            recon = model.decode_image(z)

            # Save images
            char = item['char']
            save_image(img, output_dir / f'{i}_input_{char}.png')
            save_image(recon, output_dir / f'{i}_recon_{char}.png')

            print(f"  [{i}] Character '{char}' - saved to output/")

    # Test interpolation
    print("\n=== Testing Latent Interpolation ===")
    with torch.no_grad():
        # Get two different characters
        img1 = torch.from_numpy(data[0]['image']).float().unsqueeze(0).unsqueeze(0)
        img2 = torch.from_numpy(data[10]['image']).float().unsqueeze(0).unsqueeze(0)

        mu1, _ = model.encode(img1)
        mu2, _ = model.encode(img2)

        # Interpolate
        for j, alpha in enumerate([0.0, 0.25, 0.5, 0.75, 1.0]):
            z = (1 - alpha) * mu1 + alpha * mu2
            interp = model.decode_image(z)
            save_image(interp, output_dir / f'interp_{j}_{alpha:.2f}.png')

        print(f"  Interpolation images saved to output/")

    # Test random generation
    print("\n=== Testing Random Generation ===")
    with torch.no_grad():
        for k in range(5):
            z = torch.randn(1, 64)
            generated = model.decode_image(z)
            save_image(generated, output_dir / f'random_{k}.png')

        print(f"  Random generation images saved to output/")

    print(f"\nAll results saved to {output_dir.absolute()}")


if __name__ == "__main__":
    main()

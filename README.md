# SVG-VAE PyTorch

PyTorch implementation of SVG-VAE (Scalable Vector Graphics Variational Autoencoder) for font generation.

## Project Structure

```
svg-vae-pytorch/
├── models/
│   ├── __init__.py
│   ├── encoder.py        # Image VAE Encoder
│   ├── decoder.py        # Image VAE Decoder
│   ├── svg_decoder.py    # SVG Decoder with LSTM + MDN
│   └── svgvae.py         # Full SVG-VAE Model
├── losses/
│   ├── __init__.py
│   ├── vae_loss.py       # VAE reconstruction + KL loss
│   └── svg_loss.py       # SVG command + MDN loss
├── train.py              # Training script
├── example.py            # Usage example
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
import torch
import torch.nn.functional as F
from models import SVGVAE

# Initialize model
model = SVGVAE(latent_dim=64, num_classes=62)

# Dummy input
batch_size = 4
img = torch.randn(batch_size, 1, 64, 64)
class_label = F.one_hot(torch.randint(0, 62, (batch_size,)), 62).float()
svg_tokens = torch.randn(batch_size, 50, 2)

# Forward pass
recon_img, cmd_logits, pi_logits, mu_args, log_sigma, mu, logvar = model(
    img, class_label, svg_tokens
)
```

## Architecture

### Image Encoder
- 6 convolutional layers with Instance Normalization
- Input: 64x64 grayscale image
- Output: latent mean and log-variance (64-dim each)

### Image Decoder
- 8 transposed convolutional layers with Instance Normalization
- Input: 64-dim latent vector
- Output: reconstructed 64x64 image

### SVG Decoder
- 4-stacked LSTM with dropout (0.7)
- Mixture Density Network (MDN) with 4 Gaussian components
- Commands: moveTo, cubicBezier, lineTo, EOS

## Loss Functions

- **VAE Loss**: MSE reconstruction + KL divergence (weight: 4.68)
- **SVG Loss**: Cross-entropy for commands + MDN negative log-likelihood for coordinates

## Training

```bash
python train.py
```

## License

MIT License

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
├── data/
│   ├── __init__.py
│   ├── font_downloader.py  # Google Fonts downloader
│   ├── font_processor.py   # TTF to image/SVG processor
│   ├── svg_tokenizer.py    # SVG path tokenizer
│   └── dataset.py          # PyTorch Dataset
├── prepare_data.py       # Data preparation script
├── train.py              # Training script
├── example.py            # Usage example
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Data Preparation

### Step 1: Download fonts from Google Fonts

```bash
python prepare_data.py --download --fonts-dir ./fonts --num-fonts 10
```

### Step 2: Process fonts and create dataset

```bash
python prepare_data.py --fonts-dir ./fonts --output ./processed_data.pkl --test
```

### Using your own fonts

Copy TTF files to `./fonts` directory, then run:

```bash
python prepare_data.py --fonts-dir ./fonts --output ./processed_data.pkl
```

## Training

### With processed data

```bash
python train.py --data ./processed_data.pkl --epochs 100 --batch-size 32
```

### With dummy data (for testing)

```bash
python train.py --epochs 10
```

### Training options

```
--data          Path to processed data file (.pkl)
--batch-size    Batch size (default: 32)
--epochs        Number of epochs (default: 100)
--lr            Learning rate (default: 1e-4)
--latent-dim    Latent dimension (default: 64)
--kl-weight     KL divergence weight (default: 4.68)
--seq-len       Max SVG sequence length (default: 50)
--save-path     Model save path (default: svgvae_model.pth)
--save-every    Save checkpoint every N epochs (default: 10)
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

## License

MIT License

"""PyTorch Dataset for SVG-VAE."""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pickle

from .font_processor import FontProcessor, CHARACTERS
from .svg_tokenizer import SVGTokenizer


class SVGVAEDataset(Dataset):
    """PyTorch Dataset for SVG-VAE training."""

    def __init__(
        self,
        data: List[Dict],
        max_seq_len: int = 100,
        num_classes: int = 62,
        transform=None,
    ):
        """
        Initialize SVGVAEDataset.

        Args:
            data: List of processed character data from FontProcessor
            max_seq_len: Maximum SVG sequence length
            num_classes: Number of character classes
            transform: Optional transform for images
        """
        self.data = data
        self.max_seq_len = max_seq_len
        self.num_classes = num_classes
        self.transform = transform
        self.tokenizer = SVGTokenizer(max_seq_len=max_seq_len)

        # Filter out samples without valid SVG
        self.valid_data = [d for d in data if d.get("svg") is not None]
        print(f"Dataset: {len(self.valid_data)}/{len(data)} samples with valid SVG")

    def __len__(self) -> int:
        return len(self.valid_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """
        Get a single sample.

        Returns:
            image: (1, 64, 64) tensor
            class_label: (num_classes,) one-hot tensor
            svg_tokens: (max_seq_len, 2) coordinate tensor for teacher forcing
            target_cmd: (max_seq_len,) command target tensor
            target_args: (max_seq_len, 2) coordinate target tensor
        """
        item = self.valid_data[idx]

        # Image
        image = item["image"]
        if self.transform:
            image = self.transform(image)
        image = torch.from_numpy(image).float().unsqueeze(0)  # Add channel dim

        # Class label (one-hot)
        class_idx = item["class_idx"]
        class_label = torch.zeros(self.num_classes)
        class_label[class_idx] = 1.0

        # SVG tokens
        svg_path = item["svg"]
        commands, coordinates = self.tokenizer.tokenize(svg_path)

        # Teacher forcing: shift coordinates by 1
        svg_tokens = np.zeros((self.max_seq_len, 2), dtype=np.float32)
        svg_tokens[1:] = coordinates[:-1]  # Previous token as input

        # Targets
        target_cmd = torch.from_numpy(commands).long()
        target_args = torch.from_numpy(coordinates).float()
        svg_tokens = torch.from_numpy(svg_tokens).float()

        return image, class_label, svg_tokens, target_cmd, target_args


class SVGVAEDatasetNoSVG(Dataset):
    """Dataset for training without SVG (image-only VAE)."""

    def __init__(
        self,
        data: List[Dict],
        num_classes: int = 62,
        transform=None,
    ):
        """
        Initialize dataset without SVG requirement.

        Args:
            data: List of processed character data
            num_classes: Number of character classes
            transform: Optional transform for images
        """
        self.data = [d for d in data if d.get("image") is not None]
        self.num_classes = num_classes
        self.transform = transform

        print(f"Dataset: {len(self.data)} samples (image-only)")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            image: (1, 64, 64) tensor
            class_label: (num_classes,) one-hot tensor
        """
        item = self.data[idx]

        # Image
        image = item["image"]
        if self.transform:
            image = self.transform(image)
        image = torch.from_numpy(image).float().unsqueeze(0)

        # Class label
        class_idx = item["class_idx"]
        class_label = torch.zeros(self.num_classes)
        class_label[class_idx] = 1.0

        return image, class_label


def create_dataloader(
    data: List[Dict],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    max_seq_len: int = 100,
    require_svg: bool = True,
) -> DataLoader:
    """
    Create DataLoader from processed font data.

    Args:
        data: List of processed character data
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes
        max_seq_len: Maximum SVG sequence length
        require_svg: Whether to require SVG data

    Returns:
        DataLoader instance
    """
    if require_svg:
        dataset = SVGVAEDataset(data, max_seq_len=max_seq_len)
    else:
        dataset = SVGVAEDatasetNoSVG(data)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def save_processed_data(data: List[Dict], path: str):
    """Save processed data to pickle file."""
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved {len(data)} samples to {path}")


def load_processed_data(path: str) -> List[Dict]:
    """Load processed data from pickle file."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded {len(data)} samples from {path}")
    return data


if __name__ == "__main__":
    # Test with dummy data
    dummy_data = []
    for i in range(100):
        char_idx = i % 62
        dummy_data.append({
            "image": np.random.rand(64, 64).astype(np.float32),
            "svg": f"M {np.random.rand()*50} {np.random.rand()*50} L {np.random.rand()*50} {np.random.rand()*50}",
            "class_idx": char_idx,
            "char": CHARACTERS[char_idx],
        })

    # Create dataset
    dataset = SVGVAEDataset(dummy_data, max_seq_len=50)
    print(f"\nDataset length: {len(dataset)}")

    # Get sample
    image, class_label, svg_tokens, target_cmd, target_args = dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Class label shape: {class_label.shape}")
    print(f"SVG tokens shape: {svg_tokens.shape}")
    print(f"Target cmd shape: {target_cmd.shape}")
    print(f"Target args shape: {target_args.shape}")

    # Create dataloader
    dataloader = create_dataloader(dummy_data, batch_size=16, max_seq_len=50)
    batch = next(iter(dataloader))
    print(f"\nBatch shapes:")
    for i, t in enumerate(batch):
        print(f"  [{i}]: {t.shape}")

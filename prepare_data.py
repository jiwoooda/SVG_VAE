"""Data preparation script for SVG-VAE."""

import argparse
from pathlib import Path

from data.font_downloader import download_multiple_fonts, get_all_ttf_files, DEFAULT_FONTS
from data.font_processor import FontProcessor
from data.dataset import save_processed_data, load_processed_data, create_dataloader


def main():
    parser = argparse.ArgumentParser(description="Prepare data for SVG-VAE training")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download fonts from Google Fonts",
    )
    parser.add_argument(
        "--fonts-dir",
        type=str,
        default="./fonts",
        help="Directory to save/load fonts",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./processed_data.pkl",
        help="Output path for processed data",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=64,
        help="Output image size",
    )
    parser.add_argument(
        "--num-fonts",
        type=int,
        default=10,
        help="Number of fonts to download",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test loading the processed data",
    )

    args = parser.parse_args()

    # Step 1: Download fonts (optional)
    if args.download:
        print("=" * 50)
        print("Step 1: Downloading fonts from Google Fonts")
        print("=" * 50)
        fonts_to_download = DEFAULT_FONTS[:args.num_fonts]
        download_multiple_fonts(fonts_to_download, args.fonts_dir)
        print()

    # Step 2: Get all TTF files
    print("=" * 50)
    print("Step 2: Finding TTF files")
    print("=" * 50)
    ttf_files = get_all_ttf_files(args.fonts_dir)

    if not ttf_files:
        print(f"No TTF files found in {args.fonts_dir}")
        print("\nOptions:")
        print("  1. Run with --download to download fonts from Google Fonts")
        print("  2. Copy TTF files to the fonts directory manually")
        print(f"\nExample: python prepare_data.py --download --fonts-dir {args.fonts_dir}")
        return

    print(f"Found {len(ttf_files)} TTF files")
    for ttf in ttf_files[:5]:
        print(f"  {ttf}")
    if len(ttf_files) > 5:
        print(f"  ... and {len(ttf_files) - 5} more")
    print()

    # Step 3: Process fonts
    print("=" * 50)
    print("Step 3: Processing fonts")
    print("=" * 50)
    processor = FontProcessor(image_size=args.image_size)
    all_data = processor.process_multiple_fonts([str(f) for f in ttf_files])
    print()

    # Step 4: Save processed data
    print("=" * 50)
    print("Step 4: Saving processed data")
    print("=" * 50)
    save_processed_data(all_data, args.output)
    print()

    # Step 5: Test loading (optional)
    if args.test:
        print("=" * 50)
        print("Step 5: Testing data loading")
        print("=" * 50)
        loaded_data = load_processed_data(args.output)

        # Create dataloader
        dataloader = create_dataloader(
            loaded_data,
            batch_size=16,
            max_seq_len=50,
            require_svg=True,
        )

        # Test batch
        batch = next(iter(dataloader))
        print(f"\nBatch shapes:")
        print(f"  Image: {batch[0].shape}")
        print(f"  Class label: {batch[1].shape}")
        print(f"  SVG tokens: {batch[2].shape}")
        print(f"  Target cmd: {batch[3].shape}")
        print(f"  Target args: {batch[4].shape}")

    print("\n" + "=" * 50)
    print("Data preparation complete!")
    print("=" * 50)
    print(f"\nProcessed data saved to: {args.output}")
    print(f"Total samples: {len(all_data)}")
    print(f"\nTo train the model, run:")
    print(f"  python train.py --data {args.output}")


if __name__ == "__main__":
    main()

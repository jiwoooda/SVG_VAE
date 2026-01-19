"""Download fonts from Google Fonts."""

import os
import requests
import zipfile
from pathlib import Path
from typing import List, Optional


# Popular Google Fonts for testing
DEFAULT_FONTS = [
    "Roboto",
    "Open+Sans",
    "Lato",
    "Montserrat",
    "Oswald",
    "Raleway",
    "Poppins",
    "Noto+Sans",
    "Ubuntu",
    "Playfair+Display",
]


def download_google_font(
    font_name: str,
    output_dir: str = "./fonts",
) -> Optional[Path]:
    """
    Download a font from Google Fonts.

    Args:
        font_name: Font name (use + for spaces, e.g., "Open+Sans")
        output_dir: Directory to save fonts

    Returns:
        Path to downloaded font file, or None if failed
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Google Fonts API URL
    api_url = f"https://fonts.google.com/download?family={font_name}"

    try:
        print(f"Downloading {font_name}...")
        response = requests.get(api_url, timeout=30)
        response.raise_for_status()

        # Save zip file
        zip_path = output_path / f"{font_name}.zip"
        with open(zip_path, "wb") as f:
            f.write(response.content)

        # Extract zip
        font_dir = output_path / font_name.replace("+", "_")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(font_dir)

        # Remove zip file
        zip_path.unlink()

        # Find TTF files
        ttf_files = list(font_dir.glob("**/*.ttf"))
        if ttf_files:
            print(f"  Downloaded {len(ttf_files)} TTF files")
            return font_dir

        print(f"  No TTF files found in {font_name}")
        return None

    except requests.RequestException as e:
        print(f"  Failed to download {font_name}: {e}")
        return None


def download_multiple_fonts(
    font_names: List[str] = None,
    output_dir: str = "./fonts",
) -> List[Path]:
    """
    Download multiple fonts from Google Fonts.

    Args:
        font_names: List of font names (default: popular fonts)
        output_dir: Directory to save fonts

    Returns:
        List of paths to downloaded font directories
    """
    if font_names is None:
        font_names = DEFAULT_FONTS

    downloaded = []
    for font_name in font_names:
        result = download_google_font(font_name, output_dir)
        if result:
            downloaded.append(result)

    print(f"\nDownloaded {len(downloaded)}/{len(font_names)} fonts")
    return downloaded


def get_all_ttf_files(fonts_dir: str = "./fonts") -> List[Path]:
    """
    Get all TTF files in the fonts directory.

    Args:
        fonts_dir: Root fonts directory

    Returns:
        List of paths to TTF files
    """
    fonts_path = Path(fonts_dir)
    if not fonts_path.exists():
        return []
    return list(fonts_path.glob("**/*.ttf"))


if __name__ == "__main__":
    # Download some fonts for testing
    downloaded = download_multiple_fonts(
        font_names=["Roboto", "Open+Sans", "Lato"],
        output_dir="./fonts",
    )

    # List all TTF files
    ttf_files = get_all_ttf_files("./fonts")
    print(f"\nTotal TTF files: {len(ttf_files)}")
    for ttf in ttf_files[:10]:
        print(f"  {ttf}")

"""Process TTF fonts to extract SVG paths and render images."""

import io
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont

try:
    from fontTools.ttLib import TTFont
    from fontTools.pens.svgPathPen import SVGPathPen
    from fontTools.pens.t2CharStringPen import T2CharStringPen
    FONTTOOLS_AVAILABLE = True
except ImportError:
    FONTTOOLS_AVAILABLE = False
    print("Warning: fontTools not installed. Run: pip install fonttools")


# 62 characters: A-Z, a-z, 0-9
CHARACTERS = (
    list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") +
    list("abcdefghijklmnopqrstuvwxyz") +
    list("0123456789")
)
CHAR_TO_IDX = {c: i for i, c in enumerate(CHARACTERS)}
IDX_TO_CHAR = {i: c for i, c in enumerate(CHARACTERS)}


class FontProcessor:
    """Process TTF fonts to extract SVG paths and render images."""

    def __init__(
        self,
        image_size: int = 64,
        padding: int = 4,
    ):
        """
        Initialize FontProcessor.

        Args:
            image_size: Output image size (default: 64x64)
            padding: Padding around character
        """
        self.image_size = image_size
        self.padding = padding
        self.characters = CHARACTERS

    def render_character_image(
        self,
        font_path: str,
        char: str,
        font_size: int = 50,
    ) -> Optional[np.ndarray]:
        """
        Render a character as a grayscale image.

        Args:
            font_path: Path to TTF font file
            char: Character to render
            font_size: Font size for rendering

        Returns:
            Grayscale image as numpy array (H, W), or None if failed
        """
        try:
            # Create image
            img = Image.new("L", (self.image_size, self.image_size), color=255)
            draw = ImageDraw.Draw(img)

            # Load font
            font = ImageFont.truetype(str(font_path), font_size)

            # Get text bounding box
            bbox = draw.textbbox((0, 0), char, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # Center text
            x = (self.image_size - text_width) // 2 - bbox[0]
            y = (self.image_size - text_height) // 2 - bbox[1]

            # Draw text
            draw.text((x, y), char, font=font, fill=0)

            # Convert to numpy array and normalize
            img_array = np.array(img, dtype=np.float32) / 255.0

            # Invert (black text on white -> white text on black)
            img_array = 1.0 - img_array

            return img_array

        except Exception as e:
            print(f"Failed to render '{char}' from {font_path}: {e}")
            return None

    def extract_svg_path(
        self,
        font_path: str,
        char: str,
    ) -> Optional[str]:
        """
        Extract SVG path data from a character in a font.

        Args:
            font_path: Path to TTF font file
            char: Character to extract

        Returns:
            SVG path string, or None if failed
        """
        if not FONTTOOLS_AVAILABLE:
            return None

        try:
            font = TTFont(font_path)
            glyph_set = font.getGlyphSet()

            # Get glyph name for character
            cmap = font.getBestCmap()
            if cmap is None or ord(char) not in cmap:
                return None

            glyph_name = cmap[ord(char)]
            if glyph_name not in glyph_set:
                return None

            # Extract path using SVGPathPen
            pen = SVGPathPen(glyph_set)
            glyph_set[glyph_name].draw(pen)
            path_data = pen.getCommands()

            font.close()
            return path_data

        except Exception as e:
            print(f"Failed to extract SVG for '{char}' from {font_path}: {e}")
            return None

    def process_font(
        self,
        font_path: str,
    ) -> Dict[str, Dict]:
        """
        Process a font file to extract all character images and SVG paths.

        Args:
            font_path: Path to TTF font file

        Returns:
            Dictionary mapping characters to their data:
            {
                'A': {'image': np.array, 'svg': str, 'class_idx': int},
                ...
            }
        """
        results = {}

        for char in self.characters:
            # Render image
            image = self.render_character_image(font_path, char)
            if image is None:
                continue

            # Extract SVG
            svg_path = self.extract_svg_path(font_path, char)

            results[char] = {
                "image": image,
                "svg": svg_path,
                "class_idx": CHAR_TO_IDX[char],
            }

        return results

    def process_multiple_fonts(
        self,
        font_paths: List[str],
        progress: bool = True,
    ) -> List[Dict]:
        """
        Process multiple font files.

        Args:
            font_paths: List of paths to TTF font files
            progress: Show progress

        Returns:
            List of processed character data
        """
        all_data = []

        for i, font_path in enumerate(font_paths):
            if progress:
                print(f"Processing [{i+1}/{len(font_paths)}]: {Path(font_path).name}")

            font_data = self.process_font(font_path)

            for char, data in font_data.items():
                data["font_path"] = str(font_path)
                data["char"] = char
                all_data.append(data)

        print(f"Processed {len(all_data)} character samples from {len(font_paths)} fonts")
        return all_data


if __name__ == "__main__":
    # Test with a system font
    import sys

    processor = FontProcessor(image_size=64)

    # Find a font to test
    if sys.platform == "win32":
        test_font = "C:/Windows/Fonts/arial.ttf"
    elif sys.platform == "darwin":
        test_font = "/System/Library/Fonts/Helvetica.ttc"
    else:
        test_font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

    if Path(test_font).exists():
        print(f"Testing with: {test_font}")
        data = processor.process_font(test_font)
        print(f"Processed {len(data)} characters")

        # Show sample
        if "A" in data:
            print(f"Character 'A':")
            print(f"  Image shape: {data['A']['image'].shape}")
            print(f"  SVG path: {data['A']['svg'][:100] if data['A']['svg'] else 'None'}...")
            print(f"  Class index: {data['A']['class_idx']}")
    else:
        print(f"Test font not found: {test_font}")

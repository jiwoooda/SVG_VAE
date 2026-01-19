"""Tokenize SVG paths for the SVG decoder."""

import re
import numpy as np
from typing import List, Tuple, Optional


# SVG command types
CMD_MOVE_TO = 0      # M (moveTo)
CMD_LINE_TO = 1      # L (lineTo)
CMD_CURVE_TO = 2     # C (cubicBezier)
CMD_EOS = 3          # End of sequence

CMD_NAMES = {
    CMD_MOVE_TO: "moveTo",
    CMD_LINE_TO: "lineTo",
    CMD_CURVE_TO: "cubicBezier",
    CMD_EOS: "EOS",
}


class SVGTokenizer:
    """Tokenize and detokenize SVG path commands."""

    def __init__(
        self,
        max_seq_len: int = 100,
        normalize: bool = True,
        coord_range: Tuple[float, float] = (-1.0, 1.0),
    ):
        """
        Initialize SVGTokenizer.

        Args:
            max_seq_len: Maximum sequence length
            normalize: Whether to normalize coordinates
            coord_range: Range for normalized coordinates
        """
        self.max_seq_len = max_seq_len
        self.normalize = normalize
        self.coord_range = coord_range

        # Regex patterns for SVG path commands
        self.cmd_pattern = re.compile(
            r"([MmLlCcZzHhVvSsQqTtAa])"
            r"([^MmLlCcZzHhVvSsQqTtAa]*)"
        )
        self.num_pattern = re.compile(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?")

    def parse_svg_path(self, path_data: str) -> List[Tuple[int, List[float]]]:
        """
        Parse SVG path string into command tokens.

        Args:
            path_data: SVG path data string (e.g., "M 10 10 L 20 20 C 30 30 40 40 50 50")

        Returns:
            List of (command_type, [coordinates]) tuples
        """
        if not path_data:
            return []

        tokens = []
        matches = self.cmd_pattern.findall(path_data)

        for cmd, args in matches:
            # Parse coordinates
            coords = [float(x) for x in self.num_pattern.findall(args)]

            # Convert command to type
            cmd_upper = cmd.upper()

            if cmd_upper == "M":
                # MoveTo
                if len(coords) >= 2:
                    tokens.append((CMD_MOVE_TO, coords[:2]))
                    # Additional coordinate pairs are implicit LineTo
                    for i in range(2, len(coords), 2):
                        if i + 1 < len(coords):
                            tokens.append((CMD_LINE_TO, coords[i:i+2]))

            elif cmd_upper == "L":
                # LineTo
                for i in range(0, len(coords), 2):
                    if i + 1 < len(coords):
                        tokens.append((CMD_LINE_TO, coords[i:i+2]))

            elif cmd_upper == "C":
                # CubicBezier (6 coordinates: control1, control2, end)
                for i in range(0, len(coords), 6):
                    if i + 5 < len(coords):
                        # We only keep the end point for simplicity
                        # Full: control1(2) + control2(2) + end(2)
                        tokens.append((CMD_CURVE_TO, coords[i+4:i+6]))

            elif cmd_upper == "H":
                # Horizontal line (only x coordinate)
                for x in coords:
                    tokens.append((CMD_LINE_TO, [x, 0.0]))

            elif cmd_upper == "V":
                # Vertical line (only y coordinate)
                for y in coords:
                    tokens.append((CMD_LINE_TO, [0.0, y]))

            elif cmd_upper == "Z":
                # Close path - treat as moveTo origin
                pass

        return tokens

    def normalize_coordinates(
        self,
        tokens: List[Tuple[int, List[float]]],
    ) -> List[Tuple[int, List[float]]]:
        """
        Normalize coordinates to specified range.

        Args:
            tokens: List of (command_type, coordinates) tuples

        Returns:
            Normalized tokens
        """
        if not tokens:
            return tokens

        # Find bounds
        all_coords = []
        for cmd, coords in tokens:
            all_coords.extend(coords)

        if not all_coords:
            return tokens

        min_val = min(all_coords)
        max_val = max(all_coords)
        range_val = max_val - min_val if max_val != min_val else 1.0

        # Normalize
        target_min, target_max = self.coord_range
        target_range = target_max - target_min

        normalized = []
        for cmd, coords in tokens:
            norm_coords = [
                ((c - min_val) / range_val) * target_range + target_min
                for c in coords
            ]
            normalized.append((cmd, norm_coords))

        return normalized

    def tokenize(
        self,
        svg_path: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tokenize SVG path to command and coordinate arrays.

        Args:
            svg_path: SVG path data string

        Returns:
            commands: Array of command types (seq_len,)
            coordinates: Array of coordinates (seq_len, 2)
        """
        # Parse path
        tokens = self.parse_svg_path(svg_path)

        # Normalize if requested
        if self.normalize:
            tokens = self.normalize_coordinates(tokens)

        # Initialize arrays
        commands = np.full(self.max_seq_len, CMD_EOS, dtype=np.int64)
        coordinates = np.zeros((self.max_seq_len, 2), dtype=np.float32)

        # Fill arrays
        for i, (cmd, coords) in enumerate(tokens[:self.max_seq_len - 1]):
            commands[i] = cmd
            if len(coords) >= 2:
                coordinates[i] = coords[:2]

        # Add EOS at the end
        if len(tokens) < self.max_seq_len:
            commands[len(tokens)] = CMD_EOS

        return commands, coordinates

    def detokenize(
        self,
        commands: np.ndarray,
        coordinates: np.ndarray,
    ) -> str:
        """
        Convert command and coordinate arrays back to SVG path string.

        Args:
            commands: Array of command types
            coordinates: Array of coordinates

        Returns:
            SVG path data string
        """
        path_parts = []

        for cmd, (x, y) in zip(commands, coordinates):
            if cmd == CMD_EOS:
                break
            elif cmd == CMD_MOVE_TO:
                path_parts.append(f"M {x:.2f} {y:.2f}")
            elif cmd == CMD_LINE_TO:
                path_parts.append(f"L {x:.2f} {y:.2f}")
            elif cmd == CMD_CURVE_TO:
                # Simplified: just endpoint, no control points
                path_parts.append(f"L {x:.2f} {y:.2f}")

        return " ".join(path_parts)

    def to_svg_element(
        self,
        commands: np.ndarray,
        coordinates: np.ndarray,
        width: int = 64,
        height: int = 64,
        stroke_width: float = 2.0,
    ) -> str:
        """
        Generate complete SVG element from tokens.

        Args:
            commands: Array of command types
            coordinates: Array of coordinates
            width: SVG width
            height: SVG height
            stroke_width: Stroke width

        Returns:
            Complete SVG XML string
        """
        # Denormalize coordinates to pixel space
        target_min, target_max = self.coord_range
        target_range = target_max - target_min

        denorm_coords = (coordinates - target_min) / target_range * width

        path_data = self.detokenize(commands, denorm_coords)

        svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
  <path d="{path_data}" fill="none" stroke="black" stroke-width="{stroke_width}"/>
</svg>'''

        return svg


if __name__ == "__main__":
    # Test tokenizer
    tokenizer = SVGTokenizer(max_seq_len=50)

    # Sample SVG path
    test_path = "M 10 20 L 30 40 L 50 20 C 60 10 70 30 80 20 Z"
    print(f"Input path: {test_path}")

    # Tokenize
    commands, coordinates = tokenizer.tokenize(test_path)
    print(f"\nCommands: {commands[:10]}")
    print(f"Coordinates:\n{coordinates[:10]}")

    # Detokenize
    reconstructed = tokenizer.detokenize(commands, coordinates)
    print(f"\nReconstructed: {reconstructed}")

    # Generate SVG
    svg_element = tokenizer.to_svg_element(commands, coordinates)
    print(f"\nSVG Element:\n{svg_element}")

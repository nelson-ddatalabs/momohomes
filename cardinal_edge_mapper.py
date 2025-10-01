#!/usr/bin/env python3
"""
Cardinal Edge Mapper
====================
Maps user measurements to cardinal direction edges.
Ensures perfect polygon closure by using exact N/S/E/W movements.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple
import logging
from cardinal_edge_detector import CardinalEdge

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CardinalEdgeMapper:
    """Maps measurements to edges with cardinal directions."""

    def __init__(self, edges: List[CardinalEdge], image: np.ndarray):
        """
        Initialize mapper with cardinal edges.

        Args:
            edges: List of CardinalEdge objects
            image: Binary or original image for display
        """
        self.edges = edges
        self.image = image
        self.measurements = {}
        self.scale_factor = None

    def display_cardinal_edges(self, save_path: str = "cardinal_edges.png") -> np.ndarray:
        """
        Display numbered edges with cardinal directions.

        Args:
            save_path: Path to save visualization

        Returns:
            Image with numbered cardinal edges
        """
        # Convert to BGR if grayscale
        if len(self.image.shape) == 2:
            display = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        else:
            display = self.image.copy()

        # Direction colors
        colors = {
            'E': (255, 0, 0),    # Blue
            'W': (255, 0, 255),  # Magenta
            'N': (0, 255, 0),    # Green
            'S': (0, 0, 255)     # Red
        }

        for i, edge in enumerate(self.edges):
            color = colors[edge.cardinal_direction]

            # Draw edge
            cv2.line(display, edge.start, edge.end, color, 3)

            # Calculate midpoint
            mid_x = (edge.start[0] + edge.end[0]) // 2
            mid_y = (edge.start[1] + edge.end[1]) // 2

            # Create label
            label = f"{i}:{edge.cardinal_direction}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2

            # Get text size
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)

            # Draw background
            cv2.rectangle(display,
                         (mid_x - text_width // 2 - 5, mid_y - text_height // 2 - 5),
                         (mid_x + text_width // 2 + 5, mid_y + text_height // 2 + 5),
                         (255, 255, 255), -1)

            # Draw text
            cv2.putText(display, label,
                       (mid_x - text_width // 2, mid_y + text_height // 2),
                       font, font_scale, (0, 0, 0), thickness)

        # Add instructions
        cv2.putText(display, "CARDINAL EDGES - Enter measurements clockwise",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Add direction legend
        legend_y = 60
        cv2.putText(display, "Directions: E=East(→) W=West(←) N=North(↑) S=South(↓)",
                   (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imwrite(save_path, display)
        logger.info(f"Cardinal edges saved to: {save_path}")

        return display

    def collect_cardinal_measurements(self) -> Dict[int, float]:
        """
        Collect measurements for cardinal edges.

        Returns:
            Dictionary mapping edge index to measurement in feet
        """
        print("\n" + "="*70)
        print("CARDINAL EDGE MEASUREMENT COLLECTION")
        print("="*70)
        print("\nEnter measurements for each edge going CLOCKWISE.")
        print("Cardinal directions ensure perfect polygon closure.\n")

        for i, edge in enumerate(self.edges):
            while True:
                try:
                    # Show edge info
                    direction_names = {
                        'E': "East (→)",
                        'W': "West (←)",
                        'N': "North (↑)",
                        'S': "South (↓)"
                    }

                    print(f"\nEdge {i}: {direction_names[edge.cardinal_direction]}")
                    print(f"  Pixel length: {edge.pixel_length:.1f}")

                    # Get measurement
                    measurement = input(f"  Enter measurement (feet): ").strip()

                    # Parse measurement
                    value = float(measurement)

                    if value <= 0:
                        print("  Measurement must be positive.")
                        continue

                    self.measurements[i] = value
                    edge.measurement = value
                    print(f"  ✓ Recorded: {value} feet {edge.cardinal_direction}")
                    break

                except ValueError:
                    print("  Invalid input. Please enter a number.")

        return self.measurements

    def calculate_scale_factor(self) -> float:
        """
        Calculate average scale factor from measurements.

        Returns:
            Scale factor in feet per pixel
        """
        if not self.measurements:
            logger.error("No measurements to calculate scale factor")
            return None

        scale_factors = []

        for i, edge in enumerate(self.edges):
            if i in self.measurements and edge.pixel_length > 0 and self.measurements[i] > 0:
                scale = self.measurements[i] / edge.pixel_length
                scale_factors.append(scale)
                logger.debug(f"Edge {i}: {self.measurements[i]}ft / {edge.pixel_length:.1f}px = {scale:.4f} ft/px")

        if scale_factors:
            self.scale_factor = np.median(scale_factors)
            logger.info(f"Scale factor: {self.scale_factor:.4f} feet/pixel")

        return self.scale_factor

    def get_cardinal_polygon(self) -> List[Tuple[float, float]]:
        """
        Build polygon using cardinal directions.

        Returns:
            List of (x, y) vertices in feet
        """
        polygon = []
        x, y = 0.0, 0.0
        polygon.append((x, y))

        # Direction vectors
        directions = {
            'E': (1, 0),
            'W': (-1, 0),
            'N': (0, 1),
            'S': (0, -1)
        }

        for i, edge in enumerate(self.edges):
            if i in self.measurements:
                dx, dy = directions[edge.cardinal_direction]
                x += dx * self.measurements[i]
                y += dy * self.measurements[i]
                polygon.append((x, y))

        # Remove duplicate last point if it closes
        if len(polygon) > 1 and polygon[-1] == polygon[0]:
            polygon = polygon[:-1]

        return polygon

    # NOTE: verify_closure method removed - we use measurement-based closure from CardinalPolygonReconstructor instead

    def save_measurements(self, filepath: str):
        """
        Save measurements with cardinal directions.

        Args:
            filepath: Path to save JSON file
        """
        import json

        data = {
            'edges': [
                {
                    'index': i,
                    'cardinal_direction': edge.cardinal_direction,
                    'pixel_length': edge.pixel_length,
                    'measurement': self.measurements.get(i)
                }
                for i, edge in enumerate(self.edges)
            ],
            'scale_factor': self.scale_factor,
            'polygon': self.get_cardinal_polygon()
        }

        # NOTE: Closure verification now done in CardinalPolygonReconstructor with measurement-based polygon

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Cardinal measurements saved to: {filepath}")
#!/usr/bin/env python3
"""
Interactive Edge Mapper
=======================
Displays numbered edges to user and collects measurements clockwise.
Maps user measurements to detected edges and calculates scale factor.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from edge_detection_system import Edge

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InteractiveEdgeMapper:
    """Maps user measurements to detected edges."""

    def __init__(self, edges: List[Edge], image: np.ndarray):
        """
        Initialize mapper with detected edges.

        Args:
            edges: List of Edge objects from edge detection
            image: Original or binary image for display
        """
        self.edges = edges
        self.image = image
        self.measurements = {}
        self.scale_factor = None

    def display_edges(self, save_path: str = "numbered_edges.png") -> np.ndarray:
        """
        Display numbered edges for user reference.

        Args:
            save_path: Path to save numbered edge image

        Returns:
            Image with numbered edges
        """
        # Convert to BGR if grayscale
        if len(self.image.shape) == 2:
            display = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        else:
            display = self.image.copy()

        # Draw each edge with its number
        for i, edge in enumerate(self.edges):
            # Draw edge
            cv2.line(display, edge.start, edge.end, (0, 255, 0), 3)

            # Calculate midpoint for label
            mid_x = (edge.start[0] + edge.end[0]) // 2
            mid_y = (edge.start[1] + edge.end[1]) // 2

            # Draw edge number with background for visibility
            label = str(i)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thickness = 2

            # Get text size
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)

            # Draw background rectangle
            cv2.rectangle(display,
                         (mid_x - text_width // 2 - 5, mid_y - text_height // 2 - 5),
                         (mid_x + text_width // 2 + 5, mid_y + text_height // 2 + 5),
                         (255, 255, 255), -1)

            # Draw text
            cv2.putText(display, label,
                       (mid_x - text_width // 2, mid_y + text_height // 2),
                       font, font_scale, (0, 0, 255), thickness)

        # Add instruction text
        cv2.putText(display, "NUMBERED EDGES - Enter measurements clockwise",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Save for reference
        cv2.imwrite(save_path, display)
        logger.info(f"Numbered edges saved to: {save_path}")

        return display

    def collect_measurements(self) -> Dict[int, float]:
        """
        Collect measurements from user for each edge.

        Returns:
            Dictionary mapping edge index to measurement in feet
        """
        print("\n" + "="*70)
        print("EDGE MEASUREMENT COLLECTION")
        print("="*70)
        print("\nPlease refer to 'numbered_edges.png' for edge numbers.")
        print("Enter measurements in feet going clockwise from Edge 0.\n")

        for i, edge in enumerate(self.edges):
            while True:
                try:
                    # Show edge info
                    print(f"\nEdge {i} ({edge.orientation}):")
                    print(f"  From: {edge.start}")
                    print(f"  To: {edge.end}")
                    print(f"  Pixel length: {edge.length:.1f}")

                    # Get measurement
                    measurement = input(f"  Enter measurement (feet): ").strip()

                    # Parse measurement
                    if '.' in measurement:
                        value = float(measurement)
                    else:
                        value = float(measurement)

                    if value <= 0:
                        print("  Measurement must be positive.")
                        continue

                    self.measurements[i] = value
                    print(f"  Recorded: {value} feet")
                    break

                except ValueError:
                    print("  Invalid input. Please enter a number.")

        return self.measurements

    def calculate_scale_factor(self) -> float:
        """
        Calculate scale factor from measurements.

        Returns:
            Scale factor in feet per pixel
        """
        if not self.measurements:
            logger.error("No measurements to calculate scale factor")
            return None

        scale_factors = []

        for i, edge in enumerate(self.edges):
            if i in self.measurements:
                pixel_length = edge.length
                real_length = self.measurements[i]

                if pixel_length > 0:
                    scale = real_length / pixel_length
                    scale_factors.append(scale)
                    logger.debug(f"Edge {i}: {real_length} ft / {pixel_length} px = {scale:.4f} ft/px")

        if scale_factors:
            # Use median for robustness
            self.scale_factor = np.median(scale_factors)
            logger.info(f"Calculated scale factor: {self.scale_factor:.4f} feet/pixel")

            # Check consistency
            std_dev = np.std(scale_factors)
            if std_dev > 0.01:  # More than 1% variation
                logger.warning(f"Scale factors vary significantly (std: {std_dev:.4f})")

        return self.scale_factor

    def validate_measurements(self) -> Tuple[bool, float]:
        """
        Validate that measurements form a closed polygon.

        Returns:
            Tuple of (is_valid, closure_error_feet)
        """
        if not self.measurements or not self.scale_factor:
            return False, float('inf')

        # Build polygon from measurements
        x, y = 0.0, 0.0
        vertices = [(x, y)]

        for i, edge in enumerate(self.edges):
            if i not in self.measurements:
                logger.warning(f"Missing measurement for edge {i}")
                return False, float('inf')

            # Calculate direction vector
            dx = edge.end[0] - edge.start[0]
            dy = edge.end[1] - edge.start[1]
            length_pixels = edge.length

            # Normalize and scale by measurement
            if length_pixels > 0:
                dx = dx / length_pixels * self.measurements[i]
                dy = dy / length_pixels * self.measurements[i]

                x += dx
                y += dy
                vertices.append((x, y))

        # Check closure
        first = vertices[0]
        last = vertices[-1]
        error = np.sqrt((last[0] - first[0])**2 + (last[1] - first[1])**2)

        is_valid = error < 1.0  # Less than 1 foot error

        if is_valid:
            logger.info(f"Polygon closes successfully (error: {error:.3f} feet)")
        else:
            logger.warning(f"Polygon does not close (error: {error:.1f} feet)")

        return is_valid, error

    def get_mapped_polygon(self) -> List[Tuple[float, float]]:
        """
        Get polygon with real-world measurements.

        Returns:
            List of (x, y) vertices in feet
        """
        polygon = []
        x, y = 0.0, 0.0
        polygon.append((x, y))

        for i, edge in enumerate(self.edges):
            if i in self.measurements:
                # Calculate direction vector
                dx = edge.end[0] - edge.start[0]
                dy = edge.end[1] - edge.start[1]
                length_pixels = edge.length

                # Normalize and scale by measurement
                if length_pixels > 0:
                    dx = dx / length_pixels * self.measurements[i]
                    dy = dy / length_pixels * self.measurements[i]

                    x += dx
                    y += dy
                    polygon.append((x, y))

        return polygon[:-1]  # Remove duplicate last point

    def save_measurements(self, filepath: str):
        """
        Save measurements to JSON file.

        Args:
            filepath: Path to save measurements
        """
        import json

        data = {
            'edges': [edge.to_dict() for edge in self.edges],
            'measurements': self.measurements,
            'scale_factor': self.scale_factor,
            'polygon': self.get_mapped_polygon()
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Measurements saved to: {filepath}")
#!/usr/bin/env python3
"""
Enhanced Binary Converter
=========================
Converts floor plan images to binary format where green areas (indoor space)
become white and everything else becomes black.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedBinaryConverter:
    """Converts floor plans to binary images for edge detection."""

    def __init__(self):
        """Initialize converter with HSV range for green detection."""
        # HSV range for green areas in floor plans
        self.lower_green = np.array([40, 40, 40])
        self.upper_green = np.array([80, 255, 255])

    def convert_to_binary(self, image_path: str) -> Optional[np.ndarray]:
        """
        Convert floor plan to binary image.

        Args:
            image_path: Path to floor plan image

        Returns:
            Binary image where white=indoor, black=outdoor, or None if error
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Cannot load image: {image_path}")
            return None

        h, w = image.shape[:2]
        logger.info(f"Processing image: {w}x{h} pixels")

        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Extract green areas
        green_mask = cv2.inRange(hsv, self.lower_green, self.upper_green)

        # Apply morphological operations to clean up
        green_mask = self._clean_binary_image(green_mask)

        logger.info(f"Binary conversion complete")
        return green_mask

    def _clean_binary_image(self, binary: np.ndarray) -> np.ndarray:
        """
        Clean binary image using morphological operations.

        Args:
            binary: Raw binary image

        Returns:
            Cleaned binary image
        """
        # Define kernel for morphological operations
        kernel = np.ones((5, 5), np.uint8)

        # Closing: fill small gaps and holes
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Opening: remove small noise
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        return binary

    def get_green_area_bounds(self, binary: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Get bounding box of white (indoor) areas.

        Args:
            binary: Binary image

        Returns:
            Tuple of (min_x, min_y, max_x, max_y)
        """
        # Find all white pixels
        white_pixels = np.where(binary > 0)

        if len(white_pixels[0]) == 0:
            logger.warning("No white pixels found in binary image")
            return (0, 0, binary.shape[1], binary.shape[0])

        min_y = white_pixels[0].min()
        max_y = white_pixels[0].max()
        min_x = white_pixels[1].min()
        max_x = white_pixels[1].max()

        return (min_x, min_y, max_x, max_y)

    def save_binary(self, binary: np.ndarray, output_path: str):
        """
        Save binary image to file.

        Args:
            binary: Binary image to save
            output_path: Path to save image
        """
        cv2.imwrite(output_path, binary)
        logger.info(f"Binary image saved to: {output_path}")
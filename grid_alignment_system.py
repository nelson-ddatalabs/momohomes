#!/usr/bin/env python3
"""
Grid Alignment System
=====================
Provides accurate coordinate transformation using grid-based reference points.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GridAlignmentSystem:
    """
    Grid-based alignment for accurate coordinate transformation.

    Uses a grid overlay to establish correspondence between world (feet)
    and image (pixels) coordinate systems.
    """

    def __init__(self, world_width: float = 78.0, world_height: float = 50.0,
                 image_width: int = 2807, image_height: int = 1983,
                 grid_spacing: float = 10.0):
        """
        Initialize grid alignment system.

        Args:
            world_width: Building width in feet
            world_height: Building height in feet
            image_width: Image width in pixels
            image_height: Image height in pixels
            grid_spacing: Grid spacing in feet
        """
        self.world_width = world_width
        self.world_height = world_height
        self.image_width = image_width
        self.image_height = image_height
        self.grid_spacing = grid_spacing

        # Known correspondences between world and image points
        self.correspondences = []

        # Homography matrix (calculated from correspondences)
        self.homography = None

        # Simple scale factors as fallback
        self.scale_x = image_width / world_width
        self.scale_y = image_height / world_height

        logger.info(f"Grid alignment initialized: {world_width}x{world_height}ft -> {image_width}x{image_height}px")

    def add_correspondence(self, world_pt: Tuple[float, float],
                          image_pt: Tuple[int, int], label: str = ""):
        """
        Add a known correspondence between world and image coordinates.

        Args:
            world_pt: (x, y) in feet
            image_pt: (x, y) in pixels
            label: Optional label for this point
        """
        self.correspondences.append({
            'world': world_pt,
            'image': image_pt,
            'label': label
        })
        logger.debug(f"Added correspondence: {world_pt} ft -> {image_pt} px ({label})")

    def add_luna_correspondences(self):
        """Add known correspondences for Luna floor plan."""
        # Key reference points for Luna building
        # Building is 78' wide x 50' tall (actual measurements)
        # Image is 2807 x 1983 pixels
        # Building occupies approximately 2000 pixels width, 1300 pixels height

        # Calculate approximate pixel positions based on building proportions
        # The building doesn't start at pixel (0,0) - there's padding around it
        padding_left = 400  # Approximate left padding
        padding_bottom = 350  # Approximate bottom padding

        building_width_pixels = 2000  # Approximate building width in pixels
        building_height_pixels = 1300  # Approximate building height in pixels

        # Building corners (world coordinates -> pixel coordinates)
        # Remember: Y-axis is inverted in image (top=0, bottom=1983)
        self.add_correspondence(
            (0, 0),  # Bottom-left corner in feet
            (padding_left, self.image_height - padding_bottom),  # Pixel position
            "Bottom-left (0,0)"
        )
        self.add_correspondence(
            (78, 0),  # Bottom-right corner in feet
            (padding_left + building_width_pixels, self.image_height - padding_bottom),
            "Bottom-right (78,0)"
        )
        self.add_correspondence(
            (78, 50),  # Top-right corner in feet
            (padding_left + building_width_pixels, self.image_height - padding_bottom - building_height_pixels),
            "Top-right (78,50)"
        )
        self.add_correspondence(
            (0, 50),  # Top-left corner in feet
            (padding_left, self.image_height - padding_bottom - building_height_pixels),
            "Top-left (0,50)"
        )

        # Indoor space key points (excluding garage and patio)
        # Indoor starts at x=15 feet (garage is 0-15 feet)
        garage_width_ratio = 15.0 / 78.0  # Garage is 15 feet of 78 feet total
        indoor_left_pixels = padding_left + int(garage_width_ratio * building_width_pixels)

        # Indoor bottom is at y=14 feet (0-14 is cutout area)
        indoor_bottom_ratio = 14.0 / 50.0  # Bottom cutout is 14 feet of 50 feet total
        indoor_bottom_pixels = self.image_height - padding_bottom - int(indoor_bottom_ratio * building_height_pixels)

        self.add_correspondence(
            (15, 14),  # Indoor bottom-left in feet
            (indoor_left_pixels, indoor_bottom_pixels),
            "Indoor bottom-left (15,14)"
        )

        # Add middle reference point for better accuracy
        self.add_correspondence(
            (39, 25),  # Center of building in feet
            (padding_left + building_width_pixels // 2,
             self.image_height - padding_bottom - building_height_pixels // 2),
            "Building center (39,25)"
        )

        logger.info(f"Added {len(self.correspondences)} Luna reference points with refined pixel positions")

    def calculate_homography(self) -> Optional[np.ndarray]:
        """
        Calculate homography matrix from correspondences.

        Returns:
            3x3 homography matrix or None if insufficient points
        """
        if len(self.correspondences) < 4:
            logger.warning(f"Need at least 4 correspondences, have {len(self.correspondences)}")
            return None

        # Extract source (world) and destination (image) points
        src_pts = np.array([c['world'] for c in self.correspondences], dtype=np.float32)
        dst_pts = np.array([c['image'] for c in self.correspondences], dtype=np.float32)

        # Calculate homography
        self.homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if self.homography is not None:
            logger.info("Homography matrix calculated successfully")
            logger.debug(f"Homography:\n{self.homography}")

            # Verify with test points
            self._verify_homography()
        else:
            logger.error("Failed to calculate homography")

        return self.homography

    def _verify_homography(self):
        """Verify homography accuracy with known points."""
        if self.homography is None:
            return

        errors = []
        logger.info("\nVerifying homography accuracy:")
        for corr in self.correspondences:
            world_pt = corr['world']
            expected = corr['image']
            actual = self.transform_point(world_pt[0], world_pt[1])

            error = np.sqrt((actual[0] - expected[0])**2 + (actual[1] - expected[1])**2)
            errors.append(error)

            status = "✓" if error < 50 else "✗"
            logger.info(f"{status} {corr['label']:30s}: Error: {error:6.1f}px  (Expected {expected}, Got {actual})")

        avg_error = np.mean(errors)
        max_error = np.max(errors)
        logger.info(f"\nAverage transformation error: {avg_error:.1f} pixels")
        logger.info(f"Maximum transformation error: {max_error:.1f} pixels")

        if avg_error < 30:
            logger.info("✓ Excellent alignment achieved!")
        elif avg_error < 50:
            logger.info("✓ Good alignment achieved")
        else:
            logger.warning("⚠ Alignment needs improvement")

    def transform_point(self, x_world: float, y_world: float) -> Tuple[int, int]:
        """
        Transform a point from world to image coordinates.

        Args:
            x_world: X coordinate in feet
            y_world: Y coordinate in feet

        Returns:
            (x_image, y_image) in pixels
        """
        if self.homography is not None:
            # Use homography transformation
            pt = np.array([[x_world, y_world]], dtype=np.float32).reshape(-1, 1, 2)
            transformed = cv2.perspectiveTransform(pt, self.homography)
            x_img, y_img = transformed[0][0]
        else:
            # Fallback to simple scaling with Y-axis inversion
            x_img = x_world * self.scale_x
            y_img = self.image_height - (y_world * self.scale_y)

        return int(x_img), int(y_img)

    def transform_rectangle(self, x: float, y: float, width: float, height: float) -> List[Tuple[int, int]]:
        """
        Transform a rectangle from world to image coordinates.

        Args:
            x: X position in feet (bottom-left corner)
            y: Y position in feet (bottom-left corner)
            width: Width in feet
            height: Height in feet

        Returns:
            List of 4 corner points in image coordinates
        """
        # Define corners in world coordinates
        corners_world = [
            (x, y),                    # Bottom-left
            (x + width, y),           # Bottom-right
            (x + width, y + height),  # Top-right
            (x, y + height)           # Top-left
        ]

        # Transform each corner
        corners_image = []
        for wx, wy in corners_world:
            ix, iy = self.transform_point(wx, wy)
            corners_image.append((ix, iy))

        return corners_image

    def create_grid_overlay(self, image: np.ndarray) -> np.ndarray:
        """
        Create a grid overlay on the image for visualization.

        Args:
            image: Input image

        Returns:
            Image with grid overlay
        """
        overlay = image.copy()
        h, w = image.shape[:2]

        # Draw vertical grid lines
        for x_world in np.arange(0, self.world_width + self.grid_spacing, self.grid_spacing):
            x_img, _ = self.transform_point(x_world, 0)
            _, y_top = self.transform_point(x_world, self.world_height)

            cv2.line(overlay, (x_img, 0), (x_img, h), (0, 255, 0), 1)
            cv2.putText(overlay, f"{int(x_world)}ft", (x_img + 5, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Draw horizontal grid lines
        for y_world in np.arange(0, self.world_height + self.grid_spacing, self.grid_spacing):
            _, y_img = self.transform_point(0, y_world)
            x_right, _ = self.transform_point(self.world_width, y_world)

            cv2.line(overlay, (0, y_img), (w, y_img), (0, 255, 0), 1)
            cv2.putText(overlay, f"{int(y_world)}ft", (10, y_img - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Mark correspondence points
        for corr in self.correspondences:
            ix, iy = corr['image']
            cv2.circle(overlay, (ix, iy), 5, (0, 0, 255), -1)
            if corr['label']:
                cv2.putText(overlay, corr['label'], (ix + 10, iy),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

        return overlay

    def calibrate_from_edges(self, edges: List, measurements: dict):
        """
        Calibrate alignment using detected edges and measurements.

        Args:
            edges: List of detected edges
            measurements: Dictionary of edge measurements in feet
        """
        # Use edge endpoints as correspondence points
        cumulative_x = 0
        cumulative_y = 0

        for i, edge in enumerate(edges):
            if i not in measurements:
                continue

            # Get edge measurement and pixel position
            length_feet = measurements[i]
            pixel_start = edge.start
            pixel_end = edge.end

            # Add as correspondence
            if edge.cardinal_direction in ['E', 'W']:
                cumulative_x += length_feet if edge.cardinal_direction == 'E' else -length_feet
            else:
                cumulative_y += length_feet if edge.cardinal_direction == 'N' else -length_feet

            self.add_correspondence(
                (cumulative_x, cumulative_y),
                pixel_end,
                f"Edge {i}"
            )

        logger.info(f"Calibrated with {len(self.correspondences)} edge points")


def test_grid_alignment():
    """Test the grid alignment system."""

    # Create alignment system
    aligner = GridAlignmentSystem()

    # Add Luna correspondences
    aligner.add_luna_correspondences()

    # Calculate homography
    H = aligner.calculate_homography()

    if H is not None:
        print("\nHomography Matrix:")
        print(H)

        # Test some cassette positions
        test_cassettes = [
            (0, 14, "Left wall bottom"),
            (15, 0, "Garage boundary"),
            (40, 25, "Center"),
            (78, 50, "Top-right corner")
        ]

        print("\nTest Transformations:")
        for x, y, label in test_cassettes:
            ix, iy = aligner.transform_point(x, y)
            print(f"{label}: ({x}, {y})ft -> ({ix}, {iy})px")

    return aligner


if __name__ == "__main__":
    test_grid_alignment()
#!/usr/bin/env python3
"""
Cassette Layout Visualizer
===========================
Creates professional visualization with cassettes overlaid on floor plan.
Includes semi-transparent colored rectangles, legend, and statistics.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

# Import grid alignment system for proper coordinate transformation
from grid_alignment_system import GridAlignmentSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CassetteLayoutVisualizer:
    """Creates professional cassette layout visualization."""

    def __init__(self):
        """Initialize visualizer with professional color scheme."""
        # Define professional color scheme (BGR format for OpenCV)
        # Colors chosen for good contrast and visibility
        self.color_scheme = {
            # Large cassettes (48 sq ft) - Deep Blue
            '6x8': (139, 69, 19),      # Dark Blue
            '8x6': (139, 69, 19),      # Dark Blue

            # Large cassettes (40 sq ft) - Purple
            '5x8': (128, 0, 128),      # Purple
            '8x5': (128, 0, 128),      # Purple

            # Medium cassettes (36 sq ft) - Green
            '6x6': (34, 139, 34),      # Forest Green

            # Medium cassettes (32 sq ft) - Orange
            '4x8': (0, 140, 255),      # Orange
            '8x4': (0, 140, 255),      # Orange

            # Medium cassettes (30 sq ft) - Teal
            '5x6': (128, 128, 0),      # Teal
            '6x5': (128, 128, 0),      # Teal

            # Small cassettes (24 sq ft) - Light Blue
            '4x6': (255, 191, 0),      # Light Blue
            '6x4': (255, 191, 0),      # Light Blue

            # Small cassettes (16 sq ft) - Pink
            '4x4': (203, 192, 255),    # Pink
            '2x8': (203, 192, 255),    # Pink
            '8x2': (203, 192, 255),    # Pink

            # Tiny cassettes (12 sq ft) - Gray
            '3x4': (169, 169, 169),    # Gray
            '4x3': (169, 169, 169),    # Gray
            '2x6': (169, 169, 169),    # Gray
            '6x2': (169, 169, 169),    # Gray

            # Smallest cassettes (8 sq ft) - Light Gray
            '2x4': (211, 211, 211),    # Light Gray
            '4x2': (211, 211, 211),    # Light Gray
        }

        self.alpha = 0.6  # Slightly more opaque for better visibility

    def world_to_image_coords(self, x_feet: float, y_feet: float,
                            image_height: int, scale_factor: float,
                            origin_offset: Tuple[float, float] = (0.0, 0.0)) -> Tuple[int, int]:
        """
        Convert world coordinates (feet) to image coordinates (pixels).

        Args:
            x_feet: X coordinate in feet (world space)
            y_feet: Y coordinate in feet (world space)
            image_height: Height of the image in pixels
            scale_factor: Feet per pixel conversion factor
            origin_offset: Offset of polygon origin in feet (x_offset, y_offset)

        Returns:
            Tuple of (x_pixels, y_pixels) in image space

        Note:
            World coordinates: Origin at bottom-left, Y increases upward
            Image coordinates: Origin at top-left, Y increases downward
        """
        # Apply origin offset
        x_adjusted = x_feet + origin_offset[0]
        y_adjusted = y_feet + origin_offset[1]

        # Convert X to pixels (no inversion needed)
        x_pixels = int(x_adjusted / scale_factor)

        # Convert Y to pixels with inversion
        y_pixels = int(image_height - (y_adjusted / scale_factor))

        return x_pixels, y_pixels

    def create_visualization(self,
                           floor_plan_path: str,
                           cassettes: List[Dict],
                           scale_factor: float,
                           polygon_pixels: List[Tuple[int, int]],
                           statistics: Dict,
                           output_path: str = "cassette_layout.png",
                           origin_offset: Tuple[float, float] = (0.0, 0.0),
                           grid_alignment: Optional[GridAlignmentSystem] = None) -> np.ndarray:
        """
        Create complete visualization with cassettes overlaid on floor plan.

        Args:
            floor_plan_path: Path to original floor plan image
            cassettes: List of cassette dictionaries from optimizer
            scale_factor: Scale factor (feet per pixel)
            polygon_pixels: Polygon vertices in pixel coordinates
            statistics: Optimization statistics
            output_path: Path to save visualization
            origin_offset: Offset of polygon origin in feet (x_offset, y_offset)
            grid_alignment: Optional grid alignment system for accurate transformation

        Returns:
            Final visualization image
        """
        # Load floor plan
        floor_plan = cv2.imread(floor_plan_path)
        if floor_plan is None:
            logger.error(f"Cannot load floor plan: {floor_plan_path}")
            return None

        h, w = floor_plan.shape[:2]
        logger.info(f"Creating visualization on {w}x{h} image")

        # Draw cassettes on floor plan
        visualization = self._draw_cassettes(floor_plan.copy(), cassettes, scale_factor, origin_offset, grid_alignment)

        # Add legend panel
        visualization = self._add_legend(visualization, cassettes)

        # Add statistics panel
        visualization = self._add_statistics(visualization, statistics)

        # Add title
        visualization = self._add_title(visualization)

        # Save visualization
        cv2.imwrite(output_path, visualization)
        logger.info(f"Visualization saved to: {output_path}")

        return visualization

    def _draw_cassettes(self, image: np.ndarray, cassettes: List[Dict],
                       scale_factor: float, origin_offset: Tuple[float, float] = (0.0, 0.0),
                       grid_alignment: Optional[GridAlignmentSystem] = None) -> np.ndarray:
        """
        Draw semi-transparent cassettes on image.

        Args:
            image: Base floor plan image
            cassettes: List of cassette dictionaries
            scale_factor: Feet per pixel
            origin_offset: Offset of polygon origin in feet (x_offset, y_offset)
            grid_alignment: Optional grid alignment system for accurate transformation

        Returns:
            Image with cassettes drawn
        """
        # Create overlay for transparency
        overlay = image.copy()

        # Get image height for Y-axis inversion
        image_height = image.shape[0]

        for cassette in cassettes:
            # Convert bottom-left corner of cassette to image coordinates
            x_feet = cassette['x']
            y_feet = cassette['y']

            # Use grid alignment if available, otherwise fall back to simple transformation
            if grid_alignment is not None:
                # Transform all four corners using homography for accuracy
                corners = [
                    (x_feet, y_feet),  # Bottom-left
                    (x_feet + cassette['width'], y_feet),  # Bottom-right
                    (x_feet + cassette['width'], y_feet + cassette['height']),  # Top-right
                    (x_feet, y_feet + cassette['height'])  # Top-left
                ]

                # Transform each corner
                transformed_corners = []
                for cx, cy in corners:
                    ix, iy = grid_alignment.transform_point(cx, cy)
                    transformed_corners.append((ix, iy))

                # Get bounding box for drawing
                xs = [c[0] for c in transformed_corners]
                ys = [c[1] for c in transformed_corners]
                x_pixels = min(xs)
                y_top_pixels = min(ys)
                width_pixels = max(xs) - min(xs)
                height_pixels = max(ys) - min(ys)
            else:
                # Fall back to simple coordinate transformation
                x_pixels, y_bottom_pixels = self.world_to_image_coords(
                    x_feet, y_feet, image_height, scale_factor, origin_offset
                )

                # For rectangle drawing, we need the top-left corner
                # So we convert the top of the cassette
                _, y_top_pixels = self.world_to_image_coords(
                    x_feet, y_feet + cassette['height'], image_height, scale_factor, origin_offset
                )

            # Calculate dimensions if not using grid alignment
            if grid_alignment is None:
                width_pixels = int(cassette['width'] / scale_factor)
                height_pixels = int(cassette['height'] / scale_factor)

            # Get color for cassette size
            size = cassette['size']
            color = self.color_scheme.get(size, (128, 128, 128))

            # Draw filled rectangle on overlay
            # Rectangle is drawn from top-left to bottom-right in image coordinates
            cv2.rectangle(overlay,
                         (x_pixels, y_top_pixels),
                         (x_pixels + width_pixels, y_top_pixels + height_pixels),
                         color, -1)

            # Draw white border
            cv2.rectangle(overlay,
                         (x_pixels, y_top_pixels),
                         (x_pixels + width_pixels, y_top_pixels + height_pixels),
                         (255, 255, 255), 1)

        # Blend overlay with original
        result = cv2.addWeighted(image, 1 - self.alpha, overlay, self.alpha, 0)

        return result

    def _add_legend(self, image: np.ndarray, cassettes: List[Dict]) -> np.ndarray:
        """
        Add legend panel to image.

        Args:
            image: Image to add legend to
            cassettes: List of cassettes for counting

        Returns:
            Image with legend
        """
        # Count cassettes by size
        size_counts = {}
        for cassette in cassettes:
            size = cassette['size']
            size_counts[size] = size_counts.get(size, 0) + 1

        # Create legend panel at bottom-left
        legend_width = 260
        legend_height = min(len(size_counts) * 32 + 70, 400)  # Limit max height
        h, w = image.shape[:2]

        # Position at bottom-left
        legend_x = 20
        legend_y = h - legend_height - 20

        # Draw white background
        cv2.rectangle(image,
                     (legend_x, legend_y),
                     (legend_x + legend_width, legend_y + legend_height),
                     (255, 255, 255), -1)

        # Draw border
        cv2.rectangle(image,
                     (legend_x, legend_y),
                     (legend_x + legend_width, legend_y + legend_height),
                     (0, 0, 0), 2)

        # Add legend title with better formatting
        cv2.putText(image, "CASSETTE SIZES",
                   (legend_x + 10, legend_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Add cassette sizes with improved formatting
        y_offset = legend_y + 55
        for size in sorted(size_counts.keys(), key=lambda x: -size_counts[x])[:12]:  # Top 12 by count
            count = size_counts[size]
            color = self.color_scheme.get(size, (128, 128, 128))

            # Draw color box with border
            cv2.rectangle(image,
                         (legend_x + 15, y_offset - 12),
                         (legend_x + 35, y_offset),
                         color, -1)
            cv2.rectangle(image,
                         (legend_x + 15, y_offset - 12),
                         (legend_x + 35, y_offset),
                         (0, 0, 0), 1)

            # Add text with better formatting
            w, h = map(float, size.split('x'))
            area = w * h
            weight = area * 10.4
            text = f"{size:5} ft: {count:3} units ({area:4.0f} sq ft)"
            cv2.putText(image, text,
                       (legend_x + 45, y_offset - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

            y_offset += 28

        return image

    def _add_statistics(self, image: np.ndarray, statistics: Dict) -> np.ndarray:
        """
        Add statistics panel to image.

        Args:
            image: Image to add statistics to
            statistics: Optimization statistics

        Returns:
            Image with statistics
        """
        # Create statistics panel on the right side
        stats_width = 280
        stats_height = 240
        h, w = image.shape[:2]
        stats_x = w - stats_width - 20
        stats_y = 80  # Below title

        # Draw semi-transparent white background
        overlay = image.copy()
        cv2.rectangle(overlay,
                     (stats_x, stats_y),
                     (stats_x + stats_width, stats_y + stats_height),
                     (255, 255, 255), -1)
        image = cv2.addWeighted(image, 0.3, overlay, 0.7, 0)

        # Draw border
        cv2.rectangle(image,
                     (stats_x, stats_y),
                     (stats_x + stats_width, stats_y + stats_height),
                     (0, 0, 0), 2)

        # Add statistics title with larger font
        cv2.putText(image, "STATISTICS",
                   (stats_x + 15, stats_y + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        # Separator line
        cv2.line(image,
                (stats_x + 15, stats_y + 50),
                (stats_x + stats_width - 15, stats_y + 50),
                (0, 0, 0), 1)

        # Format metrics with better layout
        coverage = statistics.get('coverage_percent', 0)
        total_area = statistics.get('total_area', 0)
        covered_area = statistics.get('covered_area', 0)
        gap_area = statistics.get('gap_area', 0)

        # Primary metrics with larger font
        y_pos = stats_y + 85

        # Coverage - prominent display
        cv2.putText(image, "Coverage:",
                   (stats_x + 20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        coverage_color = (0, 150, 0) if coverage >= 94 else (0, 0, 0)
        cv2.putText(image, f"{coverage:.1f}%",
                   (stats_x + 140, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, coverage_color, 2)

        y_pos += 35

        # Secondary metrics
        metrics = [
            ("Total Area:", f"{total_area:,.0f} sq ft"),
            ("Covered:", f"{covered_area:,.0f} sq ft"),
            ("Gap Area:", f"{gap_area:,.0f} sq ft"),
            ("Cassettes:", f"{statistics.get('num_cassettes', 0)} units"),
            ("Total Weight:", f"{statistics.get('total_weight', 0):,.0f} lbs")
        ]

        for label, value in metrics:
            cv2.putText(image, label,
                       (stats_x + 20, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(image, value,
                       (stats_x + 120, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            y_pos += 28

        return image

    def _add_title(self, image: np.ndarray) -> np.ndarray:
        """
        Add title to image.

        Args:
            image: Image to add title to

        Returns:
            Image with title
        """
        h, w = image.shape[:2]

        # Add title background with gradient effect
        cv2.rectangle(image, (0, 0), (w, 60), (40, 40, 40), -1)

        # Add title text with better centering
        title_text = "CASSETTE FLOOR JOIST LAYOUT"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        font_thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(title_text, font, font_scale, font_thickness)
        text_x = (w - text_width) // 2

        cv2.putText(image, title_text,
                   (text_x, 40),
                   font, font_scale, (255, 255, 255), font_thickness)

        return image
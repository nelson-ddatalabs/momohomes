#!/usr/bin/env python3
"""
Polygon Cassette Optimizer
==========================
Places cassettes within polygon boundary with no overlap or overhang.
Implements grid-based placement with boundary validation.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import logging
from polygon_reconstructor import PolygonReconstructor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Cassette:
    """Represents a placed cassette."""

    def __init__(self, x: float, y: float, width: float, height: float, size_name: str):
        """
        Initialize cassette.

        Args:
            x: X position (bottom-left corner) in feet
            y: Y position (bottom-left corner) in feet
            width: Width in feet
            height: Height in feet
            size_name: Size identifier (e.g., "6x8")
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.size_name = size_name
        self.area = width * height
        self.weight = self.area * 10.4  # 10.4 lbs per sq ft

    def get_corners(self) -> List[Tuple[float, float]]:
        """Get all four corners of cassette."""
        return [
            (self.x, self.y),
            (self.x + self.width, self.y),
            (self.x + self.width, self.y + self.height),
            (self.x, self.y + self.height)
        ]

    def overlaps_with(self, other: 'Cassette') -> bool:
        """Check if this cassette overlaps with another."""
        return not (
            self.x + self.width <= other.x or
            other.x + other.width <= self.x or
            self.y + self.height <= other.y or
            other.y + other.height <= self.y
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'x': self.x,
            'y': self.y,
            'width': self.width,
            'height': self.height,
            'size': self.size_name,
            'area': self.area,
            'weight': self.weight
        }


class PolygonCassetteOptimizer:
    """Optimizes cassette placement within polygon boundary."""

    def __init__(self, polygon: List[Tuple[float, float]], grid_resolution: float = 1.0):
        """
        Initialize optimizer.

        Args:
            polygon: List of (x, y) vertices in feet
            grid_resolution: Grid resolution in feet (default 1.0)
        """
        self.polygon = polygon
        self.grid_resolution = grid_resolution
        self.placed_cassettes = []

        # Cassette sizes ordered by area (largest first)
        self.cassette_sizes = [
            (6, 8, "6x8"),  # 48 sq ft
            (5, 8, "5x8"),  # 40 sq ft
            (6, 6, "6x6"),  # 36 sq ft
            (4, 8, "4x8"),  # 32 sq ft
            (5, 6, "5x6"),  # 30 sq ft
            (4, 6, "4x6"),  # 24 sq ft
            (4, 4, "4x4"),  # 16 sq ft
            (3, 4, "3x4"),  # 12 sq ft
        ]

        # Create polygon reconstructor for point-in-polygon tests
        self.reconstructor = PolygonReconstructor()
        self.reconstructor.vertices = polygon

        # Calculate bounds
        self.min_x, self.min_y, self.max_x, self.max_y = self._get_bounds()

        # Create placement grid
        self.grid_width = int((self.max_x - self.min_x) / grid_resolution) + 1
        self.grid_height = int((self.max_y - self.min_y) / grid_resolution) + 1
        self.coverage_grid = np.zeros((self.grid_height, self.grid_width), dtype=bool)

        logger.info(f"Initialized optimizer with {len(polygon)} vertices")
        logger.info(f"Grid size: {self.grid_width}x{self.grid_height} cells")

    def _get_bounds(self) -> Tuple[float, float, float, float]:
        """Get polygon bounding box."""
        xs = [v[0] for v in self.polygon]
        ys = [v[1] for v in self.polygon]
        return min(xs), min(ys), max(xs), max(ys)

    def optimize(self) -> Dict:
        """
        Run cassette placement optimization.

        Returns:
            Dictionary with optimization results
        """
        logger.info("Starting cassette optimization")

        # Try each cassette size (largest first)
        for width, height, size_name in self.cassette_sizes:
            # Try both orientations
            for w, h in [(width, height), (height, width)]:
                self._place_cassettes_of_size(w, h, size_name if w == width else f"{height}x{width}")

        # Calculate statistics
        results = self._calculate_statistics()

        logger.info(f"Optimization complete: {len(self.placed_cassettes)} cassettes placed")
        logger.info(f"Coverage: {results['coverage_percent']:.1f}%")

        return results

    def _place_cassettes_of_size(self, width: float, height: float, size_name: str):
        """
        Place all possible cassettes of given size.

        Args:
            width: Cassette width in feet
            height: Cassette height in feet
            size_name: Size identifier
        """
        # Calculate grid cells for this cassette
        cells_w = int(width / self.grid_resolution)
        cells_h = int(height / self.grid_resolution)

        # Scan grid for valid positions
        for grid_y in range(self.grid_height - cells_h + 1):
            for grid_x in range(self.grid_width - cells_w + 1):
                # Check if space is available
                if self._is_space_available(grid_x, grid_y, cells_w, cells_h):
                    # Convert to world coordinates
                    x = self.min_x + grid_x * self.grid_resolution
                    y = self.min_y + grid_y * self.grid_resolution

                    # Create cassette
                    cassette = Cassette(x, y, width, height, size_name)

                    # Validate placement
                    if self._is_valid_placement(cassette):
                        # Place cassette
                        self._place_cassette(cassette, grid_x, grid_y, cells_w, cells_h)

    def _is_space_available(self, grid_x: int, grid_y: int, cells_w: int, cells_h: int) -> bool:
        """Check if grid space is available."""
        return not self.coverage_grid[grid_y:grid_y+cells_h, grid_x:grid_x+cells_w].any()

    def _is_valid_placement(self, cassette: Cassette) -> bool:
        """
        Validate cassette placement.

        Args:
            cassette: Cassette to validate

        Returns:
            True if placement is valid
        """
        # Check all corners are inside polygon
        for corner in cassette.get_corners():
            if not self.reconstructor.point_in_polygon(corner[0], corner[1]):
                return False

        # Check no overlap with existing cassettes
        for existing in self.placed_cassettes:
            if cassette.overlaps_with(existing):
                return False

        return True

    def _place_cassette(self, cassette: Cassette, grid_x: int, grid_y: int,
                       cells_w: int, cells_h: int):
        """Place cassette and mark grid cells as occupied."""
        self.placed_cassettes.append(cassette)
        self.coverage_grid[grid_y:grid_y+cells_h, grid_x:grid_x+cells_w] = True

    def _calculate_statistics(self) -> Dict:
        """Calculate optimization statistics."""
        # Calculate areas
        total_area = self.reconstructor._calculate_area() if self.polygon else 0
        covered_area = sum(c.area for c in self.placed_cassettes)
        gap_area = total_area - covered_area

        # Calculate coverage
        coverage = covered_area / total_area if total_area > 0 else 0

        # Count by size
        size_counts = {}
        for cassette in self.placed_cassettes:
            size_counts[cassette.size_name] = size_counts.get(cassette.size_name, 0) + 1

        # Calculate total weight
        total_weight = sum(c.weight for c in self.placed_cassettes)

        return {
            'cassettes': [c.to_dict() for c in self.placed_cassettes],
            'num_cassettes': len(self.placed_cassettes),
            'coverage': coverage,
            'coverage_percent': coverage * 100,
            'gap_percent': (gap_area / total_area * 100) if total_area > 0 else 0,
            'total_area': total_area,
            'covered_area': covered_area,
            'gap_area': gap_area,
            'size_distribution': size_counts,
            'total_weight': total_weight
        }

    def get_cassettes_by_size(self) -> Dict[str, List[Cassette]]:
        """Get cassettes grouped by size."""
        grouped = {}
        for cassette in self.placed_cassettes:
            if cassette.size_name not in grouped:
                grouped[cassette.size_name] = []
            grouped[cassette.size_name].append(cassette)
        return grouped
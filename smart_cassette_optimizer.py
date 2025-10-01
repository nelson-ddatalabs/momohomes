#!/usr/bin/env python3
"""
Smart Cassette Optimizer
========================
Places cassettes starting from corners with longest edges,
prioritizing largest sizes for maximum coverage with minimum count.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SmartCassette:
    """Represents a cassette with smart placement."""

    def __init__(self, x: float, y: float, width: float, height: float, size_name: str):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.size_name = size_name
        self.area = width * height
        self.weight = self.area * 10.4  # 10.4 lbs per sq ft

    def get_corners(self) -> List[Tuple[float, float]]:
        """Get all four corners."""
        return [
            (self.x, self.y),
            (self.x + self.width, self.y),
            (self.x + self.width, self.y + self.height),
            (self.x, self.y + self.height)
        ]

    def overlaps_with(self, other: 'SmartCassette') -> bool:
        """Check overlap with another cassette."""
        return not (
            self.x + self.width <= other.x or
            other.x + other.width <= self.x or
            self.y + self.height <= other.y or
            other.y + other.height <= self.y
        )

    def to_dict(self) -> Dict:
        return {
            'x': self.x,
            'y': self.y,
            'width': self.width,
            'height': self.height,
            'size': self.size_name,
            'area': self.area,
            'weight': self.weight
        }


class SmartCassetteOptimizer:
    """
    Optimizes cassette placement for maximum coverage with minimum count.
    Starts from corners with longest edges and prioritizes largest sizes.
    """

    def __init__(self, polygon: List[Tuple[float, float]]):
        """
        Initialize optimizer with indoor-only polygon.

        Args:
            polygon: List of (x, y) vertices representing indoor space only
        """
        self.polygon = polygon
        self.placed_cassettes = []

        # Cassette sizes - LARGEST FIRST for maximum coverage
        self.cassette_sizes = [
            (8, 6, "8x6"),  # 48 sq ft - try wide orientation first
            (6, 8, "6x8"),  # 48 sq ft - standard orientation
            (8, 5, "8x5"),  # 40 sq ft - wide
            (5, 8, "5x8"),  # 40 sq ft - tall
            (8, 4, "8x4"),  # 32 sq ft - wide
            (4, 8, "4x8"),  # 32 sq ft - tall
            (6, 6, "6x6"),  # 36 sq ft - square
            (6, 5, "6x5"),  # 30 sq ft
            (5, 6, "5x6"),  # 30 sq ft
            (6, 4, "6x4"),  # 24 sq ft
            (4, 6, "4x6"),  # 24 sq ft
            (5, 5, "5x5"),  # 25 sq ft
            (5, 4, "5x4"),  # 20 sq ft
            (4, 5, "4x5"),  # 20 sq ft
            (4, 4, "4x4"),  # 16 sq ft
            (4, 3, "4x3"),  # 12 sq ft
            (3, 4, "3x4"),  # 12 sq ft
            (3, 3, "3x3"),  # 9 sq ft - smallest
        ]

        # Calculate bounds
        self.min_x, self.min_y, self.max_x, self.max_y = self._get_bounds()

        # Find corners and edges
        self.corners = self._find_corners()
        self.edge_lengths = self._calculate_edge_lengths()

        logger.info(f"Smart optimizer initialized with {len(polygon)} vertices")

    def _get_bounds(self) -> Tuple[float, float, float, float]:
        """Get polygon bounding box."""
        xs = [v[0] for v in self.polygon]
        ys = [v[1] for v in self.polygon]
        return min(xs), min(ys), max(xs), max(ys)

    def _find_corners(self) -> List[Tuple[int, Tuple[float, float]]]:
        """
        Find all corners of the polygon with their indices.

        Returns:
            List of (index, (x, y)) tuples
        """
        corners = []
        n = len(self.polygon)

        for i in range(n):
            prev_idx = (i - 1) % n
            next_idx = (i + 1) % n

            curr = self.polygon[i]
            prev = self.polygon[prev_idx]
            next_v = self.polygon[next_idx]

            # Check if this is a corner (direction changes)
            dx1 = curr[0] - prev[0]
            dy1 = curr[1] - prev[1]
            dx2 = next_v[0] - curr[0]
            dy2 = next_v[1] - curr[1]

            # If direction changes, it's a corner
            if (dx1 * dy2 - dy1 * dx2) != 0:
                corners.append((i, curr))

        return corners

    def _calculate_edge_lengths(self) -> Dict[int, float]:
        """
        Calculate length of each edge.

        Returns:
            Dictionary mapping edge index to length
        """
        lengths = {}
        n = len(self.polygon)

        for i in range(n):
            curr = self.polygon[i]
            next_v = self.polygon[(i + 1) % n]
            length = abs(next_v[0] - curr[0]) + abs(next_v[1] - curr[1])
            lengths[i] = length

        return lengths

    def _find_longest_edge_corner(self) -> Tuple[float, float]:
        """
        Find the corner that starts the longest edge.

        Returns:
            (x, y) coordinates of the corner
        """
        if not self.edge_lengths:
            return self.polygon[0]

        # Find edge with maximum length
        longest_edge_idx = max(self.edge_lengths, key=self.edge_lengths.get)
        return self.polygon[longest_edge_idx]

    def _is_point_inside_polygon(self, x: float, y: float) -> bool:
        """
        Check if point is inside polygon using ray casting algorithm.
        """
        n = len(self.polygon)
        inside = False

        j = n - 1
        for i in range(n):
            xi, yi = self.polygon[i]
            xj, yj = self.polygon[j]

            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside

            j = i

        return inside

    def _is_valid_placement(self, cassette: SmartCassette) -> bool:
        """
        Validate cassette placement - ALL corners must be inside polygon.
        """
        # Check all corners are inside
        for corner in cassette.get_corners():
            if not self._is_point_inside_polygon(corner[0], corner[1]):
                return False

        # Check no overlap with existing cassettes
        for existing in self.placed_cassettes:
            if cassette.overlaps_with(existing):
                return False

        return True

    def _place_from_corner(self, start_x: float, start_y: float,
                          width: float, height: float, size_name: str) -> bool:
        """
        Try to place a cassette starting from a corner position.

        Returns:
            True if successfully placed
        """
        # Try different anchor points (all 4 corners of the cassette)
        test_positions = [
            (start_x, start_y),  # Bottom-left anchor
            (start_x - width, start_y),  # Bottom-right anchor
            (start_x, start_y - height),  # Top-left anchor
            (start_x - width, start_y - height),  # Top-right anchor
        ]

        for x, y in test_positions:
            cassette = SmartCassette(x, y, width, height, size_name)
            if self._is_valid_placement(cassette):
                self.placed_cassettes.append(cassette)
                return True

        return False

    def _fill_area_with_cassettes(self, width: float, height: float, size_name: str):
        """
        Fill the polygon area with cassettes of given size using grid approach.
        """
        # Create a grid of potential positions
        step = 1.0  # 1-foot grid resolution

        # Start from bottom-left and work systematically
        y = self.min_y
        while y <= self.max_y - height:
            x = self.min_x
            while x <= self.max_x - width:
                cassette = SmartCassette(x, y, width, height, size_name)
                if self._is_valid_placement(cassette):
                    self.placed_cassettes.append(cassette)
                    # Skip ahead by cassette width to avoid overlaps
                    x += width
                else:
                    x += step
            y += step

    def optimize(self) -> Dict:
        """
        Run smart optimization for maximum coverage with minimum count.

        Strategy:
        1. Start from corner with longest edge
        2. Place largest cassettes first
        3. Fill systematically to minimize count
        """
        logger.info("Starting smart cassette optimization")

        # Find starting corner (with longest edge)
        start_corner = self._find_longest_edge_corner()
        logger.info(f"Starting from corner at {start_corner} (longest edge)")

        # Phase 1: Place from corners with largest sizes
        logger.info("Phase 1: Corner placement with largest cassettes")
        for corner_idx, corner_pos in self.corners[:4]:  # Try first 4 corners
            for width, height, size_name in self.cassette_sizes[:6]:  # Try largest 6 sizes
                self._place_from_corner(corner_pos[0], corner_pos[1],
                                       width, height, size_name)

        # Phase 2: Fill remaining area with largest possible cassettes
        logger.info("Phase 2: Systematic fill with decreasing sizes")
        for width, height, size_name in self.cassette_sizes:
            self._fill_area_with_cassettes(width, height, size_name)

        # Calculate statistics
        results = self._calculate_statistics()

        logger.info(f"Optimization complete: {len(self.placed_cassettes)} cassettes")
        logger.info(f"Coverage: {results['coverage_percent']:.1f}%")
        logger.info(f"Average cassette size: {results['avg_cassette_area']:.1f} sq ft")

        return results

    def _calculate_polygon_area(self) -> float:
        """Calculate polygon area using shoelace formula."""
        n = len(self.polygon)
        area = 0.0

        for i in range(n):
            j = (i + 1) % n
            area += self.polygon[i][0] * self.polygon[j][1]
            area -= self.polygon[j][0] * self.polygon[i][1]

        return abs(area) / 2.0

    def _calculate_statistics(self) -> Dict:
        """Calculate optimization statistics."""
        total_area = self._calculate_polygon_area()
        covered_area = sum(c.area for c in self.placed_cassettes)
        gap_area = total_area - covered_area

        # Count by size
        size_counts = {}
        for cassette in self.placed_cassettes:
            size_counts[cassette.size_name] = size_counts.get(cassette.size_name, 0) + 1

        # Calculate average cassette size
        avg_size = covered_area / len(self.placed_cassettes) if self.placed_cassettes else 0

        return {
            'cassettes': [c.to_dict() for c in self.placed_cassettes],
            'num_cassettes': len(self.placed_cassettes),
            'coverage': covered_area / total_area if total_area > 0 else 0,
            'coverage_percent': (covered_area / total_area * 100) if total_area > 0 else 0,
            'gap_percent': (gap_area / total_area * 100) if total_area > 0 else 0,
            'total_area': total_area,
            'covered_area': covered_area,
            'gap_area': gap_area,
            'size_distribution': size_counts,
            'total_weight': sum(c.weight for c in self.placed_cassettes),
            'avg_cassette_area': avg_size,
            'meets_requirement': (covered_area / total_area * 100) >= 94 if total_area > 0 else False
        }


if __name__ == "__main__":
    from fix_polygon_for_indoor_only import get_corrected_luna_polygon

    print("TESTING SMART CASSETTE OPTIMIZER")
    print("="*70)

    # Get corrected polygon (indoor only)
    polygon = get_corrected_luna_polygon()

    # Run optimization
    optimizer = SmartCassetteOptimizer(polygon)
    results = optimizer.optimize()

    # Display results
    print("\nOPTIMIZATION RESULTS:")
    print("-"*40)
    print(f"Total cassettes: {results['num_cassettes']}")
    print(f"Coverage: {results['coverage_percent']:.1f}%")
    print(f"Gap: {results['gap_percent']:.1f}%")
    print(f"Total weight: {results['total_weight']:.0f} lbs")
    print(f"Average cassette size: {results['avg_cassette_area']:.1f} sq ft")
    print(f"Meets 94% requirement: {'YES' if results['meets_requirement'] else 'NO'}")

    print("\nSIZE DISTRIBUTION:")
    for size, count in sorted(results['size_distribution'].items(),
                              key=lambda x: -x[1]):
        print(f"  {size}: {count} cassettes")
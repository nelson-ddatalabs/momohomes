#!/usr/bin/env python3
"""
Ultra Smart Optimizer
=====================
Enhanced cassette optimizer that achieves 94%+ coverage
with minimum cassette count through intelligent placement strategies.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Set
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UltraSmartCassette:
    """Enhanced cassette with placement intelligence."""

    def __init__(self, x: float, y: float, width: float, height: float, size_name: str):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.size_name = size_name
        self.area = width * height
        self.weight = self.area * 10.4

    def get_corners(self) -> List[Tuple[float, float]]:
        return [
            (self.x, self.y),
            (self.x + self.width, self.y),
            (self.x + self.width, self.y + self.height),
            (self.x, self.y + self.height)
        ]

    def overlaps_with(self, other: 'UltraSmartCassette') -> bool:
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


class UltraSmartOptimizer:
    """
    Ultra-optimized placement for 94%+ coverage with minimum cassettes.

    Key strategies:
    1. Row-based placement for maximum efficiency
    2. Intelligent size selection based on available space
    3. Gap detection and targeted filling
    4. Edge optimization
    """

    def __init__(self, polygon: List[Tuple[float, float]]):
        self.polygon = polygon
        self.placed_cassettes = []

        # Enhanced cassette sizes - more variety for better gap filling
        self.cassette_sizes = [
            # Primary sizes (largest first)
            (8, 6, "8x6"),  # 48 sq ft
            (6, 8, "6x8"),  # 48 sq ft

            # Secondary large sizes
            (8, 5, "8x5"),  # 40 sq ft
            (5, 8, "5x8"),  # 40 sq ft
            (6, 6, "6x6"),  # 36 sq ft

            # Medium sizes
            (8, 4, "8x4"),  # 32 sq ft
            (4, 8, "4x8"),  # 32 sq ft
            (6, 5, "6x5"),  # 30 sq ft
            (5, 6, "5x6"),  # 30 sq ft

            # Gap fillers
            (6, 4, "6x4"),  # 24 sq ft
            (4, 6, "4x6"),  # 24 sq ft
            (5, 4, "5x4"),  # 20 sq ft
            (4, 5, "4x5"),  # 20 sq ft
            (4, 4, "4x4"),  # 16 sq ft

            # Small gap fillers
            (4, 3, "4x3"),  # 12 sq ft
            (3, 4, "3x4"),  # 12 sq ft
            (3, 3, "3x3"),  # 9 sq ft

            # Tiny fillers for edges
            (3, 2, "3x2"),  # 6 sq ft
            (2, 3, "2x3"),  # 6 sq ft
            (2, 2, "2x2"),  # 4 sq ft
        ]

        # Calculate bounds
        self.min_x, self.min_y, self.max_x, self.max_y = self._get_bounds()

        # Create occupancy grid for fine-grained tracking
        self.grid_resolution = 0.5  # Half-foot resolution for precise placement
        self.grid_width = int((self.max_x - self.min_x) / self.grid_resolution) + 1
        self.grid_height = int((self.max_y - self.min_y) / self.grid_resolution) + 1
        self.occupancy = np.zeros((self.grid_height, self.grid_width), dtype=bool)

        logger.info(f"Ultra optimizer initialized: bounds [{self.min_x},{self.max_x}] x [{self.min_y},{self.max_y}]")

    def _get_bounds(self) -> Tuple[float, float, float, float]:
        xs = [v[0] for v in self.polygon]
        ys = [v[1] for v in self.polygon]
        return min(xs), min(ys), max(xs), max(ys)

    def _world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid indices."""
        gx = int((x - self.min_x) / self.grid_resolution)
        gy = int((y - self.min_y) / self.grid_resolution)
        return gx, gy

    def _grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        """Convert grid indices to world coordinates."""
        x = self.min_x + gx * self.grid_resolution
        y = self.min_y + gy * self.grid_resolution
        return x, y

    def _is_point_inside(self, x: float, y: float) -> bool:
        """Ray casting algorithm for point-in-polygon test."""
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

    def _is_point_on_edge(self, x: float, y: float, tolerance: float = 0.01) -> bool:
        """Check if point is on polygon edge within tolerance."""
        n = len(self.polygon)
        for i in range(n):
            x1, y1 = self.polygon[i]
            x2, y2 = self.polygon[(i + 1) % n]

            # Check if point is on line segment
            # Calculate distance from point to line
            if abs(x2 - x1) < tolerance:  # Vertical edge
                if abs(x - x1) < tolerance and min(y1, y2) - tolerance <= y <= max(y1, y2) + tolerance:
                    return True
            elif abs(y2 - y1) < tolerance:  # Horizontal edge
                if abs(y - y1) < tolerance and min(x1, x2) - tolerance <= x <= max(x1, x2) + tolerance:
                    return True
        return False

    def _is_cassette_valid(self, cassette: UltraSmartCassette, allow_edge_touch: bool = True) -> bool:
        """Check if cassette placement is valid."""
        # Check corners - allow them to be on edge or inside
        for corner in cassette.get_corners():
            x, y = corner
            if allow_edge_touch:
                # Allow corner to be inside OR on the edge
                if not (self._is_point_inside(x, y) or self._is_point_on_edge(x, y)):
                    return False
            else:
                # Original strict validation - must be inside
                if not self._is_point_inside(x, y):
                    return False

        # No overlaps
        for existing in self.placed_cassettes:
            if cassette.overlaps_with(existing):
                return False

        return True

    def _mark_occupied(self, cassette: UltraSmartCassette):
        """Mark grid cells as occupied by cassette."""
        gx1, gy1 = self._world_to_grid(cassette.x, cassette.y)
        gx2, gy2 = self._world_to_grid(cassette.x + cassette.width,
                                       cassette.y + cassette.height)

        # Ensure bounds
        gx1 = max(0, gx1)
        gy1 = max(0, gy1)
        gx2 = min(self.grid_width, gx2)
        gy2 = min(self.grid_height, gy2)

        self.occupancy[gy1:gy2, gx1:gx2] = True

    def _find_best_cassette_for_position(self, x: float, y: float) -> Optional[UltraSmartCassette]:
        """
        Find the best (largest) cassette that fits at given position.
        """
        for width, height, size_name in self.cassette_sizes:
            # Try both orientations
            for w, h, name in [(width, height, size_name),
                               (height, width, f"{height}x{width}")]:
                cassette = UltraSmartCassette(x, y, w, h, name)
                # Use edge-tolerant validation
                if self._is_cassette_valid(cassette, allow_edge_touch=True):
                    return cassette
        return None

    def _optimize_row(self, y: float) -> List[UltraSmartCassette]:
        """
        Optimize cassette placement for a single row at height y.
        Returns list of cassettes placed in this row.
        """
        row_cassettes = []
        x = self.min_x

        while x <= self.max_x:
            # Skip if position is already occupied
            gx, gy = self._world_to_grid(x, y)
            if gx < self.grid_width and gy < self.grid_height and self.occupancy[gy, gx]:
                x += self.grid_resolution
                continue

            # Find best cassette for this position
            cassette = self._find_best_cassette_for_position(x, y)

            if cassette:
                self.placed_cassettes.append(cassette)
                row_cassettes.append(cassette)
                self._mark_occupied(cassette)
                # Jump to end of placed cassette
                x = cassette.x + cassette.width
            else:
                x += self.grid_resolution

        return row_cassettes

    def _fill_gaps(self):
        """
        Intelligently fill remaining gaps with appropriate cassettes.
        """
        # Scan for gaps
        for gy in range(self.grid_height):
            for gx in range(self.grid_width):
                if self.occupancy[gy, gx]:
                    continue

                x, y = self._grid_to_world(gx, gy)

                # Try to place smallest cassettes in gaps
                for width, height, size_name in reversed(self.cassette_sizes):
                    for w, h, name in [(width, height, size_name),
                                      (height, width, f"{height}x{width}")]:
                        cassette = UltraSmartCassette(x, y, w, h, name)
                        # Use edge-tolerant validation for gap filling
                        if self._is_cassette_valid(cassette, allow_edge_touch=True):
                            self.placed_cassettes.append(cassette)
                            self._mark_occupied(cassette)
                            break

    def optimize(self) -> Dict:
        """
        Run ultra-smart optimization for 94%+ coverage.
        """
        logger.info("Starting ultra-smart optimization")

        # Phase 1: Row-by-row placement with largest cassettes
        logger.info("Phase 1: Row-based placement")
        y = self.min_y
        row_count = 0

        while y <= self.max_y:
            row_cassettes = self._optimize_row(y)
            if row_cassettes:
                row_count += 1
                logger.debug(f"Row at y={y:.1f}: placed {len(row_cassettes)} cassettes")

            # Advance by smallest possible increment to check next row
            y += self.grid_resolution

        initial_count = len(self.placed_cassettes)
        logger.info(f"Phase 1 complete: {initial_count} cassettes in {row_count} rows")

        # Phase 2: Fill remaining gaps
        logger.info("Phase 2: Gap filling")
        self._fill_gaps()

        gap_filled = len(self.placed_cassettes) - initial_count
        logger.info(f"Phase 2 complete: filled {gap_filled} gaps")

        # Calculate results
        results = self._calculate_statistics()

        logger.info(f"Optimization complete: {results['num_cassettes']} cassettes")
        logger.info(f"Coverage: {results['coverage_percent']:.1f}%")
        logger.info(f"Meets requirement: {results['meets_requirement']}")

        return results

    def _calculate_polygon_area(self) -> float:
        """Shoelace formula for polygon area."""
        n = len(self.polygon)
        area = 0.0

        for i in range(n):
            j = (i + 1) % n
            area += self.polygon[i][0] * self.polygon[j][1]
            area -= self.polygon[j][0] * self.polygon[i][1]

        return abs(area) / 2.0

    def _calculate_statistics(self) -> Dict:
        """Calculate comprehensive statistics."""
        total_area = self._calculate_polygon_area()
        covered_area = sum(c.area for c in self.placed_cassettes)
        gap_area = total_area - covered_area
        coverage_percent = (covered_area / total_area * 100) if total_area > 0 else 0

        # Size distribution
        size_counts = {}
        for cassette in self.placed_cassettes:
            size = cassette.size_name
            # Normalize size names (6x8 and 8x6 are the same)
            dims = sorted([int(x) for x in size.split('x')])
            normalized_size = f"{dims[0]}x{dims[1]}"
            size_counts[normalized_size] = size_counts.get(normalized_size, 0) + 1

        # Sort by area then count
        size_areas = {}
        for size in size_counts:
            w, h = map(int, size.split('x'))
            size_areas[size] = w * h

        sorted_sizes = sorted(size_counts.items(),
                            key=lambda x: (-size_areas[x[0]], -x[1]))

        return {
            'cassettes': [c.to_dict() for c in self.placed_cassettes],
            'num_cassettes': len(self.placed_cassettes),
            'coverage': covered_area / total_area if total_area > 0 else 0,
            'coverage_percent': coverage_percent,
            'gap_percent': (gap_area / total_area * 100) if total_area > 0 else 0,
            'total_area': total_area,
            'covered_area': covered_area,
            'gap_area': gap_area,
            'size_distribution': dict(sorted_sizes),
            'total_weight': sum(c.weight for c in self.placed_cassettes),
            'avg_cassette_area': covered_area / len(self.placed_cassettes) if self.placed_cassettes else 0,
            'meets_requirement': coverage_percent >= 94.0
        }


if __name__ == "__main__":
    from fix_polygon_for_indoor_only import get_corrected_luna_polygon

    print("ULTRA-SMART CASSETTE OPTIMIZER TEST")
    print("="*70)

    # Get corrected indoor-only polygon
    polygon = get_corrected_luna_polygon()

    # Run optimization
    optimizer = UltraSmartOptimizer(polygon)
    results = optimizer.optimize()

    # Display results
    print("\n" + "="*70)
    print("OPTIMIZATION RESULTS")
    print("="*70)
    print(f"✓ Total cassettes: {results['num_cassettes']}")
    print(f"✓ Coverage: {results['coverage_percent']:.2f}%")
    print(f"✓ Gap: {results['gap_percent']:.2f}%")
    print(f"✓ Total area: {results['total_area']:.1f} sq ft")
    print(f"✓ Covered area: {results['covered_area']:.1f} sq ft")
    print(f"✓ Total weight: {results['total_weight']:.0f} lbs")
    print(f"✓ Avg cassette: {results['avg_cassette_area']:.1f} sq ft")

    if results['meets_requirement']:
        print(f"\n✅✅✅ MEETS 94% REQUIREMENT!")
    else:
        print(f"\n❌ BELOW 94% REQUIREMENT")

    print("\nSIZE DISTRIBUTION:")
    print("-"*40)
    for size, count in results['size_distribution'].items():
        w, h = map(int, size.split('x'))
        area = w * h
        total = count * area
        print(f"  {size:4s}: {count:3d} cassettes ({area:2d} sq ft each, {total:4d} sq ft total)")
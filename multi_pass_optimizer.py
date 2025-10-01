#!/usr/bin/env python3
"""
Multi-Pass Optimizer
====================
Advanced cassette optimization with multiple passes and backtracking.
"""

import logging
from typing import List, Dict, Tuple, Optional
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiPassOptimizer:
    """
    Multi-pass optimization for cassette placement with backtracking.

    Implements 5-phase optimization:
    1. Large cassettes for main coverage
    2. Medium cassettes for semi-gaps
    3. Small cassettes for edge gaps
    4. Backtracking to reposition for better coverage
    5. Micro-fill with smallest sizes
    """

    # Cassette sizes in feet (width x height)
    CASSETTE_SIZES = [
        (8, 6), (6, 8),  # Large (48 sq ft)
        (8, 5), (5, 8),  # Medium-large (40 sq ft)
        (6, 6),          # Square large (36 sq ft)
        (8, 4), (4, 8),  # Medium (32 sq ft)
        (6, 5), (5, 6),  # Medium (30 sq ft)
        (6, 4), (4, 6),  # Small-medium (24 sq ft)
        (5, 4), (4, 5),  # Small-medium (20 sq ft)
        (4, 4),          # Small square (16 sq ft)
        (4, 3), (3, 4),  # Small (12 sq ft)
        (3, 3),          # Tiny square (9 sq ft)
        (3, 2), (2, 3),  # Micro (6 sq ft)
    ]

    # Weight per square foot
    WEIGHT_PER_SQ_FT = 10.4  # lbs/sq ft

    # Maximum cassette weight
    MAX_WEIGHT = 500  # lbs (48 sq ft max)

    def __init__(self, polygon: List[Tuple[float, float]],
                 min_gap: float = 0.125,  # 1.5 inches
                 edge_buffer: float = 0.0):  # No buffer - exact fit
        """
        Initialize optimizer with polygon.

        Args:
            polygon: List of (x, y) vertices in feet
            min_gap: Minimum gap between cassettes in feet
            edge_buffer: Buffer from polygon edge in feet
        """
        self.polygon = polygon
        self.min_gap = min_gap
        self.edge_buffer = edge_buffer

        # Create Shapely polygon
        self.shape_polygon = Polygon(polygon)
        self.total_area = self.shape_polygon.area

        # Storage for cassettes
        self.cassettes = []
        self.coverage_history = []

        # Optimization parameters
        self.enable_backtracking = True
        self.max_backtrack_attempts = 10

        logger.info(f"Multi-pass optimizer initialized: area={self.total_area:.1f} sq ft")

    def optimize(self) -> Dict:
        """
        Run multi-pass optimization.

        Returns:
            Dictionary with optimization results
        """
        logger.info("Starting multi-pass optimization...")

        # Phase 1: Large cassettes (6x8, 8x6, 6x6)
        logger.info("Phase 1: Placing large cassettes...")
        large_sizes = [(8, 6), (6, 8), (6, 6)]
        phase1_cassettes = self._place_cassettes_phase(large_sizes, "large")
        self.cassettes.extend(phase1_cassettes)
        self._log_coverage("Phase 1")

        # Phase 2: Medium cassettes (5x8, 8x5, 6x5, 5x6, 4x8, 8x4)
        logger.info("Phase 2: Placing medium cassettes...")
        medium_sizes = [(5, 8), (8, 5), (6, 5), (5, 6), (4, 8), (8, 4)]
        phase2_cassettes = self._place_cassettes_phase(medium_sizes, "medium")
        self.cassettes.extend(phase2_cassettes)
        self._log_coverage("Phase 2")

        # Phase 3: Small cassettes for edge gaps (4x6, 6x4, 4x4, 3x4, 4x3)
        logger.info("Phase 3: Filling edge gaps with small cassettes...")
        small_sizes = [(4, 6), (6, 4), (4, 4), (3, 4), (4, 3)]
        phase3_cassettes = self._place_edge_cassettes(small_sizes)
        self.cassettes.extend(phase3_cassettes)
        self._log_coverage("Phase 3")

        # Phase 4: Backtracking and repositioning
        if self.enable_backtracking:
            logger.info("Phase 4: Backtracking and repositioning...")
            improvement = self._backtrack_and_reposition()
            if improvement > 0:
                logger.info(f"Backtracking improved coverage by {improvement:.2f} sq ft")
            self._log_coverage("Phase 4")

        # Phase 5: Micro-fill with smallest sizes (3x3, 3x2, 2x3)
        logger.info("Phase 5: Micro-filling remaining gaps...")
        micro_sizes = [(3, 3), (3, 2), (2, 3)]
        phase5_cassettes = self._micro_fill(micro_sizes)
        self.cassettes.extend(phase5_cassettes)
        self._log_coverage("Phase 5")

        # Calculate final statistics
        return self._calculate_statistics()

    def _place_cassettes_phase(self, sizes: List[Tuple[int, int]],
                               phase_name: str) -> List[Dict]:
        """
        Place cassettes for a specific phase.

        Args:
            sizes: List of cassette sizes to try
            phase_name: Name of the phase for logging

        Returns:
            List of placed cassettes
        """
        placed = []

        # Try systematic row-based placement
        rows = self._calculate_rows()

        for row_y in rows:
            x = self._find_left_edge(row_y)

            while x < self._find_right_edge(row_y):
                # Try each size at this position
                best_cassette = None
                best_coverage = 0

                for size in sizes:
                    for rotation in [size, (size[1], size[0])]:  # Try both orientations
                        cassette = self._create_cassette(x, row_y, rotation[0], rotation[1])

                        if self._is_valid_placement(cassette):
                            coverage = rotation[0] * rotation[1]
                            if coverage > best_coverage:
                                best_cassette = cassette
                                best_coverage = coverage

                if best_cassette:
                    placed.append(best_cassette)
                    x += best_cassette['width'] + self.min_gap
                else:
                    x += 1.0  # Move forward if no cassette fits

        logger.info(f"{phase_name}: placed {len(placed)} cassettes")
        return placed

    def _place_edge_cassettes(self, sizes: List[Tuple[int, int]]) -> List[Dict]:
        """
        Specifically target edge gaps with small cassettes.

        Args:
            sizes: List of small cassette sizes

        Returns:
            List of placed cassettes
        """
        placed = []

        # Sample points along the perimeter
        perimeter_points = self._sample_perimeter_points(spacing=2.0)

        for px, py in perimeter_points:
            # Try placing cassettes near this perimeter point
            for size in sizes:
                for rotation in [size, (size[1], size[0])]:
                    # Try different offsets from the edge
                    for offset in [0.0, 0.5, 1.0]:
                        # Calculate position slightly inside the edge
                        cx = px - rotation[0]/2 + offset
                        cy = py - rotation[1]/2 + offset

                        cassette = self._create_cassette(cx, cy, rotation[0], rotation[1])

                        if self._is_valid_placement(cassette):
                            placed.append(cassette)
                            break  # Move to next perimeter point

        logger.info(f"Edge placement: {len(placed)} cassettes")
        return placed

    def _backtrack_and_reposition(self) -> float:
        """
        Try removing and repositioning cassettes for better coverage.

        Returns:
            Improvement in coverage (sq ft)
        """
        initial_coverage = self._calculate_coverage()
        best_cassettes = self.cassettes.copy()
        best_coverage = initial_coverage

        for attempt in range(self.max_backtrack_attempts):
            # Try removing some cassettes and repositioning
            if len(self.cassettes) > 5:
                # Remove a random subset of cassettes
                num_to_remove = min(5, len(self.cassettes) // 4)
                indices_to_remove = np.random.choice(
                    len(self.cassettes), num_to_remove, replace=False
                )

                # Create new cassette list without removed ones
                new_cassettes = [
                    c for i, c in enumerate(self.cassettes)
                    if i not in indices_to_remove
                ]

                # Try to place new cassettes in the freed space
                self.cassettes = new_cassettes

                # Try all sizes to fill the gaps
                for size in self.CASSETTE_SIZES:
                    new_placement = self._try_place_single_cassette(size)
                    if new_placement:
                        self.cassettes.append(new_placement)

                # Check if this is better
                new_coverage = self._calculate_coverage()
                if new_coverage > best_coverage:
                    best_cassettes = self.cassettes.copy()
                    best_coverage = new_coverage
                    logger.debug(f"Backtrack attempt {attempt}: improved coverage to {best_coverage:.1f} sq ft")

        self.cassettes = best_cassettes
        improvement = best_coverage - initial_coverage

        return improvement

    def _micro_fill(self, sizes: List[Tuple[int, int]]) -> List[Dict]:
        """
        Fill tiny gaps with smallest cassettes.

        Args:
            sizes: List of micro cassette sizes

        Returns:
            List of placed cassettes
        """
        placed = []

        # Create a grid of test points
        grid_spacing = 1.0  # Test every foot

        min_x = min(p[0] for p in self.polygon)
        max_x = max(p[0] for p in self.polygon)
        min_y = min(p[1] for p in self.polygon)
        max_y = max(p[1] for p in self.polygon)

        for x in np.arange(min_x, max_x, grid_spacing):
            for y in np.arange(min_y, max_y, grid_spacing):
                # Try smallest sizes first for better gap filling
                for size in sorted(sizes, key=lambda s: s[0] * s[1]):
                    cassette = self._create_cassette(x, y, size[0], size[1])

                    if self._is_valid_placement(cassette):
                        placed.append(cassette)
                        break  # Move to next grid point

        logger.info(f"Micro-fill: {len(placed)} cassettes")
        return placed

    def _is_valid_placement(self, cassette: Dict) -> bool:
        """
        Check if cassette placement is valid.

        Args:
            cassette: Cassette dictionary

        Returns:
            True if placement is valid
        """
        # Create cassette shape
        cass_box = box(
            cassette['x'], cassette['y'],
            cassette['x'] + cassette['width'],
            cassette['y'] + cassette['height']
        )

        # Check if within polygon
        if not self.shape_polygon.contains(cass_box):
            return False

        # Check overlap with existing cassettes
        for existing in self.cassettes:
            exist_box = box(
                existing['x'], existing['y'],
                existing['x'] + existing['width'],
                existing['y'] + existing['height']
            )

            # Add gap between cassettes
            if cass_box.distance(exist_box) < self.min_gap:
                return False

        return True

    def _create_cassette(self, x: float, y: float, width: float, height: float) -> Dict:
        """Create cassette dictionary."""
        return {
            'x': x,
            'y': y,
            'width': width,
            'height': height,
            'size': f"{int(width)}x{int(height)}",
            'area': width * height,
            'weight': width * height * self.WEIGHT_PER_SQ_FT
        }

    def _calculate_rows(self) -> List[float]:
        """Calculate row positions for systematic placement."""
        min_y = min(p[1] for p in self.polygon)
        max_y = max(p[1] for p in self.polygon)

        rows = []
        y = min_y
        while y < max_y:
            rows.append(y)
            y += 6.0  # Default row height for 6-foot cassettes

        return rows

    def _find_left_edge(self, y: float) -> float:
        """Find leftmost x coordinate at given y."""
        # This is simplified - would need proper polygon intersection
        return min(p[0] for p in self.polygon)

    def _find_right_edge(self, y: float) -> float:
        """Find rightmost x coordinate at given y."""
        # This is simplified - would need proper polygon intersection
        return max(p[0] for p in self.polygon)

    def _sample_perimeter_points(self, spacing: float = 2.0) -> List[Tuple[float, float]]:
        """Sample points along the polygon perimeter."""
        points = []

        # Sample along each edge
        for i in range(len(self.polygon)):
            p1 = self.polygon[i]
            p2 = self.polygon[(i + 1) % len(self.polygon)]

            # Calculate edge length
            edge_length = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

            # Sample points along this edge
            num_samples = max(1, int(edge_length / spacing))
            for j in range(num_samples):
                t = j / num_samples
                x = p1[0] + t * (p2[0] - p1[0])
                y = p1[1] + t * (p2[1] - p1[1])
                points.append((x, y))

        return points

    def _try_place_single_cassette(self, size: Tuple[int, int]) -> Optional[Dict]:
        """Try to place a single cassette of given size."""
        # Random placement attempt
        min_x = min(p[0] for p in self.polygon)
        max_x = max(p[0] for p in self.polygon) - size[0]
        min_y = min(p[1] for p in self.polygon)
        max_y = max(p[1] for p in self.polygon) - size[1]

        if max_x > min_x and max_y > min_y:
            x = np.random.uniform(min_x, max_x)
            y = np.random.uniform(min_y, max_y)

            cassette = self._create_cassette(x, y, size[0], size[1])

            if self._is_valid_placement(cassette):
                return cassette

        return None

    def _calculate_coverage(self) -> float:
        """Calculate current coverage area."""
        if not self.cassettes:
            return 0.0

        # Create union of all cassette shapes
        cassette_shapes = []
        for c in self.cassettes:
            cassette_shapes.append(
                box(c['x'], c['y'], c['x'] + c['width'], c['y'] + c['height'])
            )

        if cassette_shapes:
            union = unary_union(cassette_shapes)
            # Intersection with polygon to get actual coverage
            covered = self.shape_polygon.intersection(union)
            return covered.area

        return 0.0

    def _log_coverage(self, phase_name: str):
        """Log coverage statistics for a phase."""
        coverage = self._calculate_coverage()
        coverage_pct = (coverage / self.total_area) * 100
        self.coverage_history.append({
            'phase': phase_name,
            'coverage': coverage,
            'coverage_pct': coverage_pct,
            'num_cassettes': len(self.cassettes)
        })
        logger.info(f"{phase_name} complete: {len(self.cassettes)} cassettes, {coverage_pct:.1f}% coverage")

    def _calculate_statistics(self) -> Dict:
        """Calculate final optimization statistics."""
        coverage = self._calculate_coverage()
        coverage_pct = (coverage / self.total_area) * 100
        gap_area = self.total_area - coverage
        gap_pct = (gap_area / self.total_area) * 100

        # Calculate size distribution
        size_dist = {}
        total_weight = 0.0

        for c in self.cassettes:
            size = c['size']
            size_dist[size] = size_dist.get(size, 0) + 1
            total_weight += c['weight']

        stats = {
            'cassettes': self.cassettes,
            'num_cassettes': len(self.cassettes),
            'coverage': coverage,
            'coverage_percent': coverage_pct,
            'gap_percent': gap_pct,
            'total_area': self.total_area,
            'covered_area': coverage,
            'gap_area': gap_area,
            'size_distribution': size_dist,
            'total_weight': total_weight,
            'avg_cassette_area': coverage / len(self.cassettes) if self.cassettes else 0,
            'meets_requirement': coverage_pct >= 94.0,
            'coverage_history': self.coverage_history
        }

        logger.info(f"Optimization complete: {coverage_pct:.1f}% coverage, {len(self.cassettes)} cassettes")

        return stats


def test_multi_pass_optimizer():
    """Test the multi-pass optimizer."""

    # Test with Bungalow polygon
    polygon = [
        (0.0, 37.0),
        (23.5, 37.0),
        (23.5, 30.5),
        (42.0, 30.5),
        (42.0, 22.5),
        (42.0, 14.5),
        (42.0, 6.5),
        (29.5, 6.5),
        (29.5, 0.0),
        (8.5, 0.0),
        (8.5, 15.5),
        (0.0, 15.5),
    ]

    optimizer = MultiPassOptimizer(polygon)
    results = optimizer.optimize()

    print("\n" + "="*60)
    print("MULTI-PASS OPTIMIZATION RESULTS")
    print("="*60)
    print(f"Total area: {results['total_area']:.1f} sq ft")
    print(f"Covered area: {results['covered_area']:.1f} sq ft")
    print(f"Coverage: {results['coverage_percent']:.1f}%")
    print(f"Number of cassettes: {results['num_cassettes']}")
    print(f"Total weight: {results['total_weight']:.0f} lbs")
    print(f"Meets 94% requirement: {results['meets_requirement']}")

    print("\nCoverage by phase:")
    for phase in results['coverage_history']:
        print(f"  {phase['phase']}: {phase['coverage_pct']:.1f}% ({phase['num_cassettes']} cassettes)")

    print("\nSize distribution:")
    for size, count in sorted(results['size_distribution'].items()):
        print(f"  {size}: {count} cassettes")

    return optimizer


if __name__ == "__main__":
    test_multi_pass_optimizer()
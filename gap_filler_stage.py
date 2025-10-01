#!/usr/bin/env python3
"""
Gap Filler Stage
================
Identifies and fills gaps in cassette coverage to maximize total coverage.
"""

import logging
import math
from typing import List, Tuple, Optional, Set

from optimization_pipeline import PipelineStage, PipelineContext, Cassette

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GapFiller(PipelineStage):
    """
    Identifies and fills gaps in cassette placement.

    Strategy:
    1. Scan polygon with fine grid (0.5 ft resolution)
    2. Identify contiguous gap regions
    3. Classify gaps by size
    4. Fill with appropriate cassettes (smallest to largest)
    5. Use aggressive micro-adjustments
    """

    # Small cassettes for gap filling
    GAP_FILLER_SIZES = [
        (2, 4), (2, 6), (2, 8),  # 2 ft wide
        (3, 4), (3, 6), (3, 8),  # 3 ft wide
        (4, 4), (4, 6), (4, 8),  # 4 ft wide
        (5, 5), (5, 6), (5, 8),  # 5 ft wide
        (6, 6), (6, 8),          # 6 ft wide
    ]

    def __init__(self, grid_resolution: float = 0.5, min_gap_area: float = 8.0):
        """
        Initialize GapFiller.

        Args:
            grid_resolution: Resolution for gap detection grid in feet
            min_gap_area: Minimum gap area to attempt filling (sq ft)
        """
        self.grid_resolution = grid_resolution
        self.min_gap_area = min_gap_area

    @property
    def name(self) -> str:
        return "GapFiller"

    def process(self, context: PipelineContext) -> PipelineContext:
        """Identify and fill gaps in coverage."""
        polygon = context.polygon
        if len(polygon) < 3:
            logger.warning("Polygon has less than 3 vertices")
            return context

        initial_count = len(context.cassettes)
        initial_coverage = context.get_coverage()

        # Find gaps
        gaps = self._identify_gaps(polygon, context.cassettes)
        logger.info(f"  Identified {len(gaps)} gap regions")

        # Sort gaps by size (largest first for better efficiency)
        gaps.sort(key=lambda g: g['area'], reverse=True)

        # Try to fill each gap
        filled_count = 0
        for gap_idx, gap in enumerate(gaps):
            if gap['area'] < self.min_gap_area:
                continue

            cassettes_placed = self._fill_gap(
                gap,
                polygon,
                context.cassettes
            )

            for cassette in cassettes_placed:
                context.cassettes.append(cassette)

            if cassettes_placed:
                filled_count += 1
                logger.info(f"  Gap {gap_idx}: area={gap['area']:.1f} sq ft, "
                          f"filled with {len(cassettes_placed)} cassettes")

        # Aggressive final pass - try smallest cassettes everywhere
        final_cassettes = self._aggressive_fill(polygon, context.cassettes)
        context.cassettes.extend(final_cassettes)

        if final_cassettes:
            logger.info(f"  Aggressive fill: placed {len(final_cassettes)} additional cassettes")

        # Update metadata
        placed_count = len(context.cassettes) - initial_count
        final_coverage = context.get_coverage()

        context.metadata['gap_filler'] = {
            'gaps_identified': len(gaps),
            'gaps_filled': filled_count,
            'cassettes_placed': placed_count,
            'coverage_before': initial_coverage,
            'coverage_after': final_coverage,
            'coverage_gain': final_coverage - initial_coverage
        }

        logger.info(f"  Gap filling complete: {placed_count} cassettes placed, "
                   f"coverage {initial_coverage:.1f}% -> {final_coverage:.1f}%")

        return context

    def _identify_gaps(self, polygon: List[Tuple[float, float]],
                      cassettes: List[Cassette]) -> List[dict]:
        """
        Identify gap regions using grid scanning.

        Args:
            polygon: Building polygon
            cassettes: Placed cassettes

        Returns:
            List of gap regions with metadata
        """
        # Get bounding box
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # Create grid
        grid_points = []
        y = min_y
        while y <= max_y:
            x = min_x
            while x <= max_x:
                point = (x, y)
                # Check if point is in polygon but not covered
                if self._point_in_polygon(point, polygon):
                    if not self._point_covered(point, cassettes):
                        grid_points.append(point)
                x += self.grid_resolution
            y += self.grid_resolution

        # Group contiguous points into gap regions
        gaps = self._group_contiguous_points(grid_points)

        # Calculate gap metadata
        gap_info = []
        for gap_points in gaps:
            if len(gap_points) < 4:  # Too small
                continue

            # Find bounding box of gap
            gap_xs = [p[0] for p in gap_points]
            gap_ys = [p[1] for p in gap_points]

            gap_info.append({
                'points': gap_points,
                'min_x': min(gap_xs),
                'max_x': max(gap_xs),
                'min_y': min(gap_ys),
                'max_y': max(gap_ys),
                'width': max(gap_xs) - min(gap_xs),
                'height': max(gap_ys) - min(gap_ys),
                'area': len(gap_points) * self.grid_resolution**2
            })

        return gap_info

    def _group_contiguous_points(self, points: List[Tuple[float, float]]) -> List[List[Tuple[float, float]]]:
        """
        Group contiguous grid points into regions.

        Args:
            points: List of uncovered grid points

        Returns:
            List of contiguous regions
        """
        if not points:
            return []

        # Create a set for fast lookup
        point_set = set(points)
        visited = set()
        regions = []

        for point in points:
            if point in visited:
                continue

            # Start a new region
            region = []
            stack = [point]

            while stack:
                current = stack.pop()
                if current in visited:
                    continue

                visited.add(current)
                region.append(current)

                # Check neighbors
                x, y = current
                neighbors = [
                    (x + self.grid_resolution, y),
                    (x - self.grid_resolution, y),
                    (x, y + self.grid_resolution),
                    (x, y - self.grid_resolution)
                ]

                for neighbor in neighbors:
                    if neighbor in point_set and neighbor not in visited:
                        stack.append(neighbor)

            if region:
                regions.append(region)

        return regions

    def _fill_gap(self, gap: dict,
                 polygon: List[Tuple[float, float]],
                 existing_cassettes: List[Cassette]) -> List[Cassette]:
        """
        Fill a specific gap region.

        Args:
            gap: Gap information dictionary
            polygon: Building polygon
            existing_cassettes: Already placed cassettes

        Returns:
            List of cassettes placed in gap
        """
        placed = []

        # Try cassette sizes from smallest to largest
        for width, height in self.GAP_FILLER_SIZES:
            # Check if cassette could fit in gap bounds
            if width > gap['width'] + 1.0 or height > gap['height'] + 1.0:
                continue

            # Try both orientations
            for rotated in [False, True]:
                w, h = (height, width) if rotated else (width, height)

                # Try positions within gap bounds
                y = gap['min_y']
                while y + h <= gap['max_y'] + self.grid_resolution:
                    x = gap['min_x']
                    while x + w <= gap['max_x'] + self.grid_resolution:
                        cassette = Cassette(x, y, w, h)

                        if self._is_valid_placement(
                            cassette,
                            polygon,
                            existing_cassettes + placed
                        ):
                            placed.append(cassette)

                        x += w - 0.5  # Small overlap to maximize coverage
                    y += h - 0.5

        return placed

    def _aggressive_fill(self, polygon: List[Tuple[float, float]],
                        existing_cassettes: List[Cassette]) -> List[Cassette]:
        """
        Aggressively try to place smallest cassettes in any remaining gaps.

        Args:
            polygon: Building polygon
            existing_cassettes: Already placed cassettes

        Returns:
            List of additional cassettes placed
        """
        placed = []

        # Use only smallest cassettes
        small_sizes = [(2, 4), (2, 6), (3, 4), (3, 6), (4, 4)]

        # Get bounding box
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # Fine-grained scanning
        step = 0.25  # Very fine step for aggressive placement

        for width, height in small_sizes:
            y = min_y
            while y + height <= max_y + 0.5:
                x = min_x
                while x + width <= max_x + 0.5:
                    cassette = Cassette(x, y, width, height)

                    if self._is_valid_placement(
                        cassette,
                        polygon,
                        existing_cassettes + placed
                    ):
                        placed.append(cassette)
                        x += width  # Skip ahead
                    else:
                        x += step  # Small step
                y += step

        return placed

    def _point_covered(self, point: Tuple[float, float],
                      cassettes: List[Cassette]) -> bool:
        """
        Check if a point is covered by any cassette.

        Args:
            point: (x, y) coordinates
            cassettes: List of cassettes

        Returns:
            True if point is covered
        """
        x, y = point
        for cassette in cassettes:
            if (cassette.x <= x <= cassette.x + cassette.width and
                cassette.y <= y <= cassette.y + cassette.height):
                return True
        return False

    def _is_valid_placement(self, cassette: Cassette,
                           polygon: List[Tuple[float, float]],
                           existing_cassettes: List[Cassette]) -> bool:
        """
        Check if cassette placement is valid.

        Args:
            cassette: Cassette to place
            polygon: Building polygon
            existing_cassettes: Already placed cassettes

        Returns:
            True if placement is valid
        """
        # Check overlap
        for existing in existing_cassettes:
            if cassette.overlaps(existing):
                return False

        # Check boundaries - all corners must be inside
        corners = cassette.get_corners()
        for corner in corners:
            if not self._point_in_polygon(corner, polygon):
                return False

        return True

    def _point_in_polygon(self, point: Tuple[float, float],
                         polygon: List[Tuple[float, float]]) -> bool:
        """
        Check if a point is inside the polygon using ray casting.

        Args:
            point: (x, y) coordinates
            polygon: List of vertices

        Returns:
            True if point is inside polygon
        """
        x, y = point
        n = len(polygon)
        inside = False

        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside


def test_gap_filler():
    """Test the GapFiller stage."""
    from optimization_pipeline import create_standard_pipeline

    # Test polygon: Rectangle
    test_polygon = [
        (0, 0),
        (40, 0),
        (40, 30),
        (0, 30)
    ]

    # Create and run full pipeline up to GapFiller
    pipeline = create_standard_pipeline()

    # Add GapFiller
    gap_filler = GapFiller()
    pipeline.add_stage(gap_filler)

    # Run optimization
    results = pipeline.optimize(test_polygon)

    print("\n" + "="*60)
    print("GAP FILLER TEST RESULTS")
    print("="*60)
    print(f"Total cassettes: {results['num_cassettes']}")
    print(f"Coverage: {results['coverage_percent']:.1f}%")
    print(f"Meets 94% requirement: {results['meets_requirement']}")

    # Show stage progression
    print("\nStage progression:")
    for stage_name, stage_data in results['stage_results'].items():
        if 'coverage_after' in stage_data:
            print(f"  {stage_name}: {stage_data.get('coverage_after', 0):.1f}% coverage "
                  f"(+{stage_data.get('coverage_gain', 0):.1f}%)")

    # Show size distribution
    print("\nCassette size distribution:")
    for size, count in sorted(results['size_distribution'].items()):
        print(f"  {size}: {count} cassettes")

    return results


if __name__ == "__main__":
    test_gap_filler()
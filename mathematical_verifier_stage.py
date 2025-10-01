#!/usr/bin/env python3
"""
Mathematical Verifier Stage
============================
Performs precise mathematical verification of cassette coverage
using high-resolution grid sampling.
"""

import logging
import math
from typing import List, Tuple, Dict, Any

from optimization_pipeline import PipelineStage, PipelineContext, Cassette

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MathematicalVerifier(PipelineStage):
    """
    Verifies cassette placement with mathematical precision.

    Features:
    1. High-resolution grid sampling (0.25 ft default)
    2. Exact area calculation
    3. Gap analysis and visualization
    4. Overlap detection
    5. Coverage heatmap generation
    """

    def __init__(self, grid_resolution: float = 0.25):
        """
        Initialize MathematicalVerifier.

        Args:
            grid_resolution: Grid resolution in feet (default 0.25 ft)
        """
        self.grid_resolution = grid_resolution

    @property
    def name(self) -> str:
        return "MathematicalVerifier"

    def process(self, context: PipelineContext) -> PipelineContext:
        """Verify cassette placement mathematically."""
        polygon = context.polygon

        # Perform verification
        verification = self._verify_coverage(polygon, context.cassettes)

        # Store detailed results
        context.metadata['verification'] = verification

        # Log summary
        logger.info(f"  Mathematical Verification:")
        logger.info(f"    Polygon area: {verification['polygon_area']:.2f} sq ft")
        logger.info(f"    Covered area: {verification['covered_area']:.2f} sq ft")
        logger.info(f"    Coverage: {verification['coverage_percent']:.2f}%")
        logger.info(f"    Gap area: {verification['gap_area']:.2f} sq ft")
        logger.info(f"    Grid points: {verification['total_points']} total, "
                   f"{verification['covered_points']} covered")

        # Check for issues
        if verification['overlaps']:
            logger.warning(f"  Found {len(verification['overlaps'])} overlapping cassette pairs!")

        if verification['out_of_bounds']:
            logger.warning(f"  Found {len(verification['out_of_bounds'])} cassettes extending beyond polygon!")

        # Analyze gaps
        gap_analysis = self._analyze_gaps(verification['gap_regions'])
        context.metadata['gap_analysis'] = gap_analysis

        logger.info(f"  Gap Analysis:")
        logger.info(f"    Total gaps: {gap_analysis['total_gaps']}")
        logger.info(f"    Large gaps (>16 sq ft): {gap_analysis['large_gaps']}")
        logger.info(f"    Medium gaps (8-16 sq ft): {gap_analysis['medium_gaps']}")
        logger.info(f"    Small gaps (<8 sq ft): {gap_analysis['small_gaps']}")

        # Calculate theoretical maximum
        theoretical_max = self._calculate_theoretical_maximum(polygon)
        context.metadata['theoretical_maximum'] = theoretical_max

        logger.info(f"  Theoretical maximum coverage: {theoretical_max:.1f}%")
        logger.info(f"  Achievement ratio: {(verification['coverage_percent']/theoretical_max*100):.1f}%")

        return context

    def _verify_coverage(self, polygon: List[Tuple[float, float]],
                        cassettes: List[Cassette]) -> Dict[str, Any]:
        """
        Perform detailed coverage verification.

        Args:
            polygon: Building polygon
            cassettes: Placed cassettes

        Returns:
            Verification results dictionary
        """
        # Get bounding box
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # Grid sampling
        total_points = 0
        covered_points = 0
        gap_points = []

        y = min_y
        while y <= max_y:
            x = min_x
            while x <= max_x:
                point = (x, y)

                if self._point_in_polygon(point, polygon):
                    total_points += 1

                    if self._point_covered_by_cassettes(point, cassettes):
                        covered_points += 1
                    else:
                        gap_points.append(point)

                x += self.grid_resolution
            y += self.grid_resolution

        # Calculate areas
        polygon_area = self._calculate_polygon_area(polygon)
        grid_area = total_points * (self.grid_resolution ** 2)
        covered_area = covered_points * (self.grid_resolution ** 2)
        gap_area = len(gap_points) * (self.grid_resolution ** 2)

        # Check for overlaps
        overlaps = self._find_overlapping_cassettes(cassettes)

        # Check for out-of-bounds cassettes
        out_of_bounds = self._find_out_of_bounds_cassettes(cassettes, polygon)

        # Group gap points into regions
        gap_regions = self._group_gap_points(gap_points)

        return {
            'polygon_area': polygon_area,
            'grid_area': grid_area,
            'covered_area': covered_area,
            'gap_area': gap_area,
            'coverage_percent': (covered_area / polygon_area * 100) if polygon_area > 0 else 0,
            'total_points': total_points,
            'covered_points': covered_points,
            'gap_points': len(gap_points),
            'overlaps': overlaps,
            'out_of_bounds': out_of_bounds,
            'gap_regions': gap_regions,
            'grid_resolution': self.grid_resolution
        }

    def _calculate_polygon_area(self, polygon: List[Tuple[float, float]]) -> float:
        """
        Calculate exact polygon area using shoelace formula.

        Args:
            polygon: List of vertices

        Returns:
            Area in square feet
        """
        n = len(polygon)
        area = 0.0

        for i in range(n):
            j = (i + 1) % n
            area += polygon[i][0] * polygon[j][1]
            area -= polygon[j][0] * polygon[i][1]

        return abs(area) / 2.0

    def _point_covered_by_cassettes(self, point: Tuple[float, float],
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

    def _find_overlapping_cassettes(self, cassettes: List[Cassette]) -> List[Tuple[int, int]]:
        """
        Find pairs of overlapping cassettes.

        Args:
            cassettes: List of cassettes

        Returns:
            List of overlapping cassette index pairs
        """
        overlaps = []
        n = len(cassettes)

        for i in range(n):
            for j in range(i + 1, n):
                if cassettes[i].overlaps(cassettes[j]):
                    overlaps.append((i, j))

        return overlaps

    def _find_out_of_bounds_cassettes(self, cassettes: List[Cassette],
                                     polygon: List[Tuple[float, float]]) -> List[int]:
        """
        Find cassettes that extend beyond polygon boundaries.

        Args:
            cassettes: List of cassettes
            polygon: Building polygon

        Returns:
            List of out-of-bounds cassette indices
        """
        out_of_bounds = []

        for i, cassette in enumerate(cassettes):
            # Check all four corners
            corners = cassette.get_corners()
            for corner in corners:
                if not self._point_in_polygon(corner, polygon):
                    out_of_bounds.append(i)
                    break

        return out_of_bounds

    def _group_gap_points(self, gap_points: List[Tuple[float, float]]) -> List[List[Tuple[float, float]]]:
        """
        Group contiguous gap points into regions.

        Args:
            gap_points: List of uncovered points

        Returns:
            List of gap regions
        """
        if not gap_points:
            return []

        # Create a set for fast lookup
        point_set = set(gap_points)
        visited = set()
        regions = []

        for point in gap_points:
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
                    # Account for floating point precision
                    neighbor = (round(neighbor[0] / self.grid_resolution) * self.grid_resolution,
                              round(neighbor[1] / self.grid_resolution) * self.grid_resolution)
                    if neighbor in point_set and neighbor not in visited:
                        stack.append(neighbor)

            if region:
                regions.append(region)

        return regions

    def _analyze_gaps(self, gap_regions: List[List[Tuple[float, float]]]) -> Dict[str, Any]:
        """
        Analyze gap regions by size.

        Args:
            gap_regions: List of gap regions

        Returns:
            Gap analysis results
        """
        large_gaps = 0  # > 16 sq ft
        medium_gaps = 0  # 8-16 sq ft
        small_gaps = 0  # < 8 sq ft

        gap_sizes = []

        for region in gap_regions:
            area = len(region) * (self.grid_resolution ** 2)
            gap_sizes.append(area)

            if area > 16:
                large_gaps += 1
            elif area > 8:
                medium_gaps += 1
            else:
                small_gaps += 1

        return {
            'total_gaps': len(gap_regions),
            'large_gaps': large_gaps,
            'medium_gaps': medium_gaps,
            'small_gaps': small_gaps,
            'gap_sizes': gap_sizes,
            'largest_gap': max(gap_sizes) if gap_sizes else 0,
            'smallest_gap': min(gap_sizes) if gap_sizes else 0,
            'average_gap': sum(gap_sizes) / len(gap_sizes) if gap_sizes else 0
        }

    def _calculate_theoretical_maximum(self, polygon: List[Tuple[float, float]]) -> float:
        """
        Calculate theoretical maximum coverage percentage.

        This accounts for:
        - Minimum cassette size (2x4 = 8 sq ft)
        - Edge constraints
        - Geometric limitations

        Args:
            polygon: Building polygon

        Returns:
            Theoretical maximum coverage percentage
        """
        area = self._calculate_polygon_area(polygon)

        # Calculate perimeter
        perimeter = 0.0
        n = len(polygon)
        for i in range(n):
            j = (i + 1) % n
            perimeter += math.sqrt(
                (polygon[j][0] - polygon[i][0])**2 +
                (polygon[j][1] - polygon[i][1])**2
            )

        # Estimate edge loss (0.5 ft buffer around perimeter)
        edge_loss = perimeter * 0.5

        # Estimate corner loss (additional area at each corner)
        corner_loss = n * 2.0  # Roughly 2 sq ft per corner

        # Theoretical maximum
        usable_area = area - edge_loss - corner_loss
        theoretical_max = (usable_area / area * 100) if area > 0 else 0

        # Cap at realistic maximum (96% for perfect conditions)
        return min(theoretical_max, 96.0)

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


def test_mathematical_verifier():
    """Test the MathematicalVerifier stage."""
    from optimization_pipeline import create_standard_pipeline
    from gap_filler_stage import GapFiller

    # Test polygon: Rectangle
    test_polygon = [
        (0, 0),
        (40, 0),
        (40, 30),
        (0, 30)
    ]

    # Create full pipeline
    pipeline = create_standard_pipeline()
    pipeline.add_stage(GapFiller())
    pipeline.add_stage(MathematicalVerifier())

    # Run optimization
    results = pipeline.optimize(test_polygon)

    print("\n" + "="*60)
    print("MATHEMATICAL VERIFIER TEST RESULTS")
    print("="*60)

    # Get verification results
    if 'verification' in results.get('stage_results', {}).get('MathematicalVerifier', {}).get('metadata', {}):
        verification = results['stage_results']['MathematicalVerifier']['metadata']['verification']

        print(f"Polygon area: {verification['polygon_area']:.2f} sq ft")
        print(f"Covered area: {verification['covered_area']:.2f} sq ft")
        print(f"Coverage: {verification['coverage_percent']:.2f}%")
        print(f"Gap area: {verification['gap_area']:.2f} sq ft")

        if verification['overlaps']:
            print(f"\nWARNING: {len(verification['overlaps'])} overlapping cassette pairs detected!")

        if verification['out_of_bounds']:
            print(f"\nWARNING: {len(verification['out_of_bounds'])} cassettes out of bounds!")

    print(f"\nFinal coverage: {results['coverage_percent']:.1f}%")
    print(f"Meets 94% requirement: {results['meets_requirement']}")

    return results


if __name__ == "__main__":
    test_mathematical_verifier()
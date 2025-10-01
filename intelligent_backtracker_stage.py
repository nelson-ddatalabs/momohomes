#!/usr/bin/env python3
"""
Intelligent Backtracker Stage
==============================
Identifies and replaces inefficiently placed cassettes to improve coverage.
"""

import logging
import math
import random
from typing import List, Tuple, Optional, Dict, Any

from optimization_pipeline import PipelineStage, PipelineContext, Cassette

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntelligentBacktracker(PipelineStage):
    """
    Optimizes cassette placement through intelligent backtracking.

    Strategy:
    1. Identify inefficiently placed cassettes (low coverage contribution)
    2. Remove and replace with better configurations
    3. Focus on areas with large gaps
    4. Try different cassette combinations
    5. Iterative improvement until convergence
    """

    # Cassette sizes ordered by preference for replacement
    REPLACEMENT_SIZES = [
        (6, 8), (5, 8), (6, 6), (5, 6),  # Large
        (4, 8), (4, 6), (3, 8), (3, 6),  # Medium
        (2, 8), (2, 6), (2, 4)           # Small
    ]

    def __init__(self, max_iterations: int = 10, min_improvement: float = 0.5):
        """
        Initialize IntelligentBacktracker.

        Args:
            max_iterations: Maximum optimization iterations
            min_improvement: Minimum coverage improvement to continue (percentage points)
        """
        self.max_iterations = max_iterations
        self.min_improvement = min_improvement

    @property
    def name(self) -> str:
        return "IntelligentBacktracker"

    def process(self, context: PipelineContext) -> PipelineContext:
        """Optimize cassette placement through backtracking."""
        initial_coverage = context.get_coverage()
        initial_count = len(context.cassettes)

        logger.info(f"  Starting intelligent backtracking")
        logger.info(f"    Initial: {initial_count} cassettes, {initial_coverage:.1f}% coverage")

        best_cassettes = context.cassettes.copy()
        best_coverage = initial_coverage

        for iteration in range(self.max_iterations):
            # Identify problem areas
            problem_areas = self._identify_problem_areas(
                context.polygon,
                context.cassettes
            )

            if not problem_areas:
                logger.info(f"    Iteration {iteration + 1}: No problem areas found")
                break

            # Try to improve each problem area
            improved = False
            for area in problem_areas:
                new_cassettes = self._optimize_area(
                    area,
                    context.polygon,
                    context.cassettes
                )

                if new_cassettes:
                    # Test new configuration
                    test_coverage = self._calculate_coverage(
                        context.polygon,
                        new_cassettes
                    )

                    if test_coverage > best_coverage + 0.1:  # Minimal improvement threshold
                        context.cassettes = new_cassettes
                        best_cassettes = new_cassettes.copy()
                        best_coverage = test_coverage
                        improved = True
                        logger.info(f"    Iteration {iteration + 1}: Improved to {best_coverage:.1f}%")
                        break

            if not improved:
                # Try global optimization
                new_cassettes = self._global_optimization(
                    context.polygon,
                    context.cassettes
                )

                test_coverage = self._calculate_coverage(
                    context.polygon,
                    new_cassettes
                )

                if test_coverage > best_coverage:
                    context.cassettes = new_cassettes
                    best_cassettes = new_cassettes.copy()
                    best_coverage = test_coverage
                    logger.info(f"    Iteration {iteration + 1}: Global optimization to {best_coverage:.1f}%")
                else:
                    logger.info(f"    Iteration {iteration + 1}: No improvement found")

            # Check for convergence
            if best_coverage - initial_coverage < self.min_improvement and iteration > 2:
                logger.info(f"    Converged after {iteration + 1} iterations")
                break

        # Use best configuration found
        context.cassettes = best_cassettes
        final_coverage = context.get_coverage()
        final_count = len(context.cassettes)

        # Update metadata
        context.metadata['intelligent_backtracker'] = {
            'initial_coverage': initial_coverage,
            'final_coverage': final_coverage,
            'coverage_gain': final_coverage - initial_coverage,
            'initial_count': initial_count,
            'final_count': final_count,
            'iterations': min(iteration + 1, self.max_iterations)
        }

        logger.info(f"  Backtracking complete: {final_count} cassettes, "
                   f"{final_coverage:.1f}% coverage (+{final_coverage - initial_coverage:.1f}%)")

        return context

    def _identify_problem_areas(self, polygon: List[Tuple[float, float]],
                               cassettes: List[Cassette]) -> List[Dict[str, Any]]:
        """
        Identify areas with poor coverage.

        Args:
            polygon: Building polygon
            cassettes: Current cassettes

        Returns:
            List of problem area definitions
        """
        problem_areas = []

        # Find large gaps
        gaps = self._find_gaps(polygon, cassettes, min_gap_size=16.0)

        for gap in gaps:
            # Find cassettes near this gap
            nearby_cassettes = self._find_nearby_cassettes(
                gap['center'],
                cassettes,
                radius=10.0
            )

            if nearby_cassettes:
                problem_areas.append({
                    'type': 'large_gap',
                    'gap': gap,
                    'nearby_cassettes': nearby_cassettes,
                    'priority': gap['area']
                })

        # Find overlapping cassettes
        overlaps = self._find_overlaps(cassettes)
        for overlap in overlaps:
            problem_areas.append({
                'type': 'overlap',
                'cassettes': overlap,
                'priority': 100  # High priority
            })

        # Find inefficient cassettes (small cassettes in areas that could fit larger ones)
        inefficient = self._find_inefficient_cassettes(polygon, cassettes)
        for ineff in inefficient:
            problem_areas.append({
                'type': 'inefficient',
                'cassette_index': ineff,
                'priority': 50
            })

        # Sort by priority
        problem_areas.sort(key=lambda x: x['priority'], reverse=True)

        return problem_areas[:5]  # Focus on top 5 problems

    def _optimize_area(self, problem_area: Dict[str, Any],
                      polygon: List[Tuple[float, float]],
                      cassettes: List[Cassette]) -> Optional[List[Cassette]]:
        """
        Optimize a specific problem area.

        Args:
            problem_area: Problem area definition
            polygon: Building polygon
            cassettes: Current cassettes

        Returns:
            Optimized cassette list or None if no improvement
        """
        if problem_area['type'] == 'large_gap':
            return self._optimize_gap(
                problem_area['gap'],
                problem_area['nearby_cassettes'],
                polygon,
                cassettes
            )
        elif problem_area['type'] == 'overlap':
            return self._fix_overlap(
                problem_area['cassettes'],
                polygon,
                cassettes
            )
        elif problem_area['type'] == 'inefficient':
            return self._replace_inefficient(
                problem_area['cassette_index'],
                polygon,
                cassettes
            )

        return None

    def _optimize_gap(self, gap: Dict[str, Any],
                     nearby_indices: List[int],
                     polygon: List[Tuple[float, float]],
                     cassettes: List[Cassette]) -> Optional[List[Cassette]]:
        """
        Optimize a gap by rearranging nearby cassettes.

        Args:
            gap: Gap information
            nearby_indices: Indices of nearby cassettes
            polygon: Building polygon
            cassettes: Current cassettes

        Returns:
            Optimized cassette list or None
        """
        # Remove nearby cassettes
        new_cassettes = [c for i, c in enumerate(cassettes) if i not in nearby_indices]

        # Calculate area to refill
        removed_area = sum(cassettes[i].area for i in nearby_indices)
        gap_area = gap['area']
        total_area = removed_area + gap_area

        # Try to fill with better configuration
        fill_region = {
            'min_x': gap['min_x'] - 4,
            'max_x': gap['max_x'] + 4,
            'min_y': gap['min_y'] - 4,
            'max_y': gap['max_y'] + 4
        }

        new_placement = self._fill_region(
            fill_region,
            polygon,
            new_cassettes,
            target_area=total_area
        )

        if new_placement:
            return new_cassettes + new_placement

        return None

    def _fix_overlap(self, overlap_indices: Tuple[int, int],
                    polygon: List[Tuple[float, float]],
                    cassettes: List[Cassette]) -> Optional[List[Cassette]]:
        """
        Fix overlapping cassettes.

        Args:
            overlap_indices: Indices of overlapping cassettes
            polygon: Building polygon
            cassettes: Current cassettes

        Returns:
            Fixed cassette list or None
        """
        i, j = overlap_indices

        # Remove both overlapping cassettes
        new_cassettes = [c for idx, c in enumerate(cassettes) if idx not in (i, j)]

        # Try to place non-overlapping cassettes in the same area
        c1, c2 = cassettes[i], cassettes[j]

        # Define search area
        search_area = {
            'min_x': min(c1.x, c2.x),
            'max_x': max(c1.x + c1.width, c2.x + c2.width),
            'min_y': min(c1.y, c2.y),
            'max_y': max(c1.y + c1.height, c2.y + c2.height)
        }

        # Try to fill with non-overlapping cassettes
        new_placement = self._fill_region(
            search_area,
            polygon,
            new_cassettes,
            target_area=c1.area + c2.area
        )

        if new_placement:
            return new_cassettes + new_placement

        return None

    def _replace_inefficient(self, cassette_index: int,
                            polygon: List[Tuple[float, float]],
                            cassettes: List[Cassette]) -> Optional[List[Cassette]]:
        """
        Replace an inefficient cassette with a better configuration.

        Args:
            cassette_index: Index of inefficient cassette
            polygon: Building polygon
            cassettes: Current cassettes

        Returns:
            Optimized cassette list or None
        """
        # Remove the inefficient cassette
        inefficient = cassettes[cassette_index]
        new_cassettes = [c for i, c in enumerate(cassettes) if i != cassette_index]

        # Try larger cassettes in the same area
        for width, height in self.REPLACEMENT_SIZES:
            if width <= inefficient.width and height <= inefficient.height:
                continue  # Not larger

            # Try different positions around the original
            for dx in [-2, -1, 0, 1, 2]:
                for dy in [-2, -1, 0, 1, 2]:
                    x = inefficient.x + dx
                    y = inefficient.y + dy

                    new_cassette = Cassette(x, y, width, height)

                    if self._is_valid_placement(new_cassette, polygon, new_cassettes):
                        return new_cassettes + [new_cassette]

        return None

    def _global_optimization(self, polygon: List[Tuple[float, float]],
                           cassettes: List[Cassette]) -> List[Cassette]:
        """
        Perform global optimization by removing and refilling low-density areas.

        Args:
            polygon: Building polygon
            cassettes: Current cassettes

        Returns:
            Optimized cassette list
        """
        # Calculate density map
        density_map = self._calculate_density_map(polygon, cassettes)

        # Find low-density region
        low_density_region = self._find_low_density_region(density_map)

        if not low_density_region:
            return cassettes

        # Remove cassettes in low-density region
        new_cassettes = []
        removed_area = 0

        for cassette in cassettes:
            center = (cassette.x + cassette.width/2, cassette.y + cassette.height/2)
            if self._point_in_region(center, low_density_region):
                removed_area += cassette.area
            else:
                new_cassettes.append(cassette)

        # Refill the region
        new_placement = self._fill_region(
            low_density_region,
            polygon,
            new_cassettes,
            target_area=removed_area * 1.1  # Try to improve by 10%
        )

        if new_placement:
            return new_cassettes + new_placement

        return cassettes

    def _fill_region(self, region: Dict[str, float],
                    polygon: List[Tuple[float, float]],
                    existing_cassettes: List[Cassette],
                    target_area: float) -> List[Cassette]:
        """
        Fill a region with cassettes.

        Args:
            region: Region bounds (min_x, max_x, min_y, max_y)
            polygon: Building polygon
            existing_cassettes: Already placed cassettes
            target_area: Target area to fill

        Returns:
            List of new cassettes
        """
        placed = []
        filled_area = 0

        # Try cassettes from largest to smallest
        for width, height in self.REPLACEMENT_SIZES:
            if filled_area >= target_area:
                break

            y = region['min_y']
            while y + height <= region['max_y'] and filled_area < target_area:
                x = region['min_x']
                while x + width <= region['max_x'] and filled_area < target_area:
                    cassette = Cassette(x, y, width, height)

                    if self._is_valid_placement(cassette, polygon, existing_cassettes + placed):
                        placed.append(cassette)
                        filled_area += cassette.area

                    x += width - 0.5
                y += height - 0.5

        return placed

    def _find_gaps(self, polygon: List[Tuple[float, float]],
                  cassettes: List[Cassette],
                  min_gap_size: float) -> List[Dict[str, Any]]:
        """Find gaps larger than minimum size."""
        # Simplified gap detection
        gaps = []

        # Get bounding box
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # Grid-based gap detection
        grid_size = 2.0
        y = min_y
        while y < max_y:
            x = min_x
            while x < max_x:
                # Check 4x4 region
                region_covered = False
                for cassette in cassettes:
                    if (cassette.x <= x <= cassette.x + cassette.width and
                        cassette.y <= y <= cassette.y + cassette.height):
                        region_covered = True
                        break

                if not region_covered and self._point_in_polygon((x + 2, y + 2), polygon):
                    gaps.append({
                        'min_x': x,
                        'max_x': x + 4,
                        'min_y': y,
                        'max_y': y + 4,
                        'center': (x + 2, y + 2),
                        'area': 16.0
                    })

                x += grid_size
            y += grid_size

        return [g for g in gaps if g['area'] >= min_gap_size]

    def _find_nearby_cassettes(self, point: Tuple[float, float],
                              cassettes: List[Cassette],
                              radius: float) -> List[int]:
        """Find cassettes within radius of a point."""
        nearby = []
        x, y = point

        for i, cassette in enumerate(cassettes):
            center = (cassette.x + cassette.width/2, cassette.y + cassette.height/2)
            distance = math.sqrt((center[0] - x)**2 + (center[1] - y)**2)
            if distance <= radius:
                nearby.append(i)

        return nearby

    def _find_overlaps(self, cassettes: List[Cassette]) -> List[Tuple[int, int]]:
        """Find overlapping cassette pairs."""
        overlaps = []
        n = len(cassettes)

        for i in range(n):
            for j in range(i + 1, n):
                if cassettes[i].overlaps(cassettes[j]):
                    overlaps.append((i, j))

        return overlaps

    def _find_inefficient_cassettes(self, polygon: List[Tuple[float, float]],
                                   cassettes: List[Cassette]) -> List[int]:
        """Find cassettes that could be replaced with larger ones."""
        inefficient = []

        for i, cassette in enumerate(cassettes):
            if cassette.area < 24:  # Small cassette
                # Check if a larger cassette could fit
                for width, height in [(6, 8), (5, 8), (6, 6)]:
                    if width <= cassette.width and height <= cassette.height:
                        continue

                    test_cassette = Cassette(cassette.x, cassette.y, width, height)
                    other_cassettes = [c for j, c in enumerate(cassettes) if j != i]

                    if self._is_valid_placement(test_cassette, polygon, other_cassettes):
                        inefficient.append(i)
                        break

        return inefficient

    def _calculate_density_map(self, polygon: List[Tuple[float, float]],
                              cassettes: List[Cassette]) -> Dict[str, Any]:
        """Calculate coverage density map."""
        # Simplified density calculation
        return {
            'grid_size': 5.0,
            'densities': []
        }

    def _find_low_density_region(self, density_map: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Find the lowest density region."""
        # Simplified - return a random region for testing
        return None

    def _point_in_region(self, point: Tuple[float, float], region: Dict[str, float]) -> bool:
        """Check if point is in region."""
        if not region:
            return False
        x, y = point
        return (region['min_x'] <= x <= region['max_x'] and
                region['min_y'] <= y <= region['max_y'])

    def _calculate_coverage(self, polygon: List[Tuple[float, float]],
                          cassettes: List[Cassette]) -> float:
        """Calculate coverage percentage."""
        polygon_area = self._calculate_polygon_area(polygon)
        covered_area = sum(c.area for c in cassettes)
        return (covered_area / polygon_area * 100) if polygon_area > 0 else 0

    def _calculate_polygon_area(self, polygon: List[Tuple[float, float]]) -> float:
        """Calculate polygon area using shoelace formula."""
        n = len(polygon)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += polygon[i][0] * polygon[j][1]
            area -= polygon[j][0] * polygon[i][1]
        return abs(area) / 2.0

    def _is_valid_placement(self, cassette: Cassette,
                           polygon: List[Tuple[float, float]],
                           existing_cassettes: List[Cassette]) -> bool:
        """Check if cassette placement is valid."""
        # Check overlaps
        for existing in existing_cassettes:
            if cassette.overlaps(existing):
                return False

        # Check boundaries
        corners = cassette.get_corners()
        for corner in corners:
            if not self._point_in_polygon(corner, polygon):
                return False

        return True

    def _point_in_polygon(self, point: Tuple[float, float],
                         polygon: List[Tuple[float, float]]) -> bool:
        """Check if point is in polygon."""
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


def test_intelligent_backtracker():
    """Test the IntelligentBacktracker stage."""
    from optimization_pipeline import create_standard_pipeline
    from gap_filler_stage import GapFiller

    # Test polygon
    test_polygon = [
        (0, 0),
        (40, 0),
        (40, 30),
        (0, 30)
    ]

    # Create full pipeline
    pipeline = create_standard_pipeline()
    pipeline.add_stage(GapFiller())
    pipeline.add_stage(IntelligentBacktracker())

    # Run optimization
    results = pipeline.optimize(test_polygon)

    print("\n" + "="*60)
    print("INTELLIGENT BACKTRACKER TEST RESULTS")
    print("="*60)
    print(f"Total cassettes: {results['num_cassettes']}")
    print(f"Coverage: {results['coverage_percent']:.1f}%")
    print(f"Meets 94% requirement: {results['meets_requirement']}")

    # Show optimization improvement
    if 'intelligent_backtracker' in results['stage_results']:
        bt_data = results['stage_results']['intelligent_backtracker']
        print(f"\nBacktracker improvement: {bt_data.get('coverage_gain', 0):.1f}% "
              f"in {bt_data.get('iterations', 0)} iterations")

    return results


if __name__ == "__main__":
    test_intelligent_backtracker()
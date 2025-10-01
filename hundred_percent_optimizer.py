"""
100% Coverage Optimizer - Production grade implementation
Primary goal: 100% coverage
Secondary goal: Minimize cassette count
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Cassette:
    """Floor cassette module"""
    x: float
    y: float
    width: float
    height: float

    def __post_init__(self):
        # Round to 0.5 ft increments
        self.x = round(self.x * 2) / 2
        self.y = round(self.y * 2) / 2
        self.width = round(self.width * 2) / 2
        self.height = round(self.height * 2) / 2

    @property
    def area(self):
        return self.width * self.height

    @property
    def weight(self):
        return self.area * 10.4

    @property
    def size_name(self):
        return f"{self.width}x{self.height}"

    def contains_point(self, x: float, y: float) -> bool:
        """Check if point is inside cassette (inclusive of boundaries)"""
        return (self.x <= x <= self.x + self.width and
                self.y <= y <= self.y + self.height)

    def overlaps_with(self, other: 'Cassette') -> bool:
        """Check overlap with another cassette"""
        return not (self.x >= other.x + other.width or
                   self.x + self.width <= other.x or
                   self.y >= other.y + other.height or
                   self.y + self.height <= other.y)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'x': self.x,
            'y': self.y,
            'width': self.width,
            'height': self.height,
            'size': self.size_name,
            'area': self.area,
            'weight': self.weight
        }


class HundredPercentOptimizer:
    """Achieves 100% coverage with minimum cassette count"""

    # 9 cassette sizes in 0.5 ft increments
    CASSETTE_SIZES = [
        (6.0, 8.0),  # 48.0 sq ft - Max size
        (5.5, 7.5),  # 41.25 sq ft
        (5.0, 7.0),  # 35.0 sq ft
        (6.0, 6.0),  # 36.0 sq ft
        (5.0, 5.5),  # 27.5 sq ft
        (4.5, 5.0),  # 22.5 sq ft
        (4.0, 4.5),  # 18.0 sq ft
        (3.5, 4.0),  # 14.0 sq ft
        (3.5, 3.5),  # 12.25 sq ft - Min size (42"x42")
    ]

    def __init__(self, polygon: List[Tuple[float, float]]):
        self.polygon = polygon
        self.min_x = min(p[0] for p in polygon)
        self.max_x = max(p[0] for p in polygon)
        self.min_y = min(p[1] for p in polygon)
        self.max_y = max(p[1] for p in polygon)
        self.total_area = self._calculate_area()
        self.grid_resolution = 0.5

    def _calculate_area(self) -> float:
        """Calculate polygon area using shoelace formula"""
        n = len(self.polygon)
        area = 0
        for i in range(n):
            j = (i + 1) % n
            area += self.polygon[i][0] * self.polygon[j][1]
            area -= self.polygon[j][0] * self.polygon[i][1]
        return abs(area) / 2

    def _point_in_polygon(self, x: float, y: float) -> bool:
        """Check if point is inside polygon"""
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

    def _point_on_edge(self, x: float, y: float, tolerance: float = 0.01) -> bool:
        """Check if point is on polygon edge"""
        n = len(self.polygon)
        for i in range(n):
            x1, y1 = self.polygon[i]
            x2, y2 = self.polygon[(i + 1) % n]

            # Check if on horizontal or vertical edge
            if abs(x2 - x1) < tolerance:  # Vertical
                if abs(x - x1) < tolerance and min(y1, y2) <= y <= max(y1, y2):
                    return True
            elif abs(y2 - y1) < tolerance:  # Horizontal
                if abs(y - y1) < tolerance and min(x1, x2) <= x <= max(x1, x2):
                    return True

        return False

    def _is_cassette_valid(self, cassette: Cassette, placed: List[Cassette]) -> bool:
        """Validate cassette placement"""
        # Check weight limit
        if cassette.weight > 500:
            return False

        # Check overlap with existing
        for existing in placed:
            if cassette.overlaps_with(existing):
                return False

        # Check all corners are inside or on edge
        corners = [
            (cassette.x, cassette.y),
            (cassette.x + cassette.width, cassette.y),
            (cassette.x + cassette.width, cassette.y + cassette.height),
            (cassette.x, cassette.y + cassette.height)
        ]

        for x, y in corners:
            if not (self._point_in_polygon(x, y) or self._point_on_edge(x, y)):
                return False

        return True

    def _calculate_coverage(self, cassettes: List[Cassette]) -> Dict:
        """Calculate accurate coverage using grid sampling"""
        if not cassettes:
            return {'covered_area': 0, 'gap_area': self.total_area, 'coverage_percent': 0}

        # Use fine grid for accurate measurement
        resolution = 0.25
        covered_points = 0
        total_points = 0

        y = self.min_y
        while y <= self.max_y:
            x = self.min_x
            while x <= self.max_x:
                # Check if point is inside or on edge of polygon
                if self._point_in_polygon(x, y) or self._point_on_edge(x, y):
                    total_points += 1

                    # Check if covered by any cassette
                    covered = False
                    for cassette in cassettes:
                        if cassette.contains_point(x, y):
                            covered = True
                            break

                    if covered:
                        covered_points += 1

                x += resolution
            y += resolution

        if total_points == 0:
            return {'covered_area': 0, 'gap_area': self.total_area, 'coverage_percent': 0}

        coverage_ratio = covered_points / total_points
        covered_area = coverage_ratio * self.total_area
        gap_area = self.total_area - covered_area
        coverage_percent = coverage_ratio * 100

        return {
            'covered_area': covered_area,
            'gap_area': gap_area,
            'coverage_percent': coverage_percent
        }

    def _place_largest_first(self) -> List[Cassette]:
        """Strategy: Place largest cassettes first"""
        cassettes = []

        # Scan polygon in rows
        y = self.min_y
        while y <= self.max_y:
            x = self.min_x
            row_height = 0

            while x <= self.max_x:
                best_cassette = None
                best_area = 0

                # Try each size, largest first
                for width, height in self.CASSETTE_SIZES:
                    # Try both orientations
                    for w, h in [(width, height), (height, width)]:
                        cassette = Cassette(x, y, w, h)
                        if self._is_cassette_valid(cassette, cassettes):
                            if cassette.area > best_area:
                                best_cassette = cassette
                                best_area = cassette.area

                if best_cassette:
                    cassettes.append(best_cassette)
                    x += best_cassette.width
                    row_height = max(row_height, best_cassette.height)
                else:
                    x += self.grid_resolution

            if row_height > 0:
                y += row_height
            else:
                y += self.grid_resolution

        return cassettes

    def _place_balanced(self) -> List[Cassette]:
        """Strategy: Use medium sizes for balance"""
        cassettes = []

        # Prefer middle sizes
        preferred_indices = [3, 4, 5]  # Medium sizes

        y = self.min_y
        while y <= self.max_y:
            x = self.min_x
            row_height = 0

            while x <= self.max_x:
                placed = False

                # Try preferred sizes first
                for idx in preferred_indices:
                    width, height = self.CASSETTE_SIZES[idx]
                    for w, h in [(width, height), (height, width)]:
                        cassette = Cassette(x, y, w, h)
                        if self._is_cassette_valid(cassette, cassettes):
                            cassettes.append(cassette)
                            x += w
                            row_height = max(row_height, h)
                            placed = True
                            break
                    if placed:
                        break

                # Try other sizes if preferred didn't work
                if not placed:
                    for i, (width, height) in enumerate(self.CASSETTE_SIZES):
                        if i in preferred_indices:
                            continue
                        for w, h in [(width, height), (height, width)]:
                            cassette = Cassette(x, y, w, h)
                            if self._is_cassette_valid(cassette, cassettes):
                                cassettes.append(cassette)
                                x += w
                                row_height = max(row_height, h)
                                placed = True
                                break
                        if placed:
                            break

                if not placed:
                    x += self.grid_resolution

            if row_height > 0:
                y += row_height
            else:
                y += self.grid_resolution

        return cassettes

    def _fill_gaps(self, cassettes: List[Cassette]) -> List[Cassette]:
        """Fill remaining gaps with smallest fitting cassettes - aggressive approach"""
        max_iterations = 50  # More iterations for thorough coverage
        iteration = 0
        resolution = 0.5  # Use grid resolution for scanning

        while iteration < max_iterations:
            gaps_filled = False
            uncovered_points = []

            # First pass: identify all uncovered points
            y = self.min_y
            while y <= self.max_y:
                x = self.min_x
                while x <= self.max_x:
                    # Include edge points in coverage check
                    if self._point_in_polygon(x, y) or self._point_on_edge(x, y):
                        covered = any(c.contains_point(x, y) for c in cassettes)
                        if not covered:
                            uncovered_points.append((x, y))
                    x += resolution
                y += resolution

            if not uncovered_points:
                break  # 100% coverage achieved!

            # Second pass: try to place cassettes at uncovered points
            for x, y in uncovered_points:
                # Try all positions around this uncovered point with finer steps
                for dx in np.arange(0, -3.5, -0.5):
                    for dy in np.arange(0, -3.5, -0.5):
                        test_x = round((x + dx) * 2) / 2  # Ensure 0.5 ft increments
                        test_y = round((y + dy) * 2) / 2

                        # Try all cassette sizes, smallest first for better gap filling
                        for width, height in reversed(self.CASSETTE_SIZES):
                            for w, h in [(width, height), (height, width)]:
                                cassette = Cassette(test_x, test_y, w, h)

                                # Check if this cassette covers the uncovered point
                                if cassette.contains_point(x, y):
                                    if self._is_cassette_valid(cassette, cassettes):
                                        cassettes.append(cassette)
                                        gaps_filled = True
                                        break
                            if gaps_filled:
                                break
                        if gaps_filled:
                            break
                    if gaps_filled:
                        break
                if gaps_filled:
                    break  # Move to next iteration to rescan

            if not gaps_filled:
                # Try micro-adjustments of existing cassettes
                for i, cassette in enumerate(cassettes):
                    for dx in [-0.5, 0.5]:
                        for dy in [-0.5, 0.5]:
                            adjusted = Cassette(
                                cassette.x + dx,
                                cassette.y + dy,
                                cassette.width,
                                cassette.height
                            )
                            temp_cassettes = cassettes[:i] + cassettes[i+1:]
                            if self._is_cassette_valid(adjusted, temp_cassettes):
                                # Check if this improves coverage
                                temp_cassettes.append(adjusted)
                                new_uncovered = 0
                                y = self.min_y
                                while y <= self.max_y:
                                    x = self.min_x
                                    while x <= self.max_x:
                                        if self._point_in_polygon(x, y) or self._point_on_edge(x, y):
                                            if not any(c.contains_point(x, y) for c in temp_cassettes):
                                                new_uncovered += 1
                                        x += resolution
                                    y += resolution

                                if new_uncovered < len(uncovered_points):
                                    cassettes = temp_cassettes
                                    gaps_filled = True
                                    break
                        if gaps_filled:
                            break
                    if gaps_filled:
                        break

            if not gaps_filled:
                # Log details about remaining gaps
                logger.info(f"Gap filling stopped - {len(uncovered_points)} points remain uncovered")

                # Analyze gap locations
                if uncovered_points:
                    sample_points = uncovered_points[:5]  # Show first 5
                    for px, py in sample_points:
                        logger.debug(f"  Uncovered point at ({px:.1f}, {py:.1f})")

                break  # Can't fill any more gaps

            iteration += 1

        # Final report on coverage
        final_uncovered = 0
        y = self.min_y
        while y <= self.max_y:
            x = self.min_x
            while x <= self.max_x:
                if self._point_in_polygon(x, y) or self._point_on_edge(x, y):
                    if not any(c.contains_point(x, y) for c in cassettes):
                        final_uncovered += 1
                x += resolution
            y += resolution

        if final_uncovered > 0:
            logger.info(f"Final gap filling result: {final_uncovered} points remain uncovered")

        return cassettes

    def optimize(self) -> Dict:
        """Main optimization: achieve 100% coverage"""
        logger.info("Starting 100% coverage optimization")
        logger.info(f"Total area: {self.total_area:.1f} sq ft")

        # Try multiple strategies
        strategies = [
            ("Largest First", self._place_largest_first()),
            ("Balanced", self._place_balanced()),
        ]

        best_solution = None
        best_coverage = 0
        best_count = float('inf')

        for name, cassettes in strategies:
            coverage_info = self._calculate_coverage(cassettes)
            coverage = coverage_info['coverage_percent']
            count = len(cassettes)

            logger.info(f"{name}: {coverage:.1f}% coverage, {count} cassettes")

            # Select best: max coverage, then min count
            if coverage > best_coverage or (coverage == best_coverage and count < best_count):
                best_solution = cassettes
                best_coverage = coverage
                best_count = count

        # Fill gaps if not 100%
        if best_coverage < 100:
            logger.info("Filling gaps for 100% coverage...")
            best_solution = self._fill_gaps(best_solution)
            coverage_info = self._calculate_coverage(best_solution)
            best_coverage = coverage_info['coverage_percent']

        # Format results
        coverage_info = self._calculate_coverage(best_solution)

        # Size distribution
        size_dist = {}
        for c in best_solution:
            size_dist[c.size_name] = size_dist.get(c.size_name, 0) + 1

        return {
            'success': True,
            'cassettes': [c.to_dict() for c in best_solution],
            'num_cassettes': len(best_solution),
            'total_area': self.total_area,
            'covered_area': coverage_info['covered_area'],
            'gap_area': coverage_info['gap_area'],
            'coverage': coverage_info['coverage_percent'] / 100,
            'coverage_percent': coverage_info['coverage_percent'],
            'gap_percent': coverage_info['gap_area'] / self.total_area * 100,
            'size_distribution': size_dist,
            'total_weight': sum(c.weight for c in best_solution),
            'avg_cassette_area': sum(c.area for c in best_solution) / len(best_solution) if best_solution else 0,
            'meets_requirement': coverage_info['coverage_percent'] >= 100,
            'polygon': self.polygon
        }


if __name__ == "__main__":
    # Test with rectangle
    test_polygon = [
        (0, 0),
        (20, 0),
        (20, 15),
        (0, 15)
    ]

    optimizer = HundredPercentOptimizer(test_polygon)
    results = optimizer.optimize()

    print("\n" + "="*70)
    print("100% COVERAGE OPTIMIZATION RESULTS")
    print("="*70)
    print(f"Coverage: {results['coverage_percent']:.1f}%")
    print(f"Cassettes: {results['num_cassettes']}")
    print(f"Total area: {results['total_area']:.1f} sq ft")
    print(f"Covered area: {results['covered_area']:.1f} sq ft")
    print(f"Gap area: {results['gap_area']:.1f} sq ft")
    print(f"Total weight: {results['total_weight']:.0f} lbs")

    if results['meets_requirement']:
        print("\nACHIEVED 100% COVERAGE!")
    else:
        print(f"\nCoverage: {results['coverage_percent']:.1f}% (Target: 100%)")

    print("\nSize distribution:")
    for size, count in sorted(results['size_distribution'].items()):
        print(f"  {size}: {count} cassettes")
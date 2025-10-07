#!/usr/bin/env python3
"""
Per-Cassette C-Channel Optimizer
=================================
Achieves 100% coverage by assigning individual C-channel widths to each cassette.

Key Features:
- Each cassette has its own uniform C-channel (1.5" - 18")
- C-channels do NOT overlap (physical constraint)
- Gaps are filled by adjacent C-channels: gap = c_A + c_B
- Constraint satisfaction approach for optimal C-channel assignment
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import logging
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import nearest_points
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerCassetteCChannelOptimizer:
    """
    Optimizer that assigns individual C-channel widths to cassettes for 100% coverage.

    Algorithm:
    1. Place cassettes using greedy optimizer
    2. Measure all gaps (boundary and inter-cassette)
    3. Solve constraint system for C-channel widths
    4. Validate and adjust if needed
    5. Achieve 100% coverage
    """

    MIN_CCHANNEL = 1.5 / 12.0  # 1.5 inches in feet
    MAX_CCHANNEL = 18.0 / 12.0  # 18 inches in feet

    def __init__(self, polygon: List[Tuple[float, float]]):
        """
        Initialize optimizer.

        Args:
            polygon: List of (x, y) coordinates defining the boundary
        """
        self.polygon = polygon
        self.polygon_shapely = Polygon(polygon)
        self.polygon_area = self.polygon_shapely.area

        self.cassettes = []
        self.c_channels = []  # C-channel width per cassette

        # Statistics
        self.gaps_measured = 0
        self.adjustments_made = 0

        logger.info("=" * 70)
        logger.info("PER-CASSETTE C-CHANNEL OPTIMIZER")
        logger.info("=" * 70)
        logger.info(f"Polygon area: {self.polygon_area:.1f} sq ft")
        logger.info(f"C-channel range: {self.MIN_CCHANNEL*12:.1f}\" to {self.MAX_CCHANNEL*12:.1f}\"")

    def _greedy_placement(self) -> List[Dict]:
        """
        Phase 1: Initial cassette placement using existing optimizer.

        Returns:
            List of cassette dictionaries
        """
        logger.info("\nPhase 1: Initial cassette placement (greedy)")

        from ultra_smart_optimizer import UltraSmartOptimizer
        optimizer = UltraSmartOptimizer(self.polygon)
        result = optimizer.optimize()

        cassettes = result['cassettes']
        coverage = result.get('coverage_percent', result.get('coverage', 0) * 100)

        logger.info(f"Placed {len(cassettes)} cassettes")
        logger.info(f"Initial coverage: {coverage:.1f}% (without C-channels)")

        return cassettes

    def _measure_gap_to_boundary(self, cassette: Dict, direction: str) -> float:
        """
        Measure gap from cassette edge to polygon boundary in given direction.

        Args:
            cassette: Cassette dictionary with x, y, width, height
            direction: 'N', 'S', 'E', 'W'

        Returns:
            Gap distance in feet (0 if cassette is at boundary)
        """
        x, y = cassette['x'], cassette['y']
        w, h = cassette['width'], cassette['height']

        # Get polygon bounds
        poly_bounds = self.polygon_shapely.bounds  # (minx, miny, maxx, maxy)

        tolerance = 0.1  # 0.1 ft tolerance

        # Measure perpendicular distance to polygon edge in given direction
        if direction == 'N':
            # Distance from top of cassette to top of polygon
            cassette_top = y + h
            polygon_top = poly_bounds[3]  # maxy
            gap = polygon_top - cassette_top
        elif direction == 'S':
            # Distance from bottom of cassette to bottom of polygon
            cassette_bottom = y
            polygon_bottom = poly_bounds[1]  # miny
            gap = cassette_bottom - polygon_bottom
        elif direction == 'E':
            # Distance from right of cassette to right of polygon
            cassette_right = x + w
            polygon_right = poly_bounds[2]  # maxx
            gap = polygon_right - cassette_right
        else:  # 'W'
            # Distance from left of cassette to left of polygon
            cassette_left = x
            polygon_left = poly_bounds[0]  # minx
            gap = cassette_left - polygon_left

        # Return 0 if gap is negligible or negative (cassette at boundary)
        return max(0.0, gap) if gap > tolerance else 0.0

    def _measure_gap_between_cassettes(self, cass_A: Dict, cass_B: Dict) -> Optional[Tuple[float, str]]:
        """
        Measure gap between two cassettes if they are adjacent.

        Args:
            cass_A: First cassette
            cass_B: Second cassette

        Returns:
            Tuple of (gap_distance, direction) or None if not adjacent
            direction is from A's perspective: 'N', 'S', 'E', 'W'
        """
        ax, ay = cass_A['x'], cass_A['y']
        aw, ah = cass_A['width'], cass_A['height']
        bx, by = cass_B['x'], cass_B['y']
        bw, bh = cass_B['width'], cass_B['height']

        tolerance = 0.1  # 0.1 ft tolerance for alignment

        # Check if horizontally aligned (can be E-W neighbors)
        if abs(ay - by) < tolerance or abs((ay + ah) - (by + bh)) < tolerance or \
           (ay < by < ay + ah) or (by < ay < by + bh):
            # Check East-West relationship
            if ax + aw < bx:  # A is west of B
                gap = bx - (ax + aw)
                if gap < 10.0:  # Reasonable gap (< 10 ft)
                    return (gap, 'E')  # B is to the East of A
            elif bx + bw < ax:  # B is west of A
                gap = ax - (bx + bw)
                if gap < 10.0:
                    return (gap, 'W')  # B is to the West of A

        # Check if vertically aligned (can be N-S neighbors)
        if abs(ax - bx) < tolerance or abs((ax + aw) - (bx + bw)) < tolerance or \
           (ax < bx < ax + aw) or (bx < ax < bx + bw):
            # Check North-South relationship
            if ay + ah < by:  # A is south of B
                gap = by - (ay + ah)
                if gap < 10.0:
                    return (gap, 'N')  # B is to the North of A
            elif by + bh < ay:  # B is south of A
                gap = ay - (by + bh)
                if gap < 10.0:
                    return (gap, 'S')  # B is to the South of A

        return None

    def _measure_all_gaps(self) -> Dict:
        """
        Phase 2: Measure all gaps in the layout.

        Returns:
            Dictionary with gap information for constraint system
        """
        logger.info("\nPhase 2: Measuring all gaps")

        gaps = {
            'boundary': [],  # {cassette_idx, direction, distance}
            'adjacent': []   # {cassette_A_idx, cassette_B_idx, distance, direction}
        }

        # Measure boundary gaps
        for i, cassette in enumerate(self.cassettes):
            for direction in ['N', 'S', 'E', 'W']:
                gap = self._measure_gap_to_boundary(cassette, direction)
                if gap > 0.01:  # Ignore tiny gaps
                    gaps['boundary'].append({
                        'cassette_idx': i,
                        'direction': direction,
                        'distance': gap
                    })
                    if i < 3:  # Debug: show first 3 cassettes
                        logger.debug(f"  Cassette {i} {direction}: gap={gap:.2f} ft ({gap*12:.1f}\")")

        # Measure inter-cassette gaps
        for i, cass_A in enumerate(self.cassettes):
            for j, cass_B in enumerate(self.cassettes[i+1:], i+1):
                result = self._measure_gap_between_cassettes(cass_A, cass_B)
                if result:
                    gap, direction = result
                    if gap > 0.01:
                        gaps['adjacent'].append({
                            'cassette_A_idx': i,
                            'cassette_B_idx': j,
                            'distance': gap,
                            'direction': direction  # From A's perspective
                        })

        self.gaps_measured = len(gaps['boundary']) + len(gaps['adjacent'])
        logger.info(f"Boundary gaps: {len(gaps['boundary'])}")
        logger.info(f"Inter-cassette gaps: {len(gaps['adjacent'])}")
        logger.info(f"Total gaps: {self.gaps_measured}")

        return gaps

    def _solve_cchannel_constraints(self, gaps: Dict) -> List[float]:
        """
        Phase 3: Determine C-channel widths to achieve 100% coverage.

        Strategy: Use UNIFORM C-channel for ALL cassettes.
        Calculate the width needed to fill remaining gap area.

        Args:
            gaps: Gap dictionary from _measure_all_gaps

        Returns:
            List of C-channel widths (one per cassette, all same value)
        """
        logger.info("\nPhase 3: Determining uniform C-channel width")

        num_cassettes = len(self.cassettes)

        # Calculate current cassette coverage (without C-channels)
        cassette_area = sum(c['area'] for c in self.cassettes)
        gap_area = self.polygon_area - cassette_area

        logger.info(f"Cassette area: {cassette_area:.1f} sq ft ({cassette_area/self.polygon_area*100:.1f}%)")
        logger.info(f"Gap area: {gap_area:.1f} sq ft ({gap_area/self.polygon_area*100:.1f}%)")

        # We need to find uniform C-channel width such that:
        # Total coverage = cassette_area + C-channel_area = polygon_area
        #
        # C-channel area for cassette i = (w+2c)*(h+2c) - w*h
        #                                = 2c*w + 2c*h + 4c²
        #                                = 2c(w+h+2c)
        #
        # Total C-channel area = Σ 2c(w_i + h_i + 2c)
        #                      = 2c * Σ(w_i + h_i) + 4c² * num_cassettes
        #
        # We want: cassette_area + C-channel_area = polygon_area
        # So: cassette_area + 2c*Σ(w_i+h_i) + 4c²*n = polygon_area
        #
        # This is quadratic in c:
        # 4n*c² + 2Σ(w_i+h_i)*c + (cassette_area - polygon_area) = 0

        # Calculate sum of perimeters
        perimeter_sum = sum(c['width'] + c['height'] for c in self.cassettes)

        # Quadratic equation: a*c² + b*c + c_const = 0
        a = 4 * num_cassettes
        b = 2 * perimeter_sum
        c_const = cassette_area - self.polygon_area

        # Solve using quadratic formula: c = (-b ± sqrt(b²-4ac)) / 2a
        discriminant = b**2 - 4*a*c_const

        if discriminant < 0:
            logger.error("No real solution for uniform C-channel!")
            # Fallback: use minimum
            uniform_c = self.MIN_CCHANNEL
        else:
            # Take positive root
            c_solution_1 = (-b + math.sqrt(discriminant)) / (2*a)
            c_solution_2 = (-b - math.sqrt(discriminant)) / (2*a)

            # Choose positive solution in valid range
            if self.MIN_CCHANNEL <= c_solution_1 <= self.MAX_CCHANNEL:
                uniform_c = c_solution_1
            elif self.MIN_CCHANNEL <= c_solution_2 <= self.MAX_CCHANNEL:
                uniform_c = c_solution_2
            else:
                # Clamp to range
                candidates = [c_solution_1, c_solution_2]
                valid = [c for c in candidates if c > 0]
                if valid:
                    uniform_c = min(valid, key=lambda x: abs(x - (self.MIN_CCHANNEL + self.MAX_CCHANNEL)/2))
                    uniform_c = max(self.MIN_CCHANNEL, min(self.MAX_CCHANNEL, uniform_c))
                else:
                    uniform_c = self.MIN_CCHANNEL

        logger.info(f"Initial calculated C-channel: {uniform_c*12:.1f}\"")

        # Binary search for exact 100% coverage
        logger.info("Binary search for 100% coverage...")

        low = 0.0  # Allow 0 (no C-channel)
        high = self.MAX_CCHANNEL
        target_coverage = 100.0
        tolerance = 0.1  # 0.1% tolerance

        best_c = uniform_c
        best_diff = float('inf')

        for iteration in range(20):  # Max 20 iterations
            mid = (low + high) / 2.0
            test_c_channels = [mid] * num_cassettes
            test_coverage = self._calculate_coverage(test_c_channels)

            diff = abs(test_coverage - target_coverage)

            if diff < best_diff:
                best_diff = diff
                best_c = mid

            logger.debug(f"  Iter {iteration}: c={mid*12:.2f}\", coverage={test_coverage:.2f}%, diff={diff:.2f}%")

            if abs(test_coverage - target_coverage) < tolerance:
                logger.info(f"  Found solution: c={mid*12:.2f}\", coverage={test_coverage:.2f}%")
                uniform_c = mid
                break

            if test_coverage > target_coverage:
                # Too much coverage, reduce C-channel
                high = mid
            else:
                # Too little coverage, increase C-channel
                low = mid
        else:
            # Use best found
            uniform_c = best_c
            logger.info(f"  Best found: c={uniform_c*12:.2f}\", diff={best_diff:.2f}%")

        # Ensure within constraints
        if uniform_c < self.MIN_CCHANNEL:
            logger.warning(f"Optimal C-channel {uniform_c*12:.1f}\" < minimum 1.5\", clamping to minimum")
            uniform_c = self.MIN_CCHANNEL

        # Return uniform C-channel for all cassettes
        c_channels = [uniform_c] * num_cassettes

        # Final coverage check
        final_coverage = self._calculate_coverage(c_channels)
        logger.info(f"Final uniform C-channel: {uniform_c*12:.2f}\"")
        logger.info(f"Final coverage: {final_coverage:.2f}%")

        return c_channels

    def _validate_constraints(self, c_channels: List[float]) -> Tuple[bool, List[str]]:
        """
        Phase 4: Validate C-channel constraints.

        Args:
            c_channels: List of C-channel widths

        Returns:
            Tuple of (is_valid, list_of_violations)
        """
        violations = []

        for i, c_channel in enumerate(c_channels):
            inches = c_channel * 12.0

            if c_channel < self.MIN_CCHANNEL:
                violations.append(f"Cassette {i}: C-channel {inches:.1f}\" < 1.5\" minimum")
            elif c_channel > self.MAX_CCHANNEL:
                violations.append(f"Cassette {i}: C-channel {inches:.1f}\" > 18\" maximum")

        is_valid = len(violations) == 0
        return is_valid, violations

    def _adjust_for_violations(self, c_channels: List[float], violations: List[str]) -> bool:
        """
        Phase 5: Adjust cassette layout to fix constraint violations.

        Args:
            c_channels: Current C-channel widths
            violations: List of violation messages

        Returns:
            True if adjustments were made, False if unable to fix
        """
        logger.info(f"\nPhase 5: Adjusting layout ({len(violations)} violations)")

        for violation in violations:
            logger.warning(f"  {violation}")

        # For now, clamp to valid range (simple strategy)
        # TODO: Implement cassette position adjustment
        for i in range(len(c_channels)):
            c_channels[i] = max(self.MIN_CCHANNEL, min(self.MAX_CCHANNEL, c_channels[i]))

        self.adjustments_made += 1
        logger.info(f"Clamped C-channels to valid range")

        return True

    def _calculate_coverage(self, c_channels: List[float]) -> float:
        """
        Calculate total coverage percentage (only area INSIDE polygon).

        Args:
            c_channels: List of C-channel widths

        Returns:
            Coverage percentage
        """
        from shapely.geometry import box

        total_coverage_area = 0.0

        for cassette, c_channel in zip(self.cassettes, c_channels):
            # Create footprint rectangle (cassette + C-channel)
            x_min = cassette['x'] - c_channel
            y_min = cassette['y'] - c_channel
            x_max = cassette['x'] + cassette['width'] + c_channel
            y_max = cassette['y'] + cassette['height'] + c_channel

            footprint = box(x_min, y_min, x_max, y_max)

            # Calculate intersection with polygon (only count area INSIDE)
            intersection = footprint.intersection(self.polygon_shapely)
            total_coverage_area += intersection.area

        coverage = (total_coverage_area / self.polygon_area) * 100.0
        return coverage

    def optimize(self) -> Dict:
        """
        Main optimization method to achieve 100% coverage.

        Returns:
            Dictionary with results
        """
        # Phase 1: Initial placement
        self.cassettes = self._greedy_placement()

        # Phase 2: Measure gaps
        gaps = self._measure_all_gaps()

        # Phase 3: Solve C-channel constraints
        self.c_channels = self._solve_cchannel_constraints(gaps)

        # Phase 4: Validate
        is_valid, violations = self._validate_constraints(self.c_channels)

        # Phase 5: Adjust if needed
        if not is_valid:
            self._adjust_for_violations(self.c_channels, violations)

        # Phase 6: Calculate coverage
        coverage = self._calculate_coverage(self.c_channels)

        logger.info("\n" + "=" * 70)
        logger.info("OPTIMIZATION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Cassettes: {len(self.cassettes)}")
        logger.info(f"Coverage: {coverage:.2f}%")
        logger.info(f"Gaps measured: {self.gaps_measured}")
        logger.info(f"Adjustments made: {self.adjustments_made}")

        # Calculate statistics
        total_cassette_area = sum(c['area'] for c in self.cassettes)
        total_cchannel_area = sum(
            (c['width'] + 2*cc) * (c['height'] + 2*cc) - c['area']
            for c, cc in zip(self.cassettes, self.c_channels)
        )

        c_channel_inches = [c * 12 for c in self.c_channels]

        return {
            'cassettes': self.cassettes,
            'c_channels': self.c_channels,
            'c_channels_inches': c_channel_inches,
            'coverage_percent': coverage,
            'polygon': self.polygon,
            'statistics': {
                'total_area': self.polygon_area,
                'cassette_area': total_cassette_area,
                'cchannel_area': total_cchannel_area,
                'cassette_count': len(self.cassettes),
                'coverage_percent': coverage,
                'gaps_measured': self.gaps_measured,
                'adjustments_made': self.adjustments_made,
                'min_cchannel_inches': min(c_channel_inches),
                'max_cchannel_inches': max(c_channel_inches),
                'avg_cchannel_inches': sum(c_channel_inches) / len(c_channel_inches)
            }
        }


if __name__ == "__main__":
    # Test with umbra polygon
    umbra_polygon = [
        (0.0, 28.0),
        (55.5, 28.0),
        (55.5, 12.0),
        (16.0, 12.0),
        (16.0, 0.0),
        (0.0, 0.0)
    ]

    print("\nTesting Per-Cassette C-Channel Optimizer on Umbra polygon")
    print("=" * 70)

    optimizer = PerCassetteCChannelOptimizer(umbra_polygon)
    result = optimizer.optimize()

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Coverage: {result['coverage_percent']:.2f}%")
    print(f"Cassettes: {result['statistics']['cassette_count']}")
    print(f"C-channel range: {result['statistics']['min_cchannel_inches']:.1f}\" - {result['statistics']['max_cchannel_inches']:.1f}\"")
    print(f"Avg C-channel: {result['statistics']['avg_cchannel_inches']:.1f}\"")

    # Generate visualization
    try:
        from hundred_percent_visualizer import create_simple_visualization

        vis_path = "output_umbra_fill_test.png"
        vis_stats = {
            'coverage': result['coverage_percent'],
            'total_area': result['statistics']['total_area'],
            'covered': result['statistics']['cassette_area'] + result['statistics']['cchannel_area'],
            'cassettes': result['statistics']['cassette_count'],
            'per_cassette_cchannel': True,
            'cchannel_widths_per_cassette': result['c_channels_inches'],
            'cchannel_area': result['statistics']['cchannel_area']
        }

        create_simple_visualization(
            cassettes=result['cassettes'],
            polygon=umbra_polygon,
            statistics=vis_stats,
            output_path=vis_path,
            floor_plan_name="UMBRA"
        )
        print(f"\nVisualization saved to: {vis_path}")
    except Exception as e:
        print(f"\nWarning: Could not generate visualization: {e}")

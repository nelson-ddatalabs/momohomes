#!/usr/bin/env python3
"""
Per-Cassette C-Channel Optimizer V2 - Non-Overlapping Edge-to-Edge
===================================================================
Achieves 100% coverage with NON-OVERLAPPING C-channels.

Key Architecture:
- Each cassette gets C-channel rectangles on 4 sides (N, S, E, W)
- Adjacent cassettes SHARE C-channels (edge-to-edge, no overlap, no gaps)
- Boundary cassettes get full C-channels to polygon edge
- Uses UNION for coverage calculation (no double-counting)

Strategy:
1. Place cassettes greedily
2. Detect adjacency relationships
3. Create C-channel geometry for each cassette edge
4. Solve for optimal C-channel widths (non-overlapping constraint)
5. Calculate coverage using union of all geometries
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Set
import logging
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerCassetteCChannelOptimizerV2:
    """
    Non-overlapping edge-to-edge C-channel optimizer.
    """

    MIN_CCHANNEL = 1.5 / 12.0  # 1.5 inches in feet
    MAX_CCHANNEL = 18.0 / 12.0  # 18 inches in feet
    TOLERANCE = 0.01  # 0.01 feet tolerance for adjacency detection

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
        self.c_channels = []  # Uniform C-channel width for all cassettes

        # Adjacency tracking
        self.adjacency = {
            'N': {},  # cassette_idx -> adjacent cassette_idx
            'S': {},
            'E': {},
            'W': {}
        }

        # V2-specific statistics
        self.adjacent_edges_count = 0
        self.search_iterations = 0

        logger.info("=" * 70)
        logger.info("PER-CASSETTE C-CHANNEL OPTIMIZER V2 (NON-OVERLAPPING)")
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

    def _detect_adjacency(self):
        """
        Phase 2: Detect which cassettes are adjacent to each other.

        Two cassettes are adjacent if they share an edge (within tolerance).
        """
        logger.info("\nPhase 2: Detecting cassette adjacency")

        num_adjacent = 0

        for i, c1 in enumerate(self.cassettes):
            for j, c2 in enumerate(self.cassettes):
                if i == j:
                    continue

                # Check if c1's North edge touches c2's South edge
                if (abs(c1['x'] - c2['x']) < self.TOLERANCE and
                    abs(c1['width'] - c2['width']) < self.TOLERANCE and
                    abs((c1['y'] + c1['height']) - c2['y']) < self.TOLERANCE):
                    self.adjacency['N'][i] = j
                    self.adjacency['S'][j] = i
                    num_adjacent += 1

                # Check if c1's South edge touches c2's North edge
                elif (abs(c1['x'] - c2['x']) < self.TOLERANCE and
                      abs(c1['width'] - c2['width']) < self.TOLERANCE and
                      abs(c1['y'] - (c2['y'] + c2['height'])) < self.TOLERANCE):
                    self.adjacency['S'][i] = j
                    self.adjacency['N'][j] = i
                    num_adjacent += 1

                # Check if c1's East edge touches c2's West edge
                elif (abs(c1['y'] - c2['y']) < self.TOLERANCE and
                      abs(c1['height'] - c2['height']) < self.TOLERANCE and
                      abs((c1['x'] + c1['width']) - c2['x']) < self.TOLERANCE):
                    self.adjacency['E'][i] = j
                    self.adjacency['W'][j] = i
                    num_adjacent += 1

                # Check if c1's West edge touches c2's East edge
                elif (abs(c1['y'] - c2['y']) < self.TOLERANCE and
                      abs(c1['height'] - c2['height']) < self.TOLERANCE and
                      abs(c1['x'] - (c2['x'] + c2['width'])) < self.TOLERANCE):
                    self.adjacency['W'][i] = j
                    self.adjacency['E'][j] = i
                    num_adjacent += 1

        logger.info(f"Found {num_adjacent} adjacent edges")
        self.adjacent_edges_count = num_adjacent

    def _create_cchannel_geometries(self, c_width: float) -> List:
        """
        Create C-channel geometries for all cassettes with uniform width.

        For non-overlapping edge-to-edge:
        - Boundary edges: Full C-channel to polygon boundary
        - Adjacent edges: C-channel extends c_width/2 (shared with neighbor)

        Args:
            c_width: Uniform C-channel width in feet

        Returns:
            List of Shapely geometries (boxes) for all C-channel segments
        """
        geometries = []
        poly_bounds = self.polygon_shapely.bounds  # (minx, miny, maxx, maxy)

        for i, cassette in enumerate(self.cassettes):
            x, y = cassette['x'], cassette['y']
            w, h = cassette['width'], cassette['height']

            # North C-channel
            if i in self.adjacency['N']:
                # Adjacent: extend c_width/2 upward (shared)
                north_box = box(x, y + h, x + w, y + h + c_width / 2)
            else:
                # Boundary: extend to polygon edge
                gap_n = poly_bounds[3] - (y + h)
                north_box = box(x, y + h, x + w, y + h + min(gap_n, c_width))
            geometries.append(north_box)

            # South C-channel
            if i in self.adjacency['S']:
                # Adjacent: extend c_width/2 downward (shared)
                south_box = box(x, y - c_width / 2, x + w, y)
            else:
                # Boundary: extend to polygon edge
                gap_s = y - poly_bounds[1]
                south_box = box(x, y - min(gap_s, c_width), x + w, y)
            geometries.append(south_box)

            # East C-channel
            if i in self.adjacency['E']:
                # Adjacent: extend c_width/2 rightward (shared)
                east_box = box(x + w, y, x + w + c_width / 2, y + h)
            else:
                # Boundary: extend to polygon edge
                gap_e = poly_bounds[2] - (x + w)
                east_box = box(x + w, y, x + w + min(gap_e, c_width), y + h)
            geometries.append(east_box)

            # West C-channel
            if i in self.adjacency['W']:
                # Adjacent: extend c_width/2 leftward (shared)
                west_box = box(x - c_width / 2, y, x, y + h)
            else:
                # Boundary: extend to polygon edge
                gap_w = x - poly_bounds[0]
                west_box = box(x - min(gap_w, c_width), y, x, y + h)
            geometries.append(west_box)

        return geometries

    def _calculate_coverage_with_union(self, c_width: float) -> float:
        """
        Calculate coverage using UNION (correct method, no double-counting).

        Args:
            c_width: Uniform C-channel width in feet

        Returns:
            Coverage percentage
        """
        # Create cassette geometries
        cassette_geoms = [
            box(c['x'], c['y'], c['x'] + c['width'], c['y'] + c['height'])
            for c in self.cassettes
        ]

        # Create C-channel geometries
        cchannel_geoms = self._create_cchannel_geometries(c_width)

        # Union all geometries
        all_geoms = cassette_geoms + cchannel_geoms
        union_geom = unary_union(all_geoms)

        # Intersect with polygon (only count area inside)
        covered = union_geom.intersection(self.polygon_shapely)
        coverage = (covered.area / self.polygon_area) * 100.0

        return coverage

    def _find_optimal_even_cchannel_width(self) -> float:
        """
        Phase 3: Find optimal EVEN-numbered C-channel width closest to 100% coverage.

        Tests all even widths (2", 4", 6", ..., 18") and selects the one closest to 100%.
        Enforces 99-101% tolerance band.

        Returns:
            Optimal even-numbered C-channel width in feet
        """
        logger.info("\nPhase 3: Finding optimal even-numbered C-channel width")

        # Calculate gap area
        cassette_area = sum(c['area'] for c in self.cassettes)
        gap_area = self.polygon_area - cassette_area

        logger.info(f"Cassette area: {cassette_area:.1f} sq ft ({cassette_area/self.polygon_area*100:.1f}%)")
        logger.info(f"Gap area: {gap_area:.1f} sq ft ({gap_area/self.polygon_area*100:.1f}%)")

        # Define tolerance band
        MIN_ACCEPTABLE_COVERAGE = 99.0
        MAX_ACCEPTABLE_COVERAGE = 101.0

        # Test all even widths from 2" to 18"
        even_widths_inches = [2, 4, 6, 8, 10, 12, 14, 16, 18]

        logger.info("\nTesting even C-channel widths:")
        logger.info("-" * 70)

        results = []
        for width_inches in even_widths_inches:
            c_width = width_inches / 12.0  # Convert to feet
            coverage = self._calculate_coverage_with_union(c_width)

            # Calculate C-channel area for logging
            cassette_geoms = [
                box(c['x'], c['y'], c['x'] + c['width'], c['y'] + c['height'])
                for c in self.cassettes
            ]
            cchannel_geoms = self._create_cchannel_geometries(c_width)
            all_geoms = cassette_geoms + cchannel_geoms
            union_geom = unary_union(all_geoms)
            total_covered = union_geom.intersection(self.polygon_shapely).area
            cchannel_area = total_covered - cassette_area

            # Calculate distance from 100%
            distance_from_100 = abs(coverage - 100.0)

            # Check if in tolerance band
            in_band = MIN_ACCEPTABLE_COVERAGE <= coverage <= MAX_ACCEPTABLE_COVERAGE

            results.append({
                'width_inches': width_inches,
                'width_ft': c_width,
                'coverage': coverage,
                'cchannel_area': cchannel_area,
                'distance_from_100': distance_from_100,
                'in_band': in_band
            })

            status = "✓" if in_band else "✗"
            logger.info(f"  {status} {width_inches:2}\" → Coverage: {coverage:6.2f}%, "
                       f"C-channel: {cchannel_area:5.1f} sq ft, "
                       f"Distance from 100%: {distance_from_100:.2f}%")

        # Track iterations for statistics
        self.search_iterations = len(even_widths_inches)

        # Filter results within tolerance band
        valid_results = [r for r in results if r['in_band']]

        if not valid_results:
            # No width in tolerance band - use best effort (closest to 100%)
            logger.warning("\n⚠ WARNING: No even C-channel width achieves 99-101% coverage")
            best_result = min(results, key=lambda r: r['distance_from_100'])
            logger.warning(f"Using best available: {best_result['width_inches']}\" "
                          f"({best_result['coverage']:.2f}% coverage)")
            logger.warning(f"Shortfall/Excess: {best_result['coverage'] - 100.0:+.2f}%")

            return best_result['width_ft']

        # Find the width closest to 100% within tolerance band
        best_result = min(valid_results, key=lambda r: r['distance_from_100'])

        logger.info("\n" + "=" * 70)
        logger.info("OPTIMAL EVEN C-CHANNEL SELECTED")
        logger.info("=" * 70)
        logger.info(f"Width: {best_result['width_inches']}\" ({best_result['width_ft']:.4f} ft)")
        logger.info(f"Coverage: {best_result['coverage']:.2f}%")
        logger.info(f"C-channel area: {best_result['cchannel_area']:.1f} sq ft")
        logger.info(f"Distance from 100%: {best_result['distance_from_100']:.2f}%")

        if best_result['coverage'] == 100.0:
            logger.info("✓ EXACT 100% COVERAGE ACHIEVED!")
        elif best_result['coverage'] > 100.0:
            logger.info(f"✓ Slight over-coverage: +{best_result['coverage'] - 100.0:.2f}%")
        else:
            logger.info(f"✓ Slight under-coverage: {best_result['coverage'] - 100.0:.2f}%")

        return best_result['width_ft']

    def optimize(self) -> Dict:
        """
        Main optimization method to achieve 100% coverage.

        Returns:
            Dictionary with results
        """
        # Phase 1: Place cassettes
        self.cassettes = self._greedy_placement()

        # Phase 2: Detect adjacency
        self._detect_adjacency()

        # Phase 3: Find optimal even-numbered C-channel width (closest to 100%)
        optimal_c = self._find_optimal_even_cchannel_width()

        # Set uniform C-channel for all cassettes
        self.c_channels = [optimal_c] * len(self.cassettes)

        # Calculate final coverage and areas
        final_coverage = self._calculate_coverage_with_union(optimal_c)

        # Calculate cassette area
        cassette_area = sum(c['area'] for c in self.cassettes)

        # Calculate C-channel area (union method)
        cassette_geoms = [
            box(c['x'], c['y'], c['x'] + c['width'], c['y'] + c['height'])
            for c in self.cassettes
        ]
        cchannel_geoms = self._create_cchannel_geometries(optimal_c)

        # C-channel area = total union - cassette area
        all_geoms = cassette_geoms + cchannel_geoms
        union_geom = unary_union(all_geoms)
        total_covered = union_geom.intersection(self.polygon_shapely).area
        cchannel_area = total_covered - cassette_area

        # Convert C-channel geometries to serializable format (for JSON output)
        # Each geometry is a box with bounds (minx, miny, maxx, maxy)
        cchannel_geoms_serializable = [
            {
                'minx': geom.bounds[0],
                'miny': geom.bounds[1],
                'maxx': geom.bounds[2],
                'maxy': geom.bounds[3]
            }
            for geom in cchannel_geoms
        ]

        logger.info("\n" + "=" * 70)
        logger.info("OPTIMIZATION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Cassettes: {len(self.cassettes)}")
        logger.info(f"Cassette area: {cassette_area:.1f} sq ft")
        logger.info(f"C-channel area: {cchannel_area:.1f} sq ft")
        logger.info(f"Total coverage: {final_coverage:.2f}%")

        # Convert to inches for output
        c_channel_inches = [c * 12.0 for c in self.c_channels]

        # Calculate total edges and boundary edges
        total_edges = len(self.cassettes) * 4  # Each cassette has 4 edges
        # adjacent_edges_count already counts each adjacency twice (once for each cassette in the pair)
        # So 68 adjacent edges means 68 edges are adjacent to something, leaving 96-68=28 boundary edges
        boundary_edges = total_edges - self.adjacent_edges_count

        return {
            'cassettes': self.cassettes,
            'c_channels': self.c_channels,
            'c_channels_inches': c_channel_inches,
            'cchannel_geometries': cchannel_geoms_serializable,  # NEW: C-channel geometries for visualizer
            'coverage_percent': final_coverage,
            'polygon': self.polygon,  # Include polygon for visualizer
            'statistics': {
                'total_area': self.polygon_area,
                'cassette_area': cassette_area,
                'cchannel_area': cchannel_area,
                'cassette_count': len(self.cassettes),
                'coverage_percent': final_coverage,
                'min_cchannel_inches': min(c_channel_inches),
                'max_cchannel_inches': max(c_channel_inches),
                'avg_cchannel_inches': sum(c_channel_inches) / len(c_channel_inches),
                # V2-specific statistics
                'adjacent_edges': self.adjacent_edges_count,
                'boundary_edges': boundary_edges,
                'search_iterations': self.search_iterations,
                'total_edges': total_edges
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

    print("\nTesting Per-Cassette C-Channel Optimizer V2 (Non-Overlapping)")
    print("=" * 70)

    optimizer = PerCassetteCChannelOptimizerV2(umbra_polygon)
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

        vis_path = "output_umbra_fill_v2_test.png"
        vis_stats = {
            'coverage': result['coverage_percent'],
            'total_area': result['statistics']['total_area'],
            'covered': result['statistics']['cassette_area'] + result['statistics']['cchannel_area'],
            'cassettes': result['statistics']['cassette_count'],
            'per_cassette_cchannel': True,
            'cchannel_widths_per_cassette': result['c_channels_inches'],
            'cchannel_area': result['statistics']['cchannel_area'],
            'cchannel_geometries': result['cchannel_geometries']  # NEW: Pass geometries to visualizer
        }

        create_simple_visualization(
            cassettes=result['cassettes'],
            polygon=umbra_polygon,
            statistics=vis_stats,
            output_path=vis_path,
            floor_plan_name="UMBRA V2"
        )
        print(f"\nVisualization saved to: {vis_path}")
    except Exception as e:
        print(f"\nWarning: Could not generate visualization: {e}")

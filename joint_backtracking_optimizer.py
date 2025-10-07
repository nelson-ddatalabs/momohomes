#!/usr/bin/env python3
"""
Joint Backtracking Optimizer - Optimizes Both Cassettes AND C-Channel Widths
=============================================================================
This optimizer jointly optimizes:
1. Cassette placements (size, orientation, position)
2. C-channel widths (1.5" to 18" per cardinal direction)

Uses smart sampling of C-channel configurations + backtracking for cassettes.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import logging
import time
import math
from cchannel_utils import (
    create_inset_polygon,
    calculate_cchannel_areas
)
from backtracking_optimizer import BacktrackingOptimizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JointBacktrackingOptimizer:
    """
    Joint optimizer for cassettes and C-channel widths.

    Explores different C-channel width configurations (1.5" to 18"),
    and for each configuration, runs backtracking to optimize cassette placement.
    """

    MIN_CCHANNEL_WIDTH = 1.5 / 12.0  # 1.5 inches in feet
    MAX_CCHANNEL_WIDTH = 18.0 / 12.0  # 18 inches in feet

    def __init__(self, polygon: List[Tuple[float, float]]):
        """
        Initialize joint optimizer.

        Args:
            polygon: List of (x, y) coordinates defining the boundary
        """
        self.original_polygon = polygon
        self.polygon_area = self._calculate_polygon_area(polygon)

        # Best solution tracking
        self.best_solution = None
        self.best_coverage = 0.0

        # Statistics
        self.configs_explored = 0
        self.configs_pruned = 0

        logger.info("=" * 70)
        logger.info("JOINT BACKTRACKING OPTIMIZER")
        logger.info("=" * 70)
        logger.info(f"Original polygon area: {self.polygon_area:.1f} sq ft")
        logger.info(f"C-channel width range: {self.MIN_CCHANNEL_WIDTH*12:.1f}\" to {self.MAX_CCHANNEL_WIDTH*12:.1f}\"")

    def _calculate_polygon_area(self, polygon: List[Tuple[float, float]]) -> float:
        """Calculate polygon area using shoelace formula."""
        n = len(polygon)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += polygon[i][0] * polygon[j][1]
            area -= polygon[j][0] * polygon[i][1]
        return abs(area) / 2.0

    def _generate_cchannel_candidates(self) -> List[Dict]:
        """
        Generate smart C-channel width candidates using uniform inset strategy.

        Note: create_inset_polygon only supports uniform offsets, so we explore
        different uniform C-channel widths.

        Returns:
            List of candidate configurations with priority ordering
        """
        candidates = []

        logger.info("\nGenerating C-channel width candidates...")

        # Strategy: Uniform widths (all sides equal)
        # Sample at key widths: 1.5", 2", 3", 4", 6", 8", 10", 12", 15", 18"
        uniform_widths_inches = [1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 18.0]

        for width_inches in uniform_widths_inches:
            width_ft = width_inches / 12.0
            candidates.append({
                'width': width_ft,  # Uniform width for all sides
                'priority': 0 if width_inches == 1.5 else 1,
                'name': f'Uniform {width_inches:.1f}" C-channel'
            })

        # Sort by priority (minimum first)
        candidates = sorted(candidates, key=lambda x: x['priority'])

        logger.info(f"Generated {len(candidates)} C-channel candidates")
        return candidates


    def _estimate_upper_bound(self, config: Dict) -> float:
        """
        Estimate upper bound on coverage for a C-channel configuration.
        Used for pruning without running full optimization.

        Args:
            config: C-channel configuration dict with uniform 'width'

        Returns:
            Estimated maximum coverage percentage
        """
        width = config['width']

        # Calculate C-channel area (uniform width for all sides)
        try:
            widths_dict = {'N': width, 'S': width, 'E': width, 'W': width}
            cchannel_areas = calculate_cchannel_areas(self.original_polygon, widths_dict)
            cchannel_area = cchannel_areas['total']
        except Exception as e:
            # If C-channel calculation fails, this config is invalid
            return 0.0

        # Calculate inset polygon area
        try:
            inset_polygon = create_inset_polygon(self.original_polygon, width)
            inset_area = self._calculate_polygon_area(inset_polygon)
        except Exception as e:
            # If inset creation fails, config is invalid
            return 0.0

        # Check if inset is too small
        if inset_area < 4.0:  # Smaller than smallest cassette (2x2)
            return (cchannel_area / self.polygon_area) * 100.0

        # Optimistic upper bound: assume we can fill 100% of inset with cassettes
        cassette_area_upper = inset_area

        # Total upper bound
        total_area_upper = cassette_area_upper + cchannel_area
        coverage_upper = (total_area_upper / self.polygon_area) * 100.0

        # Can't exceed 100%
        return min(coverage_upper, 100.0)

    def _is_config_viable(self, config: Dict) -> bool:
        """
        Quick viability check for C-channel configuration.

        Args:
            config: C-channel configuration dict with uniform 'width'

        Returns:
            True if configuration is viable
        """
        width = config['width']
        width_inches = width * 12.0

        # Check width is in valid range
        if width_inches < 1.5 or width_inches > 18.0:
            logger.debug(f"  Config invalid: width {width_inches:.1f}\" out of range")
            return False

        # Try to create inset polygon
        try:
            inset_polygon = create_inset_polygon(self.original_polygon, width)
            inset_area = self._calculate_polygon_area(inset_polygon)

            # Check if inset has enough area for at least one cassette
            if inset_area < 4.0:  # 2x2 minimum
                logger.debug(f"  Config invalid: inset area {inset_area:.1f} too small")
                return False

        except Exception as e:
            logger.debug(f"  Config invalid: {e}")
            return False

        return True

    def optimize(self, max_time: int = 120, max_depth: int = 10) -> Dict:
        """
        Main optimization: explore C-channel configurations and optimize cassettes.

        Args:
            max_time: Maximum time in seconds (default 120 = 2 minutes)
            max_depth: Maximum backtracking depth per configuration (default 10)

        Returns:
            Dictionary with best solution including cassettes and C-channel info
        """
        start_time = time.time()

        # Generate C-channel candidates
        candidates = self._generate_cchannel_candidates()

        logger.info("\n" + "=" * 70)
        logger.info("EXPLORING C-CHANNEL CONFIGURATIONS")
        logger.info("=" * 70)
        logger.info(f"Total candidates: {len(candidates)}")
        logger.info(f"Time budget: {max_time}s")
        logger.info(f"Max depth per config: {max_depth}")

        # Explore each configuration
        for i, config in enumerate(candidates):
            # Check time budget
            elapsed = time.time() - start_time
            if elapsed > max_time:
                logger.info(f"\nTime limit reached after {self.configs_explored} configurations")
                break

            remaining_time = max_time - elapsed
            logger.info(f"\n[{i+1}/{len(candidates)}] {config['name']}")
            logger.info(f"  Width: {config['width']*12:.1f}\" (uniform all sides)")

            # Viability check
            if not self._is_config_viable(config):
                logger.info("  ✗ Configuration not viable, skipping")
                self.configs_pruned += 1
                continue

            # Upper bound pruning
            upper_bound = self._estimate_upper_bound(config)
            logger.info(f"  Upper bound: {upper_bound:.1f}%")

            if upper_bound <= self.best_coverage:
                logger.info(f"  ✗ Pruned (upper bound {upper_bound:.1f}% ≤ best {self.best_coverage:.1f}%)")
                self.configs_pruned += 1
                continue

            # Create inset polygon
            try:
                inset_polygon = create_inset_polygon(self.original_polygon, config['width'])
            except Exception as e:
                logger.info(f"  ✗ Failed to create inset: {e}")
                self.configs_pruned += 1
                continue

            # Run backtracking on inset polygon
            logger.info(f"  Running backtracking optimizer on inset...")
            per_config_time_limit = min(remaining_time * 0.5, 30)  # Cap at 30s per config

            cassette_optimizer = BacktrackingOptimizer(inset_polygon)
            cassette_optimizer.time_limit = per_config_time_limit
            cassette_result = cassette_optimizer.optimize(max_depth=max_depth)

            # Calculate C-channel areas (convert uniform width to dict)
            widths_dict = {'N': config['width'], 'S': config['width'],
                          'E': config['width'], 'W': config['width']}
            cchannel_areas = calculate_cchannel_areas(self.original_polygon, widths_dict)
            cchannel_area = cchannel_areas['total']

            # Calculate total coverage
            cassette_area = cassette_result['covered_area']
            total_covered = cassette_area + cchannel_area
            total_coverage = (total_covered / self.polygon_area) * 100.0

            cassette_percent = (cassette_area / self.polygon_area) * 100.0
            cchannel_percent = (cchannel_area / self.polygon_area) * 100.0

            logger.info(f"  Cassettes: {cassette_result['num_cassettes']} units, "
                       f"{cassette_area:.1f} sq ft ({cassette_percent:.1f}%)")
            logger.info(f"  C-channel: {cchannel_area:.1f} sq ft ({cchannel_percent:.1f}%)")
            logger.info(f"  Total coverage: {total_coverage:.1f}%")

            self.configs_explored += 1

            # Update best solution
            if total_coverage > self.best_coverage:
                logger.info(f"  ✓ NEW BEST SOLUTION! (previous: {self.best_coverage:.1f}%)")
                self.best_coverage = total_coverage
                self.best_solution = {
                    'cassettes': cassette_result['cassettes'],
                    'cchannel_widths': widths_dict,  # Store as dict for compatibility
                    'cchannel_areas': cchannel_areas,
                    'inset_polygon': inset_polygon,
                    'original_polygon': self.original_polygon,
                    'cassette_area': cassette_area,
                    'cchannel_area': cchannel_area,
                    'total_coverage': total_coverage,
                    'cassette_percent': cassette_percent,
                    'cchannel_percent': cchannel_percent,
                    'cassette_search_stats': cassette_result['search_stats'],
                    'config_name': config['name']
                }

            # Early termination if excellent solution found
            if total_coverage >= 99.5:
                logger.info(f"\n✓ Excellent solution found (≥99.5%), terminating early")
                break

        # Final statistics
        elapsed = time.time() - start_time

        logger.info("\n" + "=" * 70)
        logger.info("JOINT OPTIMIZATION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Total time: {elapsed:.1f}s")
        logger.info(f"Configurations explored: {self.configs_explored}")
        logger.info(f"Configurations pruned: {self.configs_pruned}")
        logger.info(f"Pruning efficiency: {(self.configs_pruned/len(candidates)*100):.1f}%")

        if self.best_solution:
            logger.info(f"\nBest solution: {self.best_solution['config_name']}")
            logger.info(f"  Total coverage: {self.best_solution['total_coverage']:.1f}%")
            logger.info(f"  Cassettes: {len(self.best_solution['cassettes'])} units "
                       f"({self.best_solution['cassette_percent']:.1f}%)")
            logger.info(f"  C-channel: {self.best_solution['cchannel_percent']:.1f}%")
            logger.info(f"  C-channel widths:")
            for direction in ['N', 'S', 'E', 'W']:
                width_inches = self.best_solution['cchannel_widths'][direction] * 12.0
                logger.info(f"    {direction}: {width_inches:.1f}\"")

            # Format return structure
            return self._format_results(elapsed)
        else:
            logger.error("No valid solution found!")
            return None

    def _format_results(self, elapsed_time: float) -> Dict:
        """Format results to match expected output structure."""
        sol = self.best_solution

        # Calculate size distribution
        size_counts = {}
        for cassette in sol['cassettes']:
            size = cassette['size']
            size_counts[size] = size_counts.get(size, 0) + 1

        # Calculate total weight
        total_weight = sum(c['weight'] for c in sol['cassettes'])

        return {
            'cassettes': sol['cassettes'],
            'cchannel_widths': sol['cchannel_widths'],
            'cchannel_areas': sol['cchannel_areas'],
            'original_polygon': sol['original_polygon'],
            'inset_polygon': sol['inset_polygon'],
            'statistics': {
                'total_area': self.polygon_area,
                'cassette_area': sol['cassette_area'],
                'cassette_percent': sol['cassette_percent'],
                'cchannel_area': sol['cchannel_area'],
                'cchannel_percent': sol['cchannel_percent'],
                'coverage_percent': sol['total_coverage'],
                'cassette_count': len(sol['cassettes']),
                'total_weight': total_weight,
                'cchannel_widths_inches': {
                    k: v * 12.0 for k, v in sol['cchannel_widths'].items()
                },
                'size_distribution': size_counts,
                'gap_area': self.polygon_area - sol['cassette_area'] - sol['cchannel_area']
            },
            'search_stats': {
                'total_time': elapsed_time,
                'configs_explored': self.configs_explored,
                'configs_pruned': self.configs_pruned,
                'cassette_nodes_explored': sol['cassette_search_stats']['nodes_explored'],
                'cassette_nodes_pruned': sol['cassette_search_stats']['nodes_pruned'],
                'best_config': sol['config_name']
            }
        }


if __name__ == "__main__":
    # Test on umbra polygon
    umbra_polygon = [
        (0.0, 28.0),
        (55.5, 28.0),
        (55.5, 12.0),
        (16.0, 12.0),
        (16.0, 0.0),
        (0.0, 0.0)
    ]

    print("\nTesting Joint Backtracking Optimizer on Umbra polygon")
    print("=" * 70)

    optimizer = JointBacktrackingOptimizer(umbra_polygon)
    result = optimizer.optimize(max_time=120, max_depth=10)

    if result:
        print("\n" + "=" * 70)
        print("TEST COMPLETE - FINAL RESULTS")
        print("=" * 70)
        print(f"Coverage: {result['statistics']['coverage_percent']:.1f}%")
        print(f"Cassettes: {result['statistics']['cassette_count']} units "
              f"({result['statistics']['cassette_percent']:.1f}%)")
        print(f"C-channel: {result['statistics']['cchannel_percent']:.1f}%")

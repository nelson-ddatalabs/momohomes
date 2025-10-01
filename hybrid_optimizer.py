#!/usr/bin/env python3
"""
Hybrid Optimizer
================
Extends polygon cassette optimizer with edge filling and gap detection.
Uses grid placement for main areas and tight packing for edges.
"""

import numpy as np
from typing import List, Tuple, Dict
import logging
from polygon_cassette_optimizer import PolygonCassetteOptimizer, Cassette

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridOptimizer(PolygonCassetteOptimizer):
    """Hybrid optimization with grid and edge filling strategies."""

    def __init__(self, polygon: List[Tuple[float, float]], grid_resolution: float = 1.0):
        """Initialize hybrid optimizer."""
        super().__init__(polygon, grid_resolution)
        self.gaps = []

    def optimize_hybrid(self) -> Dict:
        """
        Run hybrid optimization with multiple phases.

        Returns:
            Dictionary with optimization results
        """
        logger.info("Starting hybrid optimization")

        # Phase 1: Grid placement with large cassettes
        logger.info("Phase 1: Grid placement with large cassettes")
        self._phase_grid_placement()

        # Phase 2: Edge filling with smaller cassettes
        logger.info("Phase 2: Edge filling with smaller cassettes")
        self._phase_edge_filling()

        # Phase 3: Gap filling for spaces > 3x3 feet
        logger.info("Phase 3: Gap filling")
        self._phase_gap_filling()

        # Calculate final statistics
        results = self._calculate_statistics()

        logger.info(f"Hybrid optimization complete: {len(self.placed_cassettes)} cassettes")
        logger.info(f"Coverage: {results['coverage_percent']:.1f}%")

        return results

    def _phase_grid_placement(self):
        """Phase 1: Place large cassettes in grid pattern."""
        # Use only larger cassettes for grid placement
        large_sizes = [
            (6, 8, "6x8"),  # 48 sq ft
            (5, 8, "5x8"),  # 40 sq ft
            (6, 6, "6x6"),  # 36 sq ft
            (4, 8, "4x8"),  # 32 sq ft
        ]

        for width, height, size_name in large_sizes:
            # Try both orientations
            for w, h in [(width, height), (height, width)]:
                self._place_cassettes_of_size(w, h, size_name if w == width else f"{height}x{width}")

    def _phase_edge_filling(self):
        """Phase 2: Fill edges with smaller cassettes."""
        # Use smaller cassettes for edges
        edge_sizes = [
            (5, 6, "5x6"),  # 30 sq ft
            (4, 6, "4x6"),  # 24 sq ft
            (4, 4, "4x4"),  # 16 sq ft
            (3, 4, "3x4"),  # 12 sq ft
        ]

        # Focus on boundary regions
        for width, height, size_name in edge_sizes:
            # Try both orientations
            for w, h in [(width, height), (height, width)]:
                self._place_edge_cassettes(w, h, size_name if w == width else f"{height}x{width}")

    def _place_edge_cassettes(self, width: float, height: float, size_name: str):
        """
        Place cassettes focusing on edge regions.

        Args:
            width: Cassette width in feet
            height: Cassette height in feet
            size_name: Size identifier
        """
        cells_w = int(width / self.grid_resolution)
        cells_h = int(height / self.grid_resolution)

        # Prioritize positions near polygon edges
        edge_positions = self._get_edge_positions(cells_w, cells_h)

        for grid_x, grid_y in edge_positions:
            if self._is_space_available(grid_x, grid_y, cells_w, cells_h):
                x = self.min_x + grid_x * self.grid_resolution
                y = self.min_y + grid_y * self.grid_resolution

                cassette = Cassette(x, y, width, height, size_name)

                if self._is_valid_placement(cassette):
                    self._place_cassette(cassette, grid_x, grid_y, cells_w, cells_h)

    def _get_edge_positions(self, cells_w: int, cells_h: int) -> List[Tuple[int, int]]:
        """Get positions near polygon edges, sorted by distance to edge."""
        positions = []

        for grid_y in range(self.grid_height - cells_h + 1):
            for grid_x in range(self.grid_width - cells_w + 1):
                # Calculate distance to nearest polygon edge
                x = self.min_x + (grid_x + cells_w/2) * self.grid_resolution
                y = self.min_y + (grid_y + cells_h/2) * self.grid_resolution
                dist = self._distance_to_polygon_edge(x, y)
                positions.append((grid_x, grid_y, dist))

        # Sort by distance (closest to edge first)
        positions.sort(key=lambda p: p[2])

        return [(p[0], p[1]) for p in positions]

    def _distance_to_polygon_edge(self, x: float, y: float) -> float:
        """Calculate minimum distance from point to polygon edge."""
        min_dist = float('inf')

        for i in range(len(self.polygon)):
            p1 = self.polygon[i]
            p2 = self.polygon[(i + 1) % len(self.polygon)]

            # Distance to line segment
            dist = self._point_to_segment_distance(x, y, p1, p2)
            min_dist = min(min_dist, dist)

        return min_dist

    def _point_to_segment_distance(self, px: float, py: float,
                                  p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Calculate distance from point to line segment."""
        x1, y1 = p1
        x2, y2 = p2

        # Vector from p1 to p2
        dx = x2 - x1
        dy = y2 - y1

        # Vector from p1 to point
        dpx = px - x1
        dpy = py - y1

        # Dot product
        dot = dpx * dx + dpy * dy
        len_sq = dx * dx + dy * dy

        if len_sq == 0:
            # p1 == p2
            return np.sqrt(dpx * dpx + dpy * dpy)

        # Parameter along segment
        t = max(0, min(1, dot / len_sq))

        # Closest point on segment
        cx = x1 + t * dx
        cy = y1 + t * dy

        # Distance to closest point
        return np.sqrt((px - cx)**2 + (py - cy)**2)

    def _phase_gap_filling(self):
        """Phase 3: Fill gaps larger than 3x3 feet."""
        self.gaps = self._find_gaps()

        if not self.gaps:
            logger.info("No significant gaps found")
            return

        logger.info(f"Found {len(self.gaps)} gaps to fill")

        # Try to fill gaps with smallest cassettes
        small_sizes = [
            (3, 4, "3x4"),  # 12 sq ft
            (4, 4, "4x4"),  # 16 sq ft
        ]

        for gap in self.gaps:
            self._fill_gap(gap, small_sizes)

    def _find_gaps(self) -> List[Dict]:
        """Find gaps larger than 3x3 feet."""
        gaps = []
        min_gap_size = 3  # 3x3 feet minimum

        # Scan for rectangular gaps
        for grid_y in range(self.grid_height - min_gap_size + 1):
            for grid_x in range(self.grid_width - min_gap_size + 1):
                # Check if this could be a gap
                if not self.coverage_grid[grid_y:grid_y+min_gap_size,
                                         grid_x:grid_x+min_gap_size].any():
                    # Find maximum gap size from this position
                    max_w = min_gap_size
                    max_h = min_gap_size

                    # Expand width
                    while (grid_x + max_w < self.grid_width and
                           not self.coverage_grid[grid_y:grid_y+min_gap_size,
                                                grid_x+max_w].any()):
                        max_w += 1

                    # Expand height
                    while (grid_y + max_h < self.grid_height and
                           not self.coverage_grid[grid_y+max_h,
                                                grid_x:grid_x+max_w].any()):
                        max_h += 1

                    # Convert to world coordinates
                    x = self.min_x + grid_x * self.grid_resolution
                    y = self.min_y + grid_y * self.grid_resolution
                    width = max_w * self.grid_resolution
                    height = max_h * self.grid_resolution

                    # Check if center is inside polygon
                    cx = x + width / 2
                    cy = y + height / 2

                    if self.reconstructor.point_in_polygon(cx, cy):
                        gaps.append({
                            'x': x,
                            'y': y,
                            'width': width,
                            'height': height,
                            'grid_x': grid_x,
                            'grid_y': grid_y,
                            'grid_w': max_w,
                            'grid_h': max_h,
                            'area': width * height
                        })

        # Remove overlapping gaps
        gaps = self._remove_overlapping_gaps(gaps)

        return gaps

    def _remove_overlapping_gaps(self, gaps: List[Dict]) -> List[Dict]:
        """Remove overlapping gaps, keeping larger ones."""
        # Sort by area (largest first)
        gaps.sort(key=lambda g: g['area'], reverse=True)

        non_overlapping = []
        for gap in gaps:
            overlaps = False
            for existing in non_overlapping:
                if (gap['x'] < existing['x'] + existing['width'] and
                    gap['x'] + gap['width'] > existing['x'] and
                    gap['y'] < existing['y'] + existing['height'] and
                    gap['y'] + gap['height'] > existing['y']):
                    overlaps = True
                    break

            if not overlaps:
                non_overlapping.append(gap)

        return non_overlapping

    def _fill_gap(self, gap: Dict, sizes: List[Tuple[float, float, str]]):
        """Try to fill a gap with available cassette sizes."""
        gap_filled = False

        for width, height, size_name in sizes:
            # Try both orientations
            for w, h in [(width, height), (height, width)]:
                if w <= gap['width'] and h <= gap['height']:
                    # Try to place cassette in gap
                    cassette = Cassette(gap['x'], gap['y'], w, h,
                                      size_name if w == width else f"{height}x{width}")

                    if self._is_valid_placement(cassette):
                        # Calculate grid position
                        grid_x = int((cassette.x - self.min_x) / self.grid_resolution)
                        grid_y = int((cassette.y - self.min_y) / self.grid_resolution)
                        cells_w = int(w / self.grid_resolution)
                        cells_h = int(h / self.grid_resolution)

                        self._place_cassette(cassette, grid_x, grid_y, cells_w, cells_h)
                        gap_filled = True
                        break

            if gap_filled:
                break
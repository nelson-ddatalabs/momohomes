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

        # Enhanced cassette sizes - sorted by area descending (critical for optimal packing)
        unsorted_sizes = [
            (8, 6, "8x6"),  # 48 sq ft
            (6, 8, "6x8"),  # 48 sq ft
            (8, 5, "8x5"),  # 40 sq ft
            (5, 8, "5x8"),  # 40 sq ft
            (6, 6, "6x6"),  # 36 sq ft
            (8, 4, "8x4"),  # 32 sq ft
            (4, 8, "4x8"),  # 32 sq ft
            (6, 5, "6x5"),  # 30 sq ft
            (5, 6, "5x6"),  # 30 sq ft
            (6, 4, "6x4"),  # 24 sq ft
            (4, 6, "4x6"),  # 24 sq ft
            (5, 4, "5x4"),  # 20 sq ft
            (4, 5, "4x5"),  # 20 sq ft
            (4, 4, "4x4"),  # 16 sq ft
            (4, 3, "4x3"),  # 12 sq ft
            (3, 4, "3x4"),  # 12 sq ft
            (3, 3, "3x3"),  # 9 sq ft
            (3, 2, "3x2"),  # 6 sq ft
            (2, 3, "2x3"),  # 6 sq ft
            (2, 2, "2x2"),  # 4 sq ft
        ]
        # Sort by area descending, then by max dimension descending
        self.cassette_sizes = sorted(
            unsorted_sizes,
            key=lambda x: (x[0] * x[1], max(x[0], x[1])),
            reverse=True
        )

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

    def _is_rect_fully_inside(self, x: float, y: float, width: float, height: float) -> bool:
        """Check if a rectangle is fully inside the polygon."""
        # Sample points within the rectangle to verify it's inside
        # Check corners and center
        test_points = [
            (x + 0.1, y + 0.1),
            (x + width - 0.1, y + 0.1),
            (x + width - 0.1, y + height - 0.1),
            (x + 0.1, y + height - 0.1),
            (x + width/2, y + height/2)
        ]

        for px, py in test_points:
            if not self._is_point_inside(px, py):
                return False
        return True

    def _find_free_rectangles(self) -> List[Tuple[float, float, float, float]]:
        """
        Find maximal free rectangular spaces that are inside the polygon.
        Returns list of (x, y, width, height) tuples.
        """
        free_rects = []
        visited = np.zeros_like(self.occupancy, dtype=bool)

        # Scan for free spaces and expand to maximal rectangles
        for gy in range(self.grid_height):
            for gx in range(self.grid_width):
                if self.occupancy[gy, gx] or visited[gy, gx]:
                    continue

                # Check if this grid cell is inside polygon
                x, y = self._grid_to_world(gx, gy)
                if not self._is_point_inside(x, y):
                    visited[gy, gx] = True
                    continue

                # Find maximal rectangle starting from this point
                # Expand right (but only while staying inside polygon)
                max_width = 0
                for gx2 in range(gx, self.grid_width):
                    if self.occupancy[gy, gx2]:
                        break
                    # Check if this position is inside polygon
                    test_x, test_y = self._grid_to_world(gx2, gy)
                    if not self._is_point_inside(test_x, test_y):
                        break
                    max_width = gx2 - gx + 1

                if max_width == 0:
                    visited[gy, gx] = True
                    continue

                # Expand down with same width (but only while staying inside polygon)
                max_height = 0
                for gy2 in range(gy, self.grid_height):
                    # Check if entire row is free AND inside polygon
                    can_expand = True
                    for gx2 in range(gx, gx + max_width):
                        if gx2 >= self.grid_width or self.occupancy[gy2, gx2]:
                            can_expand = False
                            break
                        test_x, test_y = self._grid_to_world(gx2, gy2)
                        if not self._is_point_inside(test_x, test_y):
                            can_expand = False
                            break
                    if not can_expand:
                        break
                    max_height = gy2 - gy + 1

                if max_width > 0 and max_height > 0:
                    # Convert to world coordinates
                    x, y = self._grid_to_world(gx, gy)
                    width = max_width * self.grid_resolution
                    height = max_height * self.grid_resolution

                    # Final validation: check if rectangle is fully inside
                    if self._is_rect_fully_inside(x, y, width, height):
                        free_rects.append((x, y, width, height))

                    # Mark as visited
                    visited[gy:gy+max_height, gx:gx+max_width] = True

        # Sort by area descending (fill larger gaps first)
        free_rects.sort(key=lambda r: r[2] * r[3], reverse=True)
        return free_rects

    def _score_cassette_placement(self, cassette: UltraSmartCassette,
                                  free_rect: Tuple[float, float, float, float]) -> float:
        """
        Score a cassette placement. Higher score = better fit.
        Considers: area utilization, corner alignment, minimal waste.
        """
        fx, fy, fw, fh = free_rect

        # Calculate area utilization
        free_area = fw * fh
        cassette_area = cassette.area
        utilization = cassette_area / free_area if free_area > 0 else 0

        # Bonus for bottom-left positioning (corner-searching strategy)
        corner_score = 0
        if abs(cassette.x - fx) < 0.1:  # Aligned with left edge
            corner_score += 0.1
        if abs(cassette.y - fy) < 0.1:  # Aligned with bottom edge
            corner_score += 0.1

        # Penalty for excessive waste
        waste = free_area - cassette_area
        waste_penalty = waste / free_area if free_area > 0 else 1.0

        return utilization + corner_score - (waste_penalty * 0.2)

    def _try_place_in_free_rect(self, free_rect: Tuple[float, float, float, float]) -> bool:
        """
        Try to place best-fitting cassette in a free rectangle.
        Uses fine-grained search with expanded offset range.
        Returns True if placement succeeded.
        """
        fx, fy, fw, fh = free_rect

        best_placement = None
        best_score = -1

        # Try all cassette sizes (smallest to largest for gap filling)
        for width, height, size_name in reversed(self.cassette_sizes):
            for w, h, name in [(width, height, size_name), (height, width, f"{height}x{width}")]:
                # Skip if cassette is larger than free rectangle
                if w > fw + 0.5 or h > fh + 0.5:
                    continue

                # Try positions with finer granularity (0.1 ft steps)
                # Expanded search: from -1.0 to +1.0 ft in 0.1 ft increments
                search_range = np.arange(-1.0, 1.1, 0.1)

                for dx in search_range:
                    for dy in search_range:
                        test_x = fx + dx
                        test_y = fy + dy

                        # Round to 0.1 ft precision
                        test_x = round(test_x * 10) / 10
                        test_y = round(test_y * 10) / 10

                        cassette = UltraSmartCassette(test_x, test_y, w, h, name)

                        if self._is_cassette_valid(cassette, allow_edge_touch=True):
                            score = self._score_cassette_placement(cassette, free_rect)
                            if score > best_score:
                                best_score = score
                                best_placement = cassette

        # Place the best cassette found
        if best_placement:
            self.placed_cassettes.append(best_placement)
            self._mark_occupied(best_placement)
            logger.info(f"      ✓ Placed {best_placement.size_name} at ({best_placement.x:.1f}, {best_placement.y:.1f}) "
                        f"with score {best_score:.2f}")
            return True

        logger.info(f"      ✗ No cassette fits in this rectangle")
        return False

    def _aggressive_point_search(self) -> int:
        """
        Aggressive point-by-point search for irregular gaps.
        Uses fine-grained positioning and smaller cassettes.
        Returns number of cassettes placed.
        """
        placements = 0

        # Collect all unoccupied points inside polygon
        gap_points = []
        for gy in range(self.grid_height):
            for gx in range(self.grid_width):
                if not self.occupancy[gy, gx]:
                    x, y = self._grid_to_world(gx, gy)
                    if self._is_point_inside(x, y):
                        gap_points.append((x, y))

        if not gap_points:
            return 0

        logger.info(f"  Aggressive search: Found {len(gap_points)} gap points")

        # Try to fill each gap point
        attempts = 0
        valid_checks_failed = 0

        for gx, gy in gap_points[:50]:  # Limit to first 50 points to avoid excessive runtime
            # Try smallest cassettes first (better for irregular gaps)
            for width, height, size_name in reversed(self.cassette_sizes):
                placed = False

                for w, h, name in [(width, height, size_name), (height, width, f"{height}x{width}")]:
                    # Skip large cassettes for irregular gaps
                    if w * h > 12:  # Only try cassettes <= 12 sq ft
                        continue

                    # Fine-grained search around this point
                    # Try offsets from -2.0 to +2.0 ft in 0.25 ft steps
                    search_offsets = np.arange(-2.0, 2.25, 0.25)

                    best_cassette = None
                    best_coverage = 0

                    for dx in search_offsets:
                        for dy in search_offsets:
                            test_x = gx + dx
                            test_y = gy + dy

                            # Round to 0.25 ft precision
                            test_x = round(test_x * 4) / 4
                            test_y = round(test_y * 4) / 4

                            cassette = UltraSmartCassette(test_x, test_y, w, h, name)
                            attempts += 1

                            if self._is_cassette_valid(cassette, allow_edge_touch=True):
                                # Count how many currently empty grid cells it would cover
                                gx1, gy1 = self._world_to_grid(cassette.x, cassette.y)
                                gx2, gy2 = self._world_to_grid(cassette.x + cassette.width,
                                                               cassette.y + cassette.height)
                                gx1, gy1 = max(0, gx1), max(0, gy1)
                                gx2, gy2 = min(self.grid_width, gx2), min(self.grid_height, gy2)

                                coverage = np.sum(~self.occupancy[gy1:gy2, gx1:gx2])
                                if coverage > best_coverage:
                                    best_coverage = coverage
                                    best_cassette = cassette
                            else:
                                valid_checks_failed += 1

                    if best_cassette:
                        self.placed_cassettes.append(best_cassette)
                        self._mark_occupied(best_cassette)
                        logger.info(f"      ✓ Placed {best_cassette.size_name} at ({best_cassette.x:.1f}, {best_cassette.y:.1f})")
                        placements += 1
                        placed = True
                        break

                if placed:
                    break

        logger.info(f"  Search stats: {attempts} positions tried, {valid_checks_failed} failed validation")
        return placements

    def _fill_gaps(self):
        """
        Intelligently fill remaining gaps with appropriate cassettes.
        Enhanced with Maximal Rectangles algorithm and corner-searching strategy.
        """
        max_passes = 5
        pass_num = 0

        while pass_num < max_passes:
            # Find all maximal free rectangles
            free_rects = self._find_free_rectangles()

            if not free_rects:
                logger.info(f"  Pass {pass_num + 1}: No free rectangles found, trying aggressive search...")
                # Fall back to aggressive point-by-point search
                placements_made = self._aggressive_point_search()
                logger.info(f"  Pass {pass_num + 1}: Aggressive search placed {placements_made} cassettes")

                if placements_made == 0:
                    break
                else:
                    pass_num += 1
                    continue

            logger.info(f"  Pass {pass_num + 1}: Found {len(free_rects)} free rectangles")

            # Try to fill each free rectangle
            placements_made = 0
            for free_rect in free_rects:
                fx, fy, fw, fh = free_rect
                area = fw * fh

                # Skip very small rectangles (< 3 sq ft)
                if area < 3.0:
                    continue

                logger.info(f"    Trying free rect at ({fx:.1f}, {fy:.1f}) size {fw:.1f}x{fh:.1f} = {area:.1f} sq ft")

                if self._try_place_in_free_rect(free_rect):
                    placements_made += 1

            logger.info(f"  Pass {pass_num + 1}: Placed {placements_made} cassettes")

            # If no placements made, stop
            if placements_made == 0:
                break

            pass_num += 1

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
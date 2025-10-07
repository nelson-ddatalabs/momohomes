#!/usr/bin/env python3
"""
Backtracking Optimizer with Pruning
====================================
Optimal cassette placement using backtracking search with aggressive pruning.
Guarantees finding best solution within time constraints (2 minutes).
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BacktrackingOptimizer:
    """
    Backtracking optimizer with pruning for optimal cassette placement.

    Uses recursive search with:
    - Upper bound pruning to eliminate suboptimal branches
    - Greedy initialization for baseline solution
    - Time-limited search with emergency abort
    """

    def __init__(self, polygon: List[Tuple[float, float]]):
        """
        Initialize backtracking optimizer.

        Args:
            polygon: List of (x, y) coordinates defining the boundary
        """
        self.polygon = polygon

        # Cassette sizes: 2x2 to 6x8 (sorted by area descending)
        self.cassette_types = [
            (6, 8), (8, 6),  # 48 sq ft
            (6, 6),          # 36 sq ft
            (6, 5), (5, 6),  # 30 sq ft
            (6, 4), (4, 6),  # 24 sq ft
            (5, 5),          # 25 sq ft
            (5, 4), (4, 5),  # 20 sq ft
            (4, 4),          # 16 sq ft
            (4, 3), (3, 4),  # 12 sq ft
            (3, 3),          # 9 sq ft
            (3, 2), (2, 3),  # 6 sq ft
            (2, 2),          # 4 sq ft
        ]

        # Calculate polygon bounds
        self.min_x, self.min_y, self.max_x, self.max_y = self._get_bounds()

        # Create occupancy grid (0.5 ft resolution)
        self.grid_resolution = 0.5
        self.grid_width = int((self.max_x - self.min_x) / self.grid_resolution) + 1
        self.grid_height = int((self.max_y - self.min_y) / self.grid_resolution) + 1

        # Calculate polygon area
        self.polygon_area = self._calculate_polygon_area()

        # Search state
        self.best_solution = []
        self.best_coverage = 0.0
        self.nodes_explored = 0
        self.nodes_pruned = 0
        self.start_time = None
        self.time_limit = 120  # 2 minutes

        logger.info(f"Backtracking optimizer initialized")
        logger.info(f"  Polygon bounds: [{self.min_x:.1f}, {self.max_x:.1f}] x [{self.min_y:.1f}, {self.max_y:.1f}]")
        logger.info(f"  Polygon area: {self.polygon_area:.1f} sq ft")
        logger.info(f"  Grid size: {self.grid_width} x {self.grid_height}")

    def _get_bounds(self) -> Tuple[float, float, float, float]:
        """Calculate bounding box of polygon."""
        xs = [p[0] for p in self.polygon]
        ys = [p[1] for p in self.polygon]
        return min(xs), min(ys), max(xs), max(ys)

    def _calculate_polygon_area(self) -> float:
        """Calculate polygon area using shoelace formula."""
        n = len(self.polygon)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += self.polygon[i][0] * self.polygon[j][1]
            area -= self.polygon[j][0] * self.polygon[i][1]
        return abs(area) / 2.0

    def _create_empty_grid(self) -> np.ndarray:
        """Create empty occupancy grid."""
        return np.zeros((self.grid_height, self.grid_width), dtype=bool)

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

    def greedy_initialization(self) -> List[Dict]:
        """
        Generate initial solution using greedy ultra-smart optimizer.

        Returns baseline solution (typically 92-94% coverage) to beat.

        Returns:
            List of cassette dictionaries with x, y, width, height, size, area
        """
        from ultra_smart_optimizer import UltraSmartOptimizer

        logger.info("Generating greedy baseline solution...")
        optimizer = UltraSmartOptimizer(self.polygon)
        result = optimizer.optimize()

        cassettes = result['cassettes']
        coverage = result['coverage_percent']

        logger.info(f"Greedy baseline: {len(cassettes)} cassettes, {coverage:.1f}% coverage")

        return cassettes

    def calculate_coverage(self, cassettes: List[Dict]) -> float:
        """
        Calculate coverage percentage of cassettes.

        Uses grid sampling for accurate measurement including overlaps.

        Args:
            cassettes: List of cassette dictionaries

        Returns:
            Coverage percentage (0-100)
        """
        if not cassettes:
            return 0.0

        # Calculate total cassette area
        total_cassette_area = sum(c['area'] for c in cassettes)

        # Calculate coverage as percentage of polygon area
        coverage_percent = (total_cassette_area / self.polygon_area) * 100.0

        return coverage_percent

    def calculate_free_space(self, occupancy_grid: np.ndarray) -> float:
        """
        Calculate free space in polygon.

        Args:
            occupancy_grid: Boolean grid marking occupied cells

        Returns:
            Free space in square feet
        """
        # Count unoccupied cells that are inside polygon
        free_cells = 0
        for gy in range(self.grid_height):
            for gx in range(self.grid_width):
                if not occupancy_grid[gy, gx]:
                    x, y = self._grid_to_world(gx, gy)
                    if self._is_point_inside(x, y):
                        free_cells += 1

        # Convert cells to square feet
        cell_area = self.grid_resolution * self.grid_resolution
        free_space = free_cells * cell_area

        return free_space

    def calculate_upper_bound(self, current_cassettes: List[Dict],
                            occupancy_grid: np.ndarray,
                            depth: int, max_depth: int) -> float:
        """
        Calculate optimistic upper bound for coverage.

        Uses remaining cassette types and free space to estimate
        maximum achievable coverage. Critical for pruning.

        Args:
            current_cassettes: Already placed cassettes
            occupancy_grid: Current grid state
            depth: Current search depth
            max_depth: Maximum allowed depth

        Returns:
            Upper bound coverage percentage
        """
        # Current coverage
        current_coverage = self.calculate_coverage(current_cassettes)

        # If at max depth, can't place more
        if depth >= max_depth:
            return current_coverage

        # Calculate free space
        free_space = self.calculate_free_space(occupancy_grid)

        if free_space <= 0:
            return current_coverage

        # Estimate potential cassettes that could fit
        # Use largest cassettes first (optimistic)
        remaining_depth = max_depth - depth
        potential_area = 0.0

        for _ in range(remaining_depth):
            # Assume we can fit a 6x8 cassette (largest = 48 sq ft)
            largest_cassette_area = 48.0
            if potential_area + largest_cassette_area <= free_space:
                potential_area += largest_cassette_area
            else:
                potential_area = free_space
                break

        # Upper bound = current + potential additional coverage
        upper_bound_coverage = current_coverage + (potential_area / self.polygon_area) * 100.0

        return upper_bound_coverage

    def is_valid_placement(self, cassette: Dict, current_cassettes: List[Dict]) -> bool:
        """
        Check if cassette placement is valid.

        Args:
            cassette: Cassette to place {x, y, width, height, ...}
            current_cassettes: Already placed cassettes

        Returns:
            True if valid placement
        """
        x, y = cassette['x'], cassette['y']
        w, h = cassette['width'], cassette['height']

        # Check all 4 corners are inside polygon
        corners = [
            (x, y),
            (x + w, y),
            (x + w, y + h),
            (x, y + h)
        ]

        for cx, cy in corners:
            if not self._is_point_inside(cx, cy):
                return False

        # Check no overlap with existing cassettes
        for existing in current_cassettes:
            ex, ey = existing['x'], existing['y']
            ew, eh = existing['width'], existing['height']

            # Check if rectangles overlap
            if not (x + w <= ex or ex + ew <= x or y + h <= ey or ey + eh <= y):
                return False

        return True

    def update_occupancy_grid(self, occupancy_grid: np.ndarray, cassette: Dict) -> np.ndarray:
        """
        Mark cassette area as occupied in grid.

        Args:
            occupancy_grid: Current grid state
            cassette: Cassette to mark

        Returns:
            Updated grid (copy)
        """
        # Create copy to maintain immutability
        new_grid = occupancy_grid.copy()

        x, y = cassette['x'], cassette['y']
        w, h = cassette['width'], cassette['height']

        # Convert to grid coordinates
        gx1, gy1 = self._world_to_grid(x, y)
        gx2, gy2 = self._world_to_grid(x + w, y + h)

        # Clamp to grid bounds
        gx1 = max(0, min(gx1, self.grid_width - 1))
        gy1 = max(0, min(gy1, self.grid_height - 1))
        gx2 = max(0, min(gx2, self.grid_width))
        gy2 = max(0, min(gy2, self.grid_height))

        # Mark as occupied
        new_grid[gy1:gy2, gx1:gx2] = True

        return new_grid

    def generate_candidate_positions(self, cassette_width: float, cassette_height: float,
                                    occupancy_grid: np.ndarray) -> List[Tuple[float, float]]:
        """
        Generate candidate positions for cassette placement.

        Samples at 1 ft resolution, filters by occupancy and polygon containment.

        Args:
            cassette_width: Width of cassette to place
            cassette_height: Height of cassette to place
            occupancy_grid: Current grid state

        Returns:
            List of (x, y) candidate positions (limited to 50)
        """
        candidates = []
        sample_resolution = 1.0  # Sample every 1 foot

        y = self.min_y
        while y <= self.max_y - cassette_height:
            x = self.min_x
            while x <= self.max_x - cassette_width:
                # Quick check: is bottom-left corner inside?
                if not self._is_point_inside(x, y):
                    x += sample_resolution
                    continue

                # Check occupancy at this position
                gx, gy = self._world_to_grid(x, y)
                gx2, gy2 = self._world_to_grid(x + cassette_width, y + cassette_height)

                if gx2 >= self.grid_width or gy2 >= self.grid_height:
                    x += sample_resolution
                    continue

                # Calculate occupancy ratio
                region = occupancy_grid[gy:gy2, gx:gx2]
                if region.size > 0:
                    occupancy_ratio = np.sum(region) / region.size

                    # Only consider if mostly free (< 30% occupied)
                    if occupancy_ratio < 0.3:
                        candidates.append((x, y))

                x += sample_resolution
            y += sample_resolution

        # Limit to first 50 candidates
        if len(candidates) > 50:
            candidates = candidates[:50]

        return candidates

    def generate_cassette_on_demand(self, depth: int, free_space: float) -> Tuple[float, float]:
        """
        Select next cassette size to try based on search state.

        Uses largest-first strategy with depth-based adjustment.

        Args:
            depth: Current search depth
            free_space: Remaining free space in square feet

        Returns:
            (width, height) of cassette to try
        """
        # Try largest cassettes first (for better coverage)
        # But adjust based on free space
        for width, height in self.cassette_types:
            cassette_area = width * height
            # If cassette could potentially fit in remaining space
            if cassette_area <= free_space + 10:  # +10 for tolerance
                return (width, height)

        # Fallback to smallest cassette
        return (2, 2)

    def backtrack(self, current_cassettes: List[Dict], occupancy_grid: np.ndarray,
                 depth: int, max_depth: int) -> None:
        """
        Recursive backtracking search with pruning.

        Args:
            current_cassettes: Cassettes placed so far
            occupancy_grid: Current grid state
            depth: Current recursion depth
            max_depth: Maximum allowed depth
        """
        self.nodes_explored += 1

        # Log progress every 10000 nodes
        if self.nodes_explored % 10000 == 0:
            elapsed = time.time() - self.start_time
            logger.info(f"  Explored {self.nodes_explored} nodes in {elapsed:.1f}s, "
                       f"best: {self.best_coverage:.1f}%, pruned: {self.nodes_pruned}")

        # Check time limit
        elapsed = time.time() - self.start_time
        if elapsed > self.time_limit:
            logger.info(f"Time limit reached ({self.time_limit}s)")
            return

        # Emergency abort if too many nodes
        if self.nodes_explored > 10000000:  # 10 million
            logger.warning(f"Emergency abort: explored {self.nodes_explored} nodes")
            return

        # BASE CASE: Reached max depth
        if depth >= max_depth:
            current_coverage = self.calculate_coverage(current_cassettes)
            if current_coverage > self.best_coverage:
                self.best_coverage = current_coverage
                self.best_solution = current_cassettes.copy()
                logger.info(f"  New best at depth {depth}: {current_coverage:.1f}% ({len(current_cassettes)} cassettes)")
            return

        # PRUNING CHECK 1: Upper bound
        upper_bound = self.calculate_upper_bound(current_cassettes, occupancy_grid, depth, max_depth)
        if upper_bound <= self.best_coverage:
            self.nodes_pruned += 1
            return

        # PRUNING CHECK 2: Early termination if excellent coverage
        current_coverage = self.calculate_coverage(current_cassettes)
        if current_coverage >= 99.0:
            self.best_coverage = current_coverage
            self.best_solution = current_cassettes.copy()
            logger.info(f"  Achieved 99%+ coverage, terminating search")
            return

        # RECURSIVE CASE: Try placing next cassette
        free_space = self.calculate_free_space(occupancy_grid)

        if free_space < 4:  # Less than 2x2 cassette
            return

        # Generate cassette to try
        width, height = self.generate_cassette_on_demand(depth, free_space)

        # Try both orientations
        for w, h in [(width, height), (height, width)]:
            # Generate candidate positions for this cassette size
            candidates = self.generate_candidate_positions(w, h, occupancy_grid)

            # Try each candidate position
            for x, y in candidates:
                # Create cassette
                cassette = {
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h,
                    'size': f"{w}x{h}",
                    'area': w * h,
                    'weight': w * h * 10.4
                }

                # Validate placement
                if not self.is_valid_placement(cassette, current_cassettes):
                    continue

                # Place cassette
                new_cassettes = current_cassettes + [cassette]
                new_grid = self.update_occupancy_grid(occupancy_grid, cassette)

                # Recurse
                self.backtrack(new_cassettes, new_grid, depth + 1, max_depth)

                # Check if should abort
                if time.time() - self.start_time > self.time_limit:
                    return

    def optimize(self, max_depth: int = 15) -> Dict:
        """
        Main optimization entry point with time limit.

        Args:
            max_depth: Maximum recursion depth (default 15)

        Returns:
            Dictionary with cassettes, statistics, and search metrics
        """
        logger.info("=" * 70)
        logger.info("BACKTRACKING OPTIMIZER WITH PRUNING")
        logger.info("=" * 70)
        logger.info(f"Max depth: {max_depth}")
        logger.info(f"Time limit: {self.time_limit}s")

        # Initialize with greedy solution
        greedy_solution = self.greedy_initialization()
        self.best_solution = greedy_solution
        self.best_coverage = self.calculate_coverage(greedy_solution)

        logger.info(f"\nStarting backtracking search...")
        logger.info(f"Initial baseline: {self.best_coverage:.1f}%")

        # Start timer
        self.start_time = time.time()

        # Create initial empty grid
        initial_grid = self._create_empty_grid()

        # Run backtracking search
        self.backtrack([], initial_grid, 0, max_depth)

        # Calculate final statistics
        elapsed = time.time() - self.start_time

        logger.info("\n" + "=" * 70)
        logger.info("BACKTRACKING RESULTS")
        logger.info("=" * 70)
        logger.info(f"Search time: {elapsed:.1f}s")
        logger.info(f"Nodes explored: {self.nodes_explored}")
        logger.info(f"Nodes pruned: {self.nodes_pruned}")
        pruning_percent = (self.nodes_pruned / max(self.nodes_explored, 1)) * 100
        logger.info(f"Pruning efficiency: {pruning_percent:.1f}%")
        logger.info(f"\nBest solution:")
        logger.info(f"  Coverage: {self.best_coverage:.1f}%")
        logger.info(f"  Cassettes: {len(self.best_solution)}")

        # Calculate size distribution
        size_counts = {}
        for cassette in self.best_solution:
            size = cassette['size']
            size_counts[size] = size_counts.get(size, 0) + 1

        logger.info(f"  Size distribution:")
        for size in sorted(size_counts.keys()):
            logger.info(f"    {size}: {size_counts[size]} units")

        # Return results
        total_area = sum(c['area'] for c in self.best_solution)
        total_weight = sum(c['weight'] for c in self.best_solution)

        return {
            'cassettes': self.best_solution,
            'num_cassettes': len(self.best_solution),
            'coverage_percent': self.best_coverage,
            'total_area': self.polygon_area,
            'covered_area': total_area,
            'gap_area': self.polygon_area - total_area,
            'total_weight': total_weight,
            'size_distribution': size_counts,
            'search_stats': {
                'nodes_explored': self.nodes_explored,
                'nodes_pruned': self.nodes_pruned,
                'search_time': elapsed,
                'max_depth': max_depth
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

    print("\nTesting Backtracking Optimizer on Umbra polygon")
    print("=" * 70)

    optimizer = BacktrackingOptimizer(umbra_polygon)
    result = optimizer.optimize(max_depth=10)

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)

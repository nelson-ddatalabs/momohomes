#!/usr/bin/env python3
"""
dp_grid.py - Dynamic Programming Grid Discretization System
===========================================================
Production-ready grid system for DP-based panel optimization.
Provides adaptive resolution, coordinate mapping, and efficient occupancy tracking.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, FrozenSet
from math import gcd, ceil, floor
import numpy as np
from functools import reduce

from models import Room, PanelSize
from advanced_packing import PanelPlacement, PackingState


@dataclass(frozen=True)
class GridResolution:
    """
    Represents grid resolution parameters for DP discretization.
    Immutable for use in state hashing and caching.
    """
    x_step: float
    y_step: float
    x_points: int
    y_points: int
    room_width: float
    room_height: float
    
    @property
    def total_cells(self) -> int:
        """Total number of grid cells."""
        return self.x_points * self.y_points
    
    @property
    def cell_area(self) -> float:
        """Area of each grid cell."""
        return self.x_step * self.y_step
    
    def is_valid_position(self, grid_x: int, grid_y: int) -> bool:
        """Check if grid coordinates are within bounds."""
        return 0 <= grid_x < self.x_points and 0 <= grid_y < self.y_points


class AdaptiveGridResolver:
    """
    Determines optimal grid resolution for DP optimization.
    Balances accuracy with computational efficiency.
    """
    
    def __init__(self, max_cells: int = 10000, min_resolution: float = 0.1):
        self.max_cells = max_cells
        self.min_resolution = min_resolution
    
    def calculate_resolution(self, room: Room, panel_sizes: List[PanelSize]) -> GridResolution:
        """
        Calculate optimal grid resolution for given room and panels.
        Uses GCD-based approach for panel-aligned grid.
        """
        # Get all panel dimensions
        dimensions = self._extract_dimensions(panel_sizes)
        
        # Find optimal resolution using GCD
        x_step = self._find_optimal_step(dimensions, room.width)
        y_step = self._find_optimal_step(dimensions, room.height)
        
        # Ensure minimum resolution
        x_step = max(x_step, self.min_resolution)
        y_step = max(y_step, self.min_resolution)
        
        # Calculate grid points
        x_points = max(1, int(ceil(room.width / x_step)))
        y_points = max(1, int(ceil(room.height / y_step)))
        
        # Check cell count limit
        total_cells = x_points * y_points
        if total_cells > self.max_cells:
            scale_factor = np.sqrt(self.max_cells / total_cells)
            x_step *= (1 / scale_factor)
            y_step *= (1 / scale_factor)
            x_points = max(1, int(ceil(room.width / x_step)))
            y_points = max(1, int(ceil(room.height / y_step)))
        
        return GridResolution(
            x_step=x_step,
            y_step=y_step,
            x_points=x_points,
            y_points=y_points,
            room_width=room.width,
            room_height=room.height
        )
    
    def _extract_dimensions(self, panel_sizes: List[PanelSize]) -> Set[float]:
        """Extract all unique dimensions from panel sizes."""
        dimensions = set()
        for panel_size in panel_sizes:
            dimensions.add(panel_size.width)
            dimensions.add(panel_size.length)
        return dimensions
    
    def _find_optimal_step(self, dimensions: Set[float], room_dimension: float) -> float:
        """Find optimal step size using GCD of dimensions."""
        if not dimensions:
            return 1.0
        
        # Convert to integers (multiply by 10 to handle 0.1 precision)
        int_dims = [int(d * 10) for d in dimensions]
        int_room = int(room_dimension * 10)
        
        # Find GCD of all dimensions
        dimension_gcd = reduce(gcd, int_dims)
        combined_gcd = gcd(dimension_gcd, int_room)
        
        # Convert back to float
        step = combined_gcd / 10.0
        
        # Ensure reasonable bounds
        return min(step, room_dimension / 10.0)


class CoordinateMapper:
    """
    Maps between continuous world coordinates and discrete grid coordinates.
    Provides bidirectional conversion with proper alignment.
    """
    
    def __init__(self, resolution: GridResolution, room_origin: Tuple[float, float]):
        self.resolution = resolution
        self.origin_x, self.origin_y = room_origin
    
    def world_to_grid(self, world_x: float, world_y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid indices."""
        # Relative to room origin
        rel_x = world_x - self.origin_x
        rel_y = world_y - self.origin_y
        
        # Convert to grid coordinates
        grid_x = int(floor(rel_x / self.resolution.x_step))
        grid_y = int(floor(rel_y / self.resolution.y_step))
        
        # Clamp to bounds
        grid_x = max(0, min(grid_x, self.resolution.x_points - 1))
        grid_y = max(0, min(grid_y, self.resolution.y_points - 1))
        
        return grid_x, grid_y
    
    def grid_to_world(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """Convert grid indices to world coordinates (bottom-left of cell)."""
        world_x = self.origin_x + grid_x * self.resolution.x_step
        world_y = self.origin_y + grid_y * self.resolution.y_step
        return world_x, world_y
    
    def get_grid_bounds(self, placement: PanelPlacement) -> Tuple[int, int, int, int]:
        """Get grid bounds for panel placement (min_x, min_y, max_x, max_y)."""
        x1, y1, x2, y2 = placement.bounds
        
        min_x, min_y = self.world_to_grid(x1, y1)
        max_x, max_y = self.world_to_grid(x2 - 0.001, y2 - 0.001)  # Small offset to avoid boundary issues
        
        return min_x, min_y, max_x, max_y
    
    def get_valid_positions(self, panel_width: float, panel_height: float) -> List[Tuple[float, float]]:
        """Get all valid world positions for panel placement on grid."""
        positions = []
        
        for grid_x in range(self.resolution.x_points):
            for grid_y in range(self.resolution.y_points):
                world_x, world_y = self.grid_to_world(grid_x, grid_y)
                
                # Check if panel fits within room
                if (world_x + panel_width <= self.origin_x + self.resolution.room_width and
                    world_y + panel_height <= self.origin_y + self.resolution.room_height):
                    positions.append((world_x, world_y))
        
        return positions


class PanelToGridConverter:
    """
    Converts panel placements to grid occupancy patterns.
    Handles partial cell coverage and overlap detection.
    """
    
    def __init__(self, mapper: CoordinateMapper):
        self.mapper = mapper
        self.resolution = mapper.resolution
    
    def placement_to_grid_cells(self, placement: PanelPlacement) -> Set[Tuple[int, int]]:
        """Convert panel placement to set of occupied grid cells."""
        min_x, min_y, max_x, max_y = self.mapper.get_grid_bounds(placement)
        
        cells = set()
        for grid_x in range(min_x, max_x + 1):
            for grid_y in range(min_y, max_y + 1):
                if self.resolution.is_valid_position(grid_x, grid_y):
                    cells.add((grid_x, grid_y))
        
        return cells
    
    def placements_to_occupancy(self, placements: List[PanelPlacement]) -> np.ndarray:
        """Convert list of placements to occupancy grid."""
        grid = np.zeros((self.resolution.y_points, self.resolution.x_points), dtype=int)
        
        for i, placement in enumerate(placements, 1):
            cells = self.placement_to_grid_cells(placement)
            for grid_x, grid_y in cells:
                grid[grid_y, grid_x] = i  # Use placement index as value
        
        return grid
    
    def get_coverage_ratio(self, placements: List[PanelPlacement]) -> float:
        """Calculate grid-based coverage ratio."""
        if not placements:
            return 0.0
        
        occupied_cells = set()
        for placement in placements:
            occupied_cells.update(self.placement_to_grid_cells(placement))
        
        return len(occupied_cells) / self.resolution.total_cells
    
    def check_overlap(self, placement1: PanelPlacement, placement2: PanelPlacement) -> bool:
        """Check if two placements overlap on the grid."""
        cells1 = self.placement_to_grid_cells(placement1)
        cells2 = self.placement_to_grid_cells(placement2)
        return bool(cells1 & cells2)


class DPOccupancyTracker:
    """
    Enhanced occupancy tracking optimized for DP algorithms.
    Provides efficient state encoding and transition operations.
    """
    
    def __init__(self, resolution: GridResolution, room_origin: Tuple[float, float]):
        self.resolution = resolution
        self.mapper = CoordinateMapper(resolution, room_origin)
        self.converter = PanelToGridConverter(self.mapper)
        self.grid = np.zeros((resolution.y_points, resolution.x_points), dtype=int)
        self.placement_count = 0
    
    def add_placement(self, placement: PanelPlacement) -> 'DPOccupancyTracker':
        """Create new tracker with additional placement (immutable operation)."""
        new_tracker = DPOccupancyTracker(self.resolution, (self.mapper.origin_x, self.mapper.origin_y))
        new_tracker.grid = self.grid.copy()
        new_tracker.placement_count = self.placement_count
        
        # Add new placement
        cells = self.converter.placement_to_grid_cells(placement)
        new_tracker.placement_count += 1
        
        for grid_x, grid_y in cells:
            new_tracker.grid[grid_y, grid_x] = new_tracker.placement_count
        
        return new_tracker
    
    def can_place_panel(self, placement: PanelPlacement) -> bool:
        """Check if panel can be placed without overlap."""
        cells = self.converter.placement_to_grid_cells(placement)
        
        for grid_x, grid_y in cells:
            if self.grid[grid_y, grid_x] != 0:  # Cell already occupied
                return False
        
        return True
    
    def get_free_cells(self) -> List[Tuple[int, int]]:
        """Get list of all free grid cells."""
        free_cells = []
        for grid_x in range(self.resolution.x_points):
            for grid_y in range(self.resolution.y_points):
                if self.grid[grid_y, grid_x] == 0:
                    free_cells.append((grid_x, grid_y))
        return free_cells
    
    def get_largest_free_rectangle(self) -> Optional[Tuple[int, int, int, int]]:
        """Find largest free rectangle using largest rectangle in histogram approach."""
        if self.resolution.total_cells == 0:
            return None
        
        # Convert occupancy to binary (0 = free, 1 = occupied)
        binary_grid = (self.grid > 0).astype(int)
        
        max_area = 0
        best_rect = None
        
        # For each row, find largest rectangle ending at this row
        heights = np.zeros(self.resolution.x_points, dtype=int)
        
        for row in range(self.resolution.y_points):
            # Update heights
            for col in range(self.resolution.x_points):
                if binary_grid[row, col] == 0:
                    heights[col] += 1
                else:
                    heights[col] = 0
            
            # Find largest rectangle in this histogram
            rect_area, rect_bounds = self._largest_rectangle_in_histogram(heights, row)
            
            if rect_area > max_area:
                max_area = rect_area
                best_rect = rect_bounds
        
        return best_rect
    
    def _largest_rectangle_in_histogram(self, heights: np.ndarray, row: int) -> Tuple[int, Optional[Tuple[int, int, int, int]]]:
        """Find largest rectangle in histogram using stack-based algorithm."""
        stack = []
        max_area = 0
        best_rect = None
        
        for i in range(len(heights)):
            while stack and heights[i] < heights[stack[-1]]:
                h = heights[stack.pop()]
                w = i if not stack else i - stack[-1] - 1
                area = h * w
                
                if area > max_area:
                    max_area = area
                    left = 0 if not stack else stack[-1] + 1
                    right = i - 1
                    top = row - h + 1
                    bottom = row
                    best_rect = (left, top, right, bottom)
            
            stack.append(i)
        
        # Process remaining items in stack
        while stack:
            h = heights[stack.pop()]
            w = len(heights) if not stack else len(heights) - stack[-1] - 1
            area = h * w
            
            if area > max_area:
                max_area = area
                left = 0 if not stack else stack[-1] + 1
                right = len(heights) - 1
                top = row - h + 1
                bottom = row
                best_rect = (left, top, right, bottom)
        
        return max_area, best_rect
    
    def get_state_signature(self) -> str:
        """Get compact signature for state comparison."""
        # Use hash of non-zero positions for efficiency
        occupied_positions = []
        for grid_x in range(self.resolution.x_points):
            for grid_y in range(self.resolution.y_points):
                if self.grid[grid_y, grid_x] > 0:
                    occupied_positions.append((grid_x, grid_y))
        
        return str(hash(tuple(sorted(occupied_positions))))
    
    def get_coverage_ratio(self) -> float:
        """Get ratio of occupied cells."""
        occupied = np.sum(self.grid > 0)
        return occupied / self.resolution.total_cells


class GridBasedDPOptimizer:
    """
    Integration class that combines all grid components for DP optimization.
    Provides high-level interface for grid-based DP algorithms.
    """
    
    def __init__(self, room: Room, panel_sizes: List[PanelSize], max_cells: int = 10000):
        self.room = room
        self.panel_sizes = panel_sizes
        
        # Calculate optimal resolution
        resolver = AdaptiveGridResolver(max_cells=max_cells)
        self.resolution = resolver.calculate_resolution(room, panel_sizes)
        
        # Initialize components
        self.mapper = CoordinateMapper(self.resolution, (room.position.x, room.position.y))
        self.converter = PanelToGridConverter(self.mapper)
        
        # Create initial occupancy tracker
        self.initial_tracker = DPOccupancyTracker(self.resolution, (room.position.x, room.position.y))
    
    def create_tracker_from_state(self, state: PackingState) -> DPOccupancyTracker:
        """Create occupancy tracker from packing state."""
        tracker = DPOccupancyTracker(self.resolution, (self.room.position.x, self.room.position.y))
        
        # Add all placements
        for placement in state.placements:
            tracker = tracker.add_placement(placement)
        
        return tracker
    
    def get_valid_placements(self, tracker: DPOccupancyTracker, panel_size: PanelSize) -> List[PanelPlacement]:
        """Get all valid placements for panel on current grid state."""
        placements = []
        
        for orientation in ["horizontal", "vertical"]:
            pw, ph = panel_size.get_dimensions(orientation)
            positions = self.mapper.get_valid_positions(pw, ph)
            
            for world_x, world_y in positions:
                placement = PanelPlacement(
                    panel_size=panel_size,
                    position=(world_x, world_y),
                    orientation=orientation
                )
                
                if tracker.can_place_panel(placement):
                    placements.append(placement)
        
        return placements
    
    def analyze_free_space(self, tracker: DPOccupancyTracker) -> Dict[str, any]:
        """Analyze remaining free space in grid."""
        free_cells = tracker.get_free_cells()
        largest_rect = tracker.get_largest_free_rectangle()
        
        analysis = {
            'free_cells': len(free_cells),
            'free_ratio': len(free_cells) / self.resolution.total_cells,
            'largest_rectangle': largest_rect,
            'coverage': tracker.get_coverage_ratio()
        }
        
        # Add rectangle area in world coordinates if available
        if largest_rect:
            left, top, right, bottom = largest_rect
            world_width = (right - left + 1) * self.resolution.x_step
            world_height = (bottom - top + 1) * self.resolution.y_step
            analysis['largest_rect_area'] = world_width * world_height
        
        return analysis


def create_dp_grid_system(room: Room, panel_sizes: List[PanelSize], max_cells: int = 10000) -> GridBasedDPOptimizer:
    """
    Factory function to create complete DP grid system.
    Returns configured optimizer ready for DP algorithms.
    """
    return GridBasedDPOptimizer(room, panel_sizes, max_cells)
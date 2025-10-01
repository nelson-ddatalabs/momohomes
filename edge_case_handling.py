from typing import List, Dict, Optional, Set, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from datetime import datetime
import logging
import math
from collections import defaultdict

from core import PackingState, Panel, Room, Position, PlacedPanel
from spatial_index import SpatialIndex, OccupancyGrid
from algorithm_interface import OptimizerConfig, OptimizerResult

logger = logging.getLogger(__name__)


class EdgeCaseType(Enum):
    EXTREME_ASPECT_RATIO = "extreme_aspect_ratio"
    VERY_SMALL_PANEL = "very_small_panel"
    VERY_LARGE_PANEL = "very_large_panel"
    MINIMAL_SPACE = "minimal_space"
    IRREGULAR_BOUNDARY = "irregular_boundary"
    OBSTACLE_PRESENCE = "obstacle_presence"
    NEAR_PERFECT_FIT = "near_perfect_fit"
    FRAGMENTED_SPACE = "fragmented_space"


class ShapeClassification(Enum):
    SQUARE = "square"
    HORIZONTAL_STRIP = "horizontal_strip"
    VERTICAL_STRIP = "vertical_strip"
    TINY = "tiny"
    MASSIVE = "massive"
    NORMAL = "normal"


@dataclass
class EdgeCaseDetection:
    case_type: EdgeCaseType
    severity: float  # 0-1 scale
    description: str
    affected_items: List[Any]
    handling_strategy: str


@dataclass
class IrregularBoundary:
    vertices: List[Tuple[float, float]]
    holes: List[List[Tuple[float, float]]]
    obstacles: List[Tuple[float, float, float, float]]  # (x, y, width, height)
    
    def contains_point(self, x: float, y: float) -> bool:
        """Check if point is inside irregular boundary"""
        # Simple point-in-polygon test using ray casting
        inside = False
        n = len(self.vertices)
        p1x, p1y = self.vertices[0]
        
        for i in range(1, n + 1):
            p2x, p2y = self.vertices[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        # Check if point is in any hole
        for hole in self.holes:
            if self._point_in_polygon(x, y, hole):
                return False
        
        # Check if point is in any obstacle
        for ox, oy, ow, oh in self.obstacles:
            if ox <= x <= ox + ow and oy <= y <= oy + oh:
                return False
        
        return inside
    
    def _point_in_polygon(self, x: float, y: float, vertices: List[Tuple[float, float]]) -> bool:
        """Helper for point-in-polygon test"""
        inside = False
        n = len(vertices)
        p1x, p1y = vertices[0]
        
        for i in range(1, n + 1):
            p2x, p2y = vertices[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside


class ExtremeShapeHandler:
    """Handles panels with extreme shapes"""
    
    def __init__(self):
        self.aspect_ratio_threshold = 5.0  # Width/height > 5 is extreme
        self.tiny_area_threshold = 100.0   # Area < 100 is tiny
        self.massive_area_threshold = 10000.0  # Area > 10000 is massive
        self.handling_stats = defaultdict(int)
    
    def classify_shape(self, panel: Panel) -> ShapeClassification:
        """Classify panel shape"""
        area = panel.width * panel.height
        aspect_ratio = max(panel.width, panel.height) / min(panel.width, panel.height)
        
        if area < self.tiny_area_threshold:
            return ShapeClassification.TINY
        elif area > self.massive_area_threshold:
            return ShapeClassification.MASSIVE
        elif aspect_ratio > self.aspect_ratio_threshold:
            if panel.width > panel.height:
                return ShapeClassification.HORIZONTAL_STRIP
            else:
                return ShapeClassification.VERTICAL_STRIP
        elif aspect_ratio < 1.2:  # Nearly square
            return ShapeClassification.SQUARE
        else:
            return ShapeClassification.NORMAL
    
    def handle_extreme_panel(self, panel: Panel, state: PackingState, 
                            room: Room) -> Optional[PlacedPanel]:
        """Handle placement of extreme shape panel"""
        classification = self.classify_shape(panel)
        self.handling_stats[classification] += 1
        
        if classification == ShapeClassification.HORIZONTAL_STRIP:
            return self._place_horizontal_strip(panel, state, room)
        elif classification == ShapeClassification.VERTICAL_STRIP:
            return self._place_vertical_strip(panel, state, room)
        elif classification == ShapeClassification.TINY:
            return self._place_tiny_panel(panel, state, room)
        elif classification == ShapeClassification.MASSIVE:
            return self._place_massive_panel(panel, state, room)
        else:
            return None
    
    def _place_horizontal_strip(self, panel: Panel, state: PackingState, 
                               room: Room) -> Optional[PlacedPanel]:
        """Place horizontal strip panel"""
        # Try to place along top or bottom edges
        positions = [
            Position(0, 0),  # Top edge
            Position(0, room.height - panel.height)  # Bottom edge
        ]
        
        # Also try stacking on existing panels
        for placed in state.placed_panels:
            positions.append(Position(placed.position.x, 
                                    placed.position.y + placed.panel.height))
        
        for pos in positions:
            if self._is_valid_position(panel, pos, state, room):
                return PlacedPanel(panel=panel, position=pos, rotated=False)
        
        return None
    
    def _place_vertical_strip(self, panel: Panel, state: PackingState,
                             room: Room) -> Optional[PlacedPanel]:
        """Place vertical strip panel"""
        # Try to place along left or right edges
        positions = [
            Position(0, 0),  # Left edge
            Position(room.width - panel.width, 0)  # Right edge
        ]
        
        # Also try adjacent to existing panels
        for placed in state.placed_panels:
            positions.append(Position(placed.position.x + placed.panel.width,
                                    placed.position.y))
        
        for pos in positions:
            if self._is_valid_position(panel, pos, state, room):
                return PlacedPanel(panel=panel, position=pos, rotated=False)
        
        return None
    
    def _place_tiny_panel(self, panel: Panel, state: PackingState,
                         room: Room) -> Optional[PlacedPanel]:
        """Place tiny panel in gaps"""
        # Find small gaps in current placement
        gaps = self._find_gaps(state, room, max_size=panel.width * panel.height * 2)
        
        for gap_x, gap_y, gap_w, gap_h in gaps:
            if panel.width <= gap_w and panel.height <= gap_h:
                pos = Position(gap_x, gap_y)
                if self._is_valid_position(panel, pos, state, room):
                    return PlacedPanel(panel=panel, position=pos, rotated=False)
            
            # Try rotated
            if panel.can_rotate and panel.height <= gap_w and panel.width <= gap_h:
                pos = Position(gap_x, gap_y)
                if self._is_valid_position(panel, pos, state, room, rotated=True):
                    return PlacedPanel(panel=panel, position=pos, rotated=True)
        
        return None
    
    def _place_massive_panel(self, panel: Panel, state: PackingState,
                            room: Room) -> Optional[PlacedPanel]:
        """Place massive panel"""
        # For massive panels, try corners first
        corners = [
            Position(0, 0),
            Position(room.width - panel.width, 0),
            Position(0, room.height - panel.height),
            Position(room.width - panel.width, room.height - panel.height)
        ]
        
        for pos in corners:
            if self._is_valid_position(panel, pos, state, room):
                return PlacedPanel(panel=panel, position=pos, rotated=False)
        
        # Try rotated if possible
        if panel.can_rotate:
            for pos in corners:
                adjusted_pos = Position(pos.x, pos.y)
                if self._is_valid_position(panel, adjusted_pos, state, room, rotated=True):
                    return PlacedPanel(panel=panel, position=adjusted_pos, rotated=True)
        
        return None
    
    def _find_gaps(self, state: PackingState, room: Room,
                  max_size: float = float('inf')) -> List[Tuple[float, float, float, float]]:
        """Find gaps in current placement"""
        gaps = []
        
        # Simple gap detection using grid
        grid_size = 10
        occupied = np.zeros((int(room.height / grid_size) + 1,
                           int(room.width / grid_size) + 1), dtype=bool)
        
        # Mark occupied cells
        for placed in state.placed_panels:
            x1 = int(placed.position.x / grid_size)
            y1 = int(placed.position.y / grid_size)
            x2 = int((placed.position.x + placed.panel.width) / grid_size)
            y2 = int((placed.position.y + placed.panel.height) / grid_size)
            occupied[y1:y2+1, x1:x2+1] = True
        
        # Find rectangular gaps
        for y in range(occupied.shape[0]):
            for x in range(occupied.shape[1]):
                if not occupied[y, x]:
                    # Find extent of gap
                    width = 0
                    height = 0
                    
                    # Find width
                    for dx in range(x, occupied.shape[1]):
                        if occupied[y, dx]:
                            break
                        width += 1
                    
                    # Find height
                    for dy in range(y, occupied.shape[0]):
                        if occupied[dy, x]:
                            break
                        height += 1
                    
                    gap_area = (width * grid_size) * (height * grid_size)
                    if gap_area <= max_size:
                        gaps.append((x * grid_size, y * grid_size,
                                   width * grid_size, height * grid_size))
                    
                    # Mark as processed
                    occupied[y, x] = True
        
        return gaps
    
    def _is_valid_position(self, panel: Panel, position: Position,
                          state: PackingState, room: Room,
                          rotated: bool = False) -> bool:
        """Check if position is valid for panel"""
        width = panel.height if rotated else panel.width
        height = panel.width if rotated else panel.height
        
        # Check boundaries
        if position.x < 0 or position.y < 0:
            return False
        if position.x + width > room.width or position.y + height > room.height:
            return False
        
        # Check overlaps
        for placed in state.placed_panels:
            placed_width = placed.panel.height if placed.rotated else placed.panel.width
            placed_height = placed.panel.width if placed.rotated else placed.panel.height
            
            if not (position.x + width <= placed.position.x or
                   placed.position.x + placed_width <= position.x or
                   position.y + height <= placed.position.y or
                   placed.position.y + placed_height <= position.y):
                return False
        
        return True
    
    def suggest_decomposition(self, panel: Panel) -> List[Panel]:
        """Suggest decomposing extreme panel into smaller ones"""
        suggestions = []
        
        classification = self.classify_shape(panel)
        
        if classification == ShapeClassification.MASSIVE:
            # Suggest splitting into quarters
            half_width = panel.width / 2
            half_height = panel.height / 2
            
            for i in range(2):
                for j in range(2):
                    sub_panel = Panel(
                        id=f"{panel.id}_sub_{i}_{j}",
                        width=half_width,
                        height=half_height,
                        can_rotate=panel.can_rotate
                    )
                    suggestions.append(sub_panel)
        
        elif classification in [ShapeClassification.HORIZONTAL_STRIP,
                              ShapeClassification.VERTICAL_STRIP]:
            # Suggest splitting into smaller strips
            if classification == ShapeClassification.HORIZONTAL_STRIP:
                num_parts = int(panel.width / panel.height)
                part_width = panel.width / num_parts
                
                for i in range(num_parts):
                    sub_panel = Panel(
                        id=f"{panel.id}_strip_{i}",
                        width=part_width,
                        height=panel.height,
                        can_rotate=panel.can_rotate
                    )
                    suggestions.append(sub_panel)
            else:
                num_parts = int(panel.height / panel.width)
                part_height = panel.height / num_parts
                
                for i in range(num_parts):
                    sub_panel = Panel(
                        id=f"{panel.id}_strip_{i}",
                        width=panel.width,
                        height=part_height,
                        can_rotate=panel.can_rotate
                    )
                    suggestions.append(sub_panel)
        
        return suggestions


class MinimalSpaceSolver:
    """Solves packing in minimal remaining space"""
    
    def __init__(self):
        self.min_space_threshold = 0.1  # 10% of room area
        self.strategies = ["greedy_fill", "best_fit", "tetris_style"]
    
    def detect_minimal_space(self, state: PackingState, room: Room) -> bool:
        """Detect if we're in minimal space situation"""
        occupied_area = sum(p.panel.width * p.panel.height for p in state.placed_panels)
        room_area = room.width * room.height
        remaining_ratio = 1.0 - (occupied_area / room_area)
        
        return remaining_ratio < self.min_space_threshold
    
    def solve_minimal_space(self, state: PackingState, room: Room,
                          remaining_panels: List[Panel]) -> List[PlacedPanel]:
        """Solve packing in minimal space"""
        if not remaining_panels:
            return []
        
        # Find available spaces
        available_spaces = self._find_available_spaces(state, room)
        
        if not available_spaces:
            return []
        
        # Sort panels by area (smallest first for gap filling)
        sorted_panels = sorted(remaining_panels, key=lambda p: p.width * p.height)
        
        placed = []
        for panel in sorted_panels:
            placement = self._find_best_fit(panel, available_spaces, state, room)
            if placement:
                placed.append(placement)
                # Update available spaces
                available_spaces = self._update_available_spaces(
                    available_spaces, placement
                )
        
        return placed
    
    def _find_available_spaces(self, state: PackingState, 
                              room: Room) -> List[Tuple[float, float, float, float]]:
        """Find all available spaces"""
        spaces = []
        
        # Create bitmap of occupied space
        resolution = 1.0  # 1 unit resolution
        width_cells = int(room.width / resolution)
        height_cells = int(room.height / resolution)
        
        occupied = np.zeros((height_cells, width_cells), dtype=bool)
        
        # Mark occupied cells
        for placed in state.placed_panels:
            x1 = int(placed.position.x / resolution)
            y1 = int(placed.position.y / resolution)
            x2 = int((placed.position.x + placed.panel.width) / resolution)
            y2 = int((placed.position.y + placed.panel.height) / resolution)
            
            x1 = max(0, min(x1, width_cells - 1))
            y1 = max(0, min(y1, height_cells - 1))
            x2 = max(0, min(x2, width_cells))
            y2 = max(0, min(y2, height_cells))
            
            occupied[y1:y2, x1:x2] = True
        
        # Find maximal empty rectangles
        spaces = self._find_maximal_rectangles(occupied, resolution)
        
        return spaces
    
    def _find_maximal_rectangles(self, occupied: np.ndarray,
                                resolution: float) -> List[Tuple[float, float, float, float]]:
        """Find maximal empty rectangles in occupancy grid"""
        rectangles = []
        height, width = occupied.shape
        
        # Dynamic programming approach
        heights = np.zeros(width)
        
        for row in range(height):
            for col in range(width):
                if occupied[row, col]:
                    heights[col] = 0
                else:
                    heights[col] += 1
            
            # Find rectangles in this row
            max_rects = self._largest_rectangle_in_histogram(heights)
            
            for rect_col, rect_width, rect_height in max_rects:
                if rect_width > 0 and rect_height > 0:
                    rectangles.append((
                        rect_col * resolution,
                        (row - rect_height + 1) * resolution,
                        rect_width * resolution,
                        rect_height * resolution
                    ))
        
        # Remove duplicates and small rectangles
        unique_rects = []
        for rect in rectangles:
            if rect[2] * rect[3] > 10:  # Minimum area threshold
                is_duplicate = False
                for existing in unique_rects:
                    if (abs(rect[0] - existing[0]) < 1 and
                        abs(rect[1] - existing[1]) < 1 and
                        abs(rect[2] - existing[2]) < 1 and
                        abs(rect[3] - existing[3]) < 1):
                        is_duplicate = True
                        break
                if not is_duplicate:
                    unique_rects.append(rect)
        
        return unique_rects
    
    def _largest_rectangle_in_histogram(self, heights: np.ndarray) -> List[Tuple[int, int, int]]:
        """Find largest rectangles in histogram"""
        stack = []
        rectangles = []
        index = 0
        
        while index < len(heights):
            if not stack or heights[index] >= heights[stack[-1]]:
                stack.append(index)
                index += 1
            else:
                top = stack.pop()
                width = index if not stack else index - stack[-1] - 1
                area = heights[top] * width
                
                if area > 0:
                    start_col = stack[-1] + 1 if stack else 0
                    rectangles.append((start_col, width, int(heights[top])))
        
        while stack:
            top = stack.pop()
            width = index if not stack else index - stack[-1] - 1
            area = heights[top] * width
            
            if area > 0:
                start_col = stack[-1] + 1 if stack else 0
                rectangles.append((start_col, width, int(heights[top])))
        
        return rectangles
    
    def _find_best_fit(self, panel: Panel,
                      spaces: List[Tuple[float, float, float, float]],
                      state: PackingState, room: Room) -> Optional[PlacedPanel]:
        """Find best fit for panel in available spaces"""
        best_fit = None
        best_waste = float('inf')
        
        for space_x, space_y, space_w, space_h in spaces:
            # Try normal orientation
            if panel.width <= space_w and panel.height <= space_h:
                waste = (space_w * space_h) - (panel.width * panel.height)
                if waste < best_waste:
                    pos = Position(space_x, space_y)
                    if self._is_valid_minimal_placement(panel, pos, state, room, False):
                        best_waste = waste
                        best_fit = PlacedPanel(panel=panel, position=pos, rotated=False)
            
            # Try rotated
            if panel.can_rotate and panel.height <= space_w and panel.width <= space_h:
                waste = (space_w * space_h) - (panel.width * panel.height)
                if waste < best_waste:
                    pos = Position(space_x, space_y)
                    if self._is_valid_minimal_placement(panel, pos, state, room, True):
                        best_waste = waste
                        best_fit = PlacedPanel(panel=panel, position=pos, rotated=True)
        
        return best_fit
    
    def _is_valid_minimal_placement(self, panel: Panel, position: Position,
                                   state: PackingState, room: Room,
                                   rotated: bool) -> bool:
        """Validate placement in minimal space"""
        width = panel.height if rotated else panel.width
        height = panel.width if rotated else panel.height
        
        # Relaxed boundary check for minimal space
        if position.x < -0.1 or position.y < -0.1:
            return False
        if position.x + width > room.width + 0.1 or position.y + height > room.height + 0.1:
            return False
        
        # Check overlaps
        for placed in state.placed_panels:
            placed_width = placed.panel.height if placed.rotated else placed.panel.width
            placed_height = placed.panel.width if placed.rotated else placed.panel.height
            
            if not (position.x + width <= placed.position.x or
                   placed.position.x + placed_width <= position.x or
                   position.y + height <= placed.position.y or
                   placed.position.y + placed_height <= position.y):
                return False
        
        return True
    
    def _update_available_spaces(self, spaces: List[Tuple[float, float, float, float]],
                                placement: PlacedPanel) -> List[Tuple[float, float, float, float]]:
        """Update available spaces after placement"""
        updated = []
        
        placed_x = placement.position.x
        placed_y = placement.position.y
        placed_w = placement.panel.height if placement.rotated else placement.panel.width
        placed_h = placement.panel.width if placement.rotated else placement.panel.height
        
        for space_x, space_y, space_w, space_h in spaces:
            # Check if space intersects with placement
            if (space_x + space_w <= placed_x or placed_x + placed_w <= space_x or
                space_y + space_h <= placed_y or placed_y + placed_h <= space_y):
                # No intersection, keep space
                updated.append((space_x, space_y, space_w, space_h))
            else:
                # Split space around placement
                # Left part
                if space_x < placed_x:
                    updated.append((space_x, space_y, placed_x - space_x, space_h))
                
                # Right part
                if placed_x + placed_w < space_x + space_w:
                    updated.append((placed_x + placed_w, space_y,
                                  space_x + space_w - (placed_x + placed_w), space_h))
                
                # Top part
                if space_y < placed_y:
                    updated.append((space_x, space_y, space_w, placed_y - space_y))
                
                # Bottom part
                if placed_y + placed_h < space_y + space_h:
                    updated.append((space_x, placed_y + placed_h,
                                  space_w, space_y + space_h - (placed_y + placed_h)))
        
        return updated


class IrregularBoundaryHandler:
    """Handles rooms with irregular boundaries"""
    
    def __init__(self):
        self.boundary_tolerance = 1.0
        self.obstacle_buffer = 2.0
    
    def create_irregular_room(self, vertices: List[Tuple[float, float]],
                            obstacles: Optional[List[Tuple[float, float, float, float]]] = None,
                            holes: Optional[List[List[Tuple[float, float]]]] = None) -> IrregularBoundary:
        """Create irregular boundary representation"""
        return IrregularBoundary(
            vertices=vertices,
            holes=holes or [],
            obstacles=obstacles or []
        )
    
    def handle_irregular_room(self, boundary: IrregularBoundary,
                            panels: List[Panel]) -> List[PlacedPanel]:
        """Handle packing in irregular room"""
        placed = []
        
        # Get bounding box
        min_x = min(v[0] for v in boundary.vertices)
        max_x = max(v[0] for v in boundary.vertices)
        min_y = min(v[1] for v in boundary.vertices)
        max_y = max(v[1] for v in boundary.vertices)
        
        # Create grid for placement
        grid_size = 5.0
        
        # Sort panels by area (largest first)
        sorted_panels = sorted(panels, key=lambda p: p.width * p.height, reverse=True)
        
        for panel in sorted_panels:
            placement = self._find_irregular_placement(
                panel, boundary, placed, min_x, min_y, max_x, max_y, grid_size
            )
            if placement:
                placed.append(placement)
        
        return placed
    
    def _find_irregular_placement(self, panel: Panel, boundary: IrregularBoundary,
                                 placed: List[PlacedPanel],
                                 min_x: float, min_y: float,
                                 max_x: float, max_y: float,
                                 grid_size: float) -> Optional[PlacedPanel]:
        """Find placement in irregular boundary"""
        
        # Try positions on a grid
        for y in np.arange(min_y, max_y - panel.height + grid_size, grid_size):
            for x in np.arange(min_x, max_x - panel.width + grid_size, grid_size):
                pos = Position(x, y)
                
                if self._is_valid_irregular_placement(panel, pos, boundary, placed, False):
                    return PlacedPanel(panel=panel, position=pos, rotated=False)
                
                # Try rotated
                if panel.can_rotate:
                    if self._is_valid_irregular_placement(panel, pos, boundary, placed, True):
                        return PlacedPanel(panel=panel, position=pos, rotated=True)
        
        return None
    
    def _is_valid_irregular_placement(self, panel: Panel, position: Position,
                                     boundary: IrregularBoundary,
                                     placed: List[PlacedPanel],
                                     rotated: bool) -> bool:
        """Check if placement is valid in irregular boundary"""
        width = panel.height if rotated else panel.width
        height = panel.width if rotated else panel.height
        
        # Check corners are inside boundary
        corners = [
            (position.x, position.y),
            (position.x + width, position.y),
            (position.x, position.y + height),
            (position.x + width, position.y + height)
        ]
        
        for corner_x, corner_y in corners:
            if not boundary.contains_point(corner_x, corner_y):
                return False
        
        # Check sample points along edges
        samples_per_edge = 5
        for i in range(samples_per_edge):
            t = i / (samples_per_edge - 1) if samples_per_edge > 1 else 0
            
            # Top edge
            if not boundary.contains_point(position.x + t * width, position.y):
                return False
            
            # Bottom edge
            if not boundary.contains_point(position.x + t * width, position.y + height):
                return False
            
            # Left edge
            if not boundary.contains_point(position.x, position.y + t * height):
                return False
            
            # Right edge
            if not boundary.contains_point(position.x + width, position.y + t * height):
                return False
        
        # Check overlap with placed panels
        for placed_panel in placed:
            placed_width = placed_panel.panel.height if placed_panel.rotated else placed_panel.panel.width
            placed_height = placed_panel.panel.width if placed_panel.rotated else placed_panel.panel.height
            
            if not (position.x + width <= placed_panel.position.x or
                   placed_panel.position.x + placed_width <= position.x or
                   position.y + height <= placed_panel.position.y or
                   placed_panel.position.y + placed_height <= position.y):
                return False
        
        return True
    
    def decompose_into_rectangles(self, boundary: IrregularBoundary) -> List[Tuple[float, float, float, float]]:
        """Decompose irregular boundary into rectangles"""
        rectangles = []
        
        # Get bounding box
        min_x = min(v[0] for v in boundary.vertices)
        max_x = max(v[0] for v in boundary.vertices)
        min_y = min(v[1] for v in boundary.vertices)
        max_y = max(v[1] for v in boundary.vertices)
        
        # Grid-based decomposition
        grid_size = 10.0
        
        for y in np.arange(min_y, max_y, grid_size):
            for x in np.arange(min_x, max_x, grid_size):
                # Check if grid cell is fully inside
                cell_inside = True
                
                for dx in [0, grid_size]:
                    for dy in [0, grid_size]:
                        if not boundary.contains_point(x + dx, y + dy):
                            cell_inside = False
                            break
                    if not cell_inside:
                        break
                
                if cell_inside:
                    # Expand rectangle as much as possible
                    width = grid_size
                    height = grid_size
                    
                    # Expand width
                    while x + width < max_x:
                        test_width = width + grid_size
                        all_inside = True
                        
                        for dy in np.arange(0, height + 1, grid_size):
                            if not boundary.contains_point(x + test_width, y + dy):
                                all_inside = False
                                break
                        
                        if all_inside:
                            width = test_width
                        else:
                            break
                    
                    # Expand height
                    while y + height < max_y:
                        test_height = height + grid_size
                        all_inside = True
                        
                        for dx in np.arange(0, width + 1, grid_size):
                            if not boundary.contains_point(x + dx, y + test_height):
                                all_inside = False
                                break
                        
                        if all_inside:
                            height = test_height
                        else:
                            break
                    
                    rectangles.append((x, y, width, height))
        
        # Remove overlapping rectangles
        non_overlapping = []
        for rect in rectangles:
            overlaps = False
            for existing in non_overlapping:
                if self._rectangles_overlap(rect, existing):
                    overlaps = True
                    break
            
            if not overlaps:
                non_overlapping.append(rect)
        
        return non_overlapping
    
    def _rectangles_overlap(self, rect1: Tuple[float, float, float, float],
                          rect2: Tuple[float, float, float, float]) -> bool:
        """Check if two rectangles overlap"""
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2
        
        return not (x1 + w1 <= x2 or x2 + w2 <= x1 or
                   y1 + h1 <= y2 or y2 + h2 <= y1)


class EdgeCaseManager:
    """Main edge case handling system"""
    
    def __init__(self):
        self.shape_handler = ExtremeShapeHandler()
        self.space_solver = MinimalSpaceSolver()
        self.boundary_handler = IrregularBoundaryHandler()
        self.detection_history = []
        self.handling_stats = defaultdict(int)
    
    def detect_edge_cases(self, room: Room, panels: List[Panel],
                         state: Optional[PackingState] = None) -> List[EdgeCaseDetection]:
        """Detect all edge cases in current problem"""
        detections = []
        
        # Check for extreme shapes
        extreme_panels = []
        for panel in panels:
            classification = self.shape_handler.classify_shape(panel)
            if classification != ShapeClassification.NORMAL:
                extreme_panels.append(panel)
        
        if extreme_panels:
            detections.append(EdgeCaseDetection(
                case_type=EdgeCaseType.EXTREME_ASPECT_RATIO,
                severity=len(extreme_panels) / len(panels),
                description=f"{len(extreme_panels)} panels with extreme shapes",
                affected_items=extreme_panels,
                handling_strategy="extreme_shape_handler"
            ))
        
        # Check for minimal space
        if state and self.space_solver.detect_minimal_space(state, room):
            detections.append(EdgeCaseDetection(
                case_type=EdgeCaseType.MINIMAL_SPACE,
                severity=0.8,
                description="Less than 10% space remaining",
                affected_items=[],
                handling_strategy="minimal_space_solver"
            ))
        
        # Check for irregular boundary (if room has obstacles)
        if hasattr(room, 'obstacles') and room.obstacles:
            detections.append(EdgeCaseDetection(
                case_type=EdgeCaseType.OBSTACLE_PRESENCE,
                severity=0.6,
                description=f"Room has {len(room.obstacles)} obstacles",
                affected_items=room.obstacles,
                handling_strategy="irregular_boundary_handler"
            ))
        
        # Check for fragmented space
        if state and self._is_fragmented(state, room):
            detections.append(EdgeCaseDetection(
                case_type=EdgeCaseType.FRAGMENTED_SPACE,
                severity=0.7,
                description="Placement has created fragmented spaces",
                affected_items=[],
                handling_strategy="space_consolidation"
            ))
        
        self.detection_history.extend(detections)
        return detections
    
    def handle_edge_case(self, detection: EdgeCaseDetection,
                        state: PackingState, room: Room,
                        panels: List[Panel]) -> List[PlacedPanel]:
        """Handle specific edge case"""
        self.handling_stats[detection.case_type] += 1
        
        if detection.case_type == EdgeCaseType.EXTREME_ASPECT_RATIO:
            return self._handle_extreme_shapes(detection.affected_items, state, room)
        
        elif detection.case_type == EdgeCaseType.MINIMAL_SPACE:
            return self.space_solver.solve_minimal_space(state, room, panels)
        
        elif detection.case_type == EdgeCaseType.OBSTACLE_PRESENCE:
            # Convert to irregular boundary
            vertices = [
                (0, 0), (room.width, 0),
                (room.width, room.height), (0, room.height)
            ]
            boundary = self.boundary_handler.create_irregular_room(
                vertices, obstacles=detection.affected_items
            )
            return self.boundary_handler.handle_irregular_room(boundary, panels)
        
        elif detection.case_type == EdgeCaseType.FRAGMENTED_SPACE:
            return self._consolidate_fragmented_space(state, room, panels)
        
        return []
    
    def _handle_extreme_shapes(self, panels: List[Panel],
                              state: PackingState, room: Room) -> List[PlacedPanel]:
        """Handle panels with extreme shapes"""
        placed = []
        
        for panel in panels:
            placement = self.shape_handler.handle_extreme_panel(panel, state, room)
            if placement:
                placed.append(placement)
                # Update state for next placement
                state.placed_panels.append(placement)
        
        return placed
    
    def _is_fragmented(self, state: PackingState, room: Room) -> bool:
        """Check if space is fragmented"""
        if len(state.placed_panels) < 5:
            return False
        
        # Calculate fragmentation metric
        gaps = self.shape_handler._find_gaps(state, room)
        
        if len(gaps) > len(state.placed_panels) * 2:
            return True
        
        # Check if gaps are too small to be useful
        avg_panel_area = np.mean([p.panel.width * p.panel.height 
                                  for p in state.placed_panels])
        small_gaps = sum(1 for _, _, w, h in gaps if w * h < avg_panel_area * 0.5)
        
        return small_gaps > len(gaps) * 0.7
    
    def _consolidate_fragmented_space(self, state: PackingState,
                                     room: Room, panels: List[Panel]) -> List[PlacedPanel]:
        """Consolidate fragmented space by reorganizing placement"""
        # This would involve re-arranging existing panels to reduce fragmentation
        # For now, just use minimal space solver
        return self.space_solver.solve_minimal_space(state, room, panels)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get edge case handling statistics"""
        return {
            'detections_by_type': dict(self.handling_stats),
            'total_detections': len(self.detection_history),
            'shape_classifications': dict(self.shape_handler.handling_stats),
            'recent_detections': self.detection_history[-10:] if self.detection_history else []
        }
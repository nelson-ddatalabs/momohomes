#!/usr/bin/env python3
"""
advanced_packing.py - Advanced 2D Bin Packing Algorithms
=========================================================
Production implementation of state-of-the-art packing algorithms
for achieving 95%+ panel coverage.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Set, Dict, Optional, FrozenSet, Any, Protocol, Callable
from enum import Enum
from abc import ABC, abstractmethod
import hashlib
import numpy as np
from collections import namedtuple
import time
import json
from threading import Event

from models import Panel, PanelSize, Point, Room


# Step 1.1.1: State Representation System
# ========================================

@dataclass(frozen=True)
class PanelPlacement:
    """Immutable representation of a single panel placement."""
    panel_size: PanelSize
    position: Tuple[float, float]  # Using tuple for immutability
    orientation: str
    
    def __hash__(self) -> int:
        """Hash for use in sets and as dict keys."""
        return hash((self.panel_size, self.position, self.orientation))
    
    def to_panel(self, room_id: str = "") -> Panel:
        """Convert to Panel object for room placement."""
        return Panel(
            size=self.panel_size,
            position=Point(self.position[0], self.position[1]),
            orientation=self.orientation,
            room_id=room_id
        )
    
    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Get bounding box (x1, y1, x2, y2)."""
        w, h = self.panel_size.get_dimensions(self.orientation)
        x, y = self.position
        return (x, y, x + w, y + h)
    
    def overlaps(self, other: PanelPlacement) -> bool:
        """Check if this placement overlaps with another."""
        x1, y1, x2, y2 = self.bounds
        ox1, oy1, ox2, oy2 = other.bounds
        return not (x2 <= ox1 or ox2 <= x1 or y2 <= oy1 or oy2 <= y1)


@dataclass(frozen=True)
class PackingState:
    """
    Immutable state representing a partial or complete packing solution.
    Used for memoization and backtracking.
    """
    placements: FrozenSet[PanelPlacement]
    room_bounds: Tuple[float, float, float, float]  # (x, y, width, height)
    coverage: float
    
    def __hash__(self) -> int:
        """Fast hash for memoization."""
        # Use placement count and coverage for quick hash
        return hash((len(self.placements), round(self.coverage, 2)))
    
    def __eq__(self, other: Any) -> bool:
        """Equality check for memoization."""
        if not isinstance(other, PackingState):
            return False
        return (self.placements == other.placements and 
                self.room_bounds == other.room_bounds)
    
    @property
    def canonical_hash(self) -> str:
        """
        Canonical hash for exact state comparison.
        Sorted to ensure deterministic hashing.
        """
        sorted_placements = sorted(
            self.placements,
            key=lambda p: (p.position[0], p.position[1], str(p.panel_size))
        )
        state_str = f"{self.room_bounds}|"
        for p in sorted_placements:
            state_str += f"{p.panel_size}:{p.position}:{p.orientation}|"
        return hashlib.md5(state_str.encode()).hexdigest()
    
    def add_placement(self, placement: PanelPlacement) -> PackingState:
        """Create new state with additional placement (immutable)."""
        if not self.is_valid_placement(placement):
            raise ValueError("Invalid placement")
        
        new_placements = self.placements | {placement}
        new_coverage = self._calculate_coverage(new_placements)
        
        return PackingState(
            placements=new_placements,
            room_bounds=self.room_bounds,
            coverage=new_coverage
        )
    
    def remove_placement(self, placement: PanelPlacement) -> PackingState:
        """Create new state without specified placement (immutable)."""
        if placement not in self.placements:
            raise ValueError("Placement not in state")
        
        new_placements = self.placements - {placement}
        new_coverage = self._calculate_coverage(new_placements)
        
        return PackingState(
            placements=new_placements,
            room_bounds=self.room_bounds,
            coverage=new_coverage
        )
    
    def is_valid_placement(self, placement: PanelPlacement) -> bool:
        """Check if placement is valid in current state."""
        # Check bounds
        x1, y1, x2, y2 = placement.bounds
        rx, ry, rw, rh = self.room_bounds
        
        if x1 < rx or y1 < ry or x2 > rx + rw or y2 > ry + rh:
            return False
        
        # Check overlaps
        for existing in self.placements:
            if placement.overlaps(existing):
                return False
        
        return True
    
    def _calculate_coverage(self, placements: FrozenSet[PanelPlacement]) -> float:
        """Calculate coverage ratio for given placements."""
        if not placements:
            return 0.0
        
        total_area = sum(p.panel_size.area for p in placements)
        room_area = self.room_bounds[2] * self.room_bounds[3]
        
        return total_area / room_area if room_area > 0 else 0.0
    
    @classmethod
    def from_room(cls, room: Room) -> PackingState:
        """Create initial empty state from room."""
        return cls(
            placements=frozenset(),
            room_bounds=(room.position.x, room.position.y, room.width, room.height),
            coverage=0.0
        )
    
    def to_panels(self, room_id: str) -> List[Panel]:
        """Convert state to list of Panel objects."""
        return [p.to_panel(room_id) for p in self.placements]


class StateTransition:
    """Manages state transitions for backtracking and exploration."""
    
    def __init__(self):
        self.transition_cache: Dict[Tuple[str, PanelPlacement], Optional[PackingState]] = {}
    
    def apply_placement(self, state: PackingState, placement: PanelPlacement) -> Optional[PackingState]:
        """
        Apply placement to state, returning new state or None if invalid.
        Cached for efficiency.
        """
        cache_key = (state.canonical_hash, placement)
        
        if cache_key in self.transition_cache:
            return self.transition_cache[cache_key]
        
        try:
            if state.is_valid_placement(placement):
                new_state = state.add_placement(placement)
                self.transition_cache[cache_key] = new_state
                return new_state
        except ValueError:
            pass
        
        self.transition_cache[cache_key] = None
        return None
    
    def get_valid_placements(self, state: PackingState, 
                            panel_sizes: List[PanelSize],
                            resolution: float = 1.0) -> List[PanelPlacement]:
        """
        Get all valid placements for given panel sizes in current state.
        Uses grid resolution for position sampling.
        """
        valid_placements = []
        rx, ry, rw, rh = state.room_bounds
        
        for panel_size in panel_sizes:
            for orientation in ["horizontal", "vertical"]:
                pw, ph = panel_size.get_dimensions(orientation)
                
                # Skip if panel doesn't fit in room
                if pw > rw or ph > rh:
                    continue
                
                # Sample positions at resolution intervals
                x_positions = np.arange(rx, rx + rw - pw + 0.01, resolution)
                y_positions = np.arange(ry, ry + rh - ph + 0.01, resolution)
                
                for x in x_positions:
                    for y in y_positions:
                        placement = PanelPlacement(
                            panel_size=panel_size,
                            position=(float(x), float(y)),
                            orientation=orientation
                        )
                        
                        if state.is_valid_placement(placement):
                            valid_placements.append(placement)
        
        return valid_placements
    
    def clear_cache(self):
        """Clear transition cache to free memory."""
        self.transition_cache.clear()


class StateValidator:
    """Validates packing states against constraints."""
    
    @staticmethod
    def check_overlap(state: PackingState) -> bool:
        """Verify no panels overlap."""
        placements = list(state.placements)
        for i in range(len(placements)):
            for j in range(i + 1, len(placements)):
                if placements[i].overlaps(placements[j]):
                    return False
        return True
    
    @staticmethod
    def check_boundaries(state: PackingState) -> bool:
        """Verify all panels are within room boundaries."""
        rx, ry, rw, rh = state.room_bounds
        
        for placement in state.placements:
            x1, y1, x2, y2 = placement.bounds
            if x1 < rx or y1 < ry or x2 > rx + rw or y2 > ry + rh:
                return False
        
        return True
    
    @staticmethod
    def validate_state(state: PackingState) -> bool:
        """Complete state validation."""
        return (StateValidator.check_overlap(state) and 
                StateValidator.check_boundaries(state))
    
    @staticmethod
    def get_violations(state: PackingState) -> List[str]:
        """Get list of constraint violations."""
        violations = []
        
        if not StateValidator.check_overlap(state):
            violations.append("Panel overlap detected")
        
        if not StateValidator.check_boundaries(state):
            violations.append("Panel outside room boundaries")
        
        # Check coverage calculation
        expected_coverage = state._calculate_coverage(state.placements)
        if abs(state.coverage - expected_coverage) > 0.001:
            violations.append(f"Coverage mismatch: {state.coverage} vs {expected_coverage}")
        
        return violations


# Step 1.1.2: Spatial Index Structure
# ====================================

class OccupancyGrid:
    """
    Grid-based spatial index for fast collision detection and free space tracking.
    Uses configurable resolution for memory/speed tradeoff.
    """
    
    def __init__(self, bounds: Tuple[float, float, float, float], resolution: float = 0.5):
        """
        Initialize occupancy grid.
        
        Args:
            bounds: Room bounds (x, y, width, height)
            resolution: Grid cell size in feet (default 0.5ft = 6 inches)
        """
        self.bounds = bounds
        self.resolution = resolution
        
        # Calculate grid dimensions
        self.x_offset, self.y_offset, width, height = bounds
        self.cols = int(np.ceil(width / resolution))
        self.rows = int(np.ceil(height / resolution))
        
        # Initialize grid (0 = free, 1 = occupied)
        self.grid = np.zeros((self.rows, self.cols), dtype=np.uint8)
        
        # Track which panels occupy which cells
        self.cell_to_panels: Dict[Tuple[int, int], Set[PanelPlacement]] = {}
    
    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid indices."""
        col = int((x - self.x_offset) / self.resolution)
        row = int((y - self.y_offset) / self.resolution)
        return (row, col)
    
    def grid_to_world(self, row: int, col: int) -> Tuple[float, float]:
        """Convert grid indices to world coordinates."""
        x = self.x_offset + col * self.resolution
        y = self.y_offset + row * self.resolution
        return (x, y)
    
    def add_panel(self, placement: PanelPlacement) -> bool:
        """
        Add panel to grid, returning False if space is occupied.
        """
        cells = self._get_panel_cells(placement)
        
        # Check if any cell is occupied
        for row, col in cells:
            if self.grid[row, col] > 0:
                return False
        
        # Mark cells as occupied
        for row, col in cells:
            self.grid[row, col] = 1
            if (row, col) not in self.cell_to_panels:
                self.cell_to_panels[(row, col)] = set()
            self.cell_to_panels[(row, col)].add(placement)
        
        return True
    
    def remove_panel(self, placement: PanelPlacement):
        """Remove panel from grid."""
        cells = self._get_panel_cells(placement)
        
        for row, col in cells:
            if (row, col) in self.cell_to_panels:
                self.cell_to_panels[(row, col)].discard(placement)
                if not self.cell_to_panels[(row, col)]:
                    self.grid[row, col] = 0
                    del self.cell_to_panels[(row, col)]
    
    def can_place_panel(self, placement: PanelPlacement) -> bool:
        """Check if panel can be placed without collision."""
        cells = self._get_panel_cells(placement)
        
        for row, col in cells:
            if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
                return False
            if self.grid[row, col] > 0:
                return False
        
        return True
    
    def is_region_free(self, bounds: Tuple[float, float, float, float]) -> bool:
        """
        Check if a rectangular region is free of panels.
        
        Args:
            bounds: Region bounds as (x1, y1, x2, y2)
            
        Returns:
            True if region is completely free
        """
        x1, y1, x2, y2 = bounds
        
        # Convert to grid coordinates
        r1, c1 = self.world_to_grid(x1, y1)
        r2, c2 = self.world_to_grid(x2 - 0.01, y2 - 0.01)  # Slightly inside bounds
        
        # Check bounds
        if r1 < 0 or c1 < 0 or r2 >= self.rows or c2 >= self.cols:
            return False
        
        # Clamp to grid bounds
        r1 = max(0, min(r1, self.rows - 1))
        r2 = max(0, min(r2, self.rows - 1))
        c1 = max(0, min(c1, self.cols - 1))
        c2 = max(0, min(c2, self.cols - 1))
        
        # Check all cells in region
        for row in range(r1, r2 + 1):
            for col in range(c1, c2 + 1):
                if self.grid[row, col] > 0:
                    return False
        
        return True
    
    def mark_occupied(self, bounds: Tuple[float, float, float, float]):
        """
        Mark a rectangular region as occupied.
        
        Args:
            bounds: Region bounds as (x1, y1, x2, y2)
        """
        x1, y1, x2, y2 = bounds
        
        # Convert to grid coordinates
        r1, c1 = self.world_to_grid(x1, y1)
        r2, c2 = self.world_to_grid(x2 - 0.01, y2 - 0.01)  # Slightly inside bounds
        
        # Clamp to grid bounds
        r1 = max(0, min(r1, self.rows - 1))
        r2 = max(0, min(r2, self.rows - 1))
        c1 = max(0, min(c1, self.cols - 1))
        c2 = max(0, min(c2, self.cols - 1))
        
        # Mark all cells in region as occupied
        for row in range(r1, r2 + 1):
            for col in range(c1, c2 + 1):
                self.grid[row, col] = 1
    
    def _get_panel_cells(self, placement: PanelPlacement) -> List[Tuple[int, int]]:
        """Get all grid cells occupied by a panel."""
        x1, y1, x2, y2 = placement.bounds
        
        # Convert to grid coordinates
        r1, c1 = self.world_to_grid(x1, y1)
        r2, c2 = self.world_to_grid(x2 - 0.01, y2 - 0.01)  # Slightly inside bounds
        
        # Clamp to grid bounds
        r1 = max(0, min(r1, self.rows - 1))
        r2 = max(0, min(r2, self.rows - 1))
        c1 = max(0, min(c1, self.cols - 1))
        c2 = max(0, min(c2, self.cols - 1))
        
        # Get all cells in rectangle
        cells = []
        for row in range(r1, r2 + 1):
            for col in range(c1, c2 + 1):
                cells.append((row, col))
        
        return cells
    
    def get_free_cells(self) -> List[Tuple[int, int]]:
        """Get list of all free cells."""
        free = []
        for row in range(self.rows):
            for col in range(self.cols):
                if self.grid[row, col] == 0:
                    free.append((row, col))
        return free
    
    def get_coverage_ratio(self) -> float:
        """Get ratio of occupied cells."""
        occupied = np.sum(self.grid > 0)
        total = self.rows * self.cols
        return occupied / total if total > 0 else 0.0


class MaximalRectangle:
    """Represents a maximal empty rectangle in the packing space."""
    
    def __init__(self, x: float, y: float, width: float, height: float):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.x + self.width, self.y + self.height)
    
    def can_fit(self, panel_width: float, panel_height: float) -> bool:
        """Check if panel dimensions can fit in this rectangle."""
        return (self.width >= panel_width and self.height >= panel_height) or \
               (self.width >= panel_height and self.height >= panel_width)
    
    def overlaps(self, other: MaximalRectangle) -> bool:
        """Check if this rectangle overlaps with another."""
        x1, y1, x2, y2 = self.bounds
        ox1, oy1, ox2, oy2 = other.bounds
        return not (x2 <= ox1 or ox2 <= x1 or y2 <= oy1 or oy2 <= y1)


class MaximalRectangleTracker:
    """
    Tracks maximal empty rectangles for efficient free space management.
    Used by advanced packing algorithms.
    """
    
    def __init__(self, bounds: Tuple[float, float, float, float]):
        """Initialize with room bounds."""
        x, y, width, height = bounds
        self.bounds = bounds
        
        # Start with entire room as one maximal rectangle
        self.rectangles = [MaximalRectangle(x, y, width, height)]
    
    def add_panel(self, placement: PanelPlacement):
        """
        Update maximal rectangles after adding a panel.
        Uses guillotine cutting strategy.
        """
        x1, y1, x2, y2 = placement.bounds
        new_rectangles = []
        
        for rect in self.rectangles:
            # Check if panel intersects this rectangle
            if self._rectangles_overlap(rect, x1, y1, x2, y2):
                # Split rectangle around panel (guillotine cuts)
                splits = self._split_rectangle(rect, x1, y1, x2, y2)
                new_rectangles.extend(splits)
            else:
                # Keep rectangle unchanged
                new_rectangles.append(rect)
        
        # Remove redundant rectangles
        self.rectangles = self._remove_redundant(new_rectangles)
    
    def _rectangles_overlap(self, rect: MaximalRectangle, 
                           x1: float, y1: float, x2: float, y2: float) -> bool:
        """Check if rectangle overlaps with bounds."""
        rx1, ry1, rx2, ry2 = rect.bounds
        return not (rx2 <= x1 or x2 <= rx1 or ry2 <= y1 or y2 <= ry1)
    
    def _split_rectangle(self, rect: MaximalRectangle,
                        px1: float, py1: float, px2: float, py2: float) -> List[MaximalRectangle]:
        """Split rectangle around panel bounds."""
        splits = []
        rx, ry = rect.x, rect.y
        rw, rh = rect.width, rect.height
        
        # Left split
        if px1 > rx:
            splits.append(MaximalRectangle(rx, ry, px1 - rx, rh))
        
        # Right split
        if px2 < rx + rw:
            splits.append(MaximalRectangle(px2, ry, rx + rw - px2, rh))
        
        # Top split
        if py1 > ry:
            splits.append(MaximalRectangle(rx, ry, rw, py1 - ry))
        
        # Bottom split
        if py2 < ry + rh:
            splits.append(MaximalRectangle(rx, py2, rw, ry + rh - py2))
        
        return splits
    
    def _remove_redundant(self, rectangles: List[MaximalRectangle]) -> List[MaximalRectangle]:
        """Remove rectangles that are contained within others."""
        result = []
        
        for i, rect1 in enumerate(rectangles):
            is_redundant = False
            
            for j, rect2 in enumerate(rectangles):
                if i != j:
                    # Check if rect1 is contained in rect2
                    if (rect2.x <= rect1.x and 
                        rect2.y <= rect1.y and
                        rect2.x + rect2.width >= rect1.x + rect1.width and
                        rect2.y + rect2.height >= rect1.y + rect1.height):
                        is_redundant = True
                        break
            
            if not is_redundant:
                result.append(rect1)
        
        return result
    
    def get_best_fit(self, panel_width: float, panel_height: float) -> Optional[MaximalRectangle]:
        """Find rectangle with best fit for panel dimensions."""
        candidates = [r for r in self.rectangles if r.can_fit(panel_width, panel_height)]
        
        if not candidates:
            return None
        
        # Best fit: minimize wasted area
        def waste_score(rect: MaximalRectangle) -> float:
            return rect.area - (panel_width * panel_height)
        
        return min(candidates, key=waste_score)
    
    def get_bottom_left(self) -> Optional[MaximalRectangle]:
        """Get bottom-left-most rectangle."""
        if not self.rectangles:
            return None
        
        # Sort by y first (bottom), then x (left)
        return min(self.rectangles, key=lambda r: (r.y, r.x))


class SpatialIndex:
    """
    Unified spatial index combining grid and maximal rectangles.
    Provides efficient queries for packing algorithms.
    """
    
    def __init__(self, bounds: Tuple[float, float, float, float], resolution: float = 0.5):
        """Initialize spatial index."""
        self.bounds = bounds
        self.occupancy_grid = OccupancyGrid(bounds, resolution)
        self.rectangle_tracker = MaximalRectangleTracker(bounds)
        self.placements: Set[PanelPlacement] = set()
    
    def add_placement(self, placement: PanelPlacement) -> bool:
        """Add placement to spatial index."""
        if placement in self.placements:
            return False
        
        if not self.occupancy_grid.add_panel(placement):
            return False
        
        self.rectangle_tracker.add_panel(placement)
        self.placements.add(placement)
        return True
    
    def remove_placement(self, placement: PanelPlacement):
        """Remove placement from spatial index."""
        if placement not in self.placements:
            return
        
        self.occupancy_grid.remove_panel(placement)
        self.placements.remove(placement)
        
        # Rebuild rectangle tracker (simpler than tracking removals)
        self.rectangle_tracker = MaximalRectangleTracker(self.bounds)
        for p in self.placements:
            self.rectangle_tracker.add_panel(p)
    
    def can_place(self, placement: PanelPlacement) -> bool:
        """Quick check if placement is valid."""
        return self.occupancy_grid.can_place_panel(placement)
    
    def get_free_positions(self, panel_size: PanelSize, 
                          orientation: str) -> List[Tuple[float, float]]:
        """Get valid positions for panel placement."""
        positions = []
        pw, ph = panel_size.get_dimensions(orientation)
        
        # Use maximal rectangles for candidate positions
        for rect in self.rectangle_tracker.rectangles:
            if rect.can_fit(pw, ph):
                # Bottom-left corner of rectangle
                positions.append((rect.x, rect.y))
                
                # Try other corners if they create different placements
                if rect.width > pw:
                    positions.append((rect.x + rect.width - pw, rect.y))
                if rect.height > ph:
                    positions.append((rect.x, rect.y + rect.height - ph))
        
        return positions
    
    def get_coverage(self) -> float:
        """Get coverage ratio."""
        return self.occupancy_grid.get_coverage_ratio()


# Step 1.1.3: Decision Tree Structure
# ===================================

import heapq
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import time


@dataclass
class SearchNode:
    """
    Node in the search tree for backtracking and branch & bound.
    """
    state: PackingState
    parent: Optional['SearchNode']
    action: Optional[PanelPlacement]  # Action that led to this state
    depth: int
    bound: float  # Upper bound on achievable coverage
    cost: float  # Cost so far
    
    # Metadata for pruning and search guidance
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self) -> int:
        return hash(self.state)
    
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, SearchNode):
            return False
        return self.state == other.state
    
    def __lt__(self, other: 'SearchNode') -> bool:
        """For priority queue ordering (higher bound = higher priority)."""
        return self.bound > other.bound
    
    @property
    def coverage(self) -> float:
        return self.state.coverage
    
    def get_path(self) -> List[PanelPlacement]:
        """Get sequence of actions from root to this node."""
        path = []
        node = self
        while node.parent is not None:
            if node.action is not None:
                path.append(node.action)
            node = node.parent
        return list(reversed(path))
    
    def is_terminal(self, target_coverage: float = 0.95) -> bool:
        """Check if this is a terminal node."""
        return (self.coverage >= target_coverage or 
                self.metadata.get('no_more_placements', False))


class PriorityQueue:
    """
    Priority queue for best-first search strategies.
    Supports different priority functions.
    """
    
    def __init__(self, priority_fn=None):
        """
        Initialize priority queue.
        
        Args:
            priority_fn: Function to compute priority (lower = higher priority)
                        Default uses node.bound
        """
        self.heap = []
        self.counter = 0  # Tie-breaker for equal priorities
        self.priority_fn = priority_fn or (lambda node: -node.bound)
        self.seen_states = set()  # Track visited states
    
    def push(self, node: SearchNode):
        """Add node to queue."""
        if node.state.canonical_hash in self.seen_states:
            return  # Skip duplicate states
        
        priority = self.priority_fn(node)
        entry = (priority, self.counter, node)
        heapq.heappush(self.heap, entry)
        self.counter += 1
        self.seen_states.add(node.state.canonical_hash)
    
    def pop(self) -> Optional[SearchNode]:
        """Remove and return highest priority node."""
        while self.heap:
            priority, count, node = heapq.heappop(self.heap)
            return node
        return None
    
    def is_empty(self) -> bool:
        return len(self.heap) == 0
    
    def size(self) -> int:
        return len(self.heap)
    
    def clear(self):
        """Clear the queue."""
        self.heap.clear()
        self.seen_states.clear()
        self.counter = 0


class BranchRegistry:
    """
    Tracks branching decisions and alternatives for backtracking.
    """
    
    def __init__(self):
        self.branches: Dict[str, List[SearchNode]] = {}
        self.decision_points: List[SearchNode] = []
        self.explored_branches: Set[str] = set()
    
    def register_branch(self, node: SearchNode, alternatives: List[SearchNode]):
        """
        Register a branching point with alternatives.
        """
        branch_id = f"{node.state.canonical_hash}_{node.depth}"
        self.branches[branch_id] = alternatives
        self.decision_points.append(node)
    
    def get_alternatives(self, node: SearchNode) -> List[SearchNode]:
        """Get unexplored alternatives for a node."""
        branch_id = f"{node.state.canonical_hash}_{node.depth}"
        
        if branch_id not in self.branches:
            return []
        
        alternatives = []
        for alt in self.branches[branch_id]:
            alt_id = f"{alt.state.canonical_hash}_{alt.depth}"
            if alt_id not in self.explored_branches:
                alternatives.append(alt)
        
        return alternatives
    
    def mark_explored(self, node: SearchNode):
        """Mark a branch as explored."""
        branch_id = f"{node.state.canonical_hash}_{node.depth}"
        self.explored_branches.add(branch_id)
    
    def get_backtrack_point(self, current_depth: int) -> Optional[SearchNode]:
        """
        Find best backtrack point based on current depth.
        """
        candidates = [
            dp for dp in self.decision_points 
            if dp.depth < current_depth and self.get_alternatives(dp)
        ]
        
        if not candidates:
            return None
        
        # Choose point with best potential (highest bound)
        return max(candidates, key=lambda n: n.bound)
    
    def clear(self):
        """Clear registry."""
        self.branches.clear()
        self.decision_points.clear()
        self.explored_branches.clear()


@dataclass
class PruningMetadata:
    """
    Metadata for pruning decisions in search algorithms.
    """
    # Bounds
    upper_bound: float  # Maximum achievable coverage
    lower_bound: float  # Minimum guaranteed coverage
    
    # Dominance
    dominated_by: Optional[str] = None  # State that dominates this one
    dominates: List[str] = field(default_factory=list)  # States dominated by this
    
    # Pruning reasons
    pruned: bool = False
    prune_reason: Optional[str] = None
    
    # Search statistics
    expansions: int = 0
    backtracks: int = 0
    
    # Time tracking
    created_time: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    
    def update_access_time(self):
        """Update last accessed time."""
        self.last_accessed = time.time()
    
    def should_prune(self, best_solution: float, time_limit: float = None) -> bool:
        """
        Determine if this branch should be pruned.
        
        Args:
            best_solution: Current best solution coverage
            time_limit: Optional time limit for this branch
        """
        # Bound pruning
        if self.upper_bound <= best_solution:
            self.pruned = True
            self.prune_reason = "bound"
            return True
        
        # Dominance pruning
        if self.dominated_by is not None:
            self.pruned = True
            self.prune_reason = "dominated"
            return True
        
        # Time pruning
        if time_limit and (time.time() - self.created_time) > time_limit:
            self.pruned = True
            self.prune_reason = "timeout"
            return True
        
        return False


class SearchTree:
    """
    Manages the search tree for optimization algorithms.
    """
    
    def __init__(self, root_state: PackingState):
        """Initialize search tree with root state."""
        self.root = SearchNode(
            state=root_state,
            parent=None,
            action=None,
            depth=0,
            bound=1.0,  # Maximum possible coverage
            cost=0.0
        )
        
        self.nodes: Dict[str, SearchNode] = {root_state.canonical_hash: self.root}
        self.pruning_data: Dict[str, PruningMetadata] = {}
        self.best_solution: Optional[SearchNode] = None
        self.node_count = 1
    
    def add_node(self, parent: SearchNode, action: PanelPlacement, 
                 new_state: PackingState, bound: float) -> SearchNode:
        """Add new node to tree."""
        node = SearchNode(
            state=new_state,
            parent=parent,
            action=action,
            depth=parent.depth + 1,
            bound=bound,
            cost=parent.cost + action.panel_size.area * action.panel_size.cost_factor
        )
        
        state_hash = new_state.canonical_hash
        self.nodes[state_hash] = node
        
        # Initialize pruning metadata
        self.pruning_data[state_hash] = PruningMetadata(
            upper_bound=bound,
            lower_bound=new_state.coverage
        )
        
        self.node_count += 1
        
        # Update best solution if needed
        if self.best_solution is None or node.coverage > self.best_solution.coverage:
            self.best_solution = node
        
        return node
    
    def prune_dominated(self):
        """Prune dominated nodes from tree."""
        pruned_count = 0
        
        for hash1, node1 in list(self.nodes.items()):
            if hash1 not in self.pruning_data or self.pruning_data[hash1].pruned:
                continue
            
            for hash2, node2 in self.nodes.items():
                if hash1 == hash2 or self.pruning_data[hash2].pruned:
                    continue
                
                # Check if node2 dominates node1
                if (node2.coverage >= node1.coverage and 
                    node2.cost <= node1.cost and
                    node2.depth <= node1.depth):
                    
                    self.pruning_data[hash1].dominated_by = hash2
                    self.pruning_data[hash1].pruned = True
                    self.pruning_data[hash1].prune_reason = "dominated"
                    
                    if hash2 in self.pruning_data:
                        self.pruning_data[hash2].dominates.append(hash1)
                    
                    pruned_count += 1
                    break
        
        return pruned_count
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get tree statistics."""
        active_nodes = sum(1 for pd in self.pruning_data.values() if not pd.pruned)
        
        return {
            'total_nodes': self.node_count,
            'active_nodes': active_nodes,
            'pruned_nodes': self.node_count - active_nodes,
            'best_coverage': self.best_solution.coverage if self.best_solution else 0,
            'max_depth': max((n.depth for n in self.nodes.values()), default=0)
        }


# Step 1.2.1: Abstract Algorithm Interface
# =========================================

@dataclass
class OptimizerConfig:
    """Configuration for optimization algorithms."""
    # Time limits
    max_time_seconds: float = 5.0
    early_stop_coverage: float = 0.95  # Stop if this coverage is achieved
    
    # Algorithm-specific parameters
    max_iterations: int = 10000
    max_depth: int = 100
    beam_width: int = 10
    
    # Panel selection
    panel_sizes: List[PanelSize] = field(default_factory=lambda: [
        PanelSize.PANEL_6X8,
        PanelSize.PANEL_6X6,
        PanelSize.PANEL_4X6,
        PanelSize.PANEL_4X4
    ])
    prefer_larger_panels: bool = True
    
    # Search parameters
    grid_resolution: float = 0.5  # Position sampling resolution
    backtrack_threshold: float = 0.01  # Minimum improvement to avoid backtracking
    lookahead_depth: int = 3
    
    # Memoization and caching
    enable_memoization: bool = True
    cache_size_limit: int = 10000
    
    # Heuristics
    edge_alignment_weight: float = 1.0
    corner_preference_weight: float = 1.2
    waste_minimization_weight: float = 0.8
    
    # Parallelization
    enable_parallel: bool = False
    num_threads: int = 4
    
    @classmethod
    def from_json(cls, json_str: str) -> 'OptimizerConfig':
        """Create config from JSON string."""
        data = json.loads(json_str)
        # Convert panel size strings to enums
        if 'panel_sizes' in data:
            data['panel_sizes'] = [PanelSize[ps] for ps in data['panel_sizes']]
        return cls(**data)
    
    def to_json(self) -> str:
        """Convert config to JSON string."""
        data = self.__dict__.copy()
        # Convert panel sizes to strings
        data['panel_sizes'] = [ps.name for ps in data['panel_sizes']]
        return json.dumps(data, indent=2)
    
    def get_timeout_event(self) -> Event:
        """Create a timeout event for this config."""
        event = Event()
        # Set timer in separate thread if needed
        return event


@dataclass
class OptimizationMetrics:
    """Metrics collected during optimization."""
    # Performance metrics
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    iterations: int = 0
    nodes_explored: int = 0
    nodes_pruned: int = 0
    
    # Solution quality
    best_coverage: float = 0.0
    best_cost: float = 0.0
    panels_used: int = 0
    
    # Coverage progression
    coverage_history: List[Tuple[float, float]] = field(default_factory=list)  # (time, coverage)
    
    # Search statistics
    backtracks: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    max_depth_reached: int = 0
    
    # Memory usage
    peak_memory_mb: float = 0.0
    states_in_memory: int = 0
    
    def record_coverage(self, coverage: float):
        """Record coverage at current time."""
        current_time = time.time() - self.start_time
        self.coverage_history.append((current_time, coverage))
        if coverage > self.best_coverage:
            self.best_coverage = coverage
    
    def record_iteration(self):
        """Increment iteration counter."""
        self.iterations += 1
    
    def record_node_explored(self):
        """Increment nodes explored counter."""
        self.nodes_explored += 1
    
    def record_node_pruned(self):
        """Increment nodes pruned counter."""
        self.nodes_pruned += 1
    
    def record_backtrack(self):
        """Increment backtrack counter."""
        self.backtracks += 1
    
    def record_cache_hit(self):
        """Increment cache hit counter."""
        self.cache_hits += 1
    
    def record_cache_miss(self):
        """Increment cache miss counter."""
        self.cache_misses += 1
    
    def finalize(self):
        """Mark optimization as complete."""
        self.end_time = time.time()
    
    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time
    
    @property
    def cache_hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0
    
    @property
    def prune_rate(self) -> float:
        """Get node pruning rate."""
        total = self.nodes_explored + self.nodes_pruned
        return self.nodes_pruned / total if total > 0 else 0.0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        return {
            'elapsed_time': self.elapsed_time,
            'iterations': self.iterations,
            'best_coverage': self.best_coverage,
            'panels_used': self.panels_used,
            'nodes_explored': self.nodes_explored,
            'prune_rate': self.prune_rate,
            'cache_hit_rate': self.cache_hit_rate,
            'backtracks': self.backtracks,
            'max_depth': self.max_depth_reached
        }


class TimeoutHandler:
    """Manages timeout checking for optimization algorithms."""
    
    def __init__(self, max_seconds: float):
        """Initialize timeout handler."""
        self.max_seconds = max_seconds
        self.start_time = time.time()
        self.check_interval = 100  # Check every N iterations
        self.iteration_count = 0
    
    def should_check(self) -> bool:
        """Check if we should verify timeout (to avoid excessive time checks)."""
        self.iteration_count += 1
        return self.iteration_count % self.check_interval == 0
    
    def is_timeout(self) -> bool:
        """Check if timeout has been reached."""
        if not self.should_check():
            return False
        return (time.time() - self.start_time) >= self.max_seconds
    
    def remaining_time(self) -> float:
        """Get remaining time in seconds."""
        elapsed = time.time() - self.start_time
        return max(0, self.max_seconds - elapsed)
    
    def reset(self):
        """Reset timeout handler."""
        self.start_time = time.time()
        self.iteration_count = 0


class AbstractOptimizer(ABC):
    """
    Abstract base class for all optimization algorithms.
    Provides common interface and utilities.
    """
    
    def __init__(self, config: Optional[OptimizerConfig] = None):
        """Initialize optimizer with configuration."""
        self.config = config or OptimizerConfig()
        self.metrics = OptimizationMetrics()
        self.timeout_handler = TimeoutHandler(self.config.max_time_seconds)
        
        # Caching infrastructure
        self.state_cache: Dict[str, PackingState] = {}
        self.transition_manager = StateTransition()
        
        # Best solution tracking
        self.best_state: Optional[PackingState] = None
    
    @abstractmethod
    def optimize_room(self, room: Room) -> List[Panel]:
        """
        Optimize panel placement for a single room.
        Must be implemented by concrete algorithms.
        """
        pass
    
    @abstractmethod
    def get_algorithm_name(self) -> str:
        """Get name of the algorithm for logging."""
        pass
    
    def optimize(self, room: Room) -> Tuple[List[Panel], OptimizationMetrics]:
        """
        Main optimization entry point with metrics collection.
        
        Returns:
            Tuple of (panels, metrics)
        """
        # Reset for new optimization
        self.metrics = OptimizationMetrics()
        self.timeout_handler.reset()
        self.best_state = None
        self.state_cache.clear()
        self.transition_manager.clear_cache()
        
        try:
            # Run optimization
            panels = self.optimize_room(room)
            
            # Update final metrics
            self.metrics.panels_used = len(panels)
            if panels:
                total_area = sum(p.size.area for p in panels)
                room_area = room.width * room.height
                self.metrics.best_coverage = total_area / room_area
            
        except Exception as e:
            print(f"Optimization error in {self.get_algorithm_name()}: {e}")
            panels = []
        
        finally:
            self.metrics.finalize()
        
        return panels, self.metrics
    
    def check_timeout(self) -> bool:
        """Check if timeout has been reached."""
        return self.timeout_handler.is_timeout()
    
    def check_early_stop(self, coverage: float) -> bool:
        """Check if early stop condition is met."""
        return coverage >= self.config.early_stop_coverage
    
    def update_best_state(self, state: PackingState):
        """Update best state if current is better."""
        if self.best_state is None or state.coverage > self.best_state.coverage:
            self.best_state = state
            self.metrics.record_coverage(state.coverage)
    
    def cache_state(self, state: PackingState):
        """Add state to cache if enabled."""
        if not self.config.enable_memoization:
            return
        
        # Limit cache size
        if len(self.state_cache) >= self.config.cache_size_limit:
            # Remove oldest entries (simple FIFO for now)
            keys_to_remove = list(self.state_cache.keys())[:100]
            for key in keys_to_remove:
                del self.state_cache[key]
        
        self.state_cache[state.canonical_hash] = state
    
    def get_cached_state(self, state_hash: str) -> Optional[PackingState]:
        """Retrieve state from cache."""
        if not self.config.enable_memoization:
            return None
        
        if state_hash in self.state_cache:
            self.metrics.record_cache_hit()
            return self.state_cache[state_hash]
        
        self.metrics.record_cache_miss()
        return None
    
    def get_valid_placements(self, state: PackingState) -> List[PanelPlacement]:
        """Get valid placements for current state using config parameters."""
        return self.transition_manager.get_valid_placements(
            state,
            self.config.panel_sizes,
            self.config.grid_resolution
        )
    
    def evaluate_placement(self, placement: PanelPlacement, state: PackingState) -> float:
        """
        Evaluate placement quality using configured heuristics.
        Higher score is better.
        """
        score = 0.0
        x, y = placement.position
        w, h = placement.panel_size.get_dimensions(placement.orientation)
        rx, ry, rw, rh = state.room_bounds
        
        # Edge alignment bonus
        if self.config.edge_alignment_weight > 0:
            edge_score = 0
            if abs(x - rx) < 0.1:  # Left edge
                edge_score += 1
            if abs(y - ry) < 0.1:  # Top edge
                edge_score += 1
            if abs(x + w - (rx + rw)) < 0.1:  # Right edge
                edge_score += 1
            if abs(y + h - (ry + rh)) < 0.1:  # Bottom edge
                edge_score += 1
            
            score += edge_score * self.config.edge_alignment_weight
        
        # Corner preference
        if self.config.corner_preference_weight > 0:
            corner_score = 0
            # Check if placement is in a corner
            if (abs(x - rx) < 0.1 and abs(y - ry) < 0.1):  # Top-left
                corner_score = 1
            elif (abs(x + w - (rx + rw)) < 0.1 and abs(y - ry) < 0.1):  # Top-right
                corner_score = 1
            elif (abs(x - rx) < 0.1 and abs(y + h - (ry + rh)) < 0.1):  # Bottom-left
                corner_score = 1
            elif (abs(x + w - (rx + rw)) < 0.1 and abs(y + h - (ry + rh)) < 0.1):  # Bottom-right
                corner_score = 1
            
            score += corner_score * self.config.corner_preference_weight
        
        # Waste minimization (prefer larger panels)
        if self.config.waste_minimization_weight > 0:
            area_score = placement.panel_size.area / 48.0  # Normalize by max panel area
            score += area_score * self.config.waste_minimization_weight
        
        return score
    
    def get_report(self) -> str:
        """Generate optimization report."""
        metrics = self.metrics.get_summary()
        
        report = f"\n{self.get_algorithm_name()} Optimization Report\n"
        report += "=" * 50 + "\n"
        report += f"Time: {metrics['elapsed_time']:.2f}s\n"
        report += f"Coverage: {metrics['best_coverage']*100:.1f}%\n"
        report += f"Panels: {metrics['panels_used']}\n"
        report += f"Iterations: {metrics['iterations']}\n"
        report += f"Nodes Explored: {metrics['nodes_explored']}\n"
        report += f"Prune Rate: {metrics['prune_rate']*100:.1f}%\n"
        report += f"Cache Hit Rate: {metrics['cache_hit_rate']*100:.1f}%\n"
        
        return report


class OptimizerProtocol(Protocol):
    """Protocol defining the optimizer interface for type checking."""
    
    def optimize(self, room: Room) -> Tuple[List[Panel], OptimizationMetrics]:
        """Optimize room and return panels with metrics."""
        ...
    
    def get_algorithm_name(self) -> str:
        """Get algorithm name."""
        ...
    
    def get_report(self) -> str:
        """Get optimization report."""
        ...


# Step 1.2.2: Strategy Selector Module
# =====================================

@dataclass
class RoomCharacteristics:
    """Analyzed characteristics of a room for strategy selection."""
    room_id: str
    width: float
    height: float
    area: float
    aspect_ratio: float
    is_square: bool
    is_narrow: bool
    is_large: bool
    complexity_score: float
    
    @classmethod
    def analyze(cls, room: Room) -> 'RoomCharacteristics':
        """Analyze room and extract characteristics."""
        area = room.width * room.height
        aspect_ratio = max(room.width, room.height) / min(room.width, room.height)
        
        # Classification thresholds
        is_square = aspect_ratio < 1.2
        is_narrow = aspect_ratio > 2.5
        is_large = area > 200  # sq ft
        
        # Complexity score based on various factors
        complexity_score = 0.0
        
        # Aspect ratio complexity
        if aspect_ratio > 3:
            complexity_score += 2.0
        elif aspect_ratio > 2:
            complexity_score += 1.0
        
        # Size complexity
        if area < 50:  # Very small
            complexity_score += 2.0
        elif area > 400:  # Very large
            complexity_score += 1.5
        
        # Shape complexity
        if room.width < 6 or room.height < 6:  # Narrow dimension
            complexity_score += 1.5
        
        # Non-standard dimensions
        if room.width % 1 != 0 or room.height % 1 != 0:
            complexity_score += 0.5
        
        return cls(
            room_id=room.id,
            width=room.width,
            height=room.height,
            area=area,
            aspect_ratio=aspect_ratio,
            is_square=is_square,
            is_narrow=is_narrow,
            is_large=is_large,
            complexity_score=complexity_score
        )


class StrategyType(Enum):
    """Available optimization strategies."""
    GREEDY = "greedy"
    BOTTOM_LEFT_FILL = "blf"
    DYNAMIC_PROGRAMMING = "dp"
    BRANCH_AND_BOUND = "branch_bound"
    HYBRID = "hybrid"
    MAXIMAL_RECTANGLES = "maximal_rect"


@dataclass
class StrategyRule:
    """Rule for selecting optimization strategy based on room characteristics."""
    name: str
    condition: Callable[[RoomCharacteristics], bool]
    strategy: StrategyType
    priority: int  # Higher priority rules are checked first
    confidence: float  # Confidence in this rule (0-1)
    
    def matches(self, characteristics: RoomCharacteristics) -> bool:
        """Check if rule matches room characteristics."""
        return self.condition(characteristics)


class StrategyRulesMatrix:
    """
    Matrix of rules for selecting optimization strategies.
    Rules are evaluated in priority order.
    """
    
    def __init__(self):
        self.rules: List[StrategyRule] = []
        self._initialize_rules()
    
    def _initialize_rules(self):
        """Initialize default rule set."""
        
        # Small square rooms - use DP for optimal solution
        self.add_rule(
            name="small_square_dp",
            condition=lambda c: c.is_square and c.area < 100,
            strategy=StrategyType.DYNAMIC_PROGRAMMING,
            priority=10,
            confidence=0.95
        )
        
        # Large rooms - use BLF for speed
        self.add_rule(
            name="large_room_blf",
            condition=lambda c: c.area > 300,
            strategy=StrategyType.BOTTOM_LEFT_FILL,
            priority=8,
            confidence=0.85
        )
        
        # Narrow corridors - use greedy
        self.add_rule(
            name="narrow_corridor",
            condition=lambda c: c.is_narrow and c.aspect_ratio > 3,
            strategy=StrategyType.GREEDY,
            priority=9,
            confidence=0.90
        )
        
        # Complex rooms - use branch and bound
        self.add_rule(
            name="complex_room",
            condition=lambda c: c.complexity_score > 3,
            strategy=StrategyType.BRANCH_AND_BOUND,
            priority=7,
            confidence=0.80
        )
        
        # Medium square rooms - use maximal rectangles
        self.add_rule(
            name="medium_square",
            condition=lambda c: c.is_square and 100 <= c.area <= 200,
            strategy=StrategyType.MAXIMAL_RECTANGLES,
            priority=6,
            confidence=0.85
        )
        
        # Default fallback - hybrid approach
        self.add_rule(
            name="default_hybrid",
            condition=lambda c: True,  # Always matches
            strategy=StrategyType.HYBRID,
            priority=0,
            confidence=0.70
        )
    
    def add_rule(self, name: str, condition: Callable, strategy: StrategyType, 
                 priority: int, confidence: float):
        """Add a new rule to the matrix."""
        rule = StrategyRule(name, condition, strategy, priority, confidence)
        self.rules.append(rule)
        # Keep rules sorted by priority
        self.rules.sort(key=lambda r: r.priority, reverse=True)
    
    def select_strategy(self, characteristics: RoomCharacteristics) -> Tuple[StrategyType, float]:
        """
        Select best strategy for room characteristics.
        Returns (strategy, confidence).
        """
        for rule in self.rules:
            if rule.matches(characteristics):
                return rule.strategy, rule.confidence
        
        # Should never reach here due to default rule
        return StrategyType.HYBRID, 0.5
    
    def get_matching_rules(self, characteristics: RoomCharacteristics) -> List[StrategyRule]:
        """Get all rules that match the characteristics."""
        return [rule for rule in self.rules if rule.matches(characteristics)]


class StrategyFallbackChain:
    """
    Manages fallback chain when primary strategy fails or underperforms.
    """
    
    def __init__(self):
        # Default fallback chain
        self.chains = {
            StrategyType.DYNAMIC_PROGRAMMING: [
                StrategyType.BRANCH_AND_BOUND,
                StrategyType.MAXIMAL_RECTANGLES,
                StrategyType.BOTTOM_LEFT_FILL
            ],
            StrategyType.BRANCH_AND_BOUND: [
                StrategyType.DYNAMIC_PROGRAMMING,
                StrategyType.MAXIMAL_RECTANGLES,
                StrategyType.BOTTOM_LEFT_FILL
            ],
            StrategyType.BOTTOM_LEFT_FILL: [
                StrategyType.MAXIMAL_RECTANGLES,
                StrategyType.GREEDY
            ],
            StrategyType.MAXIMAL_RECTANGLES: [
                StrategyType.BOTTOM_LEFT_FILL,
                StrategyType.GREEDY
            ],
            StrategyType.GREEDY: [
                StrategyType.BOTTOM_LEFT_FILL
            ],
            StrategyType.HYBRID: []  # No fallback for hybrid
        }
        
        # Track failed strategies to avoid retrying
        self.failed_strategies: Set[StrategyType] = set()
    
    def get_fallback(self, strategy: StrategyType) -> Optional[StrategyType]:
        """Get next fallback strategy."""
        if strategy not in self.chains:
            return None
        
        for fallback in self.chains[strategy]:
            if fallback not in self.failed_strategies:
                return fallback
        
        return None
    
    def mark_failed(self, strategy: StrategyType):
        """Mark strategy as failed for current optimization."""
        self.failed_strategies.add(strategy)
    
    def reset(self):
        """Reset failed strategies for new optimization."""
        self.failed_strategies.clear()
    
    def set_chain(self, strategy: StrategyType, fallbacks: List[StrategyType]):
        """Set custom fallback chain for a strategy."""
        self.chains[strategy] = fallbacks


@dataclass
class StrategyResult:
    """Result from a strategy execution."""
    strategy: StrategyType
    panels: List[Panel]
    coverage: float
    time_taken: float
    metrics: OptimizationMetrics
    success: bool


class HybridCoordinator:
    """
    Coordinates hybrid optimization using multiple strategies.
    Can run strategies in parallel or sequence.
    """
    
    def __init__(self, config: Optional[OptimizerConfig] = None):
        self.config = config or OptimizerConfig()
        self.results: List[StrategyResult] = []
        
    def run_sequential(self, room: Room, strategies: List[StrategyType],
                       time_budget: float) -> StrategyResult:
        """
        Run strategies sequentially with time budget allocation.
        Stop when target coverage is reached or time runs out.
        """
        time_per_strategy = time_budget / len(strategies)
        best_result = None
        
        for strategy in strategies:
            if self._check_timeout(time_budget):
                break
            
            # Allocate remaining time proportionally
            remaining_time = time_budget - self._elapsed_time()
            strategy_time = min(time_per_strategy, remaining_time)
            
            # Run strategy (would call actual optimizer here)
            result = self._run_strategy(room, strategy, strategy_time)
            self.results.append(result)
            
            # Update best result
            if best_result is None or result.coverage > best_result.coverage:
                best_result = result
            
            # Early stop if target reached
            if result.coverage >= self.config.early_stop_coverage:
                break
        
        return best_result
    
    def run_parallel(self, room: Room, strategies: List[StrategyType],
                    time_budget: float) -> StrategyResult:
        """
        Run strategies in parallel (simplified - would use threading/multiprocessing).
        """
        # For now, just run sequentially with equal time allocation
        return self.run_sequential(room, strategies, time_budget)
    
    def run_adaptive(self, room: Room, time_budget: float) -> StrategyResult:
        """
        Adaptively allocate time based on strategy performance.
        Start with quick strategies, allocate more time to promising ones.
        """
        # Phase 1: Quick exploration (20% time)
        exploration_strategies = [StrategyType.GREEDY, StrategyType.BOTTOM_LEFT_FILL]
        exploration_time = time_budget * 0.2
        
        exploration_result = self.run_sequential(
            room, exploration_strategies, exploration_time
        )
        
        # Phase 2: Intensive optimization (80% time)
        if exploration_result.coverage < 0.85:
            # Need more sophisticated approaches
            intensive_strategies = [
                StrategyType.MAXIMAL_RECTANGLES,
                StrategyType.BRANCH_AND_BOUND,
                StrategyType.DYNAMIC_PROGRAMMING
            ]
        else:
            # Already good, just refine
            intensive_strategies = [
                StrategyType.MAXIMAL_RECTANGLES,
                StrategyType.BOTTOM_LEFT_FILL
            ]
        
        intensive_time = time_budget * 0.8
        intensive_result = self.run_sequential(
            room, intensive_strategies, intensive_time
        )
        
        # Return best overall result
        if intensive_result.coverage > exploration_result.coverage:
            return intensive_result
        return exploration_result
    
    def _run_strategy(self, room: Room, strategy: StrategyType, 
                     time_limit: float) -> StrategyResult:
        """
        Run a single strategy (placeholder - would call actual optimizer).
        """
        # This would call the actual optimizer implementation
        # For now, return a dummy result
        metrics = OptimizationMetrics()
        metrics.finalize()
        
        return StrategyResult(
            strategy=strategy,
            panels=[],
            coverage=0.0,
            time_taken=0.0,
            metrics=metrics,
            success=True
        )
    
    def _check_timeout(self, budget: float) -> bool:
        """Check if time budget exceeded."""
        return self._elapsed_time() >= budget
    
    def _elapsed_time(self) -> float:
        """Get total elapsed time from all results."""
        return sum(r.time_taken for r in self.results)
    
    def get_best_result(self) -> Optional[StrategyResult]:
        """Get best result from all executed strategies."""
        if not self.results:
            return None
        return max(self.results, key=lambda r: r.coverage)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all strategy results."""
        if not self.results:
            return {}
        
        return {
            'strategies_tried': len(self.results),
            'best_coverage': max(r.coverage for r in self.results),
            'total_time': sum(r.time_taken for r in self.results),
            'results_by_strategy': {
                r.strategy.value: {
                    'coverage': r.coverage,
                    'time': r.time_taken,
                    'panels': len(r.panels)
                }
                for r in self.results
            }
        }


class StrategySelector:
    """
    Main strategy selection and coordination system.
    Combines all components for intelligent strategy selection.
    """
    
    def __init__(self, config: Optional[OptimizerConfig] = None):
        self.config = config or OptimizerConfig()
        self.rules_matrix = StrategyRulesMatrix()
        self.fallback_chain = StrategyFallbackChain()
        self.coordinator = HybridCoordinator(config)
        
        # Tracking
        self.selection_history: List[Tuple[str, StrategyType, float]] = []
    
    def select_and_optimize(self, room: Room) -> Tuple[List[Panel], Dict[str, Any]]:
        """
        Select best strategy and optimize room.
        Returns panels and detailed report.
        """
        # Analyze room characteristics
        characteristics = RoomCharacteristics.analyze(room)
        
        # Select primary strategy
        primary_strategy, confidence = self.rules_matrix.select_strategy(characteristics)
        self.selection_history.append((room.id, primary_strategy, confidence))
        
        # Determine optimization approach based on confidence and complexity
        if confidence > 0.9 and characteristics.complexity_score < 2:
            # High confidence, simple room - use single strategy
            result = self._run_single_strategy(room, primary_strategy)
            
        elif characteristics.complexity_score > 3:
            # Complex room - use adaptive hybrid
            result = self.coordinator.run_adaptive(room, self.config.max_time_seconds)
            
        else:
            # Medium complexity - use fallback chain
            strategies = [primary_strategy]
            fallback = self.fallback_chain.get_fallback(primary_strategy)
            while fallback and len(strategies) < 3:
                strategies.append(fallback)
                fallback = self.fallback_chain.get_fallback(fallback)
            
            result = self.coordinator.run_sequential(
                room, strategies, self.config.max_time_seconds
            )
        
        # Generate report
        report = self._generate_report(characteristics, result)
        
        return result.panels if result else [], report
    
    def _run_single_strategy(self, room: Room, strategy: StrategyType) -> StrategyResult:
        """Run a single strategy on the room."""
        return self.coordinator._run_strategy(
            room, strategy, self.config.max_time_seconds
        )
    
    def _generate_report(self, characteristics: RoomCharacteristics, 
                        result: Optional[StrategyResult]) -> Dict[str, Any]:
        """Generate detailed optimization report."""
        report = {
            'room_analysis': {
                'area': characteristics.area,
                'aspect_ratio': characteristics.aspect_ratio,
                'complexity': characteristics.complexity_score,
                'classification': {
                    'is_square': characteristics.is_square,
                    'is_narrow': characteristics.is_narrow,
                    'is_large': characteristics.is_large
                }
            },
            'strategy_selection': {
                'primary': self.selection_history[-1][1].value if self.selection_history else None,
                'confidence': self.selection_history[-1][2] if self.selection_history else 0,
                'matching_rules': [
                    r.name for r in self.rules_matrix.get_matching_rules(characteristics)
                ]
            }
        }
        
        if result:
            report['optimization_result'] = {
                'strategy_used': result.strategy.value,
                'coverage': result.coverage,
                'panels_count': len(result.panels),
                'time_taken': result.time_taken,
                'success': result.success
            }
        
        # Add coordinator summary if multiple strategies were tried
        coordinator_summary = self.coordinator.get_summary()
        if coordinator_summary:
            report['multi_strategy_summary'] = coordinator_summary
        
        return report


# Step 1.2.3: Memoization Infrastructure
# =======================================

from collections import OrderedDict
import pickle
import os
from pathlib import Path


class CacheKey:
    """
    Generates consistent cache keys for various data structures.
    Ensures deterministic hashing for memoization.
    """
    
    @staticmethod
    def for_state(state: PackingState) -> str:
        """Generate cache key for a packing state."""
        return state.canonical_hash
    
    @staticmethod
    def for_room(room: Room) -> str:
        """Generate cache key for a room."""
        # Use room dimensions and ID for key
        key_str = f"{room.id}:{room.width:.2f}x{room.height:.2f}@{room.position.x:.1f},{room.position.y:.1f}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    @staticmethod
    def for_placement_set(placements: Set[PanelPlacement]) -> str:
        """Generate cache key for a set of placements."""
        # Sort for consistency
        sorted_placements = sorted(
            placements,
            key=lambda p: (p.position[0], p.position[1], str(p.panel_size))
        )
        
        key_str = "|".join(
            f"{p.panel_size}:{p.position}:{p.orientation}"
            for p in sorted_placements
        )
        return hashlib.md5(key_str.encode()).hexdigest()
    
    @staticmethod
    def for_optimization_params(room: Room, config: OptimizerConfig, 
                               strategy: Optional[str] = None) -> str:
        """Generate cache key for optimization parameters."""
        key_parts = [
            CacheKey.for_room(room),
            str(config.max_time_seconds),
            str(config.early_stop_coverage),
            str(config.grid_resolution),
            "|".join(ps.name for ps in config.panel_sizes)
        ]
        
        if strategy:
            key_parts.append(strategy)
        
        key_str = ":".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    @staticmethod
    def for_subproblem(bounds: Tuple[float, float, float, float],
                       remaining_panels: List[PanelSize]) -> str:
        """Generate cache key for a subproblem."""
        key_str = f"{bounds}|{sorted([p.name for p in remaining_panels])}"
        return hashlib.md5(key_str.encode()).hexdigest()


class LRUCache:
    """
    Least Recently Used cache with configurable size limit.
    Thread-safe implementation for parallel optimization.
    """
    
    def __init__(self, max_size: int = 1000):
        """Initialize LRU cache with max size."""
        self.max_size = max_size
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache, updating access order."""
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
        
        self.misses += 1
        return None
    
    def put(self, key: str, value: Any):
        """Put value in cache, evicting LRU if needed."""
        if key in self.cache:
            # Update existing
            self.cache.move_to_end(key)
            self.cache[key] = value
        else:
            # Add new
            if len(self.cache) >= self.max_size:
                # Evict least recently used
                self.cache.popitem(last=False)
                self.evictions += 1
            
            self.cache[key] = value
    
    def clear(self):
        """Clear all cache entries."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def resize(self, new_size: int):
        """Resize cache, evicting entries if needed."""
        self.max_size = new_size
        
        while len(self.cache) > new_size:
            self.cache.popitem(last=False)
            self.evictions += 1
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    @property
    def size(self) -> int:
        """Get current cache size."""
        return len(self.cache)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'size': self.size,
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hit_rate,
            'evictions': self.evictions
        }


class MemoizationStatistics:
    """
    Tracks statistics for memoization system.
    Helps identify optimization opportunities.
    """
    
    def __init__(self):
        self.cache_accesses: Dict[str, int] = {}  # Track access patterns
        self.cache_times: Dict[str, List[float]] = {}  # Track time saved
        self.computation_times: Dict[str, List[float]] = {}  # Track computation times
        self.memory_usage: List[Tuple[float, int]] = []  # (time, bytes)
        self.start_time = time.time()
    
    def record_cache_access(self, cache_type: str, hit: bool):
        """Record a cache access."""
        key = f"{cache_type}_{'hit' if hit else 'miss'}"
        self.cache_accesses[key] = self.cache_accesses.get(key, 0) + 1
    
    def record_time_saved(self, cache_type: str, time_saved: float):
        """Record time saved by cache hit."""
        if cache_type not in self.cache_times:
            self.cache_times[cache_type] = []
        self.cache_times[cache_type].append(time_saved)
    
    def record_computation_time(self, operation: str, time_taken: float):
        """Record computation time for operation."""
        if operation not in self.computation_times:
            self.computation_times[operation] = []
        self.computation_times[operation].append(time_taken)
    
    def record_memory_usage(self, bytes_used: int):
        """Record current memory usage."""
        elapsed = time.time() - self.start_time
        self.memory_usage.append((elapsed, bytes_used))
    
    def get_summary(self) -> Dict[str, Any]:
        """Get statistics summary."""
        total_time_saved = sum(
            sum(times) for times in self.cache_times.values()
        )
        
        avg_computation_times = {
            op: sum(times) / len(times) if times else 0
            for op, times in self.computation_times.items()
        }
        
        cache_effectiveness = {}
        for cache_type in set(k.rsplit('_', 1)[0] for k in self.cache_accesses.keys()):
            hits = self.cache_accesses.get(f"{cache_type}_hit", 0)
            misses = self.cache_accesses.get(f"{cache_type}_miss", 0)
            total = hits + misses
            
            cache_effectiveness[cache_type] = {
                'hit_rate': hits / total if total > 0 else 0,
                'total_accesses': total,
                'time_saved': sum(self.cache_times.get(cache_type, []))
            }
        
        return {
            'total_time_saved': total_time_saved,
            'cache_effectiveness': cache_effectiveness,
            'avg_computation_times': avg_computation_times,
            'peak_memory_bytes': max(self.memory_usage, key=lambda x: x[1])[1] if self.memory_usage else 0
        }


class CachePersistence:
    """
    Handles persistence of cache data to disk.
    Allows cache reuse across optimization sessions.
    """
    
    def __init__(self, cache_dir: str = ".cache/packing"):
        """Initialize cache persistence."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Separate directories for different cache types
        self.state_cache_dir = self.cache_dir / "states"
        self.result_cache_dir = self.cache_dir / "results"
        self.subproblem_cache_dir = self.cache_dir / "subproblems"
        
        for dir in [self.state_cache_dir, self.result_cache_dir, self.subproblem_cache_dir]:
            dir.mkdir(exist_ok=True)
    
    def save_state(self, key: str, state: PackingState):
        """Save packing state to disk."""
        file_path = self.state_cache_dir / f"{key}.pkl"
        
        # Convert to serializable format
        data = {
            'placements': list(state.placements),
            'room_bounds': state.room_bounds,
            'coverage': state.coverage
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
    
    def load_state(self, key: str) -> Optional[PackingState]:
        """Load packing state from disk."""
        file_path = self.state_cache_dir / f"{key}.pkl"
        
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            return PackingState(
                placements=frozenset(data['placements']),
                room_bounds=data['room_bounds'],
                coverage=data['coverage']
            )
        except Exception:
            # Handle corrupted cache files
            return None
    
    def save_result(self, key: str, panels: List[Panel], metrics: OptimizationMetrics):
        """Save optimization result to disk."""
        file_path = self.result_cache_dir / f"{key}.pkl"
        
        data = {
            'panels': panels,
            'metrics_summary': metrics.get_summary()
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
    
    def load_result(self, key: str) -> Optional[Tuple[List[Panel], Dict[str, Any]]]:
        """Load optimization result from disk."""
        file_path = self.result_cache_dir / f"{key}.pkl"
        
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            return data['panels'], data['metrics_summary']
        except Exception:
            return None
    
    def save_subproblem(self, key: str, solution: Any):
        """Save subproblem solution to disk."""
        file_path = self.subproblem_cache_dir / f"{key}.pkl"
        
        with open(file_path, 'wb') as f:
            pickle.dump(solution, f)
    
    def load_subproblem(self, key: str) -> Optional[Any]:
        """Load subproblem solution from disk."""
        file_path = self.subproblem_cache_dir / f"{key}.pkl"
        
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None
    
    def clear_cache(self, cache_type: Optional[str] = None):
        """Clear cache files from disk."""
        if cache_type == "states":
            dirs = [self.state_cache_dir]
        elif cache_type == "results":
            dirs = [self.result_cache_dir]
        elif cache_type == "subproblems":
            dirs = [self.subproblem_cache_dir]
        else:
            dirs = [self.state_cache_dir, self.result_cache_dir, self.subproblem_cache_dir]
        
        for dir in dirs:
            for file in dir.glob("*.pkl"):
                file.unlink()
    
    def get_cache_size(self) -> Dict[str, int]:
        """Get size of cache on disk in bytes."""
        sizes = {}
        
        for name, dir in [
            ("states", self.state_cache_dir),
            ("results", self.result_cache_dir),
            ("subproblems", self.subproblem_cache_dir)
        ]:
            total_size = sum(f.stat().st_size for f in dir.glob("*.pkl"))
            sizes[name] = total_size
        
        sizes["total"] = sum(sizes.values())
        return sizes
    
    def prune_old_cache(self, max_age_days: int = 7):
        """Remove cache files older than specified days."""
        import time
        max_age_seconds = max_age_days * 24 * 3600
        current_time = time.time()
        
        for dir in [self.state_cache_dir, self.result_cache_dir, self.subproblem_cache_dir]:
            for file in dir.glob("*.pkl"):
                if current_time - file.stat().st_mtime > max_age_seconds:
                    file.unlink()


class MemoizationManager:
    """
    Central manager for all memoization functionality.
    Coordinates caches, statistics, and persistence.
    """
    
    def __init__(self, config: Optional[OptimizerConfig] = None):
        """Initialize memoization manager."""
        self.config = config or OptimizerConfig()
        
        # Create caches
        cache_size = self.config.cache_size_limit
        self.state_cache = LRUCache(cache_size)
        self.result_cache = LRUCache(cache_size // 10)  # Smaller for results
        self.subproblem_cache = LRUCache(cache_size // 2)
        
        # Statistics tracking
        self.statistics = MemoizationStatistics()
        
        # Persistence
        self.persistence = CachePersistence() if self.config.enable_memoization else None
        
        # Load persisted cache if available
        if self.persistence:
            self._load_persisted_cache()
    
    def get_or_compute_state(self, key: str, compute_fn: Callable) -> PackingState:
        """Get state from cache or compute it."""
        # Check memory cache
        cached = self.state_cache.get(key)
        if cached:
            self.statistics.record_cache_access("state", True)
            return cached
        
        # Check disk cache
        if self.persistence:
            cached = self.persistence.load_state(key)
            if cached:
                self.state_cache.put(key, cached)
                self.statistics.record_cache_access("state", True)
                return cached
        
        # Compute
        self.statistics.record_cache_access("state", False)
        start_time = time.time()
        
        result = compute_fn()
        
        computation_time = time.time() - start_time
        self.statistics.record_computation_time("state_computation", computation_time)
        
        # Cache result
        self.state_cache.put(key, result)
        if self.persistence:
            self.persistence.save_state(key, result)
        
        return result
    
    def cache_optimization_result(self, room: Room, config: OptimizerConfig,
                                 strategy: str, panels: List[Panel], 
                                 metrics: OptimizationMetrics):
        """Cache optimization result."""
        key = CacheKey.for_optimization_params(room, config, strategy)
        
        self.result_cache.put(key, (panels, metrics))
        
        if self.persistence:
            self.persistence.save_result(key, panels, metrics)
    
    def get_cached_result(self, room: Room, config: OptimizerConfig,
                         strategy: str) -> Optional[Tuple[List[Panel], OptimizationMetrics]]:
        """Get cached optimization result."""
        key = CacheKey.for_optimization_params(room, config, strategy)
        
        # Check memory cache
        cached = self.result_cache.get(key)
        if cached:
            self.statistics.record_cache_access("result", True)
            return cached
        
        # Check disk cache
        if self.persistence:
            cached_data = self.persistence.load_result(key)
            if cached_data:
                panels, metrics_summary = cached_data
                # Reconstruct metrics (simplified)
                metrics = OptimizationMetrics()
                metrics.best_coverage = metrics_summary.get('best_coverage', 0)
                metrics.panels_used = metrics_summary.get('panels_used', 0)
                
                self.result_cache.put(key, (panels, metrics))
                self.statistics.record_cache_access("result", True)
                return panels, metrics
        
        self.statistics.record_cache_access("result", False)
        return None
    
    def clear_all_caches(self):
        """Clear all caches."""
        self.state_cache.clear()
        self.result_cache.clear()
        self.subproblem_cache.clear()
        
        if self.persistence:
            self.persistence.clear_cache()
    
    def get_statistics_report(self) -> str:
        """Generate memoization statistics report."""
        stats = self.statistics.get_summary()
        
        report = "\nMemoization Statistics\n"
        report += "=" * 50 + "\n"
        report += f"Total Time Saved: {stats['total_time_saved']:.2f}s\n"
        
        report += "\nCache Effectiveness:\n"
        for cache_type, effectiveness in stats['cache_effectiveness'].items():
            report += f"  {cache_type}:\n"
            report += f"    Hit Rate: {effectiveness['hit_rate']*100:.1f}%\n"
            report += f"    Accesses: {effectiveness['total_accesses']}\n"
            report += f"    Time Saved: {effectiveness['time_saved']:.2f}s\n"
        
        report += "\nMemory Usage:\n"
        report += f"  State Cache: {self.state_cache.size}/{self.state_cache.max_size}\n"
        report += f"  Result Cache: {self.result_cache.size}/{self.result_cache.max_size}\n"
        report += f"  Subproblem Cache: {self.subproblem_cache.size}/{self.subproblem_cache.max_size}\n"
        
        if self.persistence:
            cache_sizes = self.persistence.get_cache_size()
            report += f"  Disk Usage: {cache_sizes['total'] / 1024 / 1024:.1f} MB\n"
        
        return report
    
    def _load_persisted_cache(self):
        """Load most recent entries from persisted cache."""
        # This would intelligently pre-load frequently used cache entries
        # For now, we skip pre-loading to avoid startup delay
        pass
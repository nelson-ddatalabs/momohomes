#!/usr/bin/env python3
"""
dp_state.py - Dynamic Programming State Space
===========================================
Production-ready DP state representation for optimal panel placement.
Builds on existing PackingState foundation for 95%+ coverage optimization.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, FrozenSet, Any
from enum import Enum
import hashlib
import pickle
from abc import ABC, abstractmethod

from models import Room, Panel, PanelSize
from advanced_packing import PackingState, PanelPlacement, StateTransition


class DPStateType(Enum):
    """Types of DP states for different algorithms."""
    SKYLINE = "skyline"          # Y-coordinates at each X position
    GRID_BASED = "grid"          # Discretized occupancy grid
    CORNER_POINTS = "corners"    # Available placement corners
    MAXIMAL_RECTS = "rectangles" # Maximal empty rectangles


@dataclass(frozen=True)
class SkylineProfile:
    """
    Immutable skyline representation for DP state encoding.
    Tracks Y-coordinates at discrete X positions.
    """
    x_coords: Tuple[float, ...]
    y_coords: Tuple[float, ...]
    room_width: float
    room_height: float
    
    def __post_init__(self):
        if len(self.x_coords) != len(self.y_coords):
            raise ValueError("X and Y coordinates must have same length")
    
    def __hash__(self) -> int:
        """Efficient hash for memoization."""
        return hash((self.x_coords, self.y_coords, self.room_width, self.room_height))
    
    def get_height_at(self, x: float) -> float:
        """Get skyline height at specific X coordinate."""
        for i in range(len(self.x_coords) - 1):
            if self.x_coords[i] <= x < self.x_coords[i + 1]:
                return self.y_coords[i]
        return self.y_coords[-1] if self.x_coords else 0.0
    
    def can_place_panel(self, x: float, width: float, height: float) -> bool:
        """Check if panel can be placed at given position."""
        if x + width > self.room_width:
            return False
        
        max_y = 0.0
        for check_x in [x + i * 0.1 for i in range(int(width * 10) + 1)]:
            max_y = max(max_y, self.get_height_at(check_x))
        
        return max_y + height <= self.room_height


@dataclass(frozen=True) 
class DPState:
    """
    Enhanced state for dynamic programming optimization.
    Combines PackingState with DP-specific encoding and operations.
    """
    base_state: PackingState
    skyline: Optional[SkylineProfile] = None
    remaining_panels: FrozenSet[PanelSize] = field(default_factory=frozenset)
    depth: int = 0
    parent_hash: Optional[str] = None
    
    def __hash__(self) -> int:
        """Fast hash using skyline and remaining panels."""
        if self.skyline:
            return hash((self.skyline, self.remaining_panels, self.depth))
        return hash((self.base_state.canonical_hash, self.remaining_panels, self.depth))
    
    def __eq__(self, other: Any) -> bool:
        """Equality for memoization."""
        if not isinstance(other, DPState):
            return False
        return (self.skyline == other.skyline and 
                self.remaining_panels == other.remaining_panels)
    
    @property
    def canonical_hash(self) -> str:
        """Deterministic hash for exact state comparison."""
        if self.skyline:
            state_str = f"skyline:{self.skyline}|panels:{sorted(self.remaining_panels, key=str)}"
        else:
            state_str = f"base:{self.base_state.canonical_hash}|panels:{sorted(self.remaining_panels, key=str)}"
        return hashlib.md5(state_str.encode()).hexdigest()
    
    @property
    def coverage(self) -> float:
        """Get coverage from base state."""
        return self.base_state.coverage
    
    @property
    def placements(self) -> FrozenSet[PanelPlacement]:
        """Get placements from base state."""
        return self.base_state.placements
    
    @property
    def room_bounds(self) -> Tuple[float, float, float, float]:
        """Get room bounds from base state."""
        return self.base_state.room_bounds
    
    def is_terminal(self) -> bool:
        """Check if this is a terminal state (no more placements possible)."""
        return (not self.remaining_panels or 
                self._no_placement_possible())
    
    def _no_placement_possible(self) -> bool:
        """Check if no more panels can be placed."""
        if not self.remaining_panels:
            return True
        
        if self.skyline:
            # Check skyline-based placement possibility
            rx, ry, rw, rh = self.room_bounds
            for panel_size in self.remaining_panels:
                for orientation in ["horizontal", "vertical"]:
                    pw, ph = panel_size.get_dimensions(orientation)
                    
                    # Try placing at any valid skyline position
                    for i, x in enumerate(self.skyline.x_coords):
                        if x + pw <= rw and self.skyline.can_place_panel(x, pw, ph):
                            return False
            return True
        else:
            # Fallback to base state validation
            transition = StateTransition()
            valid_placements = transition.get_valid_placements(
                self.base_state, 
                list(self.remaining_panels),
                resolution=1.0
            )
            return len(valid_placements) == 0


class DPStateEncoder:
    """
    Encodes DP states for efficient storage and comparison.
    Supports multiple encoding strategies for different DP algorithms.
    """
    
    def __init__(self, encoding_type: DPStateType = DPStateType.SKYLINE):
        self.encoding_type = encoding_type
    
    def encode_state(self, state: PackingState, remaining_panels: Set[PanelSize]) -> DPState:
        """Encode PackingState into DP-optimized representation."""
        if self.encoding_type == DPStateType.SKYLINE:
            skyline = self._create_skyline(state)
            return DPState(
                base_state=state,
                skyline=skyline,
                remaining_panels=frozenset(remaining_panels)
            )
        else:
            return DPState(
                base_state=state,
                remaining_panels=frozenset(remaining_panels)
            )
    
    def _create_skyline(self, state: PackingState) -> SkylineProfile:
        """Create skyline profile from current panel placements."""
        rx, ry, rw, rh = state.room_bounds
        
        # Collect all X coordinates where skyline changes
        x_points = {rx, rx + rw}  # Room boundaries
        
        for placement in state.placements:
            x1, y1, x2, y2 = placement.bounds
            x_points.add(x1)
            x_points.add(x2)
        
        x_coords = tuple(sorted(x_points))
        
        # Calculate Y coordinate at each X position
        y_coords = []
        for x in x_coords[:-1]:  # Exclude right boundary
            max_y = ry  # Room bottom
            
            # Find highest panel at this X coordinate
            for placement in state.placements:
                x1, y1, x2, y2 = placement.bounds
                if x1 <= x < x2:  # Panel covers this X coordinate
                    max_y = max(max_y, y2)
            
            y_coords.append(max_y)
        
        return SkylineProfile(
            x_coords=x_coords[:-1],  # Exclude right boundary
            y_coords=tuple(y_coords),
            room_width=rw,
            room_height=rh
        )


class DPTransitionFunction:
    """
    Manages state transitions for dynamic programming algorithms.
    Handles both forward transitions (add panel) and state validation.
    """
    
    def __init__(self):
        self.state_encoder = DPStateEncoder()
        self.base_transition = StateTransition()
    
    def get_successor_states(self, dp_state: DPState) -> List[DPState]:
        """
        Generate all valid successor states from current DP state.
        Returns states with one additional panel placed.
        """
        successors = []
        
        if dp_state.is_terminal():
            return successors
        
        # Try placing each remaining panel type
        for panel_size in dp_state.remaining_panels:
            # Get valid placements for this panel
            valid_placements = self._get_valid_placements(dp_state, panel_size)
            
            for placement in valid_placements:
                # Create new base state with placement
                new_base_state = self.base_transition.apply_placement(
                    dp_state.base_state, placement
                )
                
                if new_base_state:
                    # Update remaining panels (remove one of this type)
                    new_remaining = set(dp_state.remaining_panels)
                    new_remaining.discard(panel_size)
                    
                    # Encode as new DP state
                    new_dp_state = self.state_encoder.encode_state(
                        new_base_state, new_remaining
                    )
                    new_dp_state = DPState(
                        base_state=new_dp_state.base_state,
                        skyline=new_dp_state.skyline,
                        remaining_panels=new_dp_state.remaining_panels,
                        depth=dp_state.depth + 1,
                        parent_hash=dp_state.canonical_hash
                    )
                    
                    successors.append(new_dp_state)
        
        return successors
    
    def _get_valid_placements(self, dp_state: DPState, panel_size: PanelSize) -> List[PanelPlacement]:
        """Get valid placements for specific panel in current state."""
        if dp_state.skyline:
            return self._get_skyline_placements(dp_state, panel_size)
        else:
            return self.base_transition.get_valid_placements(
                dp_state.base_state, 
                [panel_size],
                resolution=0.5  # Higher resolution for DP
            )
    
    def _get_skyline_placements(self, dp_state: DPState, panel_size: PanelSize) -> List[PanelPlacement]:
        """Get placements using skyline representation."""
        placements = []
        skyline = dp_state.skyline
        rx, ry, rw, rh = dp_state.room_bounds
        
        for orientation in ["horizontal", "vertical"]:
            pw, ph = panel_size.get_dimensions(orientation)
            
            # Try placing at each skyline position
            for i, x in enumerate(skyline.x_coords):
                if x + pw <= rw:
                    y = skyline.get_height_at(x)
                    
                    if y + ph <= rh and skyline.can_place_panel(x, pw, ph):
                        # Convert to absolute coordinates
                        abs_x = rx + x
                        abs_y = ry + y
                        
                        placement = PanelPlacement(
                            panel_size=panel_size,
                            position=(abs_x, abs_y),
                            orientation=orientation
                        )
                        
                        # Validate against existing placements
                        if dp_state.base_state.is_valid_placement(placement):
                            placements.append(placement)
        
        return placements


class DPTerminalDetector:
    """
    Determines when DP recursion should terminate.
    Includes various stopping criteria for optimization.
    """
    
    def __init__(self, target_coverage: float = 0.95, max_depth: int = 50):
        self.target_coverage = target_coverage
        self.max_depth = max_depth
    
    def is_terminal(self, dp_state: DPState) -> bool:
        """
        Check if state should terminate DP recursion.
        Returns True if no further exploration needed.
        """
        # Basic terminal conditions
        if dp_state.is_terminal():
            return True
        
        # Coverage target reached
        if dp_state.coverage >= self.target_coverage:
            return True
        
        # Maximum search depth reached
        if dp_state.depth >= self.max_depth:
            return True
        
        # No remaining panels
        if not dp_state.remaining_panels:
            return True
        
        return False
    
    def get_terminal_reason(self, dp_state: DPState) -> str:
        """Get human-readable reason for termination."""
        if dp_state.coverage >= self.target_coverage:
            return f"Target coverage {self.target_coverage:.1%} reached"
        
        if dp_state.depth >= self.max_depth:
            return f"Maximum depth {self.max_depth} reached"
        
        if not dp_state.remaining_panels:
            return "No remaining panels"
        
        if dp_state._no_placement_possible():
            return "No valid placements possible"
        
        return "Terminal state"
    
    def should_prune(self, dp_state: DPState, best_coverage: float) -> bool:
        """
        Check if state should be pruned from DP search.
        Returns True if this branch cannot improve best solution.
        """
        # Prune if we can't possibly beat current best
        remaining_area = sum(p.area for p in dp_state.remaining_panels)
        rx, ry, rw, rh = dp_state.room_bounds
        room_area = rw * rh
        
        current_coverage = dp_state.coverage
        max_possible_coverage = min(1.0, current_coverage + (remaining_area / room_area))
        
        # Prune if we can't beat best + small epsilon
        return max_possible_coverage < best_coverage - 0.001


class DPStateFactory:
    """
    Factory for creating and managing DP states.
    Provides convenient methods for state creation and conversion.
    """
    
    def __init__(self, encoding_type: DPStateType = DPStateType.SKYLINE):
        self.encoder = DPStateEncoder(encoding_type)
        self.terminal_detector = DPTerminalDetector()
    
    def create_initial_state(self, room: Room, panel_inventory: List[PanelSize]) -> DPState:
        """Create initial DP state for room with given panel inventory."""
        base_state = PackingState.from_room(room)
        return self.encoder.encode_state(base_state, set(panel_inventory))
    
    def from_packing_state(self, state: PackingState, remaining_panels: Set[PanelSize]) -> DPState:
        """Convert existing PackingState to DPState."""
        return self.encoder.encode_state(state, remaining_panels)
    
    def create_terminal_detector(self, target_coverage: float = 0.95, max_depth: int = 50) -> DPTerminalDetector:
        """Create terminal detector with custom parameters."""
        return DPTerminalDetector(target_coverage, max_depth)


def create_dp_state_system(
    encoding_type: DPStateType = DPStateType.SKYLINE,
    target_coverage: float = 0.95
) -> Tuple[DPStateFactory, DPTransitionFunction, DPTerminalDetector]:
    """
    Factory function to create complete DP state system.
    Returns all components needed for DP optimization.
    """
    factory = DPStateFactory(encoding_type)
    transition_fn = DPTransitionFunction()
    terminal_detector = DPTerminalDetector(target_coverage)
    
    return factory, transition_fn, terminal_detector
#!/usr/bin/env python3
"""
simple_blf_optimizer.py - Simplified Bottom-Left-Fill Optimizer
===============================================================
A fast, production-ready BLF implementation without expensive lookahead.
Focuses on achieving 95%+ coverage with simple, efficient logic.
"""

from typing import List, Optional, Tuple
import time

from models import Panel, PanelSize, Point, Room
from advanced_packing import (
    AbstractOptimizer,
    OptimizerConfig,
    PackingState,
    PanelPlacement
)


class SimpleBLFOptimizer(AbstractOptimizer):
    """
    Simplified Bottom-Left-Fill optimizer for 95%+ coverage.
    Uses basic BLF without expensive lookahead or backtracking.
    """
    
    def __init__(self, config: Optional[OptimizerConfig] = None):
        """Initialize simple BLF optimizer."""
        super().__init__(config)
        
    def get_algorithm_name(self) -> str:
        """Get algorithm name."""
        return "Simple BLF"
    
    def optimize_room(self, room: Room) -> List[Panel]:
        """
        Optimize panel placement for a room using simple BLF.
        """
        # Calculate how many panels we need for target coverage
        room_area = room.width * room.height
        target_area = room_area * self.config.early_stop_coverage
        
        print(f"Room {room.name}: area={room_area:.1f}, target={target_area:.1f}")
        
        # Sort panel sizes by efficiency (larger first)
        panel_sizes = sorted(
            self.config.panel_sizes,
            key=lambda p: p.area,
            reverse=True
        )
        
        # Initialize state
        state = PackingState.from_room(room)
        placed_panels = []
        
        # Multi-phase placement strategy - each phase returns updated state
        print(f"  Phase 1: Grid fill with largest panels (6x8)")
        phase1_panels, state = self._grid_fill_phase(state, room, PanelSize.PANEL_6X8)
        placed_panels.extend(phase1_panels)
        
        print(f"  Phase 2: Gap fill with 6x6 panels")
        phase2_panels, state = self._gap_fill_phase(state, room, PanelSize.PANEL_6X6, min_gap_area=20)
        placed_panels.extend(phase2_panels)
        
        print(f"  Phase 3: Gap fill with 4x6 panels")
        phase3_panels, state = self._gap_fill_phase(state, room, PanelSize.PANEL_4X6, min_gap_area=15)
        placed_panels.extend(phase3_panels)
        
        print(f"  Phase 4: Gap fill with 4x4 panels")
        phase4_panels, state = self._gap_fill_phase(state, room, PanelSize.PANEL_4X4, min_gap_area=8)
        placed_panels.extend(phase4_panels)
        
        print(f"  Phase 5: Aggressive final fill with any panel size")
        phase5_panels, state = self._aggressive_final_fill(state, room)
        placed_panels.extend(phase5_panels)
        
        # Check if we've reached target coverage
        current_coverage = sum(p.size.area for p in placed_panels) / room_area
        if current_coverage >= self.config.early_stop_coverage:
            print(f"  Reached {current_coverage*100:.1f}% coverage with {len(placed_panels)} panels")
            return placed_panels
        
        return placed_panels
    
    def _find_blf_position(self, panel_size: PanelSize, 
                          state: PackingState, 
                          room: Room) -> Optional[PanelPlacement]:
        """
        Find the bottom-left-most position for a panel.
        """
        best_placement = None
        best_score = float('inf')
        
        room_width = room.width
        room_height = room.height
        
        # Try both orientations
        for orientation in ["horizontal", "vertical"]:
            if orientation == "horizontal":
                width = panel_size.width
                height = panel_size.length
            else:
                width = panel_size.length
                height = panel_size.width
            
            # Skip if panel doesn't fit in this orientation
            if width > room_width or height > room_height:
                continue
            
            # Generate candidate positions
            candidate_positions = self._generate_candidate_positions(
                state, room, width, height
            )
            
            for x, y in candidate_positions:
                # Position relative to room
                position = (room.position.x + x, room.position.y + y)
                placement = PanelPlacement(
                    panel_size=panel_size,
                    position=position,
                    orientation=orientation
                )
                
                # Check if placement is valid
                if state.is_valid_placement(placement):
                    # Score based on bottom-left preference
                    score = y * 1000 + x  # Prioritize bottom, then left
                    
                    if score < best_score:
                        best_score = score
                        best_placement = placement
        
        return best_placement

    def _generate_candidate_positions(self, state: PackingState, room: Room, 
                                    width: float, height: float) -> List[Tuple[float, float]]:
        """Generate candidate positions for panel placement."""
        positions = set()
        grid_step = self.config.grid_resolution
        
        # 1. Grid-based positions (systematic scan)
        y = 0.0
        while y + height <= room.height:
            x = 0.0
            while x + width <= room.width:
                positions.add((x, y))
                x += grid_step
            y += grid_step
        
        # 2. Edge-aligned positions (against existing panels)
        for existing in state.placements:
            ex1, ey1, ex2, ey2 = existing.bounds
            # Convert to room coordinates
            ex1 -= room.position.x
            ey1 -= room.position.y
            ex2 -= room.position.x
            ey2 -= room.position.y
            
            # Positions adjacent to existing panels
            candidate_coords = [
                (ex2, ey1),       # Right edge, bottom aligned
                (ex1, ey2),       # Bottom edge, left aligned  
                (ex2, ey2),       # Bottom-right corner
                (ex1 - width, ey1),  # Left of existing panel
                (ex1, ey1 - height), # Below existing panel
            ]
            
            for cx, cy in candidate_coords:
                if (cx >= 0 and cy >= 0 and 
                    cx + width <= room.width and 
                    cy + height <= room.height):
                    positions.add((cx, cy))
        
        # 3. Corner and edge positions
        corner_positions = [
            (0.0, 0.0),  # Bottom-left corner
            (room.width - width, 0.0),  # Bottom-right corner  
            (0.0, room.height - height),  # Top-left corner
            (room.width - width, room.height - height),  # Top-right corner
        ]
        
        for cx, cy in corner_positions:
            if cx >= 0 and cy >= 0:
                positions.add((cx, cy))
        
        # Convert to sorted list (bottom-left preference)
        return sorted(list(positions), key=lambda pos: (pos[1], pos[0]))

    def _grid_fill_phase(self, state: PackingState, room: Room, panel_size: PanelSize) -> Tuple[List[Panel], PackingState]:
        """Grid fill with specific panel size using BLF placement."""
        placed_panels = []
        max_attempts = int(room.area / panel_size.area) + 10
        attempts = 0
        
        while attempts < max_attempts:
            attempts += 1
            
            # Find best position for this panel
            placement = self._find_blf_position(panel_size, state, room)
            
            if placement:
                # Apply placement
                new_state = self.transition_manager.apply_placement(state, placement)
                
                if new_state:
                    state = new_state
                    panel = placement.to_panel(room.id)
                    placed_panels.append(panel)
                    print(f"    Grid placed {panel_size.name} at {placement.position}")
                    
                    # Update best state
                    self.update_best_state(state)
                else:
                    break
            else:
                break
                
            # Check timeout
            if self.check_timeout():
                break
        
        return placed_panels, state

    def _gap_fill_phase(self, state: PackingState, room: Room, 
                       panel_size: PanelSize, min_gap_area: float) -> Tuple[List[Panel], PackingState]:
        """Fill gaps with specific panel size."""
        placed_panels = []
        
        # Calculate remaining uncovered area
        covered_area = sum(existing.panel_size.area for existing in state.placements 
                          if self._placement_in_room(existing, room))
        remaining_area = room.area - covered_area
        
        if remaining_area < min_gap_area:
            return placed_panels, state
        
        max_attempts = int(remaining_area / panel_size.area) + 5
        attempts = 0
        
        while attempts < max_attempts:
            attempts += 1
            
            # Find best position for gap filling
            placement = self._find_blf_position(panel_size, state, room)
            
            if placement:
                # Apply placement
                new_state = self.transition_manager.apply_placement(state, placement)
                
                if new_state:
                    state = new_state
                    panel = placement.to_panel(room.id)
                    placed_panels.append(panel)
                    print(f"    Gap filled {panel_size.name} at {placement.position}")
                    
                    # Update best state
                    self.update_best_state(state)
                else:
                    break
            else:
                break
                
            # Check timeout
            if self.check_timeout():
                break
        
        return placed_panels, state

    def _placement_in_room(self, placement: PanelPlacement, room: Room) -> bool:
        """Check if placement is within the specified room."""
        px, py = placement.position
        rx, ry = room.position.x, room.position.y
        rw, rh = room.width, room.height
        
        # Check if placement position is within room bounds (with small tolerance)
        tolerance = 0.01
        return (rx - tolerance <= px <= rx + rw + tolerance and 
                ry - tolerance <= py <= ry + rh + tolerance)

    def _aggressive_final_fill(self, state: PackingState, room: Room) -> Tuple[List[Panel], PackingState]:
        """Aggressively try to fill any remaining gaps with best-fitting panel."""
        placed_panels = []
        
        # Try all panel sizes in order of efficiency
        panel_sizes = [PanelSize.PANEL_4X4, PanelSize.PANEL_4X6, PanelSize.PANEL_6X6, PanelSize.PANEL_6X8]
        
        # Continue until no more panels can be placed
        max_total_attempts = 50  # Prevent infinite loops
        total_attempts = 0
        
        while total_attempts < max_total_attempts:
            placed_any = False
            total_attempts += 1
            
            for panel_size in panel_sizes:
                # Try to place one panel of this size
                placement = self._find_blf_position(panel_size, state, room)
                
                if placement:
                    # Apply placement
                    new_state = self.transition_manager.apply_placement(state, placement)
                    
                    if new_state:
                        state = new_state
                        panel = placement.to_panel(room.id)
                        placed_panels.append(panel)
                        placed_any = True
                        print(f"    Aggressively placed {panel_size.name} at {placement.position}")
                        
                        # Update best state
                        self.update_best_state(state)
                        break  # Try next iteration with all sizes again
                
                # Check timeout
                if self.check_timeout():
                    return placed_panels, state
            
            if not placed_any:
                # No more panels can be placed
                break
        
        return placed_panels, state


def create_simple_blf_optimizer(config: Optional[OptimizerConfig] = None) -> SimpleBLFOptimizer:
    """Factory function to create simple BLF optimizer."""
    if config is None:
        config = OptimizerConfig(
            max_time_seconds=5.0,
            early_stop_coverage=0.95,
            max_iterations=1000,
            grid_resolution=0.25
        )
    
    return SimpleBLFOptimizer(config)
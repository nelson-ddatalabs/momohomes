from typing import List, Dict, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from datetime import datetime
import logging
import copy

from core import PackingState, Panel, Room, Position, PlacedPanel
from spatial_index import SpatialIndex, OccupancyGrid
from solution_validation import SolutionValidator, ValidationResult

logger = logging.getLogger(__name__)


class ImprovementType(Enum):
    REFINEMENT = "refinement"
    GAP_FILLING = "gap_filling"
    CONSOLIDATION = "consolidation"
    COMPACTION = "compaction"
    ALIGNMENT = "alignment"
    ROTATION = "rotation"
    SWAP = "swap"
    SLIDE = "slide"


@dataclass
class ImprovementMove:
    type: ImprovementType
    description: str
    panels_affected: List[int]
    old_positions: List[Position]
    new_positions: List[Position]
    coverage_delta: float
    quality_delta: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ImprovementResult:
    improved_state: PackingState
    moves_applied: List[ImprovementMove]
    coverage_before: float
    coverage_after: float
    quality_before: float
    quality_after: float
    improvement_time: float
    iterations: int
    
    @property
    def coverage_improvement(self) -> float:
        return self.coverage_after - self.coverage_before
    
    @property
    def quality_improvement(self) -> float:
        return self.quality_after - self.quality_before


class PostOptimizationRefinement:
    """Refines solution after initial optimization"""
    
    def __init__(self):
        self.max_iterations = 100
        self.min_improvement = 0.001
        self.moves_tried = 0
        self.moves_accepted = 0
    
    def refine(self, state: PackingState, room: Room) -> List[ImprovementMove]:
        """Apply refinement techniques to improve solution"""
        moves = []
        
        # Try different refinement strategies
        moves.extend(self._align_panels(state, room))
        moves.extend(self._optimize_rotations(state, room))
        moves.extend(self._local_search(state, room))
        moves.extend(self._swap_panels(state, room))
        
        return moves
    
    def _align_panels(self, state: PackingState, room: Room) -> List[ImprovementMove]:
        """Align panels to common grid lines"""
        moves = []
        alignment_threshold = 10.0
        
        # Find common alignment positions
        x_positions = [p.position.x for p in state.placed_panels]
        y_positions = [p.position.y for p in state.placed_panels]
        
        # Cluster positions
        x_clusters = self._cluster_positions(x_positions, alignment_threshold)
        y_clusters = self._cluster_positions(y_positions, alignment_threshold)
        
        # Try to align panels to cluster centers
        for i, panel in enumerate(state.placed_panels):
            old_pos = panel.position
            
            # Find nearest cluster centers
            nearest_x = self._find_nearest_cluster(old_pos.x, x_clusters)
            nearest_y = self._find_nearest_cluster(old_pos.y, y_clusters)
            
            # Check if alignment improves solution
            new_pos = Position(nearest_x, nearest_y)
            if self._is_valid_move(state, i, new_pos, panel.rotated, room):
                delta = self._calculate_quality_delta(state, i, new_pos)
                if delta > 0:
                    moves.append(ImprovementMove(
                        type=ImprovementType.ALIGNMENT,
                        description=f"Align panel {i} to grid",
                        panels_affected=[i],
                        old_positions=[old_pos],
                        new_positions=[new_pos],
                        coverage_delta=0.0,
                        quality_delta=delta
                    ))
                    panel.position = new_pos
                    self.moves_accepted += 1
            
            self.moves_tried += 1
        
        return moves
    
    def _optimize_rotations(self, state: PackingState, room: Room) -> List[ImprovementMove]:
        """Try rotating panels for better fit"""
        moves = []
        
        for i, panel in enumerate(state.placed_panels):
            if not panel.panel.can_rotate:
                continue
            
            old_pos = panel.position
            old_rotated = panel.rotated
            new_rotated = not old_rotated
            
            # Check if rotation improves packing
            if self._is_valid_move(state, i, old_pos, new_rotated, room):
                delta = self._calculate_rotation_benefit(state, i, new_rotated, room)
                if delta > self.min_improvement:
                    moves.append(ImprovementMove(
                        type=ImprovementType.ROTATION,
                        description=f"Rotate panel {i}",
                        panels_affected=[i],
                        old_positions=[old_pos],
                        new_positions=[old_pos],
                        coverage_delta=0.0,
                        quality_delta=delta
                    ))
                    panel.rotated = new_rotated
                    self.moves_accepted += 1
            
            self.moves_tried += 1
        
        return moves
    
    def _local_search(self, state: PackingState, room: Room) -> List[ImprovementMove]:
        """Apply local search to improve positions"""
        moves = []
        search_radius = 20.0
        step_size = 5.0
        
        for i, panel in enumerate(state.placed_panels):
            old_pos = panel.position
            best_pos = old_pos
            best_quality = self._calculate_position_quality(state, i, old_pos)
            
            # Search neighborhood
            for dx in np.arange(-search_radius, search_radius + step_size, step_size):
                for dy in np.arange(-search_radius, search_radius + step_size, step_size):
                    if dx == 0 and dy == 0:
                        continue
                    
                    new_pos = Position(old_pos.x + dx, old_pos.y + dy)
                    
                    if self._is_valid_move(state, i, new_pos, panel.rotated, room):
                        quality = self._calculate_position_quality(state, i, new_pos)
                        if quality > best_quality:
                            best_quality = quality
                            best_pos = new_pos
            
            # Apply best move if found
            if best_pos != old_pos:
                moves.append(ImprovementMove(
                    type=ImprovementType.SLIDE,
                    description=f"Slide panel {i} to better position",
                    panels_affected=[i],
                    old_positions=[old_pos],
                    new_positions=[best_pos],
                    coverage_delta=0.0,
                    quality_delta=best_quality - self._calculate_position_quality(state, i, old_pos)
                ))
                panel.position = best_pos
                self.moves_accepted += 1
            
            self.moves_tried += 1
        
        return moves
    
    def _swap_panels(self, state: PackingState, room: Room) -> List[ImprovementMove]:
        """Try swapping positions of similar-sized panels"""
        moves = []
        
        for i in range(len(state.placed_panels)):
            for j in range(i + 1, len(state.placed_panels)):
                panel_i = state.placed_panels[i]
                panel_j = state.placed_panels[j]
                
                # Check if panels are similar size (within 20%)
                area_i = panel_i.panel.width * panel_i.panel.height
                area_j = panel_j.panel.width * panel_j.panel.height
                if abs(area_i - area_j) / max(area_i, area_j) > 0.2:
                    continue
                
                # Try swapping
                old_pos_i = panel_i.position
                old_pos_j = panel_j.position
                
                # Check if swap is valid
                panel_i.position = old_pos_j
                panel_j.position = old_pos_i
                
                if self._is_valid_swap(state, i, j, room):
                    quality_before = self._calculate_swap_quality(state, i, j, old_pos_i, old_pos_j)
                    quality_after = self._calculate_swap_quality(state, i, j, old_pos_j, old_pos_i)
                    
                    if quality_after > quality_before + self.min_improvement:
                        moves.append(ImprovementMove(
                            type=ImprovementType.SWAP,
                            description=f"Swap panels {i} and {j}",
                            panels_affected=[i, j],
                            old_positions=[old_pos_i, old_pos_j],
                            new_positions=[old_pos_j, old_pos_i],
                            coverage_delta=0.0,
                            quality_delta=quality_after - quality_before
                        ))
                        self.moves_accepted += 1
                    else:
                        # Revert swap
                        panel_i.position = old_pos_i
                        panel_j.position = old_pos_j
                else:
                    # Revert swap
                    panel_i.position = old_pos_i
                    panel_j.position = old_pos_j
                
                self.moves_tried += 1
        
        return moves
    
    def _cluster_positions(self, positions: List[float], threshold: float) -> List[float]:
        """Cluster positions to find alignment lines"""
        if not positions:
            return []
        
        sorted_pos = sorted(positions)
        clusters = []
        current_cluster = [sorted_pos[0]]
        
        for pos in sorted_pos[1:]:
            if pos - current_cluster[-1] <= threshold:
                current_cluster.append(pos)
            else:
                clusters.append(np.mean(current_cluster))
                current_cluster = [pos]
        
        clusters.append(np.mean(current_cluster))
        return clusters
    
    def _find_nearest_cluster(self, position: float, clusters: List[float]) -> float:
        """Find nearest cluster center"""
        if not clusters:
            return position
        
        distances = [abs(position - c) for c in clusters]
        min_idx = np.argmin(distances)
        return clusters[min_idx]
    
    def _is_valid_move(self, state: PackingState, panel_idx: int,
                      new_pos: Position, rotated: bool, room: Room) -> bool:
        """Check if moving panel to new position is valid"""
        panel = state.placed_panels[panel_idx]
        
        # Calculate new bounds
        if rotated:
            width, height = panel.panel.height, panel.panel.width
        else:
            width, height = panel.panel.width, panel.panel.height
        
        # Check room boundaries
        if (new_pos.x < 0 or new_pos.y < 0 or
            new_pos.x + width > room.width or
            new_pos.y + height > room.height):
            return False
        
        # Check overlaps with other panels
        new_bounds = (new_pos.x, new_pos.y, new_pos.x + width, new_pos.y + height)
        
        for i, other in enumerate(state.placed_panels):
            if i == panel_idx:
                continue
            
            other_bounds = self._get_panel_bounds(other)
            if self._rectangles_overlap(new_bounds, other_bounds):
                return False
        
        return True
    
    def _is_valid_swap(self, state: PackingState, idx1: int, idx2: int, room: Room) -> bool:
        """Check if swapping two panels is valid"""
        panel1 = state.placed_panels[idx1]
        panel2 = state.placed_panels[idx2]
        
        # Check if each panel fits in the other's position
        bounds1 = self._get_panel_bounds(panel1)
        bounds2 = self._get_panel_bounds(panel2)
        
        # Check boundaries
        if (bounds1[0] < 0 or bounds1[1] < 0 or
            bounds1[2] > room.width or bounds1[3] > room.height):
            return False
        
        if (bounds2[0] < 0 or bounds2[1] < 0 or
            bounds2[2] > room.width or bounds2[3] > room.height):
            return False
        
        # Check overlaps with other panels
        for i, other in enumerate(state.placed_panels):
            if i == idx1 or i == idx2:
                continue
            
            other_bounds = self._get_panel_bounds(other)
            if self._rectangles_overlap(bounds1, other_bounds):
                return False
            if self._rectangles_overlap(bounds2, other_bounds):
                return False
        
        return True
    
    def _rectangles_overlap(self, bounds1: Tuple[float, float, float, float],
                           bounds2: Tuple[float, float, float, float]) -> bool:
        """Check if two rectangles overlap"""
        return not (bounds1[2] <= bounds2[0] or bounds2[2] <= bounds1[0] or
                   bounds1[3] <= bounds2[1] or bounds2[3] <= bounds1[1])
    
    def _get_panel_bounds(self, panel: PlacedPanel) -> Tuple[float, float, float, float]:
        """Get bounding box of placed panel"""
        x, y = panel.position.x, panel.position.y
        if panel.rotated:
            width, height = panel.panel.height, panel.panel.width
        else:
            width, height = panel.panel.width, panel.panel.height
        
        return (x, y, x + width, y + height)
    
    def _calculate_quality_delta(self, state: PackingState, panel_idx: int,
                                new_pos: Position) -> float:
        """Calculate quality improvement from moving panel"""
        # Simple quality metric based on compactness and alignment
        quality = 0.0
        
        # Penalize distance from origin (prefer compact solutions)
        quality -= (new_pos.x + new_pos.y) * 0.01
        
        # Reward alignment with other panels
        for i, other in enumerate(state.placed_panels):
            if i == panel_idx:
                continue
            
            # Reward vertical alignment
            if abs(new_pos.x - other.position.x) < 1.0:
                quality += 1.0
            
            # Reward horizontal alignment
            if abs(new_pos.y - other.position.y) < 1.0:
                quality += 1.0
        
        return quality
    
    def _calculate_rotation_benefit(self, state: PackingState, panel_idx: int,
                                   new_rotated: bool, room: Room) -> float:
        """Calculate benefit of rotating panel"""
        panel = state.placed_panels[panel_idx]
        
        # Check if rotation creates better aspect ratio match with room
        if new_rotated:
            panel_aspect = panel.panel.height / panel.panel.width
        else:
            panel_aspect = panel.panel.width / panel.panel.height
        
        room_aspect = room.width / room.height
        
        # Closer aspect ratio match is better
        aspect_diff = abs(panel_aspect - room_aspect)
        
        return 1.0 / (1.0 + aspect_diff)
    
    def _calculate_position_quality(self, state: PackingState, panel_idx: int,
                                   pos: Position) -> float:
        """Calculate quality of panel at given position"""
        quality = 100.0
        
        # Penalize distance from origin
        quality -= np.sqrt(pos.x**2 + pos.y**2) * 0.1
        
        # Reward edge alignment
        panel = state.placed_panels[panel_idx]
        if pos.x < 1.0 or pos.y < 1.0:
            quality += 10.0
        
        return quality
    
    def _calculate_swap_quality(self, state: PackingState, idx1: int, idx2: int,
                               pos1: Position, pos2: Position) -> float:
        """Calculate quality of swap configuration"""
        return (self._calculate_position_quality(state, idx1, pos1) +
                self._calculate_position_quality(state, idx2, pos2))


class GapFiller:
    """Fills gaps in existing placement"""
    
    def __init__(self):
        self.min_gap_size = 100.0  # Minimum gap area to consider
        self.max_attempts = 1000
        self.gaps_found = 0
        self.gaps_filled = 0
    
    def fill_gaps(self, state: PackingState, room: Room,
                 available_panels: List[Panel]) -> List[ImprovementMove]:
        """Find and fill gaps with available panels"""
        moves = []
        
        # Find gaps in current placement
        gaps = self._find_gaps(state, room)
        self.gaps_found = len(gaps)
        
        # Sort gaps by size (largest first)
        gaps.sort(key=lambda g: g[2] * g[3], reverse=True)
        
        # Try to fill each gap
        for gap in gaps:
            move = self._fill_gap(state, gap, available_panels, room)
            if move:
                moves.append(move)
                self.gaps_filled += 1
        
        logger.info(f"Found {self.gaps_found} gaps, filled {self.gaps_filled}")
        
        return moves
    
    def _find_gaps(self, state: PackingState, room: Room) -> List[Tuple[float, float, float, float]]:
        """Find rectangular gaps in placement"""
        gaps = []
        
        # Create occupancy grid
        grid_resolution = 10
        grid_width = int(room.width / grid_resolution)
        grid_height = int(room.height / grid_resolution)
        occupied = np.zeros((grid_height, grid_width), dtype=bool)
        
        # Mark occupied cells
        for panel in state.placed_panels:
            x1 = int(panel.position.x / grid_resolution)
            y1 = int(panel.position.y / grid_resolution)
            if panel.rotated:
                x2 = x1 + int(panel.panel.height / grid_resolution)
                y2 = y1 + int(panel.panel.width / grid_resolution)
            else:
                x2 = x1 + int(panel.panel.width / grid_resolution)
                y2 = y1 + int(panel.panel.height / grid_resolution)
            
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(grid_width, x2)
            y2 = min(grid_height, y2)
            
            occupied[y1:y2, x1:x2] = True
        
        # Find maximal rectangles in free space
        for y in range(grid_height):
            for x in range(grid_width):
                if not occupied[y, x]:
                    # Find maximal rectangle starting at (x, y)
                    max_width = grid_width - x
                    max_height = grid_height - y
                    
                    # Find actual max width
                    for w in range(1, max_width + 1):
                        if occupied[y, x + w - 1]:
                            max_width = w - 1
                            break
                    
                    if max_width == 0:
                        continue
                    
                    # Find actual max height
                    for h in range(1, max_height + 1):
                        if np.any(occupied[y:y+h, x:x+max_width]):
                            max_height = h - 1
                            break
                    
                    if max_height > 0:
                        gap_x = x * grid_resolution
                        gap_y = y * grid_resolution
                        gap_w = max_width * grid_resolution
                        gap_h = max_height * grid_resolution
                        
                        if gap_w * gap_h >= self.min_gap_size:
                            gaps.append((gap_x, gap_y, gap_w, gap_h))
                            # Mark as processed
                            occupied[y:y+max_height, x:x+max_width] = True
        
        return gaps
    
    def _fill_gap(self, state: PackingState, gap: Tuple[float, float, float, float],
                 available_panels: List[Panel], room: Room) -> Optional[ImprovementMove]:
        """Try to fill a gap with an available panel"""
        gap_x, gap_y, gap_w, gap_h = gap
        
        # Find panels that could fit in the gap
        candidates = []
        for panel in available_panels:
            # Check if panel is already placed
            if any(p.panel == panel for p in state.placed_panels):
                continue
            
            # Check both orientations
            if panel.width <= gap_w and panel.height <= gap_h:
                candidates.append((panel, False))  # Not rotated
            
            if panel.can_rotate and panel.height <= gap_w and panel.width <= gap_h:
                candidates.append((panel, True))  # Rotated
        
        if not candidates:
            return None
        
        # Sort candidates by area (largest first)
        candidates.sort(key=lambda c: c[0].width * c[0].height, reverse=True)
        
        # Try to place best fitting panel
        for panel, rotated in candidates:
            pos = Position(gap_x, gap_y)
            
            # Create temporary placed panel
            new_placed = PlacedPanel(panel=panel, position=pos, rotated=rotated)
            
            # Validate placement
            if self._is_valid_placement(state, new_placed, room):
                # Add to state
                old_coverage = self._calculate_coverage(state, room)
                state.placed_panels.append(new_placed)
                new_coverage = self._calculate_coverage(state, room)
                
                return ImprovementMove(
                    type=ImprovementType.GAP_FILLING,
                    description=f"Fill gap at ({gap_x}, {gap_y})",
                    panels_affected=[len(state.placed_panels) - 1],
                    old_positions=[],
                    new_positions=[pos],
                    coverage_delta=new_coverage - old_coverage,
                    quality_delta=panel.width * panel.height
                )
        
        return None
    
    def _is_valid_placement(self, state: PackingState, new_panel: PlacedPanel,
                          room: Room) -> bool:
        """Check if panel placement is valid"""
        # Get bounds
        if new_panel.rotated:
            width, height = new_panel.panel.height, new_panel.panel.width
        else:
            width, height = new_panel.panel.width, new_panel.panel.height
        
        x, y = new_panel.position.x, new_panel.position.y
        
        # Check boundaries
        if x < 0 or y < 0 or x + width > room.width or y + height > room.height:
            return False
        
        # Check overlaps
        new_bounds = (x, y, x + width, y + height)
        for panel in state.placed_panels:
            panel_bounds = self._get_panel_bounds(panel)
            if self._rectangles_overlap(new_bounds, panel_bounds):
                return False
        
        return True
    
    def _get_panel_bounds(self, panel: PlacedPanel) -> Tuple[float, float, float, float]:
        """Get bounding box of placed panel"""
        x, y = panel.position.x, panel.position.y
        if panel.rotated:
            width, height = panel.panel.height, panel.panel.width
        else:
            width, height = panel.panel.width, panel.panel.height
        
        return (x, y, x + width, y + height)
    
    def _rectangles_overlap(self, bounds1: Tuple[float, float, float, float],
                           bounds2: Tuple[float, float, float, float]) -> bool:
        """Check if two rectangles overlap"""
        return not (bounds1[2] <= bounds2[0] or bounds2[2] <= bounds1[0] or
                   bounds1[3] <= bounds2[1] or bounds2[3] <= bounds1[1])
    
    def _calculate_coverage(self, state: PackingState, room: Room) -> float:
        """Calculate coverage percentage"""
        total_area = sum(p.panel.width * p.panel.height for p in state.placed_panels)
        room_area = room.width * room.height
        return total_area / room_area if room_area > 0 else 0.0


class ConsolidationOptimizer:
    """Consolidates panels to reduce fragmentation"""
    
    def __init__(self):
        self.consolidation_force = 10.0
        self.max_iterations = 50
        self.convergence_threshold = 0.1
    
    def consolidate(self, state: PackingState, room: Room) -> List[ImprovementMove]:
        """Consolidate panels to reduce fragmentation"""
        moves = []
        
        for iteration in range(self.max_iterations):
            # Calculate center of mass
            center = self._calculate_center_of_mass(state)
            
            # Apply consolidation forces
            iteration_moves = []
            total_movement = 0.0
            
            for i, panel in enumerate(state.placed_panels):
                old_pos = panel.position
                new_pos = self._apply_consolidation_force(state, i, center, room)
                
                if new_pos != old_pos:
                    movement = np.sqrt((new_pos.x - old_pos.x)**2 + 
                                     (new_pos.y - old_pos.y)**2)
                    total_movement += movement
                    
                    iteration_moves.append(ImprovementMove(
                        type=ImprovementType.CONSOLIDATION,
                        description=f"Consolidate panel {i}",
                        panels_affected=[i],
                        old_positions=[old_pos],
                        new_positions=[new_pos],
                        coverage_delta=0.0,
                        quality_delta=movement
                    ))
                    panel.position = new_pos
            
            moves.extend(iteration_moves)
            
            # Check convergence
            if total_movement < self.convergence_threshold:
                logger.debug(f"Consolidation converged at iteration {iteration}")
                break
        
        return moves
    
    def _calculate_center_of_mass(self, state: PackingState) -> Position:
        """Calculate center of mass of all panels"""
        if not state.placed_panels:
            return Position(0, 0)
        
        total_area = 0.0
        weighted_x = 0.0
        weighted_y = 0.0
        
        for panel in state.placed_panels:
            area = panel.panel.width * panel.panel.height
            center_x = panel.position.x + panel.panel.width / 2
            center_y = panel.position.y + panel.panel.height / 2
            
            total_area += area
            weighted_x += center_x * area
            weighted_y += center_y * area
        
        if total_area > 0:
            return Position(weighted_x / total_area, weighted_y / total_area)
        
        return Position(0, 0)
    
    def _apply_consolidation_force(self, state: PackingState, panel_idx: int,
                                  center: Position, room: Room) -> Position:
        """Apply consolidation force to panel"""
        panel = state.placed_panels[panel_idx]
        old_pos = panel.position
        
        # Calculate force vector toward center
        panel_center_x = old_pos.x + panel.panel.width / 2
        panel_center_y = old_pos.y + panel.panel.height / 2
        
        force_x = (center.x - panel_center_x) * self.consolidation_force / 100
        force_y = (center.y - panel_center_y) * self.consolidation_force / 100
        
        # Apply force with step limit
        max_step = 5.0
        force_x = np.clip(force_x, -max_step, max_step)
        force_y = np.clip(force_y, -max_step, max_step)
        
        new_x = old_pos.x + force_x
        new_y = old_pos.y + force_y
        
        # Ensure valid position
        new_pos = Position(new_x, new_y)
        
        if self._is_valid_move(state, panel_idx, new_pos, room):
            return new_pos
        
        # Try smaller movements if full movement is invalid
        for scale in [0.5, 0.25, 0.1]:
            scaled_x = old_pos.x + force_x * scale
            scaled_y = old_pos.y + force_y * scale
            scaled_pos = Position(scaled_x, scaled_y)
            
            if self._is_valid_move(state, panel_idx, scaled_pos, room):
                return scaled_pos
        
        return old_pos
    
    def _is_valid_move(self, state: PackingState, panel_idx: int,
                      new_pos: Position, room: Room) -> bool:
        """Check if moving panel to new position is valid"""
        panel = state.placed_panels[panel_idx]
        
        # Get dimensions
        if panel.rotated:
            width, height = panel.panel.height, panel.panel.width
        else:
            width, height = panel.panel.width, panel.panel.height
        
        # Check boundaries
        if (new_pos.x < 0 or new_pos.y < 0 or
            new_pos.x + width > room.width or
            new_pos.y + height > room.height):
            return False
        
        # Check overlaps
        new_bounds = (new_pos.x, new_pos.y, new_pos.x + width, new_pos.y + height)
        
        for i, other in enumerate(state.placed_panels):
            if i == panel_idx:
                continue
            
            other_bounds = self._get_panel_bounds(other)
            if self._rectangles_overlap(new_bounds, other_bounds):
                return False
        
        return True
    
    def _get_panel_bounds(self, panel: PlacedPanel) -> Tuple[float, float, float, float]:
        """Get bounding box of placed panel"""
        x, y = panel.position.x, panel.position.y
        if panel.rotated:
            width, height = panel.panel.height, panel.panel.width
        else:
            width, height = panel.panel.width, panel.panel.height
        
        return (x, y, x + width, y + height)
    
    def _rectangles_overlap(self, bounds1: Tuple[float, float, float, float],
                           bounds2: Tuple[float, float, float, float]) -> bool:
        """Check if two rectangles overlap"""
        return not (bounds1[2] <= bounds2[0] or bounds2[2] <= bounds1[0] or
                   bounds1[3] <= bounds2[1] or bounds2[3] <= bounds1[1])


class Compactor:
    """Compacts solution to minimize used space"""
    
    def __init__(self):
        self.compact_directions = ["left", "bottom", "diagonal"]
        self.max_iterations = 100
    
    def compact(self, state: PackingState, room: Room) -> List[ImprovementMove]:
        """Compact panels to minimize used space"""
        moves = []
        
        for direction in self.compact_directions:
            direction_moves = self._compact_direction(state, room, direction)
            moves.extend(direction_moves)
        
        return moves
    
    def _compact_direction(self, state: PackingState, room: Room,
                          direction: str) -> List[ImprovementMove]:
        """Compact panels in given direction"""
        moves = []
        
        # Sort panels based on direction
        if direction == "left":
            sorted_indices = sorted(range(len(state.placed_panels)),
                                  key=lambda i: state.placed_panels[i].position.x)
        elif direction == "bottom":
            sorted_indices = sorted(range(len(state.placed_panels)),
                                  key=lambda i: state.placed_panels[i].position.y)
        else:  # diagonal
            sorted_indices = sorted(range(len(state.placed_panels)),
                                  key=lambda i: (state.placed_panels[i].position.x +
                                                state.placed_panels[i].position.y))
        
        # Try to move each panel in order
        for idx in sorted_indices:
            panel = state.placed_panels[idx]
            old_pos = panel.position
            new_pos = self._find_compact_position(state, idx, direction, room)
            
            if new_pos != old_pos:
                distance = np.sqrt((new_pos.x - old_pos.x)**2 + 
                                 (new_pos.y - old_pos.y)**2)
                
                moves.append(ImprovementMove(
                    type=ImprovementType.COMPACTION,
                    description=f"Compact panel {idx} {direction}",
                    panels_affected=[idx],
                    old_positions=[old_pos],
                    new_positions=[new_pos],
                    coverage_delta=0.0,
                    quality_delta=distance
                ))
                panel.position = new_pos
        
        return moves
    
    def _find_compact_position(self, state: PackingState, panel_idx: int,
                              direction: str, room: Room) -> Position:
        """Find most compact position for panel in given direction"""
        panel = state.placed_panels[panel_idx]
        old_pos = panel.position
        best_pos = old_pos
        
        if direction == "left":
            # Try to move left
            for x in range(int(old_pos.x) - 1, -1, -1):
                new_pos = Position(float(x), old_pos.y)
                if self._is_valid_move(state, panel_idx, new_pos, room):
                    best_pos = new_pos
                else:
                    break
        
        elif direction == "bottom":
            # Try to move down
            for y in range(int(old_pos.y) - 1, -1, -1):
                new_pos = Position(old_pos.x, float(y))
                if self._is_valid_move(state, panel_idx, new_pos, room):
                    best_pos = new_pos
                else:
                    break
        
        else:  # diagonal
            # Try to move diagonally toward origin
            step = 1.0
            for i in range(1, int(max(old_pos.x, old_pos.y)) + 1):
                new_x = max(0, old_pos.x - step * i)
                new_y = max(0, old_pos.y - step * i)
                new_pos = Position(new_x, new_y)
                
                if self._is_valid_move(state, panel_idx, new_pos, room):
                    best_pos = new_pos
                else:
                    break
        
        return best_pos
    
    def _is_valid_move(self, state: PackingState, panel_idx: int,
                      new_pos: Position, room: Room) -> bool:
        """Check if moving panel to new position is valid"""
        panel = state.placed_panels[panel_idx]
        
        # Get dimensions
        if panel.rotated:
            width, height = panel.panel.height, panel.panel.width
        else:
            width, height = panel.panel.width, panel.panel.height
        
        # Check boundaries
        if (new_pos.x < 0 or new_pos.y < 0 or
            new_pos.x + width > room.width or
            new_pos.y + height > room.height):
            return False
        
        # Check overlaps
        new_bounds = (new_pos.x, new_pos.y, new_pos.x + width, new_pos.y + height)
        
        for i, other in enumerate(state.placed_panels):
            if i == panel_idx:
                continue
            
            other_bounds = self._get_panel_bounds(other)
            if self._rectangles_overlap(new_bounds, other_bounds):
                return False
        
        return True
    
    def _get_panel_bounds(self, panel: PlacedPanel) -> Tuple[float, float, float, float]:
        """Get bounding box of placed panel"""
        x, y = panel.position.x, panel.position.y
        if panel.rotated:
            width, height = panel.panel.height, panel.panel.width
        else:
            width, height = panel.panel.width, panel.panel.height
        
        return (x, y, x + width, y + height)
    
    def _rectangles_overlap(self, bounds1: Tuple[float, float, float, float],
                           bounds2: Tuple[float, float, float, float]) -> bool:
        """Check if two rectangles overlap"""
        return not (bounds1[2] <= bounds2[0] or bounds2[2] <= bounds1[0] or
                   bounds1[3] <= bounds2[1] or bounds2[3] <= bounds1[1])


class SolutionImprover:
    """Main improvement system that coordinates all improvement strategies"""
    
    def __init__(self):
        self.refinement = PostOptimizationRefinement()
        self.gap_filler = GapFiller()
        self.consolidator = ConsolidationOptimizer()
        self.compactor = Compactor()
        self.validator = SolutionValidator()
        self.improvement_cache = {}
    
    def improve(self, state: PackingState, room: Room,
               available_panels: Optional[List[Panel]] = None,
               time_limit: float = 10.0) -> ImprovementResult:
        """Apply all improvement strategies to solution"""
        import time
        start_time = time.time()
        
        # Create working copy
        working_state = copy.deepcopy(state)
        
        # Calculate initial metrics
        coverage_before = self._calculate_coverage(working_state, room)
        quality_before = self._calculate_quality(working_state, room)
        
        all_moves = []
        iterations = 0
        
        # Apply improvement strategies in sequence
        while time.time() - start_time < time_limit:
            iteration_moves = []
            
            # 1. Post-optimization refinement
            if time.time() - start_time < time_limit * 0.3:
                moves = self.refinement.refine(working_state, room)
                iteration_moves.extend(moves)
            
            # 2. Gap filling
            if available_panels and time.time() - start_time < time_limit * 0.5:
                moves = self.gap_filler.fill_gaps(working_state, room, available_panels)
                iteration_moves.extend(moves)
            
            # 3. Consolidation
            if time.time() - start_time < time_limit * 0.7:
                moves = self.consolidator.consolidate(working_state, room)
                iteration_moves.extend(moves)
            
            # 4. Compaction
            if time.time() - start_time < time_limit * 0.9:
                moves = self.compactor.compact(working_state, room)
                iteration_moves.extend(moves)
            
            all_moves.extend(iteration_moves)
            iterations += 1
            
            # Check if no more improvements found
            if not iteration_moves:
                logger.debug(f"No more improvements found at iteration {iterations}")
                break
            
            # Validate current state
            validation = self.validator.validate(working_state, room)
            if not validation.is_valid:
                logger.warning(f"Invalid state after improvements: {validation.error_count} errors")
                # Revert last moves
                for move in reversed(iteration_moves):
                    for i, old_pos in zip(move.panels_affected, move.old_positions):
                        if i < len(working_state.placed_panels):
                            working_state.placed_panels[i].position = old_pos
                iteration_moves.clear()
                break
        
        # Calculate final metrics
        coverage_after = self._calculate_coverage(working_state, room)
        quality_after = self._calculate_quality(working_state, room)
        improvement_time = time.time() - start_time
        
        # Log improvement summary
        logger.info(f"Improvement complete: coverage {coverage_before:.1%} -> {coverage_after:.1%}, "
                   f"quality {quality_before:.1f} -> {quality_after:.1f}, "
                   f"{len(all_moves)} moves in {iterations} iterations")
        
        return ImprovementResult(
            improved_state=working_state,
            moves_applied=all_moves,
            coverage_before=coverage_before,
            coverage_after=coverage_after,
            quality_before=quality_before,
            quality_after=quality_after,
            improvement_time=improvement_time,
            iterations=iterations
        )
    
    def improve_incremental(self, state: PackingState, room: Room,
                          improvement_type: ImprovementType) -> ImprovementResult:
        """Apply specific improvement strategy"""
        working_state = copy.deepcopy(state)
        
        coverage_before = self._calculate_coverage(working_state, room)
        quality_before = self._calculate_quality(working_state, room)
        
        moves = []
        
        if improvement_type == ImprovementType.REFINEMENT:
            moves = self.refinement.refine(working_state, room)
        elif improvement_type == ImprovementType.CONSOLIDATION:
            moves = self.consolidator.consolidate(working_state, room)
        elif improvement_type == ImprovementType.COMPACTION:
            moves = self.compactor.compact(working_state, room)
        
        coverage_after = self._calculate_coverage(working_state, room)
        quality_after = self._calculate_quality(working_state, room)
        
        return ImprovementResult(
            improved_state=working_state,
            moves_applied=moves,
            coverage_before=coverage_before,
            coverage_after=coverage_after,
            quality_before=quality_before,
            quality_after=quality_after,
            improvement_time=0.0,
            iterations=1
        )
    
    def _calculate_coverage(self, state: PackingState, room: Room) -> float:
        """Calculate coverage percentage"""
        total_area = sum(p.panel.width * p.panel.height for p in state.placed_panels)
        room_area = room.width * room.height
        return total_area / room_area if room_area > 0 else 0.0
    
    def _calculate_quality(self, state: PackingState, room: Room) -> float:
        """Calculate solution quality score"""
        quality = 0.0
        
        # Coverage component
        coverage = self._calculate_coverage(state, room)
        quality += coverage * 100
        
        # Compactness component
        if state.placed_panels:
            max_x = max(p.position.x + (p.panel.height if p.rotated else p.panel.width)
                       for p in state.placed_panels)
            max_y = max(p.position.y + (p.panel.width if p.rotated else p.panel.height)
                       for p in state.placed_panels)
            
            used_area = max_x * max_y
            compactness = sum(p.panel.width * p.panel.height for p in state.placed_panels) / used_area
            quality += compactness * 50
        
        # Alignment component
        alignment_score = self._calculate_alignment_score(state)
        quality += alignment_score * 20
        
        return quality
    
    def _calculate_alignment_score(self, state: PackingState) -> float:
        """Calculate panel alignment score"""
        if len(state.placed_panels) < 2:
            return 1.0
        
        aligned_pairs = 0
        total_pairs = 0
        
        for i in range(len(state.placed_panels)):
            for j in range(i + 1, len(state.placed_panels)):
                total_pairs += 1
                
                panel_i = state.placed_panels[i]
                panel_j = state.placed_panels[j]
                
                # Check vertical alignment
                if abs(panel_i.position.x - panel_j.position.x) < 1.0:
                    aligned_pairs += 1
                # Check horizontal alignment
                elif abs(panel_i.position.y - panel_j.position.y) < 1.0:
                    aligned_pairs += 1
        
        return aligned_pairs / total_pairs if total_pairs > 0 else 0.0
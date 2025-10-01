#!/usr/bin/env python3
"""
blf_optimizer.py - Enhanced Bottom-Left-Fill Algorithm
=======================================================
Production implementation of Bottom-Left-Fill with backtracking
for achieving 95%+ panel coverage.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Set, Optional, Dict, Any
import heapq
from collections import deque

from models import Panel, PanelSize, Point, Room
from advanced_packing import (
    AbstractOptimizer, 
    OptimizerConfig,
    PackingState,
    PanelPlacement,
    StateTransition,
    StateValidator,
    OccupancyGrid,
    OptimizationMetrics
)


# Step 2.1.1: BLF Placement Engine
# =================================

@dataclass
class SkylineSegment:
    """Represents a segment of the skyline."""
    x_start: float
    x_end: float
    y: float
    
    @property
    def width(self) -> float:
        """Width of the segment."""
        return self.x_end - self.x_start
    
    def can_fit_panel(self, panel_width: float) -> bool:
        """Check if panel can fit on this segment."""
        return self.width >= panel_width


class Skyline:
    """
    Manages the skyline structure for Bottom-Left-Fill algorithm.
    The skyline represents the upper boundary of placed panels.
    """
    
    def __init__(self, room_width: float, room_height: float):
        """Initialize skyline with room dimensions."""
        self.room_width = room_width
        self.room_height = room_height
        
        # Start with a single segment at the bottom
        self.segments: List[SkylineSegment] = [
            SkylineSegment(x_start=0, x_end=room_width, y=0)
        ]
    
    def find_lowest_position(self, panel_width: float, panel_height: float) -> Optional[Tuple[float, float]]:
        """
        Find the lowest valid position for a panel on the skyline.
        Returns (x, y) position or None if no valid position exists.
        """
        best_position = None
        best_y = float('inf')
        
        for segment in self.segments:
            # Check if panel fits on this segment
            if not segment.can_fit_panel(panel_width):
                continue
            
            # Check if panel would exceed room height
            if segment.y + panel_height > self.room_height:
                continue
            
            # Check if this is the lowest position so far
            if segment.y < best_y:
                best_y = segment.y
                best_position = (segment.x_start, segment.y)
            elif segment.y == best_y and best_position:
                # Prefer leftmost position at same height
                if segment.x_start < best_position[0]:
                    best_position = (segment.x_start, segment.y)
        
        return best_position
    
    def add_panel(self, x: float, y: float, width: float, height: float):
        """
        Update skyline after placing a panel.
        This modifies the skyline segments to reflect the new upper boundary.
        """
        panel_top = y + height
        panel_left = x
        panel_right = x + width
        
        # Find segments that intersect with the panel
        new_segments = []
        
        for segment in self.segments:
            # No intersection - keep segment as is
            if segment.x_end <= panel_left or segment.x_start >= panel_right:
                new_segments.append(segment)
                continue
            
            # Split segment if needed
            if segment.x_start < panel_left:
                # Left part remains at original height
                new_segments.append(
                    SkylineSegment(segment.x_start, panel_left, segment.y)
                )
            
            # Middle part (covered by panel) raises to panel top
            overlap_start = max(segment.x_start, panel_left)
            overlap_end = min(segment.x_end, panel_right)
            if overlap_start < overlap_end:
                new_segments.append(
                    SkylineSegment(overlap_start, overlap_end, panel_top)
                )
            
            if segment.x_end > panel_right:
                # Right part remains at original height
                new_segments.append(
                    SkylineSegment(panel_right, segment.x_end, segment.y)
                )
        
        # Merge adjacent segments with same height
        self.segments = self._merge_segments(new_segments)
    
    def _merge_segments(self, segments: List[SkylineSegment]) -> List[SkylineSegment]:
        """Merge adjacent segments with the same y-coordinate."""
        if not segments:
            return []
        
        # Sort by x_start
        segments.sort(key=lambda s: s.x_start)
        
        merged = []
        current = segments[0]
        
        for segment in segments[1:]:
            if (current.x_end == segment.x_start and 
                abs(current.y - segment.y) < 0.01):  # Same height
                # Merge segments
                current = SkylineSegment(current.x_start, segment.x_end, current.y)
            else:
                merged.append(current)
                current = segment
        
        merged.append(current)
        return merged
    
    def get_waste_area(self) -> float:
        """Calculate wasted area below the skyline."""
        waste = 0.0
        for segment in self.segments:
            waste += segment.width * segment.y
        return waste
    
    def copy(self) -> 'Skyline':
        """Create a deep copy of the skyline."""
        new_skyline = Skyline(self.room_width, self.room_height)
        new_skyline.segments = [
            SkylineSegment(s.x_start, s.x_end, s.y) 
            for s in self.segments
        ]
        return new_skyline


class BLFPositionGenerator:
    """
    Generates candidate positions for panel placement using BLF strategy.
    Positions are generated in bottom-left priority order.
    """
    
    def __init__(self, room: Room, grid_resolution: float = 0.5):
        """Initialize position generator."""
        self.room = room
        self.grid_resolution = grid_resolution
        self.skyline = Skyline(room.width, room.height)
    
    def generate_positions(self, panel_size: PanelSize, 
                          placed_panels: Set[PanelPlacement]) -> List[Tuple[float, float]]:
        """
        Generate candidate positions for a panel size.
        Returns positions sorted by bottom-left priority.
        """
        positions = []
        
        # Try both orientations
        for orientation in ["horizontal", "vertical"]:
            width, height = panel_size.get_dimensions(orientation)
            
            # Find skyline position
            skyline_pos = self.skyline.find_lowest_position(width, height)
            if skyline_pos:
                positions.append(skyline_pos)
            
            # Generate grid positions for more options
            positions.extend(
                self._generate_grid_positions(width, height, placed_panels)
            )
        
        # Sort by bottom-left priority (y first, then x)
        positions.sort(key=lambda p: (p[1], p[0]))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_positions = []
        for pos in positions:
            if pos not in seen:
                seen.add(pos)
                unique_positions.append(pos)
        
        return unique_positions
    
    def _generate_grid_positions(self, width: float, height: float,
                                placed_panels: Set[PanelPlacement]) -> List[Tuple[float, float]]:
        """Generate positions on a grid, avoiding overlaps."""
        positions = []
        
        # Create occupancy grid for collision detection
        occupancy = OccupancyGrid(
            (0, 0, self.room.width, self.room.height),
            self.grid_resolution
        )
        
        # Mark placed panels
        for panel in placed_panels:
            occupancy.mark_occupied(panel.bounds)
        
        # Sample positions
        x = 0
        while x + width <= self.room.width:
            y = 0
            while y + height <= self.room.height:
                # Check if position is valid
                bounds = (x, y, x + width, y + height)
                if occupancy.is_region_free(bounds):
                    positions.append((x, y))
                
                y += self.grid_resolution
            x += self.grid_resolution
        
        return positions
    
    def update_skyline(self, placement: PanelPlacement):
        """Update skyline after placing a panel."""
        width, height = placement.panel_size.get_dimensions(placement.orientation)
        self.skyline.add_panel(
            placement.position[0],
            placement.position[1],
            width,
            height
        )


class BLFCollisionDetector:
    """
    Efficient collision detection for BLF algorithm.
    Uses spatial indexing for fast overlap checking.
    """
    
    def __init__(self, room_bounds: Tuple[float, float, float, float]):
        """Initialize collision detector."""
        self.room_bounds = room_bounds
        self.occupancy_grid = OccupancyGrid(room_bounds, resolution=0.25)
        self.placed_panels: Set[PanelPlacement] = set()
    
    def is_valid_placement(self, placement: PanelPlacement) -> bool:
        """
        Check if placement is valid (no collisions, within bounds).
        """
        # Check room boundaries
        x1, y1, x2, y2 = placement.bounds
        rx, ry, rw, rh = self.room_bounds
        
        if x1 < rx or y1 < ry or x2 > rx + rw or y2 > ry + rh:
            return False
        
        # Check collisions with existing panels
        return self.occupancy_grid.is_region_free(placement.bounds)
    
    def add_placement(self, placement: PanelPlacement):
        """Add a placement to the collision detector."""
        self.placed_panels.add(placement)
        self.occupancy_grid.mark_occupied(placement.bounds)
    
    def remove_placement(self, placement: PanelPlacement):
        """Remove a placement from the collision detector."""
        if placement in self.placed_panels:
            self.placed_panels.remove(placement)
            self.occupancy_grid.mark_free(placement.bounds)
    
    def get_overlapping_placements(self, placement: PanelPlacement) -> List[PanelPlacement]:
        """Get all placements that overlap with the given placement."""
        overlapping = []
        for existing in self.placed_panels:
            if placement.overlaps(existing):
                overlapping.append(existing)
        return overlapping
    
    def clear(self):
        """Clear all placements."""
        self.placed_panels.clear()
        self.occupancy_grid = OccupancyGrid(self.room_bounds, resolution=0.25)


class BLFPlacementValidator:
    """
    Validates panel placements according to BLF rules and constraints.
    """
    
    def __init__(self, room: Room, config: OptimizerConfig):
        """Initialize placement validator."""
        self.room = room
        self.config = config
        self.collision_detector = BLFCollisionDetector(
            (room.position.x, room.position.y, room.width, room.height)
        )
    
    def validate_placement(self, placement: PanelPlacement, 
                          current_state: PackingState) -> Tuple[bool, Optional[str]]:
        """
        Validate a placement against all constraints.
        Returns (is_valid, error_message).
        """
        # Check basic validity
        if not current_state.is_valid_placement(placement):
            return False, "Placement violates basic constraints"
        
        # Check collision detection
        if not self.collision_detector.is_valid_placement(placement):
            return False, "Placement causes collision"
        
        # Check panel size constraints
        if placement.panel_size not in self.config.panel_sizes:
            return False, f"Panel size {placement.panel_size} not allowed"
        
        # Check orientation constraints (if any)
        if self._violates_orientation_rules(placement):
            return False, "Placement violates orientation rules"
        
        # Check structural requirements
        if self._violates_structural_rules(placement, current_state):
            return False, "Placement violates structural requirements"
        
        return True, None
    
    def _violates_orientation_rules(self, placement: PanelPlacement) -> bool:
        """Check if placement violates orientation rules."""
        # For narrow rooms, prefer panels aligned with long dimension
        room_aspect = self.room.width / self.room.height
        panel_width, panel_height = placement.panel_size.get_dimensions(placement.orientation)
        panel_aspect = panel_width / panel_height
        
        # If room is very narrow (aspect > 3), prefer aligned panels
        if room_aspect > 3 or room_aspect < 1/3:
            if abs(room_aspect - panel_aspect) > abs(room_aspect - 1/panel_aspect):
                return True  # Panel is perpendicular to room's long dimension
        
        return False
    
    def _violates_structural_rules(self, placement: PanelPlacement, 
                                   state: PackingState) -> bool:
        """Check if placement violates structural requirements."""
        # Check if panel has adequate support
        x1, y1, x2, y2 = placement.bounds
        
        # Panels should have support from below (except at floor level)
        if y1 > 0.1:  # Not at floor
            # Check for support from existing panels
            has_support = False
            for existing in state.placements:
                ex1, ey1, ex2, ey2 = existing.bounds
                
                # Check if existing panel provides support from below
                if abs(ey2 - y1) < 0.1:  # Top of existing aligns with bottom of new
                    # Check overlap in x-direction (at least 50% support)
                    overlap_start = max(x1, ex1)
                    overlap_end = min(x2, ex2)
                    overlap_width = max(0, overlap_end - overlap_start)
                    
                    if overlap_width >= (x2 - x1) * 0.5:
                        has_support = True
                        break
            
            if not has_support:
                return True
        
        return False
    
    def validate_state(self, state: PackingState) -> Tuple[bool, List[str]]:
        """
        Validate entire packing state.
        Returns (is_valid, list_of_violations).
        """
        violations = []
        
        # Use base validator for standard checks
        if not StateValidator.validate_state(state):
            violations.extend(StateValidator.get_violations(state))
        
        # Additional BLF-specific validations
        for placement in state.placements:
            is_valid, error = self.validate_placement(placement, state)
            if not is_valid:
                violations.append(f"Panel at {placement.position}: {error}")
        
        return len(violations) == 0, violations
    
    def update_collision_detector(self, state: PackingState):
        """Update collision detector with current state."""
        self.collision_detector.clear()
        for placement in state.placements:
            self.collision_detector.add_placement(placement)


# Step 2.1.2: Sorting Strategies
# ===============================

from enum import Enum
import itertools
import random


class SortCriteria(Enum):
    """Criteria for sorting panels."""
    AREA_DESC = "area_desc"  # Largest area first
    AREA_ASC = "area_asc"    # Smallest area first
    WIDTH_DESC = "width_desc"  # Widest first
    WIDTH_ASC = "width_asc"  # Narrowest first
    HEIGHT_DESC = "height_desc"  # Tallest first
    HEIGHT_ASC = "height_asc"  # Shortest first
    PERIMETER_DESC = "perimeter_desc"  # Largest perimeter first
    PERIMETER_ASC = "perimeter_asc"  # Smallest perimeter first
    ASPECT_RATIO = "aspect_ratio"  # Most square-like first
    RANDOM = "random"  # Random order


class MultiCriteriaSorter:
    """
    Sorts panels using multiple criteria with configurable weights.
    """
    
    def __init__(self):
        """Initialize multi-criteria sorter."""
        self.criteria_weights = {
            SortCriteria.AREA_DESC: 1.0,
            SortCriteria.WIDTH_DESC: 0.5,
            SortCriteria.HEIGHT_DESC: 0.5,
            SortCriteria.PERIMETER_DESC: 0.3,
            SortCriteria.ASPECT_RATIO: 0.2
        }
    
    def sort_panels(self, panel_sizes: List[PanelSize], 
                   primary_criteria: SortCriteria,
                   secondary_criteria: Optional[SortCriteria] = None) -> List[PanelSize]:
        """
        Sort panels by specified criteria.
        """
        def get_sort_key(panel_size: PanelSize) -> Tuple:
            """Generate sort key for panel based on criteria."""
            # Use horizontal dimensions for consistency
            width, height = panel_size.get_dimensions("horizontal")
            
            primary_value = self._get_criteria_value(panel_size, primary_criteria)
            
            if secondary_criteria:
                secondary_value = self._get_criteria_value(panel_size, secondary_criteria)
                return (primary_value, secondary_value)
            
            return (primary_value,)
        
        if primary_criteria == SortCriteria.RANDOM:
            sorted_panels = list(panel_sizes)
            random.shuffle(sorted_panels)
            return sorted_panels
        
        # Sort with appropriate reverse flag (DESC = reverse=True for largest first)
        reverse = primary_criteria in [
            SortCriteria.AREA_DESC,
            SortCriteria.WIDTH_DESC,
            SortCriteria.HEIGHT_DESC,
            SortCriteria.PERIMETER_DESC
        ]
        
        return sorted(panel_sizes, key=get_sort_key, reverse=reverse)
    
    def _get_criteria_value(self, panel_size: PanelSize, criteria: SortCriteria) -> float:
        """Get sort value for panel based on criteria."""
        width, height = panel_size.get_dimensions("horizontal")
        
        if criteria in [SortCriteria.AREA_DESC, SortCriteria.AREA_ASC]:
            return float(panel_size.area)
        elif criteria in [SortCriteria.WIDTH_DESC, SortCriteria.WIDTH_ASC]:
            return float(width)
        elif criteria in [SortCriteria.HEIGHT_DESC, SortCriteria.HEIGHT_ASC]:
            return float(height)
        elif criteria in [SortCriteria.PERIMETER_DESC, SortCriteria.PERIMETER_ASC]:
            return float(2 * (width + height))
        elif criteria == SortCriteria.ASPECT_RATIO:
            # Return distance from square (1.0 is perfect square)
            return float(abs(1.0 - (width / height)))
        else:
            return 0.0
    
    def get_weighted_score(self, panel_size: PanelSize) -> float:
        """
        Calculate weighted score for panel using all criteria.
        """
        score = 0.0
        
        for criteria, weight in self.criteria_weights.items():
            if weight > 0:
                value = self._get_criteria_value(panel_size, criteria)
                # Normalize value (assuming max panel dimension is 8)
                normalized = value / 64.0 if criteria == SortCriteria.AREA_DESC else value / 8.0
                score += weight * normalized
        
        return score
    
    def sort_by_weighted_score(self, panel_sizes: List[PanelSize]) -> List[PanelSize]:
        """Sort panels by weighted multi-criteria score."""
        return sorted(panel_sizes, key=self.get_weighted_score, reverse=True)


class AdaptiveSorter:
    """
    Adaptively adjusts sorting strategy based on room characteristics and performance.
    """
    
    def __init__(self, room: Room):
        """Initialize adaptive sorter."""
        self.room = room
        self.sorter = MultiCriteriaSorter()
        self.performance_history: Dict[SortCriteria, List[float]] = {}
        
        # Determine initial strategy based on room
        self.current_strategy = self._determine_initial_strategy()
    
    def _determine_initial_strategy(self) -> SortCriteria:
        """Determine initial sorting strategy based on room characteristics."""
        aspect_ratio = self.room.width / self.room.height
        area = self.room.width * self.room.height
        
        # For square rooms, prefer area-based sorting
        if 0.8 <= aspect_ratio <= 1.2:
            return SortCriteria.AREA_DESC
        
        # For narrow rooms, prefer width-based sorting
        elif aspect_ratio > 2 or aspect_ratio < 0.5:
            return SortCriteria.WIDTH_DESC if aspect_ratio > 1 else SortCriteria.HEIGHT_DESC
        
        # For small rooms, try smallest first
        elif area < 100:
            return SortCriteria.AREA_ASC
        
        # Default to largest area first
        else:
            return SortCriteria.AREA_DESC
    
    def sort_panels(self, panel_sizes: List[PanelSize]) -> List[PanelSize]:
        """Sort panels using current adaptive strategy."""
        return self.sorter.sort_panels(panel_sizes, self.current_strategy)
    
    def record_performance(self, criteria: SortCriteria, coverage: float):
        """Record performance of a sorting strategy."""
        if criteria not in self.performance_history:
            self.performance_history[criteria] = []
        
        self.performance_history[criteria].append(coverage)
    
    def adapt_strategy(self):
        """Adapt sorting strategy based on performance history."""
        if not self.performance_history:
            return
        
        # Calculate average performance for each strategy
        avg_performance = {}
        for criteria, performances in self.performance_history.items():
            if performances:
                avg_performance[criteria] = sum(performances) / len(performances)
        
        # Select best performing strategy
        if avg_performance:
            best_criteria = max(avg_performance, key=avg_performance.get)
            self.current_strategy = best_criteria
    
    def get_alternative_strategies(self) -> List[SortCriteria]:
        """Get alternative sorting strategies to try."""
        alternatives = []
        
        # Add strategies based on room characteristics
        aspect_ratio = self.room.width / self.room.height
        
        if aspect_ratio > 1.5:
            alternatives.extend([SortCriteria.WIDTH_DESC, SortCriteria.PERIMETER_DESC])
        elif aspect_ratio < 0.67:
            alternatives.extend([SortCriteria.HEIGHT_DESC, SortCriteria.PERIMETER_DESC])
        else:
            alternatives.extend([SortCriteria.AREA_DESC, SortCriteria.AREA_ASC])
        
        # Add aspect ratio sorting for mixed panel sizes
        alternatives.append(SortCriteria.ASPECT_RATIO)
        
        # Remove current strategy and duplicates
        alternatives = [s for s in alternatives if s != self.current_strategy]
        return list(set(alternatives))


class PermutationGenerator:
    """
    Generates panel permutations for exploring different placement orders.
    """
    
    def __init__(self, max_permutations: int = 100):
        """Initialize permutation generator."""
        self.max_permutations = max_permutations
    
    def generate_permutations(self, panel_sizes: List[PanelSize]) -> List[List[PanelSize]]:
        """
        Generate permutations of panel sizes.
        Limits number of permutations for large sets.
        """
        n = len(panel_sizes)
        
        if n <= 6:
            # Generate all permutations for small sets
            return list(itertools.permutations(panel_sizes))
        
        else:
            # Sample random permutations for large sets
            permutations = []
            
            # Always include sorted orders
            sorter = MultiCriteriaSorter()
            
            # Add key sorted orders
            for criteria in [SortCriteria.AREA_DESC, SortCriteria.AREA_ASC,
                           SortCriteria.WIDTH_DESC, SortCriteria.HEIGHT_DESC]:
                sorted_panels = sorter.sort_panels(panel_sizes, criteria)
                permutations.append(sorted_panels)
            
            # Add random permutations
            remaining = self.max_permutations - len(permutations)
            for _ in range(remaining):
                perm = list(panel_sizes)
                random.shuffle(perm)
                permutations.append(perm)
            
            return permutations
    
    def generate_partial_permutations(self, panel_sizes: List[PanelSize], 
                                     k: int = 3) -> List[List[PanelSize]]:
        """
        Generate permutations by varying order of first k panels only.
        More efficient for large panel sets.
        """
        if len(panel_sizes) <= k:
            return self.generate_permutations(panel_sizes)
        
        permutations = []
        
        # Split into head and tail
        head = panel_sizes[:k]
        tail = panel_sizes[k:]
        
        # Generate all permutations of head
        for head_perm in itertools.permutations(head):
            permutations.append(list(head_perm) + tail)
        
        return permutations
    
    def generate_type_based_permutations(self, panels_by_type: Dict[PanelSize, int]) -> List[List[PanelSize]]:
        """
        Generate permutations based on panel types and counts.
        Handles multiple panels of same size efficiently.
        """
        # Expand to full list
        panel_list = []
        for panel_size, count in panels_by_type.items():
            panel_list.extend([panel_size] * count)
        
        # Generate unique type orders
        unique_types = list(panels_by_type.keys())
        type_permutations = []
        
        if len(unique_types) <= 4:
            # All permutations of types
            for type_order in itertools.permutations(unique_types):
                ordered_panels = []
                for panel_type in type_order:
                    ordered_panels.extend([panel_type] * panels_by_type[panel_type])
                type_permutations.append(ordered_panels)
        else:
            # Sample some type orders
            type_permutations = self.generate_permutations(panel_list)
        
        return type_permutations


class SortingEffectivenessEvaluator:
    """
    Evaluates effectiveness of different sorting strategies.
    """
    
    def __init__(self):
        """Initialize effectiveness evaluator."""
        self.evaluation_results: Dict[str, Dict[str, float]] = {}
    
    def evaluate_sorting(self, room: Room, panel_sizes: List[PanelSize],
                        sorting_strategy: SortCriteria) -> float:
        """
        Evaluate effectiveness of a sorting strategy.
        Returns score between 0 and 1.
        """
        # Create sorted order
        sorter = MultiCriteriaSorter()
        sorted_panels = sorter.sort_panels(panel_sizes, sorting_strategy)
        
        # Calculate metrics
        total_area = sum(p.area for p in panel_sizes)
        room_area = room.width * room.height
        max_coverage = min(1.0, total_area / room_area)
        
        # Estimate packing efficiency based on order
        efficiency_score = self._estimate_packing_efficiency(room, sorted_panels)
        
        # Consider waste potential
        waste_score = self._estimate_waste_potential(room, sorted_panels)
        
        # Combined score
        effectiveness = (
            0.6 * efficiency_score +
            0.3 * (1.0 - waste_score) +
            0.1 * max_coverage
        )
        
        # Store result
        key = f"{room.id}_{sorting_strategy.value}"
        self.evaluation_results[key] = {
            'effectiveness': effectiveness,
            'efficiency': efficiency_score,
            'waste': waste_score
        }
        
        return effectiveness
    
    def _estimate_packing_efficiency(self, room: Room, sorted_panels: List[PanelSize]) -> float:
        """Estimate how efficiently panels will pack in given order."""
        if not sorted_panels:
            return 0.0
        
        # Simulate simple placement
        placed_area = 0.0
        remaining_width = room.width
        remaining_height = room.height
        current_row_height = 0.0
        
        for panel in sorted_panels:
            width, height = panel.get_dimensions("horizontal")
            
            # Try to place in current row
            if width <= remaining_width:
                placed_area += panel.area
                remaining_width -= width
                current_row_height = max(current_row_height, height)
            
            # Start new row
            elif height <= remaining_height - current_row_height:
                remaining_height -= current_row_height
                remaining_width = room.width - width
                current_row_height = height
                
                if width <= room.width:
                    placed_area += panel.area
            
            # Can't place
            else:
                break
        
        total_area = sum(p.area for p in sorted_panels)
        return placed_area / total_area if total_area > 0 else 0.0
    
    def _estimate_waste_potential(self, room: Room, sorted_panels: List[PanelSize]) -> float:
        """Estimate potential waste based on panel order."""
        if not sorted_panels:
            return 0.0
        
        # Calculate dimension mismatches
        waste_score = 0.0
        
        for i, panel in enumerate(sorted_panels):
            width, height = panel.get_dimensions("horizontal")
            
            # Check if panel dimensions align well with room
            width_waste = (room.width % width) / room.width
            height_waste = (room.height % height) / room.height
            
            # Early panels have more impact
            weight = 1.0 / (i + 1)
            waste_score += weight * (width_waste + height_waste) / 2
        
        # Normalize
        total_weight = sum(1.0 / (i + 1) for i in range(len(sorted_panels)))
        return waste_score / total_weight if total_weight > 0 else 0.0
    
    def get_best_strategy(self, room: Room, panel_sizes: List[PanelSize]) -> SortCriteria:
        """Determine best sorting strategy for room and panels."""
        best_score = -1.0
        best_strategy = SortCriteria.AREA_DESC
        
        # Evaluate each strategy
        for strategy in SortCriteria:
            if strategy == SortCriteria.RANDOM:
                continue  # Skip random for best strategy selection
            
            score = self.evaluate_sorting(room, panel_sizes, strategy)
            
            if score > best_score:
                best_score = score
                best_strategy = strategy
        
        return best_strategy


# Step 2.1.3: Placement Heuristics
# =================================

class WasteMinimizer:
    """
    Minimizes waste in panel placement through intelligent positioning.
    """
    
    def __init__(self, room: Room):
        """Initialize waste minimizer."""
        self.room = room
        self.room_area = room.width * room.height
    
    def calculate_waste(self, state: PackingState) -> float:
        """
        Calculate total waste in current packing state.
        Returns waste as percentage of room area.
        """
        # Calculate covered area
        covered_area = sum(p.panel_size.area for p in state.placements)
        
        # Calculate waste
        waste_area = self.room_area - covered_area
        waste_percentage = waste_area / self.room_area
        
        return waste_percentage
    
    def evaluate_placement_waste(self, placement: PanelPlacement, 
                                state: PackingState) -> float:
        """
        Evaluate waste created by a specific placement.
        Lower score is better.
        """
        x1, y1, x2, y2 = placement.bounds
        
        # Calculate void spaces created
        void_score = 0.0
        
        # Check for small gaps that can't fit any panel
        min_panel_width = 4.0  # Smallest panel dimension
        min_panel_height = 4.0
        
        # Left void
        if x1 > 0:
            if x1 < min_panel_width:
                void_score += x1 * self.room.height
        
        # Right void  
        if x2 < self.room.width:
            remaining = self.room.width - x2
            if remaining < min_panel_width:
                void_score += remaining * self.room.height
        
        # Top void
        if y2 < self.room.height:
            remaining = self.room.height - y2
            if remaining < min_panel_height:
                void_score += remaining * (x2 - x1)
        
        # Bottom void (less critical as we fill bottom-up)
        if y1 > 0:
            if y1 < min_panel_height:
                void_score += y1 * (x2 - x1) * 0.5  # Weight less
        
        # Normalize by room area
        return void_score / self.room_area
    
    def find_minimal_waste_position(self, panel_size: PanelSize,
                                   positions: List[Tuple[float, float]],
                                   state: PackingState) -> Optional[Tuple[float, float]]:
        """
        Find position that creates minimal waste.
        """
        if not positions:
            return None
        
        best_position = None
        best_waste = float('inf')
        
        for orientation in ["horizontal", "vertical"]:
            width, height = panel_size.get_dimensions(orientation)
            
            for x, y in positions:
                placement = PanelPlacement(
                    panel_size=panel_size,
                    position=(x, y),
                    orientation=orientation
                )
                
                # Check validity
                if not state.is_valid_placement(placement):
                    continue
                
                # Evaluate waste
                waste = self.evaluate_placement_waste(placement, state)
                
                if waste < best_waste:
                    best_waste = waste
                    best_position = (x, y)
        
        return best_position
    
    def get_waste_reduction_suggestions(self, state: PackingState) -> List[str]:
        """
        Get suggestions for reducing waste in current state.
        """
        suggestions = []
        waste = self.calculate_waste(state)
        
        if waste > 0.2:  # More than 20% waste
            suggestions.append("Consider using smaller panels to fill gaps")
            
            # Analyze gap sizes
            gaps = self._identify_gaps(state)
            for gap in gaps:
                if gap['area'] > 16:  # Significant gap
                    suggestions.append(f"Gap at ({gap['x']}, {gap['y']}) could fit a {gap['suggested_panel']}")
        
        return suggestions
    
    def _identify_gaps(self, state: PackingState) -> List[Dict]:
        """Identify significant gaps in packing."""
        # Simplified gap identification
        gaps = []
        
        # This would use spatial indexing to find empty rectangles
        # For now, return empty list
        return gaps


class EdgeAlignmentOptimizer:
    """
    Optimizes panel placement for edge alignment to improve stability.
    """
    
    def __init__(self, room: Room):
        """Initialize edge alignment optimizer."""
        self.room = room
        self.tolerance = 0.1  # Alignment tolerance in feet
    
    def score_edge_alignment(self, placement: PanelPlacement, 
                            state: PackingState) -> float:
        """
        Score placement based on edge alignment.
        Higher score means better alignment.
        """
        score = 0.0
        x1, y1, x2, y2 = placement.bounds
        
        # Check alignment with room edges
        room_edge_score = 0.0
        
        # Left edge
        if abs(x1) < self.tolerance:
            room_edge_score += 1.0
        
        # Right edge
        if abs(x2 - self.room.width) < self.tolerance:
            room_edge_score += 1.0
        
        # Bottom edge
        if abs(y1) < self.tolerance:
            room_edge_score += 1.5  # Bottom alignment more important
        
        # Top edge
        if abs(y2 - self.room.height) < self.tolerance:
            room_edge_score += 0.5
        
        score += room_edge_score
        
        # Check alignment with existing panels
        panel_alignment_score = 0.0
        alignment_count = 0
        
        for existing in state.placements:
            ex1, ey1, ex2, ey2 = existing.bounds
            
            # Vertical edge alignment
            if abs(x1 - ex1) < self.tolerance or abs(x1 - ex2) < self.tolerance:
                panel_alignment_score += 0.5
                alignment_count += 1
            
            if abs(x2 - ex1) < self.tolerance or abs(x2 - ex2) < self.tolerance:
                panel_alignment_score += 0.5
                alignment_count += 1
            
            # Horizontal edge alignment
            if abs(y1 - ey1) < self.tolerance or abs(y1 - ey2) < self.tolerance:
                panel_alignment_score += 0.5
                alignment_count += 1
            
            if abs(y2 - ey1) < self.tolerance or abs(y2 - ey2) < self.tolerance:
                panel_alignment_score += 0.5
                alignment_count += 1
        
        # Normalize panel alignment score
        if alignment_count > 0:
            score += panel_alignment_score / alignment_count
        
        return score
    
    def find_aligned_positions(self, panel_size: PanelSize,
                              state: PackingState) -> List[Tuple[float, float]]:
        """
        Find positions that align well with existing panels and room edges.
        """
        positions = []
        width, height = panel_size.get_dimensions("horizontal")
        
        # Room edge positions
        positions.extend([
            (0, 0),  # Bottom-left corner
            (self.room.width - width, 0),  # Bottom-right
            (0, self.room.height - height),  # Top-left
            ((self.room.width - width) / 2, 0)  # Bottom-center
        ])
        
        # Positions aligned with existing panels
        for existing in state.placements:
            ex1, ey1, ex2, ey2 = existing.bounds
            
            # Try positions adjacent to existing panel
            positions.extend([
                (ex2, ey1),  # Right of existing
                (ex1 - width, ey1),  # Left of existing
                (ex1, ey2),  # Above existing
                (ex1, ey1 - height)  # Below existing
            ])
        
        # Filter valid positions
        valid_positions = []
        for x, y in positions:
            if 0 <= x <= self.room.width - width and 0 <= y <= self.room.height - height:
                placement = PanelPlacement(
                    panel_size=panel_size,
                    position=(x, y),
                    orientation="horizontal"
                )
                if state.is_valid_placement(placement):
                    valid_positions.append((x, y))
        
        return valid_positions
    
    def improve_alignment(self, placement: PanelPlacement) -> PanelPlacement:
        """
        Adjust placement slightly to improve edge alignment.
        """
        x, y = placement.position
        width, height = placement.panel_size.get_dimensions(placement.orientation)
        
        # Snap to room edges if close
        if x < self.tolerance:
            x = 0
        elif self.room.width - (x + width) < self.tolerance:
            x = self.room.width - width
        
        if y < self.tolerance:
            y = 0
        elif self.room.height - (y + height) < self.tolerance:
            y = self.room.height - height
        
        return PanelPlacement(
            panel_size=placement.panel_size,
            position=(x, y),
            orientation=placement.orientation
        )


class CornerPreferenceEvaluator:
    """
    Evaluates and prefers corner placements for better structural stability.
    """
    
    def __init__(self, room: Room):
        """Initialize corner preference evaluator."""
        self.room = room
        self.corners = [
            (0, 0),  # Bottom-left
            (room.width, 0),  # Bottom-right
            (0, room.height),  # Top-left
            (room.width, room.height)  # Top-right
        ]
    
    def score_corner_placement(self, placement: PanelPlacement) -> float:
        """
        Score placement based on proximity to corners.
        Higher score for corner placements.
        """
        x1, y1, x2, y2 = placement.bounds
        panel_corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
        
        max_score = 0.0
        
        for panel_corner in panel_corners:
            for room_corner in self.corners:
                distance = ((panel_corner[0] - room_corner[0])**2 + 
                          (panel_corner[1] - room_corner[1])**2)**0.5
                
                # Score inversely proportional to distance
                if distance < 0.1:
                    corner_score = 2.0  # Perfect corner placement
                elif distance < 1.0:
                    corner_score = 1.0 / (1.0 + distance)
                else:
                    corner_score = 0.0
                
                max_score = max(max_score, corner_score)
        
        return max_score
    
    def find_corner_positions(self, panel_size: PanelSize) -> List[Tuple[float, float]]:
        """
        Find positions that place panel in or near corners.
        """
        positions = []
        
        for orientation in ["horizontal", "vertical"]:
            width, height = panel_size.get_dimensions(orientation)
            
            # Bottom-left corner
            positions.append((0, 0))
            
            # Bottom-right corner
            if width <= self.room.width:
                positions.append((self.room.width - width, 0))
            
            # Top-left corner
            if height <= self.room.height:
                positions.append((0, self.room.height - height))
            
            # Top-right corner
            if width <= self.room.width and height <= self.room.height:
                positions.append((self.room.width - width, self.room.height - height))
        
        return positions
    
    def is_corner_placement(self, placement: PanelPlacement, tolerance: float = 0.1) -> bool:
        """
        Check if placement is in a corner.
        """
        x1, y1, x2, y2 = placement.bounds
        
        # Check each corner
        corners_touched = 0
        
        # Bottom-left
        if x1 < tolerance and y1 < tolerance:
            corners_touched += 1
        
        # Bottom-right
        if abs(x2 - self.room.width) < tolerance and y1 < tolerance:
            corners_touched += 1
        
        # Top-left
        if x1 < tolerance and abs(y2 - self.room.height) < tolerance:
            corners_touched += 1
        
        # Top-right
        if abs(x2 - self.room.width) < tolerance and abs(y2 - self.room.height) < tolerance:
            corners_touched += 1
        
        return corners_touched > 0


class StabilityMetricCalculator:
    """
    Calculates stability metrics for panel placements.
    """
    
    def __init__(self, room: Room):
        """Initialize stability calculator."""
        self.room = room
    
    def calculate_stability(self, state: PackingState) -> float:
        """
        Calculate overall stability score for packing state.
        Score between 0 (unstable) and 1 (very stable).
        """
        if not state.placements:
            return 1.0
        
        stability_score = 0.0
        weights_sum = 0.0
        
        for placement in state.placements:
            panel_stability = self._calculate_panel_stability(placement, state)
            panel_weight = placement.panel_size.area  # Weight by panel size
            
            stability_score += panel_stability * panel_weight
            weights_sum += panel_weight
        
        return stability_score / weights_sum if weights_sum > 0 else 0.0
    
    def _calculate_panel_stability(self, placement: PanelPlacement, 
                                  state: PackingState) -> float:
        """
        Calculate stability for individual panel.
        """
        x1, y1, x2, y2 = placement.bounds
        width = x2 - x1
        height = y2 - y1
        
        stability = 0.0
        
        # Floor support (most stable)
        if y1 < 0.1:
            stability += 0.5
        
        # Check support from below
        support_ratio = self._calculate_support_ratio(placement, state)
        stability += 0.3 * support_ratio
        
        # Edge support
        edge_support = 0.0
        if x1 < 0.1:  # Left wall
            edge_support += 0.25
        if abs(x2 - self.room.width) < 0.1:  # Right wall
            edge_support += 0.25
        
        stability += 0.2 * min(edge_support * 2, 1.0)
        
        return min(stability, 1.0)
    
    def _calculate_support_ratio(self, placement: PanelPlacement,
                                state: PackingState) -> float:
        """
        Calculate ratio of panel bottom edge that has support.
        """
        x1, y1, x2, y2 = placement.bounds
        panel_width = x2 - x1
        
        if y1 < 0.1:  # On floor
            return 1.0
        
        supported_length = 0.0
        
        for other in state.placements:
            if other == placement:
                continue
            
            ox1, oy1, ox2, oy2 = other.bounds
            
            # Check if other panel provides support from below
            if abs(oy2 - y1) < 0.1:  # Top of other aligns with bottom of this
                # Calculate overlap
                overlap_start = max(x1, ox1)
                overlap_end = min(x2, ox2)
                
                if overlap_end > overlap_start:
                    supported_length += overlap_end - overlap_start
        
        return supported_length / panel_width if panel_width > 0 else 0.0
    
    def identify_unstable_placements(self, state: PackingState) -> List[PanelPlacement]:
        """
        Identify placements that may be unstable.
        """
        unstable = []
        
        for placement in state.placements:
            stability = self._calculate_panel_stability(placement, state)
            if stability < 0.5:  # Threshold for instability
                unstable.append(placement)
        
        return unstable
    
    def suggest_stability_improvements(self, state: PackingState) -> List[str]:
        """
        Suggest improvements for stability.
        """
        suggestions = []
        overall_stability = self.calculate_stability(state)
        
        if overall_stability < 0.7:
            suggestions.append("Consider adding more floor-level panels for foundation")
            
            unstable = self.identify_unstable_placements(state)
            if unstable:
                suggestions.append(f"Found {len(unstable)} potentially unstable panels")
                suggestions.append("Ensure panels have adequate support from below")
        
        return suggestions


class PlacementHeuristicManager:
    """
    Manages all placement heuristics for optimal panel positioning.
    """
    
    def __init__(self, room: Room, config: OptimizerConfig):
        """Initialize heuristic manager."""
        self.room = room
        self.config = config
        
        # Initialize heuristic components
        self.waste_minimizer = WasteMinimizer(room)
        self.edge_optimizer = EdgeAlignmentOptimizer(room)
        self.corner_evaluator = CornerPreferenceEvaluator(room)
        self.stability_calculator = StabilityMetricCalculator(room)
    
    def evaluate_placement(self, placement: PanelPlacement, 
                         state: PackingState) -> float:
        """
        Evaluate placement using all heuristics.
        Returns combined score (higher is better).
        """
        # Calculate individual scores
        waste_score = 1.0 - self.waste_minimizer.evaluate_placement_waste(placement, state)
        edge_score = self.edge_optimizer.score_edge_alignment(placement, state)
        corner_score = self.corner_evaluator.score_corner_placement(placement)
        
        # Estimate stability (simplified)
        x1, y1, x2, y2 = placement.bounds
        stability_score = 1.0 if y1 < 0.1 else 0.5  # Floor placement is stable
        
        # Combine with weights from config
        total_score = (
            self.config.waste_minimization_weight * waste_score +
            self.config.edge_alignment_weight * edge_score +
            self.config.corner_preference_weight * corner_score +
            0.5 * stability_score  # Fixed weight for stability
        )
        
        # Normalize
        total_weight = (
            self.config.waste_minimization_weight +
            self.config.edge_alignment_weight +
            self.config.corner_preference_weight +
            0.5
        )
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def find_best_placement(self, panel_size: PanelSize,
                          positions: List[Tuple[float, float]],
                          state: PackingState) -> Optional[PanelPlacement]:
        """
        Find best placement from candidate positions using heuristics.
        """
        best_placement = None
        best_score = -1.0
        
        for orientation in ["horizontal", "vertical"]:
            for x, y in positions:
                placement = PanelPlacement(
                    panel_size=panel_size,
                    position=(x, y),
                    orientation=orientation
                )
                
                # Check validity
                if not state.is_valid_placement(placement):
                    continue
                
                # Evaluate
                score = self.evaluate_placement(placement, state)
                
                if score > best_score:
                    best_score = score
                    best_placement = placement
        
        # Try to improve alignment if placement found
        if best_placement:
            best_placement = self.edge_optimizer.improve_alignment(best_placement)
        
        return best_placement
    
    def get_placement_report(self, state: PackingState) -> Dict[str, Any]:
        """
        Generate report on placement quality.
        """
        return {
            'waste_percentage': self.waste_minimizer.calculate_waste(state),
            'stability_score': self.stability_calculator.calculate_stability(state),
            'suggestions': {
                'waste': self.waste_minimizer.get_waste_reduction_suggestions(state),
                'stability': self.stability_calculator.suggest_stability_improvements(state)
            }
        }
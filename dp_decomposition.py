#!/usr/bin/env python3
"""
dp_decomposition.py - Room Decomposition for DP Optimization
============================================================
Production-ready room partitioning system for divide-and-conquer DP.
Breaks large rooms into manageable subproblems for optimal panel placement.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum
import numpy as np
from abc import ABC, abstractmethod

from models import Room, Point, PanelSize
from advanced_packing import PackingState, PanelPlacement
from dp_grid import GridBasedDPOptimizer, GridResolution


class PartitionStrategy(Enum):
    """Strategies for room partitioning."""
    HORIZONTAL = "horizontal"    # Split along X-axis
    VERTICAL = "vertical"       # Split along Y-axis
    GRID_BASED = "grid"        # Regular grid partitioning
    ADAPTIVE = "adaptive"      # Content-aware partitioning
    L_SHAPED = "l_shaped"      # Handle L-shaped rooms


@dataclass(frozen=True)
class SubRegion:
    """
    Represents a rectangular subregion of a room.
    Immutable for use in DP state management.
    """
    id: str
    bounds: Tuple[float, float, float, float]  # (x, y, width, height)
    parent_room_id: str
    partition_type: PartitionStrategy
    adjacent_regions: Tuple[str, ...] = ()
    
    @property
    def x(self) -> float:
        return self.bounds[0]
    
    @property
    def y(self) -> float:
        return self.bounds[1]
    
    @property
    def width(self) -> float:
        return self.bounds[2]
    
    @property
    def height(self) -> float:
        return self.bounds[3]
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height > 0 else float('inf')
    
    def contains_point(self, x: float, y: float) -> bool:
        """Check if point is within subregion."""
        return (self.x <= x <= self.x + self.width and 
                self.y <= y <= self.y + self.height)
    
    def overlaps_with(self, other: 'SubRegion') -> bool:
        """Check if this subregion overlaps with another."""
        x1, y1, w1, h1 = self.bounds
        x2, y2, w2, h2 = other.bounds
        
        return not (x1 + w1 <= x2 or x2 + w2 <= x1 or 
                   y1 + h1 <= y2 or y2 + h2 <= y1)
    
    def is_adjacent_to(self, other: 'SubRegion') -> bool:
        """Check if this subregion is adjacent to another."""
        x1, y1, w1, h1 = self.bounds
        x2, y2, w2, h2 = other.bounds
        
        # Check if they share an edge
        tolerance = 0.01
        
        # Horizontal adjacency
        if (abs(x1 + w1 - x2) < tolerance or abs(x2 + w2 - x1) < tolerance):
            return not (y1 + h1 <= y2 + tolerance or y2 + h2 <= y1 + tolerance)
        
        # Vertical adjacency  
        if (abs(y1 + h1 - y2) < tolerance or abs(y2 + h2 - y1) < tolerance):
            return not (x1 + w1 <= x2 + tolerance or x2 + w2 <= x1 + tolerance)
        
        return False


@dataclass
class PartitionResult:
    """
    Result of room partitioning operation.
    Contains subregions and metadata about the decomposition.
    """
    subregions: List[SubRegion]
    partition_strategy: PartitionStrategy
    total_area: float
    overlap_area: float = 0.0
    gap_area: float = 0.0
    quality_score: float = 0.0
    
    @property
    def num_regions(self) -> int:
        return len(self.subregions)
    
    @property
    def coverage_ratio(self) -> float:
        """Ratio of room area covered by subregions."""
        covered_area = sum(region.area for region in self.subregions)
        return covered_area / self.total_area if self.total_area > 0 else 0.0
    
    def get_region_by_id(self, region_id: str) -> Optional[SubRegion]:
        """Get subregion by ID."""
        for region in self.subregions:
            if region.id == region_id:
                return region
        return None


class RoomPartitioner(ABC):
    """
    Abstract base class for room partitioning algorithms.
    Implementations provide different strategies for room decomposition.
    """
    
    @abstractmethod
    def partition(self, room: Room, target_area: float) -> PartitionResult:
        """Partition room into subregions with target area."""
        pass
    
    @abstractmethod
    def can_partition(self, room: Room) -> bool:
        """Check if room can be partitioned by this strategy."""
        pass


class HorizontalPartitioner(RoomPartitioner):
    """Partitions room into horizontal strips."""
    
    def __init__(self, min_strip_height: float = 4.0):
        self.min_strip_height = min_strip_height
    
    def can_partition(self, room: Room) -> bool:
        """Check if room can be horizontally partitioned."""
        return room.height >= 2 * self.min_strip_height
    
    def partition(self, room: Room, target_area: float) -> PartitionResult:
        """Partition room into horizontal strips."""
        if not self.can_partition(room):
            return PartitionResult([self._room_to_subregion(room)], 
                                 PartitionStrategy.HORIZONTAL, room.area)
        
        # Calculate number of strips based on target area
        strips_needed = max(2, int(np.ceil(room.area / target_area)))
        strip_height = room.height / strips_needed
        
        # Ensure minimum strip height
        if strip_height < self.min_strip_height:
            strips_needed = int(room.height / self.min_strip_height)
            strip_height = room.height / strips_needed
        
        subregions = []
        for i in range(strips_needed):
            y_offset = i * strip_height
            region_id = f"{room.id}_h{i}"
            
            region = SubRegion(
                id=region_id,
                bounds=(room.position.x, room.position.y + y_offset, 
                       room.width, strip_height),
                parent_room_id=room.id,
                partition_type=PartitionStrategy.HORIZONTAL
            )
            subregions.append(region)
        
        return PartitionResult(subregions, PartitionStrategy.HORIZONTAL, room.area)
    
    def _room_to_subregion(self, room: Room) -> SubRegion:
        """Convert room to single subregion."""
        return SubRegion(
            id=f"{room.id}_single",
            bounds=(room.position.x, room.position.y, room.width, room.height),
            parent_room_id=room.id,
            partition_type=PartitionStrategy.HORIZONTAL
        )


class VerticalPartitioner(RoomPartitioner):
    """Partitions room into vertical strips."""
    
    def __init__(self, min_strip_width: float = 4.0):
        self.min_strip_width = min_strip_width
    
    def can_partition(self, room: Room) -> bool:
        """Check if room can be vertically partitioned."""
        return room.width >= 2 * self.min_strip_width
    
    def partition(self, room: Room, target_area: float) -> PartitionResult:
        """Partition room into vertical strips."""
        if not self.can_partition(room):
            return PartitionResult([self._room_to_subregion(room)], 
                                 PartitionStrategy.VERTICAL, room.area)
        
        # Calculate number of strips based on target area
        strips_needed = max(2, int(np.ceil(room.area / target_area)))
        strip_width = room.width / strips_needed
        
        # Ensure minimum strip width
        if strip_width < self.min_strip_width:
            strips_needed = int(room.width / self.min_strip_width)
            strip_width = room.width / strips_needed
        
        subregions = []
        for i in range(strips_needed):
            x_offset = i * strip_width
            region_id = f"{room.id}_v{i}"
            
            region = SubRegion(
                id=region_id,
                bounds=(room.position.x + x_offset, room.position.y, 
                       strip_width, room.height),
                parent_room_id=room.id,
                partition_type=PartitionStrategy.VERTICAL
            )
            subregions.append(region)
        
        return PartitionResult(subregions, PartitionStrategy.VERTICAL, room.area)
    
    def _room_to_subregion(self, room: Room) -> SubRegion:
        """Convert room to single subregion."""
        return SubRegion(
            id=f"{room.id}_single",
            bounds=(room.position.x, room.position.y, room.width, room.height),
            parent_room_id=room.id,
            partition_type=PartitionStrategy.VERTICAL
        )


class GridPartitioner(RoomPartitioner):
    """Partitions room into regular grid cells."""
    
    def __init__(self, min_cell_area: float = 40.0):
        self.min_cell_area = min_cell_area
    
    def can_partition(self, room: Room) -> bool:
        """Check if room can be grid partitioned."""
        return room.area >= 4 * self.min_cell_area
    
    def partition(self, room: Room, target_area: float) -> PartitionResult:
        """Partition room into grid cells."""
        if not self.can_partition(room):
            return PartitionResult([self._room_to_subregion(room)], 
                                 PartitionStrategy.GRID_BASED, room.area)
        
        # Calculate grid dimensions
        cells_needed = max(4, int(np.ceil(room.area / target_area)))
        grid_ratio = room.width / room.height
        
        # Try to maintain room aspect ratio in grid
        cols = max(2, int(np.ceil(np.sqrt(cells_needed * grid_ratio))))
        rows = max(2, int(np.ceil(cells_needed / cols)))
        
        cell_width = room.width / cols
        cell_height = room.height / rows
        
        # Ensure minimum cell area
        if cell_width * cell_height < self.min_cell_area:
            # Reduce number of cells
            max_cells = int(room.area / self.min_cell_area)
            cols = max(2, int(np.sqrt(max_cells * grid_ratio)))
            rows = max(2, int(max_cells / cols))
            cell_width = room.width / cols
            cell_height = room.height / rows
        
        subregions = []
        for row in range(rows):
            for col in range(cols):
                x_offset = col * cell_width
                y_offset = row * cell_height
                region_id = f"{room.id}_g{row}_{col}"
                
                region = SubRegion(
                    id=region_id,
                    bounds=(room.position.x + x_offset, room.position.y + y_offset,
                           cell_width, cell_height),
                    parent_room_id=room.id,
                    partition_type=PartitionStrategy.GRID_BASED
                )
                subregions.append(region)
        
        return PartitionResult(subregions, PartitionStrategy.GRID_BASED, room.area)
    
    def _room_to_subregion(self, room: Room) -> SubRegion:
        """Convert room to single subregion."""
        return SubRegion(
            id=f"{room.id}_single",
            bounds=(room.position.x, room.position.y, room.width, room.height),
            parent_room_id=room.id,
            partition_type=PartitionStrategy.GRID_BASED
        )


class AdaptivePartitioner(RoomPartitioner):
    """
    Intelligent partitioner that chooses optimal strategy based on room characteristics.
    Considers room dimensions, aspect ratio, and panel sizes.
    """
    
    def __init__(self, panel_sizes: List[PanelSize], target_cells: int = 1000):
        self.panel_sizes = panel_sizes
        self.target_cells = target_cells
        
        # Initialize sub-partitioners
        self.horizontal = HorizontalPartitioner()
        self.vertical = VerticalPartitioner()
        self.grid = GridPartitioner()
    
    def can_partition(self, room: Room) -> bool:
        """Check if room can be adaptively partitioned."""
        return (self.horizontal.can_partition(room) or 
                self.vertical.can_partition(room) or
                self.grid.can_partition(room))
    
    def partition(self, room: Room, target_area: float) -> PartitionResult:
        """Choose optimal partitioning strategy for room."""
        if not self.can_partition(room):
            return PartitionResult([self._room_to_subregion(room)], 
                                 PartitionStrategy.ADAPTIVE, room.area)
        
        # Evaluate different strategies
        strategies = []
        
        if self.horizontal.can_partition(room):
            h_result = self.horizontal.partition(room, target_area)
            h_score = self._score_partition(h_result, room)
            strategies.append((h_result, h_score))
        
        if self.vertical.can_partition(room):
            v_result = self.vertical.partition(room, target_area)
            v_score = self._score_partition(v_result, room)
            strategies.append((v_result, v_score))
        
        if self.grid.can_partition(room):
            g_result = self.grid.partition(room, target_area)
            g_score = self._score_partition(g_result, room)
            strategies.append((g_result, g_score))
        
        # Choose best strategy
        if strategies:
            best_result, best_score = max(strategies, key=lambda x: x[1])
            best_result.quality_score = best_score
            return best_result
        
        return PartitionResult([self._room_to_subregion(room)], 
                             PartitionStrategy.ADAPTIVE, room.area)
    
    def _score_partition(self, result: PartitionResult, room: Room) -> float:
        """Score partition quality based on multiple criteria."""
        score = 0.0
        
        # Prefer fewer subregions (simplicity)
        complexity_penalty = len(result.subregions) * 0.1
        score -= complexity_penalty
        
        # Prefer subregions that fit panel sizes well
        panel_fit_score = self._calculate_panel_fit_score(result.subregions)
        score += panel_fit_score * 2.0
        
        # Prefer balanced aspect ratios
        aspect_score = self._calculate_aspect_score(result.subregions)
        score += aspect_score
        
        # Prefer good area utilization
        coverage_score = result.coverage_ratio
        score += coverage_score * 0.5
        
        return score
    
    def _calculate_panel_fit_score(self, subregions: List[SubRegion]) -> float:
        """Calculate how well panels fit in subregions."""
        total_score = 0.0
        
        for region in subregions:
            region_score = 0.0
            
            for panel_size in self.panel_sizes:
                # Check both orientations
                for orientation in ["horizontal", "vertical"]:
                    pw, ph = panel_size.get_dimensions(orientation)
                    
                    # Calculate how many panels fit
                    panels_x = int(region.width / pw)
                    panels_y = int(region.height / ph)
                    panels_fit = panels_x * panels_y
                    
                    if panels_fit > 0:
                        # Score based on area utilization
                        used_area = panels_fit * panel_size.area
                        utilization = used_area / region.area
                        region_score = max(region_score, utilization)
            
            total_score += region_score
        
        return total_score / len(subregions) if subregions else 0.0
    
    def _calculate_aspect_score(self, subregions: List[SubRegion]) -> float:
        """Score based on aspect ratios (prefer close to 1.0)."""
        total_score = 0.0
        
        for region in subregions:
            # Ideal aspect ratio is 1.0, penalize deviations
            aspect = region.aspect_ratio
            if aspect > 1.0:
                aspect_score = 1.0 / aspect
            else:
                aspect_score = aspect
            
            total_score += aspect_score
        
        return total_score / len(subregions) if subregions else 0.0
    
    def _room_to_subregion(self, room: Room) -> SubRegion:
        """Convert room to single subregion."""
        return SubRegion(
            id=f"{room.id}_single",
            bounds=(room.position.x, room.position.y, room.width, room.height),
            parent_room_id=room.id,
            partition_type=PartitionStrategy.ADAPTIVE
        )


class SubproblemMerger:
    """
    Merges solutions from different subregions into complete room solution.
    Handles boundary conditions and validates merged results.
    """
    
    def __init__(self):
        self.tolerance = 0.01
    
    def merge_solutions(self, subregion_solutions: Dict[str, List[PanelPlacement]], 
                       partition_result: PartitionResult) -> List[PanelPlacement]:
        """
        Merge panel placements from subregions into complete solution.
        Validates boundary conditions and resolves conflicts.
        """
        merged_placements = []
        
        # Collect all placements
        for region_id, placements in subregion_solutions.items():
            region = partition_result.get_region_by_id(region_id)
            if region:
                # Validate placements are within subregion
                valid_placements = self._validate_subregion_placements(placements, region)
                merged_placements.extend(valid_placements)
        
        # Check for overlaps at boundaries
        conflict_free_placements = self._resolve_boundary_conflicts(merged_placements)
        
        return conflict_free_placements
    
    def _validate_subregion_placements(self, placements: List[PanelPlacement], 
                                     region: SubRegion) -> List[PanelPlacement]:
        """Validate that placements are within subregion bounds."""
        valid_placements = []
        
        for placement in placements:
            x1, y1, x2, y2 = placement.bounds
            
            # Check if panel is fully within subregion
            if (x1 >= region.x - self.tolerance and 
                y1 >= region.y - self.tolerance and
                x2 <= region.x + region.width + self.tolerance and
                y2 <= region.y + region.height + self.tolerance):
                valid_placements.append(placement)
        
        return valid_placements
    
    def _resolve_boundary_conflicts(self, placements: List[PanelPlacement]) -> List[PanelPlacement]:
        """Resolve overlapping placements at subregion boundaries."""
        if not placements:
            return []
        
        # Sort by placement position for consistent processing
        sorted_placements = sorted(placements, key=lambda p: (p.position[0], p.position[1]))
        
        conflict_free = []
        
        for placement in sorted_placements:
            has_conflict = False
            
            # Check against already accepted placements
            for existing in conflict_free:
                if self._placements_overlap(placement, existing):
                    has_conflict = True
                    break
            
            if not has_conflict:
                conflict_free.append(placement)
        
        return conflict_free
    
    def _placements_overlap(self, p1: PanelPlacement, p2: PanelPlacement) -> bool:
        """Check if two placements overlap."""
        x1a, y1a, x2a, y2a = p1.bounds
        x1b, y1b, x2b, y2b = p2.bounds
        
        return not (x2a <= x1b + self.tolerance or x2b <= x1a + self.tolerance or
                   y2a <= y1b + self.tolerance or y2b <= y1a + self.tolerance)


class DecompositionValidator:
    """
    Validates room decomposition for structural and optimization constraints.
    Ensures partitions don't violate panel placement requirements.
    """
    
    def __init__(self, panel_sizes: List[PanelSize]):
        self.panel_sizes = panel_sizes
        self.min_region_area = min(p.area for p in panel_sizes) if panel_sizes else 16.0
    
    def validate_decomposition(self, result: PartitionResult, room: Room) -> Tuple[bool, List[str]]:
        """
        Validate partition result for feasibility and quality.
        Returns (is_valid, list_of_issues).
        """
        issues = []
        
        # Check area coverage
        if abs(result.coverage_ratio - 1.0) > 0.01:
            issues.append(f"Incomplete coverage: {result.coverage_ratio:.2%}")
        
        # Check minimum region sizes
        for region in result.subregions:
            if region.area < self.min_region_area:
                issues.append(f"Region {region.id} too small: {region.area:.1f}")
        
        # Check for overlaps
        overlaps = self._detect_overlaps(result.subregions)
        if overlaps:
            issues.append(f"Found {len(overlaps)} overlapping regions")
        
        # Check panel fitting capability
        unusable_regions = self._check_panel_fitting(result.subregions)
        if unusable_regions:
            issues.append(f"{len(unusable_regions)} regions cannot fit any panels")
        
        return len(issues) == 0, issues
    
    def _detect_overlaps(self, subregions: List[SubRegion]) -> List[Tuple[str, str]]:
        """Detect overlapping subregions."""
        overlaps = []
        
        for i, region1 in enumerate(subregions):
            for region2 in subregions[i + 1:]:
                if region1.overlaps_with(region2):
                    overlaps.append((region1.id, region2.id))
        
        return overlaps
    
    def _check_panel_fitting(self, subregions: List[SubRegion]) -> List[str]:
        """Check which regions cannot fit any panels."""
        unusable_regions = []
        
        for region in subregions:
            can_fit_panel = False
            
            for panel_size in self.panel_sizes:
                for orientation in ["horizontal", "vertical"]:
                    pw, ph = panel_size.get_dimensions(orientation)
                    
                    if region.width >= pw and region.height >= ph:
                        can_fit_panel = True
                        break
                
                if can_fit_panel:
                    break
            
            if not can_fit_panel:
                unusable_regions.append(region.id)
        
        return unusable_regions


def create_room_decomposer(panel_sizes: List[PanelSize], 
                          strategy: PartitionStrategy = PartitionStrategy.ADAPTIVE,
                          target_area: float = 200.0) -> Tuple[RoomPartitioner, SubproblemMerger, DecompositionValidator]:
    """
    Factory function to create complete room decomposition system.
    Returns partitioner, merger, and validator components.
    """
    # Choose partitioner based on strategy
    if strategy == PartitionStrategy.HORIZONTAL:
        partitioner = HorizontalPartitioner()
    elif strategy == PartitionStrategy.VERTICAL:
        partitioner = VerticalPartitioner()
    elif strategy == PartitionStrategy.GRID_BASED:
        partitioner = GridPartitioner()
    else:  # ADAPTIVE
        partitioner = AdaptivePartitioner(panel_sizes)
    
    merger = SubproblemMerger()
    validator = DecompositionValidator(panel_sizes)
    
    return partitioner, merger, validator
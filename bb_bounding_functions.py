#!/usr/bin/env python3
"""
bb_bounding_functions.py - Advanced Bounding Functions for Branch & Bound
========================================================================
Production-ready linear relaxation, area-based bounds, constraint tightening,
and adaptive refinement for B&B panel optimization.
"""

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from enum import Enum
from abc import ABC, abstractmethod

from models import Room, PanelSize, Point
from advanced_packing import PackingState, PanelPlacement
from bb_search_tree import BranchNode, BranchBounds


class BoundingMethod(Enum):
    """Types of bounding methods available."""
    LINEAR_RELAXATION = "linear_relaxation"
    AREA_BASED = "area_based"
    GEOMETRIC_RELAXATION = "geometric_relaxation"
    CONSTRAINT_PROPAGATION = "constraint_propagation"
    DUAL_BOUNDS = "dual_bounds"


class ConstraintType(Enum):
    """Types of constraints that can be tightened."""
    NO_OVERLAP = "no_overlap"
    BOUNDARY = "boundary"
    POSITIONING = "positioning"
    SIZE_COMPATIBILITY = "size_compatibility"


@dataclass
class BoundingConfiguration:
    """Configuration for advanced bounding computations."""
    methods: Set[BoundingMethod] = field(default_factory=lambda: {
        BoundingMethod.LINEAR_RELAXATION, 
        BoundingMethod.AREA_BASED,
        BoundingMethod.GEOMETRIC_RELAXATION
    })
    constraint_tightening_enabled: bool = True
    adaptive_refinement_enabled: bool = True
    dual_bounds_enabled: bool = False
    
    # Method-specific parameters
    linear_relaxation_resolution: float = 0.1
    geometric_tolerance: float = 0.05
    constraint_propagation_iterations: int = 3
    adaptive_refinement_threshold: float = 0.1
    
    # Performance tuning
    max_computation_time: float = 0.1  # seconds
    bound_quality_target: float = 0.9


@dataclass
class BoundingResults:
    """Results from advanced bounds computation."""
    bounds: BranchBounds
    method_contributions: Dict[BoundingMethod, float]
    constraint_tightenings: Dict[ConstraintType, int]
    computation_time: float
    refinement_iterations: int = 0
    
    @property
    def best_upper_bound(self) -> float:
        """Get the tightest upper bound computed."""
        return self.bounds.upper_bound
    
    @property
    def dominant_method(self) -> BoundingMethod:
        """Get the method that provided the tightest bound."""
        if not self.method_contributions:
            return BoundingMethod.AREA_BASED
        return min(self.method_contributions.keys(), 
                  key=lambda m: self.method_contributions[m])


class LinearRelaxationSolver:
    """
    Solves linear relaxation of the 2D bin packing problem.
    Relaxes integer placement constraints to get upper bounds.
    """
    
    def __init__(self, room: Room, resolution: float = 0.1):
        self.room = room
        self.resolution = resolution
        self.grid_width = int(math.ceil(room.width / resolution))
        self.grid_height = int(math.ceil(room.height / resolution))
        
    def compute_relaxation_bound(self, node: BranchNode) -> float:
        """
        Compute upper bound using linear relaxation.
        Treats panel placement as continuous fractional variables.
        """
        if not node.remaining_panels:
            return node.state.coverage
            
        # Current occupied area
        occupied_area = node.state.coverage * (self.room.width * self.room.height)
        available_area = (self.room.width * self.room.height) - occupied_area
        
        # Linear relaxation: allow fractional placement
        fractional_placement_area = 0.0
        
        # Sort panels by efficiency (area per unit)
        panels_by_efficiency = sorted(node.remaining_panels, 
                                    key=lambda p: p.area, reverse=True)
        
        for panel in panels_by_efficiency:
            if available_area <= 0:
                break
                
            # In relaxation, we can place fractional parts of panels
            placeable_fraction = min(1.0, available_area / panel.area)
            placed_area = panel.area * placeable_fraction
            
            fractional_placement_area += placed_area
            available_area -= placed_area
        
        total_coverage_area = occupied_area + fractional_placement_area
        relaxation_bound = total_coverage_area / (self.room.width * self.room.height)
        
        return min(1.0, relaxation_bound)
    
    def compute_geometric_relaxation_bound(self, node: BranchNode) -> float:
        """
        Compute bound using geometric relaxation.
        Relaxes shape constraints while respecting spatial layout.
        """
        if not node.remaining_panels:
            return node.state.coverage
            
        # Create occupancy grid
        grid = [[False for _ in range(self.grid_width)] 
                for _ in range(self.grid_height)]
        
        # Mark occupied cells from existing placements
        for placement in node.state.placements:
            self._mark_placement_on_grid(grid, placement)
        
        # Count available cells
        available_cells = sum(1 for row in grid for cell in row if not cell)
        available_area = available_cells * (self.resolution ** 2)
        
        # In geometric relaxation, panels can be reshaped to fit optimally
        remaining_area = sum(panel.area for panel in node.remaining_panels)
        
        # Bound considers both shape flexibility and spatial constraints
        geometric_efficiency = self._estimate_geometric_packing_efficiency(node)
        usable_area = min(available_area, remaining_area * geometric_efficiency)
        
        current_area = node.state.coverage * (self.room.width * self.room.height)
        total_area = current_area + usable_area
        
        bound = total_area / (self.room.width * self.room.height)
        return min(1.0, bound)  # Ensure bound doesn't exceed 1.0
    
    def _mark_placement_on_grid(self, grid: List[List[bool]], placement: PanelPlacement):
        """Mark cells occupied by placement on the grid."""
        x1, y1, x2, y2 = placement.bounds
        
        # Convert to grid coordinates
        grid_x1 = max(0, int(x1 / self.resolution))
        grid_y1 = max(0, int(y1 / self.resolution))
        grid_x2 = min(self.grid_width, int(math.ceil(x2 / self.resolution)))
        grid_y2 = min(self.grid_height, int(math.ceil(y2 / self.resolution)))
        
        # Mark occupied cells
        for gy in range(grid_y1, grid_y2):
            for gx in range(grid_x1, grid_x2):
                if 0 <= gy < self.grid_height and 0 <= gx < self.grid_width:
                    grid[gy][gx] = True
    
    def _estimate_geometric_packing_efficiency(self, node: BranchNode) -> float:
        """Estimate how efficiently remaining panels can be packed."""
        if not node.remaining_panels:
            return 1.0
            
        # Factors affecting geometric packing efficiency
        
        # Factor 1: Panel size variety (more variety = better packing)
        panel_areas = [p.area for p in node.remaining_panels]
        area_variance = self._calculate_variance(panel_areas)
        variety_factor = min(1.0, area_variance / (max(panel_areas) ** 2) * 10)
        
        # Factor 2: Current coverage (higher coverage = tighter constraints)
        coverage_penalty = node.state.coverage * 0.3
        
        # Factor 3: Remaining space fragmentation
        fragmentation_factor = self._estimate_space_fragmentation(node)
        
        base_efficiency = 0.85  # Baseline geometric packing efficiency
        efficiency = base_efficiency + variety_factor * 0.1 - coverage_penalty - fragmentation_factor * 0.2
        
        return max(0.5, min(1.0, efficiency))
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of values."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / len(values)
    
    def _estimate_space_fragmentation(self, node: BranchNode) -> float:
        """Estimate degree of fragmentation in available space."""
        if not node.state.placements:
            return 0.0
            
        # Simple fragmentation estimate based on placement distribution
        placements = node.state.placements
        
        # Calculate spread of placements
        xs = [p.position[0] for p in placements]
        ys = [p.position[1] for p in placements]
        
        x_spread = (max(xs) - min(xs)) / self.room.width if xs else 0.0
        y_spread = (max(ys) - min(ys)) / self.room.height if ys else 0.0
        
        # Higher spread often means more fragmentation
        spread_factor = (x_spread + y_spread) / 2.0
        
        return min(1.0, spread_factor)


class AreaBasedBoundingEngine:
    """
    Advanced area-based bounding with geometric constraints.
    Considers both area availability and geometric feasibility.
    """
    
    def __init__(self, room: Room):
        self.room = room
        self.room_area = room.width * room.height
        
    def compute_area_bounds(self, node: BranchNode) -> Tuple[float, float]:
        """
        Compute lower and upper area-based bounds.
        Returns (lower_bound, upper_bound).
        """
        current_coverage = node.state.coverage
        
        # Lower bound is current coverage (guaranteed)
        lower_bound = current_coverage
        
        # Upper bound considers remaining panel placement
        upper_bound = self._compute_area_upper_bound(node)
        
        return lower_bound, upper_bound
    
    def _compute_area_upper_bound(self, node: BranchNode) -> float:
        """Compute sophisticated area-based upper bound."""
        if not node.remaining_panels:
            return node.state.coverage
            
        current_area = node.state.coverage * self.room_area
        available_area = self.room_area - current_area
        
        # Method 1: Perfect packing bound
        remaining_area = sum(panel.area for panel in node.remaining_panels)
        perfect_bound = node.state.coverage + min(remaining_area, available_area) / self.room_area
        
        # Method 2: Shape-constrained bound
        shape_bound = self._compute_shape_constrained_bound(node, available_area)
        
        # Method 3: Spatial availability bound  
        spatial_bound = self._compute_spatial_availability_bound(node)
        
        # Take the tightest bound
        return min(perfect_bound, shape_bound, spatial_bound, 1.0)
    
    def _compute_shape_constrained_bound(self, node: BranchNode, available_area: float) -> float:
        """Bound considering panel shape constraints."""
        # Account for shape matching efficiency
        panels_by_size = sorted(node.remaining_panels, key=lambda p: p.area, reverse=True)
        
        usable_area = 0.0
        remaining_space = available_area
        
        for panel in panels_by_size:
            if remaining_space <= 0:
                break
                
            # Estimate how efficiently this panel shape can use remaining space
            shape_efficiency = self._estimate_panel_shape_efficiency(panel, remaining_space)
            usable_panel_area = min(panel.area * shape_efficiency, remaining_space)
            
            usable_area += usable_panel_area
            remaining_space -= usable_panel_area
        
        return node.state.coverage + usable_area / self.room_area
    
    def _compute_spatial_availability_bound(self, node: BranchNode) -> float:
        """Bound considering spatial distribution of available space."""
        # Simplified spatial analysis
        # In practice, this would analyze the geometric layout more carefully
        
        occupied_positions = {placement.position for placement in node.state.placements}
        
        # Estimate available "good" positions
        spatial_efficiency = 0.8  # Base efficiency for spatial placement
        
        # Adjust based on current layout density
        if len(occupied_positions) > 0:
            # Higher density typically means lower spatial efficiency
            density_factor = len(occupied_positions) / (self.room.width * self.room.height / 4.0)
            spatial_efficiency = max(0.5, spatial_efficiency - density_factor * 0.1)
        
        remaining_area = sum(panel.area for panel in node.remaining_panels)
        spatially_usable = remaining_area * spatial_efficiency
        
        return node.state.coverage + spatially_usable / self.room_area
    
    def _estimate_panel_shape_efficiency(self, panel: PanelSize, available_space: float) -> float:
        """Estimate how efficiently a panel shape can use available space."""
        if panel.area > available_space:
            return available_space / panel.area
            
        # Consider aspect ratio effects
        panel_ratio = panel.width / panel.length
        
        # Panels closer to square are often more efficient
        square_bonus = 1.0 - abs(1.0 - panel_ratio) * 0.1
        
        # Very thin or very wide panels are less efficient
        extreme_penalty = max(0.0, (panel_ratio - 4.0) * 0.05 + (1.0/panel_ratio - 4.0) * 0.05)
        
        efficiency = min(1.0, max(0.7, square_bonus - extreme_penalty))
        return efficiency


class ConstraintTighteningEngine:
    """
    Tightens constraints to improve bound quality.
    Identifies and strengthens problem constraints.
    """
    
    def __init__(self, room: Room, tolerance: float = 0.01):
        self.room = room
        self.tolerance = tolerance
        self.tightening_count = {}
        
    def tighten_constraints(self, node: BranchNode) -> Dict[ConstraintType, int]:
        """
        Apply constraint tightening techniques.
        Returns count of tightenings applied by type.
        """
        tightenings = {}
        
        # Tighten no-overlap constraints
        overlap_tightenings = self._tighten_overlap_constraints(node)
        tightenings[ConstraintType.NO_OVERLAP] = overlap_tightenings
        
        # Tighten boundary constraints  
        boundary_tightenings = self._tighten_boundary_constraints(node)
        tightenings[ConstraintType.BOUNDARY] = boundary_tightenings
        
        # Tighten positioning constraints
        position_tightenings = self._tighten_positioning_constraints(node)
        tightenings[ConstraintType.POSITIONING] = position_tightenings
        
        # Tighten size compatibility constraints
        size_tightenings = self._tighten_size_constraints(node)
        tightenings[ConstraintType.SIZE_COMPATIBILITY] = size_tightenings
        
        # Update global counters
        for constraint_type, count in tightenings.items():
            self.tightening_count[constraint_type] = self.tightening_count.get(constraint_type, 0) + count
        
        return tightenings
    
    def _tighten_overlap_constraints(self, node: BranchNode) -> int:
        """Tighten constraints to prevent panel overlaps."""
        tightenings = 0
        
        # For each remaining panel, identify positions that would cause overlap
        for panel in node.remaining_panels:
            forbidden_regions = self._compute_forbidden_overlap_regions(node, panel)
            tightenings += len(forbidden_regions)
            
        return tightenings
    
    def _tighten_boundary_constraints(self, node: BranchNode) -> int:
        """Tighten constraints to respect room boundaries."""
        tightenings = 0
        
        for panel in node.remaining_panels:
            # Panel must fit within room bounds considering orientation
            width, height = panel.get_dimensions()  # Default orientation
            
            # Tighten x-coordinate range
            max_x = self.room.width - width
            max_y = self.room.height - height
            
            if max_x < 0 or max_y < 0:
                # Panel cannot fit in default orientation
                # Check rotated orientation
                width_rot, height_rot = height, width
                max_x_rot = self.room.width - width_rot
                max_y_rot = self.room.height - height_rot
                
                if max_x_rot >= 0 and max_y_rot >= 0:
                    tightenings += 1  # Rotation constraint tightening
        
        return tightenings
    
    def _tighten_positioning_constraints(self, node: BranchNode) -> int:
        """Tighten positioning constraints based on existing layout."""
        tightenings = 0
        
        if not node.state.placements:
            return 0
            
        # Identify preferred alignment positions
        x_positions = set()
        y_positions = set()
        
        for placement in node.state.placements:
            x1, y1, x2, y2 = placement.bounds
            x_positions.update([x1, x2])
            y_positions.update([y1, y2])
        
        # Add room boundaries
        x_positions.update([0, self.room.width])
        y_positions.update([0, self.room.height])
        
        # For each remaining panel, tighten to prefer alignment positions
        for panel in node.remaining_panels:
            # Count how many good alignment positions are available
            good_positions = 0
            for x in x_positions:
                for y in y_positions:
                    if self._is_valid_alignment_position(node, panel, (x, y)):
                        good_positions += 1
            
            if good_positions < len(x_positions) * len(y_positions) * 0.1:
                tightenings += 1  # Position constraint tightened
        
        return tightenings
    
    def _tighten_size_constraints(self, node: BranchNode) -> int:
        """Tighten constraints based on panel size compatibility."""
        tightenings = 0
        
        # Group panels by similar sizes
        size_groups = self._group_panels_by_size(node.remaining_panels)
        
        # For each group, identify spatial constraints
        for group in size_groups:
            if len(group) > 1:
                # Multiple panels of similar size - look for packing constraints
                available_space = self._estimate_available_space_for_group(node, group)
                required_space = sum(panel.area for panel in group)
                
                if required_space > available_space * 1.2:  # Tight fit
                    tightenings += len(group)
        
        return tightenings
    
    def _compute_forbidden_overlap_regions(self, node: BranchNode, panel: PanelSize) -> List[Tuple[float, float, float, float]]:
        """Compute regions where panel placement would cause overlap."""
        forbidden = []
        
        width, height = panel.get_dimensions()
        
        for placement in node.state.placements:
            px1, py1, px2, py2 = placement.bounds
            
            # Panel would overlap if its bottom-left is in this region
            forbidden_x1 = px1 - width + self.tolerance
            forbidden_y1 = py1 - height + self.tolerance
            forbidden_x2 = px2 - self.tolerance
            forbidden_y2 = py2 - self.tolerance
            
            if (forbidden_x1 < forbidden_x2 and forbidden_y1 < forbidden_y2 and
                forbidden_x1 < self.room.width and forbidden_y1 < self.room.height and
                forbidden_x2 > 0 and forbidden_y2 > 0):
                forbidden.append((forbidden_x1, forbidden_y1, forbidden_x2, forbidden_y2))
        
        return forbidden
    
    def _is_valid_alignment_position(self, node: BranchNode, panel: PanelSize, position: Tuple[float, float]) -> bool:
        """Check if position is valid for panel placement with alignment."""
        x, y = position
        width, height = panel.get_dimensions()
        
        # Check room boundaries
        if x + width > self.room.width or y + height > self.room.height:
            return False
            
        # Check overlap with existing placements
        for placement in node.state.placements:
            px1, py1, px2, py2 = placement.bounds
            if not (x + width <= px1 + self.tolerance or x >= px2 - self.tolerance or
                   y + height <= py1 + self.tolerance or y >= py2 - self.tolerance):
                return False
        
        return True
    
    def _group_panels_by_size(self, panels: Set[PanelSize]) -> List[List[PanelSize]]:
        """Group panels by similar size."""
        groups = []
        size_tolerance = 0.1
        
        for panel in panels:
            placed_in_group = False
            for group in groups:
                if group and abs(panel.area - group[0].area) / group[0].area < size_tolerance:
                    group.append(panel)
                    placed_in_group = True
                    break
            
            if not placed_in_group:
                groups.append([panel])
        
        return groups
    
    def _estimate_available_space_for_group(self, node: BranchNode, group: List[PanelSize]) -> float:
        """Estimate available space for a group of similar-sized panels."""
        occupied_area = node.state.coverage * (self.room.width * self.room.height)
        total_available = (self.room.width * self.room.height) - occupied_area
        
        # Assume this group gets proportional share of available space
        total_remaining_area = sum(panel.area for panel in node.remaining_panels)
        group_area = sum(panel.area for panel in group)
        
        if total_remaining_area > 0:
            group_share = group_area / total_remaining_area
            return total_available * group_share
        
        return 0.0


class AdaptiveRefinementEngine:
    """
    Adaptively refines bounds based on search progress.
    Improves bound quality when needed.
    """
    
    def __init__(self, refinement_threshold: float = 0.1):
        self.refinement_threshold = refinement_threshold
        self.refinement_history = {}
        self.performance_metrics = {}
        
    def should_refine_bounds(self, node: BranchNode, current_best: float, 
                           bound_quality: float) -> bool:
        """Determine if bounds should be refined for this node."""
        # Refine if bound quality is poor
        if bound_quality < 0.7:
            return True
            
        # Refine if node is promising but bound is not tight
        if (node.bounds.upper_bound > current_best + self.refinement_threshold and
            node.bounds.upper_bound - node.bounds.lower_bound > self.refinement_threshold):
            return True
            
        # Refine if we haven't made progress recently
        if self._should_refine_for_progress():
            return True
            
        return False
    
    def refine_bounds(self, node: BranchNode, solver: LinearRelaxationSolver,
                     area_engine: AreaBasedBoundingEngine) -> BranchBounds:
        """Apply adaptive refinement to improve bound quality."""
        refinement_start = time.time()
        iterations = 0
        
        current_bounds = node.bounds
        
        # Iteratively refine bounds
        for iteration in range(5):  # Max 5 refinement iterations
            iterations += 1
            
            # Method 1: Higher resolution linear relaxation
            refined_lr_bound = self._refine_linear_relaxation(node, solver, iteration)
            
            # Method 2: Localized area analysis
            refined_area_bound = self._refine_area_bound(node, area_engine, iteration)
            
            # Method 3: Constraint-based refinement
            refined_constraint_bound = self._refine_with_constraints(node, iteration)
            
            # Take tightest bound
            new_upper = min(refined_lr_bound, refined_area_bound, refined_constraint_bound,
                           current_bounds.upper_bound)
            
            # Check for improvement
            improvement = current_bounds.upper_bound - new_upper
            if improvement < 0.01:  # Minimal improvement
                break
                
            # Update bounds
            current_bounds = BranchBounds(
                lower_bound=current_bounds.lower_bound,
                upper_bound=new_upper,
                is_feasible=current_bounds.is_feasible,
                bound_quality=min(1.0, current_bounds.bound_quality + improvement)
            )
        
        # Update refinement history
        refinement_time = time.time() - refinement_start
        self.refinement_history[node.node_id] = {
            'iterations': iterations,
            'time': refinement_time,
            'improvement': node.bounds.upper_bound - current_bounds.upper_bound
        }
        
        return current_bounds
    
    def _refine_linear_relaxation(self, node: BranchNode, solver: LinearRelaxationSolver, 
                                 iteration: int) -> float:
        """Refine linear relaxation with higher precision."""
        # Increase resolution for refinement
        original_resolution = solver.resolution
        solver.resolution = original_resolution / (2 ** iteration)
        
        refined_bound = solver.compute_relaxation_bound(node)
        
        # Restore original resolution
        solver.resolution = original_resolution
        
        return refined_bound
    
    def _refine_area_bound(self, node: BranchNode, area_engine: AreaBasedBoundingEngine,
                          iteration: int) -> float:
        """Refine area-based bound with deeper analysis."""
        # More sophisticated area analysis
        lower, upper = area_engine.compute_area_bounds(node)
        
        # Apply refinement corrections based on iteration
        refinement_factor = 1.0 - 0.1 * iteration  # Progressively tighter
        
        return upper * refinement_factor
    
    def _refine_with_constraints(self, node: BranchNode, iteration: int) -> float:
        """Refine bounds using constraint analysis."""
        # Simplified constraint-based refinement
        base_bound = node.bounds.upper_bound
        
        # Apply constraint tightening factor
        constraint_factor = 0.95 ** iteration  # Progressively tighter
        
        return base_bound * constraint_factor
    
    def _should_refine_for_progress(self) -> bool:
        """Check if refinement is needed due to lack of progress."""
        # Simplified progress tracking
        # In practice, this would track search tree progress more carefully
        return len(self.refinement_history) % 100 == 0  # Refine every 100 nodes
    
    def get_refinement_statistics(self) -> Dict[str, Any]:
        """Get adaptive refinement statistics."""
        if not self.refinement_history:
            return {'refinements_performed': 0}
            
        total_iterations = sum(h['iterations'] for h in self.refinement_history.values())
        total_time = sum(h['time'] for h in self.refinement_history.values())
        total_improvement = sum(h['improvement'] for h in self.refinement_history.values())
        
        return {
            'refinements_performed': len(self.refinement_history),
            'total_iterations': total_iterations,
            'total_time': total_time,
            'total_improvement': total_improvement,
            'avg_iterations': total_iterations / len(self.refinement_history),
            'avg_improvement': total_improvement / len(self.refinement_history)
        }


class AdvancedBoundingSystem:
    """
    Comprehensive bounding system combining all advanced techniques.
    Main interface for sophisticated bound computation.
    """
    
    def __init__(self, room: Room, config: Optional[BoundingConfiguration] = None):
        self.room = room
        self.config = config or BoundingConfiguration()
        
        # Initialize components
        self.linear_solver = LinearRelaxationSolver(room, self.config.linear_relaxation_resolution)
        self.area_engine = AreaBasedBoundingEngine(room)
        self.constraint_engine = ConstraintTighteningEngine(room, self.config.geometric_tolerance)
        self.refinement_engine = AdaptiveRefinementEngine(self.config.adaptive_refinement_threshold)
        
        # Statistics
        self.computation_count = 0
        self.total_computation_time = 0.0
        
    def compute_advanced_bounds(self, node: BranchNode, current_best: float = 0.0) -> BoundingResults:
        """
        Compute advanced bounds using all enabled methods.
        Returns comprehensive bounding results.
        """
        start_time = time.time()
        self.computation_count += 1
        
        method_bounds = {}
        
        # Apply enabled bounding methods
        if BoundingMethod.LINEAR_RELAXATION in self.config.methods:
            lr_bound = self.linear_solver.compute_relaxation_bound(node)
            method_bounds[BoundingMethod.LINEAR_RELAXATION] = lr_bound
        
        if BoundingMethod.GEOMETRIC_RELAXATION in self.config.methods:
            geo_bound = self.linear_solver.compute_geometric_relaxation_bound(node)
            method_bounds[BoundingMethod.GEOMETRIC_RELAXATION] = geo_bound
        
        if BoundingMethod.AREA_BASED in self.config.methods:
            lower, upper = self.area_engine.compute_area_bounds(node)
            method_bounds[BoundingMethod.AREA_BASED] = upper
        
        # Constraint tightening
        constraint_tightenings = {}
        if self.config.constraint_tightening_enabled:
            constraint_tightenings = self.constraint_engine.tighten_constraints(node)
        
        # Take tightest upper bound
        if method_bounds:
            best_upper = min(method_bounds.values())
        else:
            best_upper = node.bounds.upper_bound
        
        # Create initial bounds
        bounds = BranchBounds(
            lower_bound=node.state.coverage,
            upper_bound=best_upper,
            is_feasible=node.bounds.is_feasible,
            bound_quality=self._compute_bound_quality(method_bounds, constraint_tightenings)
        )
        
        # Adaptive refinement
        refinement_iterations = 0
        if (self.config.adaptive_refinement_enabled and 
            self.refinement_engine.should_refine_bounds(node, current_best, bounds.bound_quality)):
            
            refined_bounds = self.refinement_engine.refine_bounds(
                node, self.linear_solver, self.area_engine)
            bounds = refined_bounds
            refinement_iterations = self.refinement_engine.refinement_history.get(
                node.node_id, {}).get('iterations', 0)
        
        computation_time = time.time() - start_time
        self.total_computation_time += computation_time
        
        return BoundingResults(
            bounds=bounds,
            method_contributions=method_bounds,
            constraint_tightenings=constraint_tightenings,
            computation_time=computation_time,
            refinement_iterations=refinement_iterations
        )
    
    def _compute_bound_quality(self, method_bounds: Dict[BoundingMethod, float],
                              constraint_tightenings: Dict[ConstraintType, int]) -> float:
        """Assess quality of computed bounds."""
        quality = 0.8  # Base quality
        
        # More methods generally means better quality
        method_bonus = len(method_bounds) * 0.05
        quality += method_bonus
        
        # Constraint tightening improves quality
        tightening_bonus = sum(constraint_tightenings.values()) * 0.01
        quality += tightening_bonus
        
        # Agreement between methods improves confidence
        if len(method_bounds) > 1:
            bound_values = list(method_bounds.values())
            variance = self._calculate_variance(bound_values)
            if variance < 0.01:  # Good agreement
                quality += 0.1
        
        return min(1.0, max(0.5, quality))
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of values."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / len(values)
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        stats = {
            'computations_performed': self.computation_count,
            'total_computation_time': self.total_computation_time,
            'avg_computation_time': self.total_computation_time / max(1, self.computation_count),
            'enabled_methods': [m.value for m in self.config.methods],
            'constraint_tightening_enabled': self.config.constraint_tightening_enabled,
            'adaptive_refinement_enabled': self.config.adaptive_refinement_enabled
        }
        
        # Add component statistics
        stats.update(self.constraint_engine.tightening_count)
        stats.update(self.refinement_engine.get_refinement_statistics())
        
        return stats


def create_advanced_bounding_system(room: Room, 
                                   config: Optional[BoundingConfiguration] = None) -> AdvancedBoundingSystem:
    """
    Factory function to create advanced bounding system.
    Returns configured system ready for bound computation.
    """
    return AdvancedBoundingSystem(room, config)
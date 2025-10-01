#!/usr/bin/env python3
"""
enhanced_blf_optimizer.py - Enhanced Bottom-Left-Fill Optimizer
================================================================
Main implementation that integrates all BLF components for 95%+ coverage.
"""

from typing import List, Tuple, Optional, Dict, Any
import time

from models import Panel, PanelSize, Point, Room
from advanced_packing import (
    AbstractOptimizer,
    OptimizerConfig,
    PackingState,
    PanelPlacement,
    StateTransition,
    OptimizationMetrics
)
from blf_optimizer import (
    Skyline,
    BLFPositionGenerator,
    BLFCollisionDetector,
    BLFPlacementValidator,
    SortCriteria,
    MultiCriteriaSorter,
    AdaptiveSorter,
    PlacementHeuristicManager
)
from blf_backtracking import (
    BacktrackStateManager,
    BacktrackTriggerManager,
    BacktrackReason,
    StateSnapshot,
    BacktrackStrategy
)
from blf_lookahead import KStepLookahead, LocalSearchRefinement


class EnhancedBLFOptimizer(AbstractOptimizer):
    """
    Enhanced Bottom-Left-Fill optimizer with backtracking for 95%+ coverage.
    """
    
    def __init__(self, config: Optional[OptimizerConfig] = None):
        """Initialize enhanced BLF optimizer."""
        super().__init__(config)
        
        # BLF components (initialized per room)
        self.position_generator: Optional[BLFPositionGenerator] = None
        self.collision_detector: Optional[BLFCollisionDetector] = None
        self.placement_validator: Optional[BLFPlacementValidator] = None
        self.heuristic_manager: Optional[PlacementHeuristicManager] = None
        self.adaptive_sorter: Optional[AdaptiveSorter] = None
        
        # Backtracking components
        self.backtrack_manager = BacktrackStateManager({
            'max_history': 1000,
            'max_snapshots': 100
        })
        
        self.trigger_manager = BacktrackTriggerManager({
            'plateau_threshold': 5,
            'min_improvement': 0.01,
            'max_waste_ratio': 0.25,
            'critical_waste_ratio': 0.35,
            'max_time_per_decision': 0.1,
            'total_time_limit': self.config.max_time_seconds
        })
        
        # Backtrack strategy
        self.backtrack_strategy = BacktrackStrategy({
            'max_backtrack_depth': 10,
            'min_backtrack_depth': 1,
            'adaptive_depth': True,
            'alternative_path_threshold': 0.85
        })
        
        # Lookahead mechanism
        self.lookahead = KStepLookahead(k=3, config={
            'max_branches': 5,
            'beam_width': 3,
            'time_limit': 0.5
        })
        
        # Local search refinement
        self.local_search = LocalSearchRefinement({
            'max_iterations': 50,
            'time_limit': 0.5,
            'improvement_threshold': 0.001,
            'neighborhood_size': 5
        })
        
    def get_algorithm_name(self) -> str:
        """Get algorithm name."""
        return "Enhanced BLF with Backtracking"
    
    def optimize_room(self, room: Room) -> List[Panel]:
        """
        Optimize panel placement for a room using enhanced BLF.
        """
        # Initialize components for this room
        self._initialize_room_components(room)
        
        # Debug: print config panel sizes
        print(f"Config panel_sizes: {self.config.panel_sizes}")
        
        # Determine panel budget
        panel_budget = self._calculate_panel_budget(room)
        
        # Sort panels using adaptive strategy
        sorted_panels = self.adaptive_sorter.sort_panels(panel_budget)
        
        # Debug: print sorted panels
        print(f"Sorted panels for {room.name}: {[p.name for p in sorted_panels]}")
        
        # Initialize state
        current_state = PackingState.from_room(room)
        remaining_panels = sorted_panels.copy()
        
        # Main optimization loop
        iteration = 0
        max_iterations = self.config.max_iterations
        
        print(f"  Starting main loop with {len(remaining_panels)} panels, max_iterations={max_iterations}")
        
        while remaining_panels and iteration < max_iterations:
            self.metrics.record_iteration()
            iteration += 1
            
            # Check for timeout
            if self.check_timeout():
                print(f"  Timeout at iteration {iteration}")
                break
            
            # Check backtrack triggers
            should_backtrack, reason = self.trigger_manager.check_triggers(
                current_state, remaining_panels, room.width * room.height
            )
            
            if iteration <= 5:
                print(f"    Backtrack check: should_backtrack={should_backtrack}, reason={reason}")
            
            if should_backtrack:
                # Perform backtracking
                backtrack_state, backtrack_panels = self._handle_backtrack(
                    current_state, remaining_panels, reason, room
                )
                if backtrack_state is not None:
                    current_state = backtrack_state
                    remaining_panels = backtrack_panels
                else:
                    # No valid backtrack point, continue with current state
                    pass
                continue
            
            # Try to place next panel
            panel_to_place = remaining_panels[0]
            placement = self._find_best_placement(panel_to_place, current_state, remaining_panels)
            
            # Only log first few iterations to avoid spam
            if iteration <= 5:
                print(f"  Iteration {iteration}: Trying to place {panel_to_place.name}, found placement: {placement is not None}")
            
            if placement:
                # Apply placement
                new_state = self.transition_manager.apply_placement(current_state, placement)
                
                if iteration <= 5:
                    print(f"    Applied placement, new_state is not None: {new_state is not None}")
                
                if new_state:
                    # Record placement
                    self.backtrack_manager.record_placement(placement, new_state)
                    
                    # Update state
                    current_state = new_state
                    remaining_panels.pop(0)
                    
                    # Update position generator
                    self.position_generator.update_skyline(placement)
                    
                    # Update best state
                    self.update_best_state(current_state)
                    
                    # Check early stop
                    if self.check_early_stop(current_state.coverage):
                        break
                    
                    # Save snapshot periodically
                    if iteration % 10 == 0:
                        self.backtrack_manager.save_state(
                            current_state, remaining_panels, time.time()
                        )
                else:
                    # Failed to place, try next panel
                    remaining_panels.append(remaining_panels.pop(0))
            else:
                # No valid placement found
                if len(remaining_panels) > 1:
                    # Try different panel order
                    remaining_panels.append(remaining_panels.pop(0))
                else:
                    # Trigger backtrack
                    backtrack_state, backtrack_panels = self._handle_backtrack(
                        current_state, remaining_panels, 
                        BacktrackReason.NO_VALID_POSITIONS, room
                    )
                    if backtrack_state is not None:
                        current_state = backtrack_state
                        remaining_panels = backtrack_panels
                    else:
                        break  # No valid backtrack point, exit
        
        # Use best state found
        if self.best_state:
            current_state = self.best_state
        
        # Convert to panels
        if current_state:
            panels = current_state.to_panels(room.id)
        else:
            panels = []  # No valid solution found
        
        print(f"  Main optimization placed {len(panels)} panels before refinement")
        
        # Post-optimization refinement
        panels = self._refine_solution(panels, room)
        
        print(f"  After refinement: {len(panels)} panels")
        
        return panels
    
    def _initialize_room_components(self, room: Room):
        """Initialize BLF components for room."""
        self.current_room = room  # Store for lookahead
        room_bounds = (room.position.x, room.position.y, room.width, room.height)
        
        self.position_generator = BLFPositionGenerator(room, self.config.grid_resolution)
        self.collision_detector = BLFCollisionDetector(room_bounds)
        self.placement_validator = BLFPlacementValidator(room, self.config)
        self.heuristic_manager = PlacementHeuristicManager(room, self.config)
        self.adaptive_sorter = AdaptiveSorter(room)
        
        # Reset backtracking components
        self.backtrack_manager.clear()
        self.trigger_manager.reset()
    
    def _calculate_panel_budget(self, room: Room) -> List[PanelSize]:
        """
        Calculate panel budget for room based on area and target coverage.
        """
        room_area = room.width * room.height
        target_area = room_area * self.config.early_stop_coverage
        
        panels = []
        current_area = 0.0
        
        # Prioritize larger panels for efficiency
        panel_priority = [
            PanelSize.PANEL_6X8,
            PanelSize.PANEL_6X6,
            PanelSize.PANEL_4X6,
            PanelSize.PANEL_4X4
        ]
        
        for panel_size in panel_priority:
            if panel_size not in self.config.panel_sizes:
                continue
            
            panel_area = panel_size.area
            
            # Calculate how many of this size we can use
            while current_area + panel_area <= target_area * 1.1:  # Allow 10% extra
                panels.append(panel_size)
                current_area += panel_area
        
        # Debug logging
        print(f"Room {room.name}: area={room_area:.1f}, target={target_area:.1f}, panels={len(panels)}, panel_area={current_area:.1f}")
        
        return panels
    
    def _find_best_placement(self, panel_size: PanelSize, 
                           state: PackingState,
                           remaining_panels: List[PanelSize] = None) -> Optional[PanelPlacement]:
        """
        Find best placement for panel using BLF strategy, heuristics, and lookahead.
        """
        # Generate candidate positions
        positions = self.position_generator.generate_positions(panel_size, state.placements)
        
        print(f"    Finding placement for {panel_size.name}: {len(positions)} candidate positions")
        
        if not positions:
            print(f"    No positions generated!")
            return None
        
        # Get room bounds for lookahead
        room_bounds = (self.current_room.position.x, self.current_room.position.y,
                      self.current_room.width, self.current_room.height)
        
        # Use lookahead if we have remaining panels info
        if remaining_panels and len(remaining_panels) > 1:
            # Evaluate top candidates with lookahead
            best_score = -1
            best_placement = None
            
            for position in positions[:10]:  # Evaluate top positions
                for orientation in ["horizontal", "vertical"]:
                    placement = PanelPlacement(
                        panel_size=panel_size,
                        position=position,
                        orientation=orientation
                    )
                    
                    if state.is_valid_placement(placement):
                        # Use lookahead to evaluate
                        lookahead_result = self.lookahead.evaluate_placement(
                            placement, state, remaining_panels[1:],
                            room_bounds, self.position_generator
                        )
                        
                        # Score combines immediate and future benefits
                        combined_score = (lookahead_result.sequence_score * 0.6 +
                                        lookahead_result.future_fit_probability * 0.4)
                        
                        if combined_score > best_score:
                            best_score = combined_score
                            best_placement = placement
            
            if best_placement:
                return best_placement
        
        # Fallback to heuristics
        best_placement = self.heuristic_manager.find_best_placement(
            panel_size, positions[:20], state  # Limit candidates for speed
        )
        
        if best_placement:
            # Validate placement
            is_valid, _ = self.placement_validator.validate_placement(best_placement, state)
            if is_valid:
                return best_placement
        
        # Fallback: try first valid position
        for position in positions:
            for orientation in ["horizontal", "vertical"]:
                placement = PanelPlacement(
                    panel_size=panel_size,
                    position=position,
                    orientation=orientation
                )
                
                if state.is_valid_placement(placement):
                    return placement
        
        return None
    
    def _handle_backtrack(self, current_state: PackingState,
                         remaining_panels: List[PanelSize],
                         reason: BacktrackReason,
                         room: Room) -> Tuple[Optional[PackingState], List[PanelSize]]:
        """
        Handle backtracking based on trigger reason using intelligent strategy.
        """
        self.metrics.record_backtrack()
        
        # Use strategy to determine depth
        backtrack_depth = self.backtrack_strategy.determine_backtrack_depth(
            reason, current_state, self.backtrack_manager.current_depth
        )
        
        # Handle special cases
        if reason == BacktrackReason.TIME_LIMIT:
            return None, []
        
        # Get all available snapshots
        snapshots = self.backtrack_manager.snapshot_manager.snapshot_index
        
        # Use strategy to select best backtrack point
        selected_snapshot = self.backtrack_strategy.select_backtrack_point(
            list(snapshots.values()), current_state.coverage, reason
        )
        
        if selected_snapshot:
            # Record coverage before backtrack
            coverage_before = current_state.coverage
            
            # Backtrack to selected point
            snapshot = self.backtrack_manager.backtrack_to(
                selected_snapshot.decision_point
            )
            
            if snapshot:
                # Find alternative paths if stuck
                if reason == BacktrackReason.COVERAGE_PLATEAU:
                    alternatives = self.backtrack_strategy.find_alternative_paths(
                        snapshot, list(snapshots.values()), num_alternatives=2
                    )
                    if alternatives:
                        # Try first alternative
                        snapshot = alternatives[0]
                
                # Adjust panel order for next attempt
                if len(snapshot.remaining_panels) > 1:
                    alt_strategies = self.adaptive_sorter.get_alternative_strategies()
                    if alt_strategies:
                        sorter = MultiCriteriaSorter()
                        snapshot.remaining_panels = sorter.sort_panels(
                            snapshot.remaining_panels, alt_strategies[0]
                        )
                
                # Record backtrack result
                improvement = snapshot.coverage - coverage_before
                self.backtrack_strategy.record_backtrack_result(
                    reason, backtrack_depth, improvement
                )
                
                return snapshot.state, snapshot.remaining_panels
        
        # Fallback to simple depth-based backtrack
        target_depth = max(0, self.backtrack_manager.current_depth - backtrack_depth)
        snapshot = self.backtrack_manager.backtrack_to(target_depth)
        
        if snapshot:
            return snapshot.state, snapshot.remaining_panels
        
        return None, []
    
    def _refine_solution(self, panels: List[Panel], room: Room) -> List[Panel]:
        """
        Post-optimization refinement to improve coverage using local search.
        """
        # Calculate current coverage
        current_area = sum(p.size.area for p in panels)
        room_area = room.width * room.height
        current_coverage = current_area / room_area
        
        if current_coverage >= self.config.early_stop_coverage:
            return panels  # Already meets target
        
        # First apply local search refinement to optimize existing placement
        panels = self.local_search.refine_placement(panels, room)
        
        # Recalculate coverage after local search
        current_area = sum(p.size.area for p in panels)
        current_coverage = current_area / room_area
        
        if current_coverage >= self.config.early_stop_coverage:
            return panels  # Target reached after local search
        
        # Try to fill remaining gaps with smaller panels
        state = PackingState.from_room(room)
        for panel in panels:
            placement = PanelPlacement(
                panel_size=panel.size,
                position=(panel.position.x, panel.position.y),
                orientation=panel.orientation
            )
            state = state.add_placement(placement)
        
        # Try adding small panels to gaps
        small_panels = [PanelSize.PANEL_4X4, PanelSize.PANEL_4X6]
        added_panels = []
        
        for panel_size in small_panels:
            if panel_size not in self.config.panel_sizes:
                continue
            
            # Try to place in gaps
            for _ in range(10):  # Limit attempts
                placement = self._find_best_placement(panel_size, state, [panel_size])
                if placement:
                    new_state = self.transition_manager.apply_placement(state, placement)
                    if new_state:
                        state = new_state
                        added_panels.append(placement.to_panel(room.id))
                        
                        # Check if target reached
                        if state.coverage >= self.config.early_stop_coverage:
                            break
                else:
                    break
        
        # Final refinement on complete solution
        final_panels = panels + added_panels
        if added_panels:
            final_panels = self.local_search.refine_placement(final_panels, room)
        
        return final_panels


def create_enhanced_optimizer(config: Optional[OptimizerConfig] = None) -> EnhancedBLFOptimizer:
    """Factory function to create enhanced BLF optimizer."""
    if config is None:
        config = OptimizerConfig(
            max_time_seconds=5.0,
            early_stop_coverage=0.95,
            max_iterations=1000,
            grid_resolution=0.5,
            edge_alignment_weight=1.2,
            corner_preference_weight=1.0,
            waste_minimization_weight=1.5,
            enable_memoization=True
        )
    
    return EnhancedBLFOptimizer(config)
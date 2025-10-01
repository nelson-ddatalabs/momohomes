#!/usr/bin/env python3
"""
blf_lookahead.py - Lookahead Mechanisms for BLF Algorithm
===========================================================
Implements K-step lookahead to predict future placement success.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Set
from enum import Enum
import copy
import time

from models import Panel, PanelSize, Point, Room
from advanced_packing import PackingState, PanelPlacement


# Step 2.3.1: Lookahead Mechanism
# ================================

@dataclass
class LookaheadResult:
    """
    Result of K-step lookahead analysis.
    """
    sequence_score: float
    predicted_coverage: float
    placement_success_rate: float
    future_fit_probability: float
    recommended_depth: int
    placement_sequence: List[PanelPlacement]
    metadata: Dict[str, Any] = field(default_factory=dict)


class PlacementSequenceScorer:
    """
    Scores placement sequences based on multiple criteria.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize sequence scorer."""
        config = config or {}
        
        self.coverage_weight = config.get('coverage_weight', 0.4)
        self.compactness_weight = config.get('compactness_weight', 0.2)
        self.edge_weight = config.get('edge_weight', 0.2)
        self.future_fit_weight = config.get('future_fit_weight', 0.2)
    
    def score_sequence(self, placements: List[PanelPlacement],
                      initial_state: PackingState,
                      room_bounds: Tuple[float, float, float, float]) -> float:
        """
        Score a sequence of placements.
        """
        if not placements:
            return 0.0
        
        # Simulate placement sequence
        current_state = initial_state
        scores = []
        
        for placement in placements:
            # Score individual placement
            placement_score = self._score_placement(
                placement, current_state, room_bounds
            )
            scores.append(placement_score)
            
            # Update state
            current_state = current_state.add_placement(placement)
            if not current_state:
                break
        
        # Calculate aggregate score
        if not scores:
            return 0.0
        
        # Weight early placements more heavily
        weighted_scores = [
            score * (1.0 - i * 0.1) 
            for i, score in enumerate(scores)
        ]
        
        return sum(weighted_scores) / len(weighted_scores)
    
    def _score_placement(self, placement: PanelPlacement,
                        state: PackingState,
                        room_bounds: Tuple[float, float, float, float]) -> float:
        """
        Score a single placement in context.
        """
        score = 0.0
        x, y = placement.position
        # Get dimensions based on orientation
        if placement.orientation == "horizontal":
            width = placement.panel_size.width
            height = placement.panel_size.length
        else:  # vertical
            width = placement.panel_size.length
            height = placement.panel_size.width
        room_x, room_y, room_width, room_height = room_bounds
        
        # Coverage contribution
        panel_area = width * height
        room_area = room_width * room_height
        coverage_contrib = panel_area / room_area
        score += coverage_contrib * self.coverage_weight
        
        # Compactness (prefer placements near existing panels)
        if state.placements:
            min_distance = float('inf')
            for existing in state.placements:
                ex_x, ex_y = existing.position
                distance = abs(x - ex_x) + abs(y - ex_y)
                min_distance = min(min_distance, distance)
            
            compactness = 1.0 / (1.0 + min_distance * 0.1)
            score += compactness * self.compactness_weight
        else:
            # First placement - prefer corner
            corner_distance = x + y
            compactness = 1.0 / (1.0 + corner_distance * 0.05)
            score += compactness * self.compactness_weight
        
        # Edge alignment
        edge_aligned = 0
        if x == room_x or x + width == room_x + room_width:
            edge_aligned += 1
        if y == room_y or y + height == room_y + room_height:
            edge_aligned += 1
        
        edge_score = edge_aligned / 2.0
        score += edge_score * self.edge_weight
        
        # Future fit potential (remaining space quality)
        occupied_area = sum(p.panel_size.area for p in state.placements)
        remaining_area = room_area - occupied_area - panel_area
        if remaining_area > 0:
            # Check if remaining space can fit standard panels
            can_fit_large = remaining_area >= PanelSize.PANEL_6X8.area
            can_fit_medium = remaining_area >= PanelSize.PANEL_4X6.area
            can_fit_small = remaining_area >= PanelSize.PANEL_4X4.area
            
            future_fit = (can_fit_large * 0.5 + 
                         can_fit_medium * 0.3 + 
                         can_fit_small * 0.2)
            score += future_fit * self.future_fit_weight
        
        return score


class FutureFitPredictor:
    """
    Predicts future placement success based on current state.
    """
    
    def __init__(self):
        """Initialize future fit predictor."""
        self.panel_areas = {
            PanelSize.PANEL_6X8: 48,
            PanelSize.PANEL_6X6: 36,
            PanelSize.PANEL_4X6: 24,
            PanelSize.PANEL_4X4: 16
        }
    
    def predict_future_fit(self, state: PackingState,
                          remaining_panels: List[PanelSize],
                          room_bounds: Tuple[float, float, float, float]) -> float:
        """
        Predict probability of successfully placing remaining panels.
        """
        if not remaining_panels:
            return 1.0
        
        room_x, room_y, room_width, room_height = room_bounds
        room_area = room_width * room_height
        
        # Calculate remaining area
        occupied_area = sum(p.panel_size.area for p in state.placements)
        remaining_area = room_area - occupied_area
        
        # Calculate area needed for remaining panels
        needed_area = sum(self.panel_areas[p] for p in remaining_panels)
        
        # Basic area check
        if needed_area > remaining_area * 1.2:  # Allow 20% overlap optimism
            return 0.0
        
        # Analyze remaining space quality
        free_rectangles = self._find_free_rectangles(state, room_bounds)
        
        if not free_rectangles:
            return 0.0
        
        # Check how many panels can potentially fit
        fit_count = 0
        remaining_copy = list(remaining_panels)
        
        for rect in sorted(free_rectangles, key=lambda r: r[2] * r[3], reverse=True):
            rect_x, rect_y, rect_w, rect_h = rect
            
            for i, panel in enumerate(remaining_copy):
                panel_w, panel_h = self._get_panel_dimensions(panel)
                
                # Check both orientations
                if ((panel_w <= rect_w and panel_h <= rect_h) or
                    (panel_h <= rect_w and panel_w <= rect_h)):
                    fit_count += 1
                    remaining_copy.pop(i)
                    break
            
            if not remaining_copy:
                break
        
        # Calculate fit probability
        if remaining_panels:
            fit_probability = fit_count / len(remaining_panels)
        else:
            fit_probability = 1.0
        
        # Adjust based on fragmentation
        fragmentation_penalty = self._calculate_fragmentation(free_rectangles)
        fit_probability *= (1.0 - fragmentation_penalty * 0.3)
        
        return min(1.0, max(0.0, fit_probability))
    
    def _find_free_rectangles(self, state: PackingState,
                             room_bounds: Tuple[float, float, float, float]) -> List[Tuple[float, float, float, float]]:
        """
        Find maximal free rectangles in the room.
        Simplified version - in production would use maximal rectangles algorithm.
        """
        room_x, room_y, room_width, room_height = room_bounds
        
        # For simplicity, divide into grid and find free cells
        grid_size = 2.0  # 2-foot grid
        free_rects = []
        
        # Check grid cells
        for gx in range(int(room_width / grid_size)):
            for gy in range(int(room_height / grid_size)):
                x = room_x + gx * grid_size
                y = room_y + gy * grid_size
                
                # Find maximal rectangle starting from this point
                max_w = room_x + room_width - x
                max_h = room_y + room_height - y
                
                # Check if this area is free
                test_rect = (x, y, min(grid_size * 4, max_w), min(grid_size * 4, max_h))
                if self._is_rect_free(test_rect, state):
                    free_rects.append(test_rect)
        
        return free_rects
    
    def _is_rect_free(self, rect: Tuple[float, float, float, float],
                     state: PackingState) -> bool:
        """
        Check if rectangle is free of placements.
        """
        rx, ry, rw, rh = rect
        
        for placement in state.placements:
            px, py = placement.position
            # Get dimensions based on orientation
            if placement.orientation == "horizontal":
                pw = placement.panel_size.width
                ph = placement.panel_size.length
            else:  # vertical
                pw = placement.panel_size.length
                ph = placement.panel_size.width
            
            # Check overlap
            if not (rx + rw <= px or px + pw <= rx or
                   ry + rh <= py or py + ph <= ry):
                return False
        
        return True
    
    def _get_panel_dimensions(self, panel_size: PanelSize) -> Tuple[float, float]:
        """
        Get panel dimensions.
        """
        if panel_size == PanelSize.PANEL_6X8:
            return (6, 8)
        elif panel_size == PanelSize.PANEL_6X6:
            return (6, 6)
        elif panel_size == PanelSize.PANEL_4X6:
            return (4, 6)
        else:  # PANEL_4X4
            return (4, 4)
    
    def _calculate_fragmentation(self, free_rectangles: List[Tuple[float, float, float, float]]) -> float:
        """
        Calculate fragmentation penalty based on free space distribution.
        """
        if not free_rectangles:
            return 1.0
        
        # Calculate area statistics
        areas = [r[2] * r[3] for r in free_rectangles]
        total_area = sum(areas)
        
        if total_area == 0:
            return 1.0
        
        # More rectangles = more fragmentation
        num_fragments = len(free_rectangles)
        avg_area = total_area / num_fragments
        
        # Penalty based on number of small fragments
        small_fragments = sum(1 for a in areas if a < 16)  # Smaller than 4x4 panel
        
        fragmentation = (num_fragments / 10.0) * 0.5 + (small_fragments / num_fragments) * 0.5
        
        return min(1.0, fragmentation)


class KStepLookahead:
    """
    K-step lookahead optimizer for placement decisions.
    """
    
    def __init__(self, k: int = 3, config: Dict[str, Any] = None):
        """
        Initialize K-step lookahead.
        
        Args:
            k: Number of steps to look ahead
            config: Configuration parameters
        """
        self.k = k
        config = config or {}
        
        self.max_branches = config.get('max_branches', 5)
        self.beam_width = config.get('beam_width', 3)
        self.time_limit = config.get('time_limit', 0.5)
        
        self.sequence_scorer = PlacementSequenceScorer(config)
        self.future_predictor = FutureFitPredictor()
    
    def evaluate_placement(self, placement: PanelPlacement,
                          current_state: PackingState,
                          remaining_panels: List[PanelSize],
                          room_bounds: Tuple[float, float, float, float],
                          position_generator) -> LookaheadResult:
        """
        Evaluate a placement by looking ahead K steps.
        """
        start_time = time.time()
        
        # Simulate placement
        new_state = current_state.add_placement(placement)
        if not new_state:
            return LookaheadResult(
                sequence_score=0.0,
                predicted_coverage=current_state.coverage,
                placement_success_rate=0.0,
                future_fit_probability=0.0,
                recommended_depth=0,
                placement_sequence=[]
            )
        
        # Perform K-step lookahead
        best_sequence = self._lookahead_search(
            new_state, remaining_panels[:self.k],
            room_bounds, position_generator,
            depth=0, start_time=start_time
        )
        
        # Score the sequence
        sequence_score = self.sequence_scorer.score_sequence(
            [placement] + best_sequence,
            current_state, room_bounds
        )
        
        # Predict future success
        future_state = new_state
        for p in best_sequence:
            future_state = future_state.add_placement(p)
            if not future_state:
                break
        
        if future_state:
            future_fit_prob = self.future_predictor.predict_future_fit(
                future_state, remaining_panels[len(best_sequence):],
                room_bounds
            )
            predicted_coverage = future_state.coverage
        else:
            future_fit_prob = 0.0
            predicted_coverage = new_state.coverage
        
        # Calculate success rate
        success_rate = (len(best_sequence) + 1) / (self.k + 1)
        
        # Recommend depth based on analysis
        if future_fit_prob > 0.8:
            recommended_depth = min(self.k + 2, 5)
        elif future_fit_prob > 0.5:
            recommended_depth = self.k
        else:
            recommended_depth = max(1, self.k - 1)
        
        return LookaheadResult(
            sequence_score=sequence_score,
            predicted_coverage=predicted_coverage,
            placement_success_rate=success_rate,
            future_fit_probability=future_fit_prob,
            recommended_depth=recommended_depth,
            placement_sequence=[placement] + best_sequence,
            metadata={
                'lookahead_time': time.time() - start_time,
                'branches_explored': self.branches_explored
            }
        )
    
    def _lookahead_search(self, state: PackingState,
                         panels: List[PanelSize],
                         room_bounds: Tuple[float, float, float, float],
                         position_generator,
                         depth: int,
                         start_time: float) -> List[PanelPlacement]:
        """
        Recursive lookahead search with beam search pruning.
        """
        # Check termination conditions
        if not panels or depth >= self.k:
            return []
        
        if time.time() - start_time > self.time_limit:
            return []
        
        # Track branches for statistics
        if depth == 0:
            self.branches_explored = 0
        
        # Generate candidate placements for next panel
        panel = panels[0]
        candidates = []
        
        # Get positions from generator
        positions = position_generator.generate_positions(panel, state.placements)
        
        # Limit positions to explore
        for pos in positions[:self.max_branches]:
            for orientation in ["horizontal", "vertical"]:
                placement = PanelPlacement(
                    panel_size=panel,
                    position=pos,
                    orientation=orientation
                )
                
                if state.is_valid_placement(placement):
                    candidates.append(placement)
                    self.branches_explored += 1
        
        if not candidates:
            return []
        
        # Score candidates
        scored_candidates = []
        for candidate in candidates:
            new_state = state.add_placement(candidate)
            if new_state:
                score = self.sequence_scorer._score_placement(
                    candidate, state, room_bounds
                )
                scored_candidates.append((score, candidate, new_state))
        
        # Beam search - keep top candidates
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        beam = scored_candidates[:self.beam_width]
        
        # Explore best branches
        best_sequence = []
        best_score = -1
        
        for score, candidate, new_state in beam:
            # Recursive lookahead
            future_sequence = self._lookahead_search(
                new_state, panels[1:], room_bounds,
                position_generator, depth + 1, start_time
            )
            
            # Evaluate complete sequence
            sequence = [candidate] + future_sequence
            seq_score = self.sequence_scorer.score_sequence(
                sequence, state, room_bounds
            )
            
            if seq_score > best_score:
                best_score = seq_score
                best_sequence = sequence
        
        return best_sequence
    
    def optimize_depth(self, state: PackingState,
                      remaining_panels: List[PanelSize],
                      room_bounds: Tuple[float, float, float, float]) -> int:
        """
        Dynamically optimize lookahead depth based on state.
        """
        room_area = room_bounds[2] * room_bounds[3]
        occupied_area = sum(p.panel_size.area for p in state.placements)
        remaining_area = room_area - occupied_area
        
        # More depth needed when space is tight
        occupied_area = sum(p.panel_size.area for p in state.placements)
        fill_ratio = occupied_area / room_area
        
        if fill_ratio < 0.5:
            # Early stage - less lookahead needed
            return max(2, self.k - 1)
        elif fill_ratio < 0.8:
            # Middle stage - standard lookahead
            return self.k
        else:
            # End game - more lookahead needed
            return min(self.k + 2, 6)


# Step 2.3.2: Local Search Refinement
# ====================================

class LocalSearchRefinement:
    """
    Local search methods to refine panel placements.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize local search refinement."""
        config = config or {}
        
        self.max_iterations = config.get('max_iterations', 100)
        self.time_limit = config.get('time_limit', 1.0)
        self.improvement_threshold = config.get('improvement_threshold', 0.001)
        self.neighborhood_size = config.get('neighborhood_size', 5)
    
    def refine_placement(self, panels: List[Panel], room: Room) -> List[Panel]:
        """
        Refine panel placement using local search techniques.
        """
        start_time = time.time()
        current_panels = list(panels)
        current_coverage = self._calculate_coverage(current_panels, room)
        
        iteration = 0
        no_improvement_count = 0
        
        while iteration < self.max_iterations:
            if time.time() - start_time > self.time_limit:
                break
            
            iteration += 1
            improved = False
            
            # Try different local search operators
            operators = [
                self._two_opt_swap,
                self._slide_adjustment,
                self._rotation_exploration,
                self._neighborhood_search
            ]
            
            for operator in operators:
                new_panels = operator(current_panels, room)
                new_coverage = self._calculate_coverage(new_panels, room)
                
                if new_coverage > current_coverage + self.improvement_threshold:
                    current_panels = new_panels
                    current_coverage = new_coverage
                    improved = True
                    no_improvement_count = 0
                    break
            
            if not improved:
                no_improvement_count += 1
                if no_improvement_count >= 10:
                    break  # Local optimum reached
        
        return current_panels
    
    def _two_opt_swap(self, panels: List[Panel], room: Room) -> List[Panel]:
        """
        Try swapping pairs of panels to improve placement.
        """
        best_panels = list(panels)
        best_coverage = self._calculate_coverage(panels, room)
        
        # Try swapping each pair
        for i in range(len(panels)):
            for j in range(i + 1, min(i + self.neighborhood_size, len(panels))):
                # Create swap candidate
                swapped = list(panels)
                
                # Swap positions but keep same panel sizes
                panel_i = swapped[i]
                panel_j = swapped[j]
                
                # Try swapping positions
                new_panel_i = Panel(
                    panel_id=panel_i.panel_id,
                    size=panel_i.size,
                    position=panel_j.position,
                    orientation=panel_i.orientation,
                    room_id=panel_i.room_id
                )
                
                new_panel_j = Panel(
                    panel_id=panel_j.panel_id,
                    size=panel_j.size,
                    position=panel_i.position,
                    orientation=panel_j.orientation,
                    room_id=panel_j.room_id
                )
                
                swapped[i] = new_panel_i
                swapped[j] = new_panel_j
                
                # Check validity and improvement
                if self._is_valid_configuration(swapped, room):
                    coverage = self._calculate_coverage(swapped, room)
                    if coverage > best_coverage:
                        best_panels = swapped
                        best_coverage = coverage
        
        return best_panels
    
    def _slide_adjustment(self, panels: List[Panel], room: Room) -> List[Panel]:
        """
        Try sliding panels to eliminate gaps.
        """
        best_panels = list(panels)
        best_coverage = self._calculate_coverage(panels, room)
        
        # Sort panels by position for systematic sliding
        sorted_panels = sorted(panels, key=lambda p: (p.position.y, p.position.x))
        
        for i, panel in enumerate(sorted_panels):
            # Try sliding in each direction
            directions = [
                (0, -1),   # Up
                (0, 1),    # Down
                (-1, 0),   # Left
                (1, 0),    # Right
            ]
            
            for dx, dy in directions:
                # Try different slide distances
                for distance in [0.5, 1.0, 2.0]:
                    new_x = panel.position.x + dx * distance
                    new_y = panel.position.y + dy * distance
                    
                    # Create adjusted panel
                    adjusted = list(sorted_panels)
                    adjusted[i] = Panel(
                        panel_id=panel.panel_id,
                        size=panel.size,
                        position=Point(new_x, new_y),
                        orientation=panel.orientation,
                        room_id=panel.room_id
                    )
                    
                    # Check validity and improvement
                    if self._is_valid_configuration(adjusted, room):
                        coverage = self._calculate_coverage(adjusted, room)
                        if coverage > best_coverage:
                            best_panels = adjusted
                            best_coverage = coverage
        
        return best_panels
    
    def _rotation_exploration(self, panels: List[Panel], room: Room) -> List[Panel]:
        """
        Try rotating panels to improve fit.
        """
        best_panels = list(panels)
        best_coverage = self._calculate_coverage(panels, room)
        
        for i, panel in enumerate(panels):
            # Skip square panels
            if panel.size in [PanelSize.PANEL_6X6, PanelSize.PANEL_4X4]:
                continue
            
            # Try rotation
            rotated_panels = list(panels)
            new_orientation = "vertical" if panel.orientation == "horizontal" else "horizontal"
            
            rotated_panels[i] = Panel(
                panel_id=panel.panel_id,
                size=panel.size,
                position=panel.position,
                orientation=new_orientation,
                room_id=panel.room_id
            )
            
            # Check validity and improvement
            if self._is_valid_configuration(rotated_panels, room):
                coverage = self._calculate_coverage(rotated_panels, room)
                if coverage > best_coverage:
                    best_panels = rotated_panels
                    best_coverage = coverage
        
        return best_panels
    
    def _neighborhood_search(self, panels: List[Panel], room: Room) -> List[Panel]:
        """
        Search in the neighborhood of current configuration.
        """
        best_panels = list(panels)
        best_coverage = self._calculate_coverage(panels, room)
        
        # Try small perturbations to multiple panels
        for _ in range(self.neighborhood_size):
            perturbed = list(panels)
            
            # Randomly select panels to perturb
            import random
            num_to_perturb = min(3, len(panels))
            indices = random.sample(range(len(panels)), num_to_perturb)
            
            for idx in indices:
                panel = perturbed[idx]
                
                # Small random perturbation
                dx = random.choice([-0.5, 0, 0.5])
                dy = random.choice([-0.5, 0, 0.5])
                
                new_x = panel.position.x + dx
                new_y = panel.position.y + dy
                
                perturbed[idx] = Panel(
                    panel_id=panel.panel_id,
                    size=panel.size,
                    position=Point(new_x, new_y),
                    orientation=panel.orientation,
                    room_id=panel.room_id
                )
            
            # Check validity and improvement
            if self._is_valid_configuration(perturbed, room):
                coverage = self._calculate_coverage(perturbed, room)
                if coverage > best_coverage:
                    best_panels = perturbed
                    best_coverage = coverage
        
        return best_panels
    
    def _is_valid_configuration(self, panels: List[Panel], room: Room) -> bool:
        """
        Check if panel configuration is valid (no overlaps, within bounds).
        """
        # Check room boundaries
        for panel in panels:
            x, y = panel.position.x, panel.position.y
            width, height = panel.width, panel.length
            
            # Check if within room
            if (x < room.position.x or y < room.position.y or
                x + width > room.position.x + room.width or
                y + height > room.position.y + room.height):
                return False
        
        # Check for overlaps
        for i, panel1 in enumerate(panels):
            x1, y1 = panel1.position.x, panel1.position.y
            w1, h1 = panel1.width, panel1.length
            
            for panel2 in panels[i+1:]:
                x2, y2 = panel2.position.x, panel2.position.y
                w2, h2 = panel2.width, panel2.length
                
                # Check overlap
                if not (x1 + w1 <= x2 or x2 + w2 <= x1 or
                       y1 + h1 <= y2 or y2 + h2 <= y1):
                    return False
        
        return True
    
    def _calculate_coverage(self, panels: List[Panel], room: Room) -> float:
        """
        Calculate coverage ratio for panels in room.
        """
        if not panels:
            return 0.0
        
        total_panel_area = sum(p.size.area for p in panels)
        room_area = room.width * room.height
        
        return total_panel_area / room_area if room_area > 0 else 0.0
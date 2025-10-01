#!/usr/bin/env python3
"""
result_aggregation.py - Result Aggregation System
=================================================
Production-ready best result selector, combination strategies,
coverage calculator, and quality scorer for optimal solution selection.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from collections import defaultdict
from enum import Enum
import math
import numpy as np

from models import Room, PanelSize
from advanced_packing import PackingState, PanelPlacement


class QualityMetric(Enum):
    """Quality metrics for scoring."""
    COVERAGE = "coverage"
    WASTE = "waste"
    COMPACTNESS = "compactness"
    UNIFORMITY = "uniformity"
    EDGE_ALIGNMENT = "edge_alignment"
    STRUCTURAL_INTEGRITY = "structural_integrity"
    AESTHETIC = "aesthetic"
    EFFICIENCY = "efficiency"


class CombinationStrategy(Enum):
    """Strategies for combining multiple results."""
    MERGE = "merge"  # Merge non-overlapping placements
    OVERLAY = "overlay"  # Overlay best parts
    HYBRID = "hybrid"  # Hybrid approach
    VOTING = "voting"  # Voting-based combination
    WEIGHTED = "weighted"  # Weighted combination


@dataclass
class SolutionMetrics:
    """Comprehensive metrics for a solution."""
    coverage: float
    waste: float
    num_panels: int
    compactness: float
    uniformity: float
    edge_alignment: float
    structural_score: float
    aesthetic_score: float
    efficiency: float
    gaps_area: float
    largest_gap: float
    fragmentation: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityScore:
    """Quality score with breakdown."""
    total_score: float
    metric_scores: Dict[QualityMetric, float]
    weights: Dict[QualityMetric, float]
    normalized: bool = True


@dataclass
class CombinedResult:
    """Result from combining multiple solutions."""
    final_state: PackingState
    source_states: List[PackingState]
    combination_strategy: CombinationStrategy
    metrics: SolutionMetrics
    quality_score: QualityScore
    metadata: Dict[str, Any] = field(default_factory=dict)


class CoverageCalculator:
    """Calculates various coverage metrics."""
    
    def __init__(self, room: Room):
        self.room = room
        self.room_area = room.width * room.height
    
    def calculate_coverage(self, state: PackingState) -> float:
        """Calculate basic coverage ratio."""
        if not state.placed_panels:
            return 0.0
        
        covered_area = sum(
            p.panel_size.width * p.panel_size.height
            for p in state.placed_panels
        )
        
        return covered_area / self.room_area if self.room_area > 0 else 0.0
    
    def calculate_effective_coverage(self, state: PackingState) -> float:
        """Calculate effective coverage considering overlaps."""
        if not state.placed_panels:
            return 0.0
        
        # Create occupancy grid for accurate calculation
        grid_resolution = 1.0
        grid_width = int(self.room.width / grid_resolution)
        grid_height = int(self.room.height / grid_resolution)
        
        occupied = set()
        
        for panel in state.placed_panels:
            x, y = panel.position
            w, h = panel.panel_size.width, panel.panel_size.height
            
            # Mark grid cells as occupied
            for gx in range(int(x / grid_resolution), int((x + w) / grid_resolution)):
                for gy in range(int(y / grid_resolution), int((y + h) / grid_resolution)):
                    if 0 <= gx < grid_width and 0 <= gy < grid_height:
                        occupied.add((gx, gy))
        
        effective_area = len(occupied) * grid_resolution * grid_resolution
        return effective_area / self.room_area if self.room_area > 0 else 0.0
    
    def calculate_waste(self, state: PackingState) -> float:
        """Calculate waste percentage."""
        coverage = self.calculate_coverage(state)
        return 1.0 - coverage
    
    def identify_gaps(self, state: PackingState) -> List[Tuple[float, float, float, float]]:
        """Identify gaps in placement."""
        gaps = []
        
        # Simple gap detection using grid
        grid_size = 5.0
        
        for x in range(0, int(self.room.width), int(grid_size)):
            for y in range(0, int(self.room.height), int(grid_size)):
                # Check if this grid cell is empty
                is_empty = True
                for panel in state.placed_panels:
                    px, py = panel.position
                    pw, ph = panel.panel_size.width, panel.panel_size.height
                    
                    if (px <= x < px + pw and py <= y < py + ph):
                        is_empty = False
                        break
                
                if is_empty:
                    # Find extent of gap
                    gap_width = grid_size
                    gap_height = grid_size
                    
                    # Extend horizontally
                    test_x = x + grid_size
                    while test_x < self.room.width:
                        still_empty = True
                        for panel in state.placed_panels:
                            px, py = panel.position
                            pw, ph = panel.panel_size.width, panel.panel_size.height
                            if (px <= test_x < px + pw and py <= y < py + ph):
                                still_empty = False
                                break
                        
                        if still_empty:
                            gap_width += grid_size
                            test_x += grid_size
                        else:
                            break
                    
                    gaps.append((x, y, gap_width, gap_height))
        
        # Merge adjacent gaps
        merged_gaps = self._merge_gaps(gaps)
        
        return merged_gaps
    
    def _merge_gaps(self, gaps: List[Tuple[float, float, float, float]]) -> List[Tuple[float, float, float, float]]:
        """Merge adjacent gaps."""
        if not gaps:
            return []
        
        # Simple merging - can be improved
        merged = []
        used = set()
        
        for i, gap1 in enumerate(gaps):
            if i in used:
                continue
            
            x1, y1, w1, h1 = gap1
            
            # Try to merge with adjacent gaps
            for j, gap2 in enumerate(gaps[i+1:], i+1):
                if j in used:
                    continue
                
                x2, y2, w2, h2 = gap2
                
                # Check if adjacent
                if (abs(x1 - x2) < 1 and abs(y1 - y2) < 1):
                    # Merge
                    x1 = min(x1, x2)
                    y1 = min(y1, y2)
                    w1 = max(x1 + w1, x2 + w2) - x1
                    h1 = max(y1 + h1, y2 + h2) - y1
                    used.add(j)
            
            merged.append((x1, y1, w1, h1))
            used.add(i)
        
        return merged
    
    def calculate_fragmentation(self, state: PackingState) -> float:
        """Calculate fragmentation score (0-1, lower is better)."""
        gaps = self.identify_gaps(state)
        
        if not gaps:
            return 0.0
        
        # Calculate fragmentation based on number and size of gaps
        num_gaps = len(gaps)
        avg_gap_size = sum(w * h for _, _, w, h in gaps) / num_gaps
        
        # Normalize
        max_expected_gaps = 20
        fragmentation = min(1.0, num_gaps / max_expected_gaps)
        
        # Consider gap size variance
        if num_gaps > 1:
            gap_sizes = [w * h for _, _, w, h in gaps]
            variance = np.var(gap_sizes)
            normalized_variance = variance / (self.room_area ** 2)
            fragmentation = (fragmentation + normalized_variance) / 2
        
        return fragmentation


class QualityScorer:
    """Scores solution quality."""
    
    def __init__(self, room: Room, weights: Optional[Dict[QualityMetric, float]] = None):
        self.room = room
        self.weights = weights or self._default_weights()
        self.coverage_calculator = CoverageCalculator(room)
    
    def _default_weights(self) -> Dict[QualityMetric, float]:
        """Default quality metric weights."""
        return {
            QualityMetric.COVERAGE: 0.4,
            QualityMetric.WASTE: 0.2,
            QualityMetric.COMPACTNESS: 0.1,
            QualityMetric.UNIFORMITY: 0.05,
            QualityMetric.EDGE_ALIGNMENT: 0.05,
            QualityMetric.STRUCTURAL_INTEGRITY: 0.1,
            QualityMetric.AESTHETIC: 0.05,
            QualityMetric.EFFICIENCY: 0.05
        }
    
    def score(self, state: PackingState) -> QualityScore:
        """Calculate comprehensive quality score."""
        metric_scores = {}
        
        # Coverage score
        coverage = self.coverage_calculator.calculate_coverage(state)
        metric_scores[QualityMetric.COVERAGE] = coverage
        
        # Waste score (inverted)
        waste = self.coverage_calculator.calculate_waste(state)
        metric_scores[QualityMetric.WASTE] = 1.0 - waste
        
        # Compactness score
        compactness = self._calculate_compactness(state)
        metric_scores[QualityMetric.COMPACTNESS] = compactness
        
        # Uniformity score
        uniformity = self._calculate_uniformity(state)
        metric_scores[QualityMetric.UNIFORMITY] = uniformity
        
        # Edge alignment score
        edge_alignment = self._calculate_edge_alignment(state)
        metric_scores[QualityMetric.EDGE_ALIGNMENT] = edge_alignment
        
        # Structural integrity score
        structural = self._calculate_structural_integrity(state)
        metric_scores[QualityMetric.STRUCTURAL_INTEGRITY] = structural
        
        # Aesthetic score
        aesthetic = self._calculate_aesthetic_score(state)
        metric_scores[QualityMetric.AESTHETIC] = aesthetic
        
        # Efficiency score
        efficiency = self._calculate_efficiency(state)
        metric_scores[QualityMetric.EFFICIENCY] = efficiency
        
        # Calculate weighted total
        total_score = sum(
            score * self.weights.get(metric, 0.0)
            for metric, score in metric_scores.items()
        )
        
        return QualityScore(
            total_score=total_score,
            metric_scores=metric_scores,
            weights=self.weights,
            normalized=True
        )
    
    def _calculate_compactness(self, state: PackingState) -> float:
        """Calculate compactness of placement."""
        if not state.placed_panels:
            return 0.0
        
        # Calculate bounding box of placement
        min_x = min(p.position[0] for p in state.placed_panels)
        max_x = max(p.position[0] + p.panel_size.width for p in state.placed_panels)
        min_y = min(p.position[1] for p in state.placed_panels)
        max_y = max(p.position[1] + p.panel_size.height for p in state.placed_panels)
        
        bbox_area = (max_x - min_x) * (max_y - min_y)
        covered_area = sum(
            p.panel_size.width * p.panel_size.height
            for p in state.placed_panels
        )
        
        return covered_area / bbox_area if bbox_area > 0 else 0.0
    
    def _calculate_uniformity(self, state: PackingState) -> float:
        """Calculate uniformity of panel distribution."""
        if not state.placed_panels:
            return 0.0
        
        # Divide room into quadrants
        mid_x = self.room.width / 2
        mid_y = self.room.height / 2
        
        quadrant_counts = [0, 0, 0, 0]
        
        for panel in state.placed_panels:
            cx = panel.position[0] + panel.panel_size.width / 2
            cy = panel.position[1] + panel.panel_size.height / 2
            
            if cx < mid_x and cy < mid_y:
                quadrant_counts[0] += 1
            elif cx >= mid_x and cy < mid_y:
                quadrant_counts[1] += 1
            elif cx < mid_x and cy >= mid_y:
                quadrant_counts[2] += 1
            else:
                quadrant_counts[3] += 1
        
        # Calculate uniformity
        total = sum(quadrant_counts)
        if total == 0:
            return 0.0
        
        expected = total / 4
        variance = sum((c - expected) ** 2 for c in quadrant_counts) / 4
        
        # Normalize (lower variance is better)
        max_variance = expected ** 2  # Worst case: all in one quadrant
        uniformity = 1.0 - (variance / max_variance) if max_variance > 0 else 1.0
        
        return uniformity
    
    def _calculate_edge_alignment(self, state: PackingState) -> float:
        """Calculate edge alignment score."""
        if not state.placed_panels:
            return 0.0
        
        aligned_count = 0
        total_edges = 0
        
        for panel in state.placed_panels:
            x, y = panel.position
            w, h = panel.panel_size.width, panel.panel_size.height
            
            # Check alignment with room edges
            if abs(x) < 1:  # Left edge
                aligned_count += 1
            if abs(x + w - self.room.width) < 1:  # Right edge
                aligned_count += 1
            if abs(y) < 1:  # Bottom edge
                aligned_count += 1
            if abs(y + h - self.room.height) < 1:  # Top edge
                aligned_count += 1
            
            total_edges += 4
            
            # Check alignment with other panels
            for other in state.placed_panels:
                if other == panel:
                    continue
                
                ox, oy = other.position
                ow, oh = other.panel_size.width, other.panel_size.height
                
                # Check edge alignment
                if abs(x - ox) < 1 or abs(x - (ox + ow)) < 1:
                    aligned_count += 0.5
                if abs(y - oy) < 1 or abs(y - (oy + oh)) < 1:
                    aligned_count += 0.5
        
        return aligned_count / total_edges if total_edges > 0 else 0.0
    
    def _calculate_structural_integrity(self, state: PackingState) -> float:
        """Calculate structural integrity score."""
        if not state.placed_panels:
            return 0.0
        
        supported = 0
        total = len(state.placed_panels)
        
        for panel in state.placed_panels:
            x, y = panel.position
            
            # Check if on floor
            if y <= 1:
                supported += 1
            else:
                # Check if supported by other panels
                for other in state.placed_panels:
                    if other == panel:
                        continue
                    
                    ox, oy = other.position
                    ow, oh = other.panel_size.width, other.panel_size.height
                    
                    # Check if other panel supports this one
                    if (oy + oh >= y - 1 and oy < y and
                        not (x + panel.panel_size.width <= ox or ox + ow <= x)):
                        supported += 1
                        break
        
        return supported / total if total > 0 else 1.0
    
    def _calculate_aesthetic_score(self, state: PackingState) -> float:
        """Calculate aesthetic score based on visual balance."""
        if not state.placed_panels:
            return 0.0
        
        # Calculate center of mass
        total_mass = 0
        com_x = 0
        com_y = 0
        
        for panel in state.placed_panels:
            mass = panel.panel_size.width * panel.panel_size.height
            cx = panel.position[0] + panel.panel_size.width / 2
            cy = panel.position[1] + panel.panel_size.height / 2
            
            com_x += cx * mass
            com_y += cy * mass
            total_mass += mass
        
        if total_mass > 0:
            com_x /= total_mass
            com_y /= total_mass
        
        # Calculate distance from room center
        room_cx = self.room.width / 2
        room_cy = self.room.height / 2
        
        distance = math.sqrt((com_x - room_cx) ** 2 + (com_y - room_cy) ** 2)
        max_distance = math.sqrt(room_cx ** 2 + room_cy ** 2)
        
        # Normalize (closer to center is better)
        aesthetic = 1.0 - (distance / max_distance) if max_distance > 0 else 1.0
        
        return aesthetic
    
    def _calculate_efficiency(self, state: PackingState) -> float:
        """Calculate placement efficiency."""
        if not state.placed_panels:
            return 0.0
        
        # Consider panel usage efficiency
        total_panels = len(state.placed_panels)
        unique_sizes = len(set(p.panel_size.id for p in state.placed_panels))
        
        # Efficiency based on using fewer panel types
        type_efficiency = 1.0 / unique_sizes if unique_sizes > 0 else 0.0
        
        # Efficiency based on coverage per panel
        coverage = self.coverage_calculator.calculate_coverage(state)
        coverage_efficiency = coverage / (total_panels / 100) if total_panels > 0 else 0.0
        coverage_efficiency = min(1.0, coverage_efficiency)
        
        return (type_efficiency + coverage_efficiency) / 2


class BestResultSelector:
    """Selects best result from multiple solutions."""
    
    def __init__(self, room: Room, scorer: Optional[QualityScorer] = None):
        self.room = room
        self.scorer = scorer or QualityScorer(room)
    
    def select_best(self, 
                   results: List[PackingState],
                   criteria: Optional[List[QualityMetric]] = None) -> Optional[PackingState]:
        """Select best result based on criteria."""
        if not results:
            return None
        
        if len(results) == 1:
            return results[0]
        
        # Score all results
        scored_results = []
        for result in results:
            score = self.scorer.score(result)
            scored_results.append((score.total_score, result))
        
        # Sort by score
        scored_results.sort(reverse=True, key=lambda x: x[0])
        
        # Apply additional criteria if specified
        if criteria:
            scored_results = self._apply_criteria(scored_results, criteria)
        
        return scored_results[0][1] if scored_results else None
    
    def _apply_criteria(self,
                       scored_results: List[Tuple[float, PackingState]],
                       criteria: List[QualityMetric]) -> List[Tuple[float, PackingState]]:
        """Apply additional selection criteria."""
        # Re-score based on specific criteria
        rescored = []
        
        for score, result in scored_results:
            quality_score = self.scorer.score(result)
            
            # Calculate weighted score for specified criteria
            criteria_score = sum(
                quality_score.metric_scores.get(metric, 0.0)
                for metric in criteria
            ) / len(criteria) if criteria else score
            
            rescored.append((criteria_score, result))
        
        rescored.sort(reverse=True, key=lambda x: x[0])
        return rescored
    
    def select_top_k(self, results: List[PackingState], k: int = 3) -> List[PackingState]:
        """Select top k results."""
        if not results:
            return []
        
        # Score all results
        scored_results = []
        for result in results:
            score = self.scorer.score(result)
            scored_results.append((score.total_score, result))
        
        # Sort and take top k
        scored_results.sort(reverse=True, key=lambda x: x[0])
        return [result for _, result in scored_results[:k]]


class CombinationStrategies:
    """Strategies for combining multiple solutions."""
    
    def __init__(self, room: Room):
        self.room = room
        self.coverage_calculator = CoverageCalculator(room)
        self.scorer = QualityScorer(room)
    
    def combine(self,
               states: List[PackingState],
               strategy: CombinationStrategy) -> Optional[CombinedResult]:
        """Combine multiple states using specified strategy."""
        if not states:
            return None
        
        if len(states) == 1:
            return self._wrap_single(states[0])
        
        if strategy == CombinationStrategy.MERGE:
            return self.merge_solutions(states)
        elif strategy == CombinationStrategy.OVERLAY:
            return self.overlay_solutions(states)
        elif strategy == CombinationStrategy.VOTING:
            return self.voting_combination(states)
        elif strategy == CombinationStrategy.WEIGHTED:
            return self.weighted_combination(states)
        else:  # HYBRID
            return self.hybrid_combination(states)
    
    def _wrap_single(self, state: PackingState) -> CombinedResult:
        """Wrap single state as combined result."""
        metrics = self._calculate_metrics(state)
        quality = self.scorer.score(state)
        
        return CombinedResult(
            final_state=state,
            source_states=[state],
            combination_strategy=CombinationStrategy.MERGE,
            metrics=metrics,
            quality_score=quality
        )
    
    def merge_solutions(self, states: List[PackingState]) -> Optional[CombinedResult]:
        """Merge non-overlapping placements from multiple solutions."""
        merged_state = PackingState()
        merged_panels = []
        used_positions = set()
        
        # Sort states by coverage
        states_sorted = sorted(states, 
                             key=lambda s: self.coverage_calculator.calculate_coverage(s),
                             reverse=True)
        
        for state in states_sorted:
            for panel in state.placed_panels:
                # Check if position is available
                panel_key = (panel.position, panel.panel_size.id)
                
                if panel_key not in used_positions:
                    # Check for overlaps with existing panels
                    overlaps = False
                    for existing in merged_panels:
                        if self._panels_overlap(panel, existing):
                            overlaps = True
                            break
                    
                    if not overlaps:
                        merged_panels.append(panel)
                        used_positions.add(panel_key)
        
        merged_state.placed_panels = merged_panels
        merged_state.coverage_ratio = self.coverage_calculator.calculate_coverage(merged_state)
        
        metrics = self._calculate_metrics(merged_state)
        quality = self.scorer.score(merged_state)
        
        return CombinedResult(
            final_state=merged_state,
            source_states=states,
            combination_strategy=CombinationStrategy.MERGE,
            metrics=metrics,
            quality_score=quality
        )
    
    def overlay_solutions(self, states: List[PackingState]) -> Optional[CombinedResult]:
        """Overlay best parts of multiple solutions."""
        # Create grid-based representation
        grid_size = 10
        grid_scores = defaultdict(lambda: defaultdict(float))
        grid_panels = defaultdict(lambda: defaultdict(list))
        
        # Score each grid cell based on solutions
        for state_idx, state in enumerate(states):
            state_weight = self.scorer.score(state).total_score
            
            for panel in state.placed_panels:
                x, y = panel.position
                w, h = panel.panel_size.width, panel.panel_size.height
                
                # Mark grid cells
                for gx in range(int(x / grid_size), int((x + w) / grid_size) + 1):
                    for gy in range(int(y / grid_size), int((y + h) / grid_size) + 1):
                        grid_scores[gx][gy] += state_weight
                        grid_panels[gx][gy].append((state_idx, panel))
        
        # Select best panels for each region
        overlay_state = PackingState()
        overlay_panels = []
        used_panels = set()
        
        for gx in sorted(grid_scores.keys()):
            for gy in sorted(grid_scores[gx].keys()):
                if grid_panels[gx][gy]:
                    # Select panel from best-scoring state for this cell
                    best_panel = None
                    best_score = -1
                    
                    for state_idx, panel in grid_panels[gx][gy]:
                        panel_id = id(panel)
                        if panel_id not in used_panels:
                            state_score = self.scorer.score(states[state_idx]).total_score
                            if state_score > best_score:
                                best_score = state_score
                                best_panel = panel
                    
                    if best_panel and not self._conflicts_with_existing(best_panel, overlay_panels):
                        overlay_panels.append(best_panel)
                        used_panels.add(id(best_panel))
        
        overlay_state.placed_panels = overlay_panels
        overlay_state.coverage_ratio = self.coverage_calculator.calculate_coverage(overlay_state)
        
        metrics = self._calculate_metrics(overlay_state)
        quality = self.scorer.score(overlay_state)
        
        return CombinedResult(
            final_state=overlay_state,
            source_states=states,
            combination_strategy=CombinationStrategy.OVERLAY,
            metrics=metrics,
            quality_score=quality
        )
    
    def voting_combination(self, states: List[PackingState]) -> Optional[CombinedResult]:
        """Combine using voting on panel positions."""
        position_votes = defaultdict(lambda: defaultdict(int))
        panel_candidates = defaultdict(list)
        
        # Collect votes for positions
        for state in states:
            for panel in state.placed_panels:
                pos_key = (round(panel.position[0]), round(panel.position[1]))
                size_key = panel.panel_size.id
                position_votes[size_key][pos_key] += 1
                panel_candidates[size_key].append(panel)
        
        # Select panels based on votes
        voting_state = PackingState()
        voting_panels = []
        
        for size_key, positions in position_votes.items():
            # Sort positions by votes
            sorted_positions = sorted(positions.items(), key=lambda x: x[1], reverse=True)
            
            # Take positions with high votes
            threshold = max(1, len(states) // 2)  # Majority threshold
            
            for pos_key, votes in sorted_positions:
                if votes >= threshold:
                    # Find corresponding panel
                    for panel in panel_candidates[size_key]:
                        if (round(panel.position[0]), round(panel.position[1])) == pos_key:
                            if not self._conflicts_with_existing(panel, voting_panels):
                                voting_panels.append(panel)
                                break
        
        voting_state.placed_panels = voting_panels
        voting_state.coverage_ratio = self.coverage_calculator.calculate_coverage(voting_state)
        
        metrics = self._calculate_metrics(voting_state)
        quality = self.scorer.score(voting_state)
        
        return CombinedResult(
            final_state=voting_state,
            source_states=states,
            combination_strategy=CombinationStrategy.VOTING,
            metrics=metrics,
            quality_score=quality
        )
    
    def weighted_combination(self, states: List[PackingState]) -> Optional[CombinedResult]:
        """Combine with weights based on solution quality."""
        # Calculate weights
        weights = []
        for state in states:
            score = self.scorer.score(state).total_score
            weights.append(score)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(states)] * len(states)
        
        # Weighted selection of panels
        weighted_state = PackingState()
        weighted_panels = []
        
        for state_idx, (state, weight) in enumerate(zip(states, weights)):
            # Select panels proportional to weight
            num_to_select = int(len(state.placed_panels) * weight)
            
            # Sort panels by local quality
            scored_panels = []
            for panel in state.placed_panels:
                # Simple panel score based on position
                panel_score = (self.room.width - panel.position[0]) + (self.room.height - panel.position[1])
                scored_panels.append((panel_score, panel))
            
            scored_panels.sort(reverse=True, key=lambda x: x[0])
            
            # Add best panels
            for _, panel in scored_panels[:num_to_select]:
                if not self._conflicts_with_existing(panel, weighted_panels):
                    weighted_panels.append(panel)
        
        weighted_state.placed_panels = weighted_panels
        weighted_state.coverage_ratio = self.coverage_calculator.calculate_coverage(weighted_state)
        
        metrics = self._calculate_metrics(weighted_state)
        quality = self.scorer.score(weighted_state)
        
        return CombinedResult(
            final_state=weighted_state,
            source_states=states,
            combination_strategy=CombinationStrategy.WEIGHTED,
            metrics=metrics,
            quality_score=quality
        )
    
    def hybrid_combination(self, states: List[PackingState]) -> Optional[CombinedResult]:
        """Hybrid combination using multiple strategies."""
        # Try different strategies
        strategies = [
            CombinationStrategy.MERGE,
            CombinationStrategy.OVERLAY,
            CombinationStrategy.VOTING,
            CombinationStrategy.WEIGHTED
        ]
        
        best_result = None
        best_score = -1
        
        for strategy in strategies:
            result = self.combine(states, strategy)
            if result:
                if result.quality_score.total_score > best_score:
                    best_score = result.quality_score.total_score
                    best_result = result
        
        if best_result:
            best_result.combination_strategy = CombinationStrategy.HYBRID
        
        return best_result
    
    def _panels_overlap(self, p1: PanelPlacement, p2: PanelPlacement) -> bool:
        """Check if two panels overlap."""
        x1, y1 = p1.position
        w1, h1 = p1.panel_size.width, p1.panel_size.height
        x2, y2 = p2.position
        w2, h2 = p2.panel_size.width, p2.panel_size.height
        
        return not (x1 + w1 <= x2 or x2 + w2 <= x1 or
                   y1 + h1 <= y2 or y2 + h2 <= y1)
    
    def _conflicts_with_existing(self, panel: PanelPlacement, existing: List[PanelPlacement]) -> bool:
        """Check if panel conflicts with existing panels."""
        for existing_panel in existing:
            if self._panels_overlap(panel, existing_panel):
                return True
        return False
    
    def _calculate_metrics(self, state: PackingState) -> SolutionMetrics:
        """Calculate comprehensive metrics for a state."""
        coverage = self.coverage_calculator.calculate_coverage(state)
        waste = self.coverage_calculator.calculate_waste(state)
        gaps = self.coverage_calculator.identify_gaps(state)
        fragmentation = self.coverage_calculator.calculate_fragmentation(state)
        
        gaps_area = sum(w * h for _, _, w, h in gaps)
        largest_gap = max((w * h for _, _, w, h in gaps), default=0)
        
        quality = self.scorer.score(state)
        
        return SolutionMetrics(
            coverage=coverage,
            waste=waste,
            num_panels=len(state.placed_panels),
            compactness=quality.metric_scores.get(QualityMetric.COMPACTNESS, 0),
            uniformity=quality.metric_scores.get(QualityMetric.UNIFORMITY, 0),
            edge_alignment=quality.metric_scores.get(QualityMetric.EDGE_ALIGNMENT, 0),
            structural_score=quality.metric_scores.get(QualityMetric.STRUCTURAL_INTEGRITY, 0),
            aesthetic_score=quality.metric_scores.get(QualityMetric.AESTHETIC, 0),
            efficiency=quality.metric_scores.get(QualityMetric.EFFICIENCY, 0),
            gaps_area=gaps_area,
            largest_gap=largest_gap,
            fragmentation=fragmentation
        )


class ResultAggregationSystem:
    """Main system for result aggregation."""
    
    def __init__(self, room: Room, config: Optional[Dict[str, Any]] = None):
        self.room = room
        self.config = config or {}
        
        # Initialize components
        self.coverage_calculator = CoverageCalculator(room)
        self.quality_scorer = QualityScorer(
            room,
            weights=self.config.get('quality_weights')
        )
        self.best_selector = BestResultSelector(room, self.quality_scorer)
        self.combination_strategies = CombinationStrategies(room)
        
        self.aggregated_results = []
    
    def aggregate(self,
                 results: List[PackingState],
                 strategy: Optional[CombinationStrategy] = None) -> Optional[CombinedResult]:
        """Aggregate multiple results."""
        if not results:
            return None
        
        # Select strategy if not specified
        if strategy is None:
            strategy = self._select_strategy(results)
        
        # Combine results
        combined = self.combination_strategies.combine(results, strategy)
        
        if combined:
            self.aggregated_results.append(combined)
        
        return combined
    
    def _select_strategy(self, results: List[PackingState]) -> CombinationStrategy:
        """Select appropriate combination strategy."""
        # Analyze results to select strategy
        coverages = [self.coverage_calculator.calculate_coverage(r) for r in results]
        
        # High variance suggests different approaches worked
        variance = np.var(coverages) if len(coverages) > 1 else 0
        
        if variance > 0.1:
            return CombinationStrategy.HYBRID
        elif max(coverages) > 0.9:
            return CombinationStrategy.WEIGHTED
        else:
            return CombinationStrategy.MERGE
    
    def select_final(self, candidates: List[PackingState]) -> Optional[PackingState]:
        """Select final solution from candidates."""
        return self.best_selector.select_best(candidates)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregation statistics."""
        stats = {
            'total_aggregated': len(self.aggregated_results),
            'strategies_used': {}
        }
        
        # Count strategy usage
        for result in self.aggregated_results:
            strategy = result.combination_strategy.value
            stats['strategies_used'][strategy] = stats['strategies_used'].get(strategy, 0) + 1
        
        # Best aggregated result
        if self.aggregated_results:
            best = max(self.aggregated_results, key=lambda r: r.quality_score.total_score)
            stats['best_aggregated_score'] = best.quality_score.total_score
            stats['best_aggregated_coverage'] = best.metrics.coverage
        
        return stats
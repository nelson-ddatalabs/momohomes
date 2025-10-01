#!/usr/bin/env python3
"""
bb_search_tree.py - Branch & Bound Search Tree System
===================================================
Production-ready B&B search tree with branching strategies, node evaluation,
tree traversal, and intelligent expansion prioritization.
"""

import time
import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Iterator, Union
from collections import deque, defaultdict
from enum import Enum
from abc import ABC, abstractmethod
import hashlib

from models import Room, PanelSize, Point
from advanced_packing import PackingState, PanelPlacement, StateTransition
from dp_state import DPState, DPStateFactory


class BranchingStrategy(Enum):
    """Types of branching strategies for B&B tree construction."""
    PANEL_FIRST = "panel_first"        # Branch on panel selection first
    POSITION_FIRST = "position_first"  # Branch on position selection first
    HYBRID = "hybrid"                  # Adaptive strategy mixing
    GREEDY_GUIDED = "greedy_guided"    # Use heuristics to guide branching


class TraversalStrategy(Enum):
    """Tree traversal strategies for B&B search."""
    DEPTH_FIRST = "dfs"
    BREADTH_FIRST = "bfs" 
    BEST_FIRST = "best_first"
    DEPTH_LIMITED = "depth_limited"
    BEAM_SEARCH = "beam_search"


class NodePriority(Enum):
    """Priority levels for node expansion."""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4


@dataclass
class BranchBounds:
    """
    Bounds information for Branch & Bound pruning.
    Contains lower and upper bounds on the objective function.
    """
    lower_bound: float = 0.0    # Minimum achievable coverage
    upper_bound: float = 1.0    # Maximum possible coverage
    is_feasible: bool = True    # Whether solution is feasible
    bound_quality: float = 1.0  # Quality of bound estimate (0-1)
    
    def is_prunable(self, current_best: float, epsilon: float = 1e-6) -> bool:
        """Check if node can be pruned based on bounds."""
        return self.upper_bound <= current_best + epsilon or not self.is_feasible
    
    def dominates(self, other: 'BranchBounds') -> bool:
        """Check if this bound dominates another."""
        return (self.lower_bound >= other.upper_bound and 
                self.is_feasible and 
                not other.is_feasible)


@dataclass
class BranchNode:
    """
    Node in the Branch & Bound search tree.
    Represents a partial solution with bounds and branching information.
    """
    node_id: str
    state: PackingState
    remaining_panels: Set[PanelSize]
    bounds: BranchBounds
    depth: int = 0
    parent_id: Optional[str] = None
    children_ids: Set[str] = field(default_factory=set)
    
    # Search metadata
    creation_time: float = field(default_factory=time.time)
    expansion_time: Optional[float] = None
    is_expanded: bool = False
    is_pruned: bool = False
    pruning_reason: Optional[str] = None
    
    # Branching decision that led to this node
    branch_decision: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize derived properties."""
        if not self.node_id:
            # Generate unique node ID
            state_hash = self.state.canonical_hash[:8]
            remaining_hash = str(hash(frozenset(self.remaining_panels)))[:4]
            self.node_id = f"node_{state_hash}_{remaining_hash}_{self.depth}"
    
    @property
    def coverage(self) -> float:
        """Current coverage of the partial solution."""
        return self.state.coverage
    
    @property
    def is_terminal(self) -> bool:
        """Check if this is a terminal node (complete solution)."""
        return (not self.remaining_panels or 
                self.coverage >= 0.999 or  # Near-perfect coverage
                not self.bounds.is_feasible)
    
    @property
    def priority_score(self) -> float:
        """Calculate priority score for expansion ordering."""
        # Higher score = higher priority
        base_score = self.bounds.upper_bound
        
        # Bonus for promising bounds
        if self.bounds.bound_quality > 0.8:
            base_score += 0.1 * self.bounds.bound_quality
        
        # Penalty for depth (favor breadth)
        depth_penalty = 0.01 * self.depth
        
        return base_score - depth_penalty
    
    def can_be_expanded(self) -> bool:
        """Check if node can be expanded."""
        return (not self.is_terminal and 
                not self.is_expanded and 
                not self.is_pruned and
                self.bounds.is_feasible)
    
    def mark_pruned(self, reason: str):
        """Mark node as pruned with reason."""
        self.is_pruned = True
        self.pruning_reason = reason
    
    def mark_expanded(self):
        """Mark node as expanded."""
        self.is_expanded = True
        self.expansion_time = time.time()


class BranchingRuleEngine:
    """
    Implements different branching strategies for B&B tree construction.
    Determines how to create child nodes from a parent node.
    """
    
    def __init__(self, room: Room, strategy: BranchingStrategy = BranchingStrategy.HYBRID):
        self.room = room
        self.strategy = strategy
        self.state_transition = StateTransition()
        self.branch_count = 0
        
    def generate_branches(self, node: BranchNode, max_branches: int = 10) -> List[BranchNode]:
        """
        Generate child nodes from parent node using selected branching strategy.
        Returns list of child nodes with branching decisions applied.
        """
        if not node.can_be_expanded():
            return []
        
        branches = []
        
        if self.strategy == BranchingStrategy.PANEL_FIRST:
            branches = self._branch_by_panel_selection(node, max_branches)
        elif self.strategy == BranchingStrategy.POSITION_FIRST:
            branches = self._branch_by_position_selection(node, max_branches)
        elif self.strategy == BranchingStrategy.GREEDY_GUIDED:
            branches = self._branch_greedy_guided(node, max_branches)
        else:  # HYBRID
            branches = self._branch_hybrid_strategy(node, max_branches)
        
        self.branch_count += len(branches)
        return branches
    
    def _branch_by_panel_selection(self, node: BranchNode, max_branches: int) -> List[BranchNode]:
        """Branch by selecting which panel to place next."""
        branches = []
        
        # Try each remaining panel type
        panel_options = sorted(node.remaining_panels, 
                             key=lambda p: p.area, reverse=True)  # Larger panels first
        
        for i, panel in enumerate(panel_options[:max_branches]):
            # Find best position for this panel
            valid_placements = self.state_transition.get_valid_placements(
                node.state, [panel], resolution=0.5
            )
            
            if valid_placements:
                # Use best placement for this panel
                best_placement = max(valid_placements, 
                                   key=lambda p: self._evaluate_placement_quality(p, node.state))
                
                # Create child node with this placement
                new_state = self.state_transition.apply_placement(node.state, best_placement)
                if new_state:
                    child = self._create_child_node(
                        parent=node,
                        new_state=new_state,
                        placed_panel=panel,
                        decision={'type': 'panel_selection', 'panel': panel.name, 'placement': best_placement}
                    )
                    branches.append(child)
        
        return branches
    
    def _branch_by_position_selection(self, node: BranchNode, max_branches: int) -> List[BranchNode]:
        """Branch by selecting position for the first remaining panel."""
        branches = []
        
        if not node.remaining_panels:
            return branches
        
        # Take the first panel (could be largest, or use other heuristic)
        panel = max(node.remaining_panels, key=lambda p: p.area)
        
        # Get valid positions for this panel
        valid_placements = self.state_transition.get_valid_placements(
            node.state, [panel], resolution=0.5
        )
        
        # Sort placements by quality
        sorted_placements = sorted(valid_placements, 
                                 key=lambda p: self._evaluate_placement_quality(p, node.state),
                                 reverse=True)
        
        # Create branches for best positions
        for placement in sorted_placements[:max_branches]:
            new_state = self.state_transition.apply_placement(node.state, placement)
            if new_state:
                child = self._create_child_node(
                    parent=node,
                    new_state=new_state,
                    placed_panel=panel,
                    decision={'type': 'position_selection', 'panel': panel.name, 'placement': placement}
                )
                branches.append(child)
        
        return branches
    
    def _branch_greedy_guided(self, node: BranchNode, max_branches: int) -> List[BranchNode]:
        """Branch using greedy heuristics to guide selection."""
        branches = []
        
        # Get all valid placements for all remaining panels
        all_placements = []
        for panel in node.remaining_panels:
            placements = self.state_transition.get_valid_placements(
                node.state, [panel], resolution=0.5
            )
            for placement in placements:
                quality = self._evaluate_placement_quality(placement, node.state)
                all_placements.append((placement, panel, quality))
        
        # Sort by quality and take best options
        all_placements.sort(key=lambda x: x[2], reverse=True)
        
        # Create branches for best placements
        for placement, panel, quality in all_placements[:max_branches]:
            new_state = self.state_transition.apply_placement(node.state, placement)
            if new_state:
                child = self._create_child_node(
                    parent=node,
                    new_state=new_state,
                    placed_panel=panel,
                    decision={'type': 'greedy_guided', 'panel': panel.name, 'placement': placement, 'quality': quality}
                )
                branches.append(child)
        
        return branches
    
    def _branch_hybrid_strategy(self, node: BranchNode, max_branches: int) -> List[BranchNode]:
        """Adaptive branching that combines multiple strategies."""
        # Choose strategy based on node characteristics
        if node.depth < 3:
            # Early in search - use panel selection for diversity
            return self._branch_by_panel_selection(node, max_branches)
        elif len(node.remaining_panels) <= 3:
            # Few panels left - use position selection for precision
            return self._branch_by_position_selection(node, max_branches)
        else:
            # Middle of search - use greedy guidance
            return self._branch_greedy_guided(node, max_branches // 2)
    
    def _evaluate_placement_quality(self, placement: PanelPlacement, state: PackingState) -> float:
        """Evaluate quality of a placement for branching decisions."""
        quality = 0.0
        
        # Factor 1: Coverage improvement
        area_improvement = placement.panel_size.area
        room_area = self.room.width * self.room.height
        coverage_improvement = area_improvement / room_area
        quality += coverage_improvement * 2.0
        
        # Factor 2: Position preference (bottom-left is better)
        x, y = placement.position
        position_score = 1.0 - (x / self.room.width + y / self.room.height) / 2.0
        quality += position_score * 0.3
        
        # Factor 3: Alignment with existing panels
        alignment_bonus = self._calculate_alignment_bonus(placement, state)
        quality += alignment_bonus * 0.2
        
        return quality
    
    def _calculate_alignment_bonus(self, placement: PanelPlacement, state: PackingState) -> float:
        """Calculate bonus for good alignment with existing panels."""
        if not state.placements:
            return 0.0
        
        x, y = placement.position
        pw, ph = placement.panel_size.get_dimensions(placement.orientation)
        
        alignment_bonus = 0.0
        
        for existing in state.placements:
            ex, ey, ex2, ey2 = existing.bounds
            
            # Edge alignment bonuses
            if abs(x - ex2) < 0.1:  # Left edge aligns with existing right edge
                alignment_bonus += 0.5
            if abs(y - ey2) < 0.1:  # Bottom edge aligns with existing top edge
                alignment_bonus += 0.5
            if abs(x + pw - ex) < 0.1:  # Right edge aligns with existing left edge
                alignment_bonus += 0.5
            if abs(y + ph - ey) < 0.1:  # Top edge aligns with existing bottom edge
                alignment_bonus += 0.5
        
        return min(alignment_bonus, 1.0)  # Cap at 1.0
    
    def _create_child_node(self, parent: BranchNode, new_state: PackingState, 
                          placed_panel: PanelSize, decision: Dict[str, Any]) -> BranchNode:
        """Create child node from parent with new state."""
        new_remaining = parent.remaining_panels.copy()
        new_remaining.discard(placed_panel)
        
        child = BranchNode(
            node_id="",  # Will be generated in __post_init__
            state=new_state,
            remaining_panels=new_remaining,
            bounds=BranchBounds(),  # Will be updated by evaluator
            depth=parent.depth + 1,
            parent_id=parent.node_id,
            branch_decision=decision
        )
        
        # Update parent-child relationship
        parent.children_ids.add(child.node_id)
        
        return child


class NodeEvaluator:
    """
    Evaluates B&B nodes to compute bounds and assess solution quality.
    Provides both optimistic upper bounds and pessimistic lower bounds.
    """
    
    def __init__(self, room: Room, panel_sizes: List[PanelSize]):
        self.room = room
        self.panel_sizes = panel_sizes
        self.room_area = room.width * room.height
        self.evaluation_count = 0
        
    def evaluate_node(self, node: BranchNode) -> BranchBounds:
        """
        Comprehensive node evaluation with tight bounds.
        Returns updated bounds for the node.
        """
        self.evaluation_count += 1
        
        # Current lower bound (actual coverage achieved)
        lower_bound = node.state.coverage
        
        # Upper bound estimation
        upper_bound = self._compute_upper_bound(node)
        
        # Feasibility check
        is_feasible = self._check_feasibility(node)
        
        # Bound quality assessment
        bound_quality = self._assess_bound_quality(node, upper_bound)
        
        bounds = BranchBounds(
            lower_bound=lower_bound,
            upper_bound=min(upper_bound, 1.0),
            is_feasible=is_feasible,
            bound_quality=bound_quality
        )
        
        node.bounds = bounds
        return bounds
    
    def _compute_upper_bound(self, node: BranchNode) -> float:
        """Compute optimistic upper bound on achievable coverage."""
        current_coverage = node.state.coverage
        
        if not node.remaining_panels:
            return current_coverage
        
        # Method 1: Simple area-based upper bound
        remaining_area = sum(panel.area for panel in node.remaining_panels)
        area_bound = current_coverage + (remaining_area / self.room_area)
        
        # Method 2: Space-constrained upper bound
        space_bound = self._compute_space_constrained_bound(node)
        
        # Method 3: Greedy construction upper bound
        greedy_bound = self._compute_greedy_upper_bound(node)
        
        # Take the tightest (lowest) bound
        upper_bound = min(area_bound, space_bound, greedy_bound)
        
        return upper_bound
    
    def _compute_space_constrained_bound(self, node: BranchNode) -> float:
        """Compute bound considering available space constraints."""
        # Get approximate available space
        occupied_area = node.state.coverage * self.room_area
        available_area = self.room_area - occupied_area
        
        # Sort panels by efficiency (area/footprint)
        panels_by_efficiency = sorted(node.remaining_panels, 
                                    key=lambda p: p.area, reverse=True)
        
        # Greedily pack panels into available space
        used_area = 0.0
        for panel in panels_by_efficiency:
            if used_area + panel.area <= available_area:
                used_area += panel.area
            else:
                # Try to fit partial panel (optimistic)
                remaining_space = available_area - used_area
                if remaining_space > 0:
                    used_area += remaining_space
                break
        
        return node.state.coverage + (used_area / self.room_area)
    
    def _compute_greedy_upper_bound(self, node: BranchNode) -> float:
        """Compute bound using fast greedy placement simulation."""
        # Quick greedy simulation (simplified)
        simulated_coverage = node.state.coverage
        simulated_area = simulated_coverage * self.room_area
        
        # Sort panels by area (largest first)
        remaining_sorted = sorted(node.remaining_panels, 
                                key=lambda p: p.area, reverse=True)
        
        # Assume we can place panels with 85% efficiency
        packing_efficiency = 0.85
        
        for panel in remaining_sorted[:5]:  # Consider only first 5 panels for speed
            if simulated_area + panel.area * packing_efficiency <= self.room_area:
                simulated_area += panel.area * packing_efficiency
            else:
                # Partial placement
                remaining_capacity = self.room_area - simulated_area
                if remaining_capacity > 0:
                    simulated_area += min(panel.area * packing_efficiency, remaining_capacity)
                break
        
        return simulated_area / self.room_area
    
    def _check_feasibility(self, node: BranchNode) -> bool:
        """Check if node represents a feasible partial solution."""
        # Basic feasibility checks
        
        # Check 1: Coverage doesn't exceed 100%
        if node.state.coverage > 1.0:
            return False
        
        # Check 2: All placements are within room bounds
        for placement in node.state.placements:
            x1, y1, x2, y2 = placement.bounds
            if (x1 < 0 or y1 < 0 or 
                x2 > self.room.width or y2 > self.room.height):
                return False
        
        # Check 3: No overlapping panels (should be guaranteed by StateTransition)
        # This is already enforced by the PackingState validation
        
        # Check 4: Remaining panels can theoretically fit
        remaining_area = sum(panel.area for panel in node.remaining_panels)
        occupied_area = node.state.coverage * self.room_area
        if occupied_area + remaining_area > self.room_area * 1.1:  # Allow 10% overhead
            return False
        
        return True
    
    def _assess_bound_quality(self, node: BranchNode, upper_bound: float) -> float:
        """Assess quality of the computed upper bound (0-1 scale)."""
        # Factors that affect bound quality
        quality = 1.0
        
        # Factor 1: Depth - deeper nodes have more reliable bounds
        depth_factor = min(1.0, node.depth / 10.0)
        quality *= (0.7 + 0.3 * depth_factor)
        
        # Factor 2: Coverage - higher coverage nodes have tighter bounds
        coverage_factor = node.state.coverage
        quality *= (0.8 + 0.2 * coverage_factor)
        
        # Factor 3: Remaining panel count - fewer panels = more reliable
        remaining_factor = 1.0 - (len(node.remaining_panels) / max(1, len(self.panel_sizes)))
        quality *= (0.6 + 0.4 * remaining_factor)
        
        return min(1.0, max(0.0, quality))
    
    def get_evaluation_statistics(self) -> Dict[str, Any]:
        """Get node evaluation statistics."""
        return {
            'evaluations_performed': self.evaluation_count,
            'room_area': self.room_area,
            'panel_types': len(self.panel_sizes)
        }


class TreeTraversalManager:
    """
    Manages tree traversal strategies for B&B search.
    Handles different search strategies and maintains the search frontier.
    """
    
    def __init__(self, strategy: TraversalStrategy = TraversalStrategy.BEST_FIRST,
                 max_nodes: int = 10000):
        self.strategy = strategy
        self.max_nodes = max_nodes
        self.nodes_visited = 0
        self.nodes_expanded = 0
        
        # Different data structures for different strategies
        if strategy == TraversalStrategy.DEPTH_FIRST:
            self.frontier = deque()  # Stack for DFS
        elif strategy == TraversalStrategy.BREADTH_FIRST:
            self.frontier = deque()  # Queue for BFS
        elif strategy in [TraversalStrategy.BEST_FIRST, TraversalStrategy.BEAM_SEARCH]:
            self.frontier = []  # Heap for priority-based search
        else:
            self.frontier = deque()  # Default to queue
    
    def add_nodes(self, nodes: List[BranchNode]):
        """Add nodes to the search frontier."""
        for node in nodes:
            if self.strategy == TraversalStrategy.DEPTH_FIRST:
                self.frontier.append(node)  # Add to end (stack behavior)
            elif self.strategy == TraversalStrategy.BREADTH_FIRST:
                self.frontier.appendleft(node)  # Add to beginning (queue behavior)
            elif self.strategy in [TraversalStrategy.BEST_FIRST, TraversalStrategy.BEAM_SEARCH]:
                # Add with priority (negative score for min-heap to get max priority)
                heapq.heappush(self.frontier, (-node.priority_score, time.time(), node))
            else:
                self.frontier.append(node)
        
        # Maintain size limits for beam search
        if (self.strategy == TraversalStrategy.BEAM_SEARCH and 
            len(self.frontier) > self.max_nodes):
            self._trim_frontier()
    
    def get_next_node(self) -> Optional[BranchNode]:
        """Get next node to expand according to traversal strategy."""
        if not self.frontier:
            return None
        
        self.nodes_visited += 1
        
        if self.strategy in [TraversalStrategy.BEST_FIRST, TraversalStrategy.BEAM_SEARCH]:
            if self.frontier:
                _, _, node = heapq.heappop(self.frontier)
                return node
        else:
            if self.frontier:
                return self.frontier.pop()
        
        return None
    
    def _trim_frontier(self):
        """Trim frontier to maintain size limits (for beam search)."""
        if not self.frontier:
            return
        
        # Keep only the best nodes
        keep_count = self.max_nodes // 2
        
        # Extract all nodes, sort by priority, keep best
        all_nodes = []
        while self.frontier:
            _, _, node = heapq.heappop(self.frontier)
            all_nodes.append(node)
        
        # Sort by priority score (descending)
        all_nodes.sort(key=lambda n: n.priority_score, reverse=True)
        
        # Re-add best nodes
        for node in all_nodes[:keep_count]:
            heapq.heappush(self.frontier, (-node.priority_score, time.time(), node))
    
    def frontier_size(self) -> int:
        """Get current size of search frontier."""
        return len(self.frontier)
    
    def is_empty(self) -> bool:
        """Check if frontier is empty."""
        return len(self.frontier) == 0
    
    def get_traversal_statistics(self) -> Dict[str, Any]:
        """Get traversal statistics."""
        return {
            'strategy': self.strategy.value,
            'nodes_visited': self.nodes_visited,
            'nodes_expanded': self.nodes_expanded,
            'frontier_size': self.frontier_size(),
            'max_nodes': self.max_nodes
        }


class ExpansionPrioritizer:
    """
    Intelligent prioritization system for node expansion in B&B search.
    Determines which nodes deserve computational resources.
    """
    
    def __init__(self, target_coverage: float = 0.95):
        self.target_coverage = target_coverage
        self.expansion_count = 0
        self.pruning_count = 0
        
        # Priority calculation weights
        self.weights = {
            'upper_bound': 0.4,
            'lower_bound': 0.2, 
            'bound_quality': 0.15,
            'depth_penalty': 0.1,
            'coverage_progress': 0.15
        }
        
    def prioritize_nodes(self, nodes: List[BranchNode], 
                        current_best: float) -> List[BranchNode]:
        """
        Prioritize nodes for expansion based on multiple criteria.
        Returns nodes sorted by expansion priority (highest first).
        """
        if not nodes:
            return nodes
        
        # Calculate priority scores
        scored_nodes = []
        for node in nodes:
            if not node.is_pruned and node.can_be_expanded():
                priority = self._calculate_priority_score(node, current_best)
                scored_nodes.append((priority, node))
        
        # Sort by priority (highest first)
        scored_nodes.sort(key=lambda x: x[0], reverse=True)
        
        return [node for _, node in scored_nodes]
    
    def should_expand_node(self, node: BranchNode, current_best: float,
                          available_time: float, resource_budget: float) -> bool:
        """
        Decide whether to expand a specific node based on various criteria.
        """
        # Basic expansion criteria
        if not node.can_be_expanded():
            return False
        
        # Pruning check
        if node.bounds.is_prunable(current_best):
            node.mark_pruned("bounds_pruning")
            self.pruning_count += 1
            return False
        
        # Resource constraints
        if available_time < 0.1 or resource_budget < 0.01:
            # Under resource pressure - only expand very promising nodes
            if node.priority_score < current_best + 0.1:
                return False
        
        # Target achievement check
        if current_best >= self.target_coverage:
            # Target reached - only expand nodes that might beat current best significantly
            if node.bounds.upper_bound < current_best + 0.01:
                return False
        
        self.expansion_count += 1
        return True
    
    def _calculate_priority_score(self, node: BranchNode, current_best: float) -> float:
        """Calculate comprehensive priority score for node expansion."""
        score = 0.0
        
        # Upper bound contribution
        upper_bound_score = node.bounds.upper_bound
        score += self.weights['upper_bound'] * upper_bound_score
        
        # Lower bound contribution
        lower_bound_score = node.bounds.lower_bound
        score += self.weights['lower_bound'] * lower_bound_score
        
        # Bound quality contribution
        quality_score = node.bounds.bound_quality
        score += self.weights['bound_quality'] * quality_score
        
        # Depth penalty (favor breadth-first exploration initially)
        depth_penalty = min(0.5, node.depth * 0.05)
        score -= self.weights['depth_penalty'] * depth_penalty
        
        # Coverage progress bonus
        if current_best > 0:
            progress_ratio = node.bounds.lower_bound / current_best
            progress_bonus = min(1.0, progress_ratio)
            score += self.weights['coverage_progress'] * progress_bonus
        
        # Promising node bonus
        if node.bounds.upper_bound > current_best + 0.05:
            score += 0.1  # Bonus for potentially improving nodes
        
        return score
    
    def adjust_weights(self, performance_metrics: Dict[str, float]):
        """Adaptively adjust prioritization weights based on search performance."""
        # This could implement learning-based weight adjustment
        # For now, use simple heuristics
        
        coverage_rate = performance_metrics.get('coverage_improvement_rate', 0.0)
        
        if coverage_rate > 0.1:
            # Good progress - emphasize exploration
            self.weights['upper_bound'] += 0.05
            self.weights['depth_penalty'] += 0.02
        else:
            # Slow progress - emphasize exploitation
            self.weights['lower_bound'] += 0.05
            self.weights['bound_quality'] += 0.02
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        for key in self.weights:
            self.weights[key] /= total_weight
    
    def get_prioritization_statistics(self) -> Dict[str, Any]:
        """Get expansion prioritization statistics."""
        return {
            'expansions_performed': self.expansion_count,
            'nodes_pruned': self.pruning_count,
            'pruning_rate': self.pruning_count / max(1, self.expansion_count + self.pruning_count),
            'current_weights': dict(self.weights)
        }


def create_bb_search_system(room: Room, panel_sizes: List[PanelSize],
                           branching_strategy: BranchingStrategy = BranchingStrategy.HYBRID,
                           traversal_strategy: TraversalStrategy = TraversalStrategy.BEST_FIRST,
                           target_coverage: float = 0.95) -> Tuple[BranchingRuleEngine, NodeEvaluator, TreeTraversalManager, ExpansionPrioritizer]:
    """
    Factory function to create complete B&B search tree system.
    Returns all components needed for Branch & Bound optimization.
    """
    branching_engine = BranchingRuleEngine(room, branching_strategy)
    node_evaluator = NodeEvaluator(room, panel_sizes)
    traversal_manager = TreeTraversalManager(traversal_strategy)
    expansion_prioritizer = ExpansionPrioritizer(target_coverage)
    
    return branching_engine, node_evaluator, traversal_manager, expansion_prioritizer
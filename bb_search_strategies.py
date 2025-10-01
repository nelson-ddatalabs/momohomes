#!/usr/bin/env python3
"""
bb_search_strategies.py - Advanced Search Strategies for Branch & Bound
=======================================================================
Production-ready best-first search, depth-first with iterative deepening,
hybrid mode, and adaptive strategy selection for B&B optimization.
"""

import time
import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Callable, Union
from collections import deque, defaultdict
from enum import Enum
from abc import ABC, abstractmethod
import math

from models import Room, PanelSize
from advanced_packing import PackingState
from bb_search_tree import (
    BranchNode, BranchBounds, BranchingRuleEngine, 
    NodeEvaluator, ExpansionPrioritizer, TraversalStrategy
)
from bb_bounding_functions import AdvancedBoundingSystem


class SearchStrategy(Enum):
    """Available search strategies for B&B."""
    BEST_FIRST = "best_first"
    DEPTH_FIRST = "depth_first"
    DEPTH_LIMITED = "depth_limited"
    ITERATIVE_DEEPENING = "iterative_deepening"
    BEAM_SEARCH = "beam_search"
    HYBRID_ADAPTIVE = "hybrid_adaptive"
    A_STAR = "a_star"
    UNIFORM_COST = "uniform_cost"


class SearchMode(Enum):
    """Search mode configurations."""
    EXPLORATION = "exploration"  # Prioritize finding diverse solutions
    EXPLOITATION = "exploitation"  # Focus on improving best solution
    BALANCED = "balanced"  # Balance exploration and exploitation
    ANYTIME = "anytime"  # Return improving solutions over time


@dataclass
class SearchConfiguration:
    """Configuration for search strategies."""
    strategy: SearchStrategy = SearchStrategy.BEST_FIRST
    mode: SearchMode = SearchMode.BALANCED
    
    # Strategy-specific parameters
    depth_limit: int = 100
    beam_width: int = 50
    iteration_increment: int = 5
    
    # Resource limits
    max_nodes: int = 10000
    time_limit: float = 5.0
    memory_limit: int = 1000000  # nodes in memory
    
    # Heuristic weights
    coverage_weight: float = 0.5
    bound_weight: float = 0.3
    depth_weight: float = 0.2
    
    # Adaptive parameters
    adaptation_interval: int = 100  # nodes between strategy adaptations
    performance_threshold: float = 0.01  # minimum improvement to continue strategy


@dataclass 
class SearchStatistics:
    """Statistics collected during search."""
    nodes_explored: int = 0
    nodes_pruned: int = 0
    nodes_generated: int = 0
    best_coverage: float = 0.0
    solutions_found: int = 0
    pruning_efficiency: float = 0.0
    search_depth_reached: int = 0
    time_elapsed: float = 0.0
    memory_peak: int = 0
    strategy_switches: int = 0
    
    def update_pruning_efficiency(self):
        """Update pruning efficiency metric."""
        total = self.nodes_generated + self.nodes_pruned
        if total > 0:
            self.pruning_efficiency = self.nodes_pruned / total


class SearchStrategyBase(ABC):
    """Abstract base class for search strategy implementations."""
    
    @abstractmethod
    def select_next_node(self, frontier: List[BranchNode], 
                        statistics: SearchStatistics) -> Optional[BranchNode]:
        """Select next node to expand from frontier."""
        pass
    
    @abstractmethod
    def add_to_frontier(self, nodes: List[BranchNode], frontier: List[BranchNode]):
        """Add new nodes to the frontier."""
        pass
    
    @abstractmethod
    def should_continue(self, statistics: SearchStatistics, 
                       config: SearchConfiguration) -> bool:
        """Determine if search should continue."""
        pass


class BestFirstSearch:
    """
    Best-first search strategy using node evaluation scores.
    Expands most promising nodes first based on heuristic evaluation.
    """
    
    def __init__(self, config: SearchConfiguration):
        self.config = config
        self.frontier = []  # Min-heap priority queue
        self.explored = set()
        self.node_counter = 0  # For tie-breaking
        
    def add_nodes(self, nodes: List[BranchNode]):
        """Add nodes to frontier with priority."""
        for node in nodes:
            if node.node_id not in self.explored:
                # Calculate priority (lower is better for min-heap)
                priority = self._calculate_priority(node)
                heapq.heappush(self.frontier, (priority, self.node_counter, node))
                self.node_counter += 1
    
    def get_next_node(self) -> Optional[BranchNode]:
        """Get highest priority node from frontier."""
        while self.frontier:
            _, _, node = heapq.heappop(self.frontier)
            if node.node_id not in self.explored:
                self.explored.add(node.node_id)
                return node
        return None
    
    def _calculate_priority(self, node: BranchNode) -> float:
        """Calculate node priority for best-first search."""
        # Combine multiple factors
        coverage_score = node.state.coverage * self.config.coverage_weight
        bound_score = node.bounds.upper_bound * self.config.bound_weight
        depth_penalty = node.depth * self.config.depth_weight / 100.0
        
        # Lower priority value = higher priority (min-heap)
        priority = -(coverage_score + bound_score - depth_penalty)
        return priority
    
    def is_frontier_empty(self) -> bool:
        """Check if frontier is empty."""
        return len(self.frontier) == 0
    
    def frontier_size(self) -> int:
        """Get current frontier size."""
        return len(self.frontier)


class DepthFirstIterativeDeepening:
    """
    Depth-first search with iterative deepening.
    Combines completeness of BFS with space efficiency of DFS.
    """
    
    def __init__(self, config: SearchConfiguration):
        self.config = config
        self.current_depth_limit = config.iteration_increment
        self.iteration = 0
        self.best_at_depth = {}  # Track best solution at each depth
        
    def search_iteration(self, root: BranchNode, depth_limit: int,
                        branching_engine: BranchingRuleEngine,
                        evaluator: NodeEvaluator) -> Tuple[Optional[BranchNode], int]:
        """
        Perform one iteration of depth-limited search.
        Returns best node found and nodes explored.
        """
        best_node = root  # Start with root as best
        best_coverage = root.state.coverage
        nodes_explored = 0
        
        # DFS with depth limit
        stack = [(root, 0)]
        visited = set()
        
        while stack:
            node, depth = stack.pop()
            
            if node.node_id in visited:
                continue
                
            visited.add(node.node_id)
            nodes_explored += 1
            
            # Update best if this is better
            if node.state.coverage >= best_coverage:
                best_coverage = node.state.coverage
                best_node = node
            
            # Check depth limit
            if depth >= depth_limit:
                continue
            
            # Generate children
            children = branching_engine.generate_branches(node)
            
            # Evaluate children
            for child in children:
                evaluator.evaluate_node(child)
                
                # Add to stack if promising
                if not child.is_pruned and child.bounds.is_feasible:
                    stack.append((child, depth + 1))
        
        return best_node, nodes_explored
    
    def run_iterative_deepening(self, root: BranchNode,
                               branching_engine: BranchingRuleEngine,
                               evaluator: NodeEvaluator,
                               max_iterations: int = 20) -> BranchNode:
        """
        Run iterative deepening DFS.
        Gradually increases depth limit until solution found or limit reached.
        """
        best_overall = root
        total_nodes = 0
        
        for iteration in range(max_iterations):
            depth_limit = self.current_depth_limit + iteration * self.config.iteration_increment
            
            # Run depth-limited search
            best_node, nodes_explored = self.search_iteration(
                root, depth_limit, branching_engine, evaluator)
            
            total_nodes += nodes_explored
            
            # Update best
            if best_node and best_node.state.coverage > best_overall.state.coverage:
                best_overall = best_node
                self.best_at_depth[depth_limit] = best_node
            
            # Check termination conditions
            if best_overall.state.coverage >= 0.95:  # Target reached
                break
            
            if nodes_explored == 0:  # No new nodes at this depth
                break
            
            # Check resource limits
            if total_nodes > self.config.max_nodes:
                break
        
        self.iteration = iteration
        return best_overall


class BeamSearch:
    """
    Beam search with fixed beam width.
    Maintains only top-k nodes at each level for memory efficiency.
    """
    
    def __init__(self, config: SearchConfiguration):
        self.config = config
        self.beam_width = config.beam_width
        self.current_level = []
        self.next_level = []
        self.level = 0
        
    def initialize(self, root: BranchNode):
        """Initialize beam with root node."""
        self.current_level = [root]
        self.level = 0
    
    def expand_level(self, branching_engine: BranchingRuleEngine,
                    evaluator: NodeEvaluator) -> List[BranchNode]:
        """
        Expand current level and select top nodes for next level.
        Returns newly generated nodes.
        """
        all_children = []
        
        # Generate children for all nodes in current beam
        for node in self.current_level:
            if node.can_be_expanded():
                children = branching_engine.generate_branches(node)
                
                # Evaluate children
                for child in children:
                    evaluator.evaluate_node(child)
                    if not child.is_pruned:
                        all_children.append(child)
        
        # Select top beam_width nodes for next level
        if all_children:
            # Sort by evaluation score
            all_children.sort(key=lambda n: self._score_node(n), reverse=True)
            self.next_level = all_children[:self.beam_width]
        else:
            self.next_level = []
        
        return all_children
    
    def advance_level(self):
        """Move to next level."""
        self.current_level = self.next_level
        self.next_level = []
        self.level += 1
    
    def _score_node(self, node: BranchNode) -> float:
        """Score node for beam selection."""
        return (node.state.coverage * 0.4 + 
                node.bounds.upper_bound * 0.4 +
                (1.0 - node.depth / 100.0) * 0.2)
    
    def is_complete(self) -> bool:
        """Check if beam search is complete."""
        return len(self.current_level) == 0
    
    def get_best_node(self) -> Optional[BranchNode]:
        """Get best node from current beam."""
        if not self.current_level:
            return None
        return max(self.current_level, key=lambda n: n.state.coverage)


class HybridAdaptiveSearch:
    """
    Hybrid search that adapts strategy based on search progress.
    Switches between exploration and exploitation dynamically.
    """
    
    def __init__(self, config: SearchConfiguration):
        self.config = config
        self.current_strategy = SearchStrategy.BEST_FIRST
        self.strategies = {
            SearchStrategy.BEST_FIRST: BestFirstSearch(config),
            SearchStrategy.ITERATIVE_DEEPENING: DepthFirstIterativeDeepening(config),
            SearchStrategy.BEAM_SEARCH: BeamSearch(config)
        }
        self.performance_history = []
        self.last_switch = 0
        self.nodes_since_switch = 0
        
    def select_strategy(self, statistics: SearchStatistics) -> SearchStrategy:
        """
        Select appropriate strategy based on current search state.
        Uses performance metrics to adapt.
        """
        self.nodes_since_switch += 1
        
        # Check if time to evaluate strategy
        if self.nodes_since_switch < self.config.adaptation_interval:
            return self.current_strategy
        
        # Evaluate current performance
        current_performance = self._evaluate_performance(statistics)
        self.performance_history.append(current_performance)
        
        # Determine if strategy switch needed
        if self._should_switch_strategy(current_performance):
            new_strategy = self._select_new_strategy(statistics)
            if new_strategy != self.current_strategy:
                self.current_strategy = new_strategy
                statistics.strategy_switches += 1
                self.nodes_since_switch = 0
                self.last_switch = statistics.nodes_explored
        
        return self.current_strategy
    
    def _evaluate_performance(self, statistics: SearchStatistics) -> float:
        """Evaluate current search performance."""
        # Combine multiple metrics
        coverage_progress = statistics.best_coverage
        pruning_efficiency = statistics.pruning_efficiency
        exploration_rate = statistics.nodes_explored / max(1, statistics.nodes_generated)
        
        performance = (coverage_progress * 0.5 +
                      pruning_efficiency * 0.3 +
                      exploration_rate * 0.2)
        return performance
    
    def _should_switch_strategy(self, current_performance: float) -> bool:
        """Determine if strategy switch is warranted."""
        if len(self.performance_history) < 2:
            return False
        
        # Check if performance is stagnating
        recent_improvement = current_performance - self.performance_history[-2]
        if recent_improvement < self.config.performance_threshold:
            return True
        
        # Check if stuck in local optimum
        if len(self.performance_history) >= 5:
            variance = self._calculate_variance(self.performance_history[-5:])
            if variance < 0.001:  # Very low variance suggests stuck
                return True
        
        return False
    
    def _select_new_strategy(self, statistics: SearchStatistics) -> SearchStrategy:
        """Select new strategy based on search characteristics."""
        # Early in search - use best-first for quick progress
        if statistics.nodes_explored < 1000:
            return SearchStrategy.BEST_FIRST
        
        # If making good progress - continue with beam search
        if statistics.best_coverage > 0.7:
            return SearchStrategy.BEAM_SEARCH
        
        # If stuck - try iterative deepening for systematic exploration
        if statistics.nodes_explored > 5000 and statistics.best_coverage < 0.5:
            return SearchStrategy.ITERATIVE_DEEPENING
        
        # Default to best-first
        return SearchStrategy.BEST_FIRST
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of values."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / len(values)


class AStarSearch:
    """
    A* search with admissible heuristic for optimal solutions.
    Guarantees optimal solution if heuristic is admissible.
    """
    
    def __init__(self, config: SearchConfiguration):
        self.config = config
        self.open_set = []  # Priority queue
        self.closed_set = set()
        self.g_score = {}  # Cost from start
        self.f_score = {}  # Estimated total cost
        self.node_counter = 0
        
    def initialize(self, root: BranchNode):
        """Initialize A* search with root."""
        self.g_score[root.node_id] = 0
        self.f_score[root.node_id] = self._heuristic(root)
        heapq.heappush(self.open_set, (self.f_score[root.node_id], self.node_counter, root))
        self.node_counter += 1
    
    def search_step(self, branching_engine: BranchingRuleEngine,
                   evaluator: NodeEvaluator) -> Optional[BranchNode]:
        """
        Perform one step of A* search.
        Returns current node being expanded.
        """
        if not self.open_set:
            return None
        
        # Get node with lowest f-score
        _, _, current = heapq.heappop(self.open_set)
        
        if current.node_id in self.closed_set:
            return self.search_step(branching_engine, evaluator)  # Skip already processed
        
        self.closed_set.add(current.node_id)
        
        # Check if goal reached
        if current.state.coverage >= 0.95:
            return current
        
        # Generate and evaluate children
        children = branching_engine.generate_branches(current)
        
        for child in children:
            if child.node_id in self.closed_set:
                continue
            
            evaluator.evaluate_node(child)
            
            if child.is_pruned:
                continue
            
            # Calculate tentative g-score
            tentative_g = self.g_score[current.node_id] + self._edge_cost(current, child)
            
            # Update if better path found
            if child.node_id not in self.g_score or tentative_g < self.g_score[child.node_id]:
                self.g_score[child.node_id] = tentative_g
                self.f_score[child.node_id] = tentative_g + self._heuristic(child)
                heapq.heappush(self.open_set, (self.f_score[child.node_id], self.node_counter, child))
                self.node_counter += 1
        
        return current
    
    def _heuristic(self, node: BranchNode) -> float:
        """
        Admissible heuristic for remaining cost.
        Must never overestimate true cost to goal.
        """
        # Estimate based on remaining coverage needed
        remaining_coverage = 0.95 - node.state.coverage
        
        # Optimistic estimate: assume perfect packing of remaining space
        if node.remaining_panels:
            max_panel_efficiency = 0.9  # Best possible packing efficiency
            estimated_cost = remaining_coverage / max_panel_efficiency
        else:
            estimated_cost = float('inf') if remaining_coverage > 0 else 0
        
        return estimated_cost
    
    def _edge_cost(self, parent: BranchNode, child: BranchNode) -> float:
        """Cost of edge from parent to child."""
        # Cost based on coverage improvement
        coverage_gain = child.state.coverage - parent.state.coverage
        
        # Penalize if not improving much
        if coverage_gain < 0.01:
            return 1.0
        
        # Reward good coverage gains
        return 1.0 / (1.0 + coverage_gain)


class StrategyAdapter:
    """
    Adaptive strategy selector that chooses and switches strategies dynamically.
    Monitors performance and adapts to search characteristics.
    """
    
    def __init__(self, room: Room, panel_sizes: List[PanelSize], 
                 config: Optional[SearchConfiguration] = None):
        self.room = room
        self.panel_sizes = panel_sizes
        self.config = config or SearchConfiguration()
        
        # Initialize strategies
        self.strategies = {
            SearchStrategy.BEST_FIRST: BestFirstSearch(self.config),
            SearchStrategy.ITERATIVE_DEEPENING: DepthFirstIterativeDeepening(self.config),
            SearchStrategy.BEAM_SEARCH: BeamSearch(self.config),
            SearchStrategy.A_STAR: AStarSearch(self.config)
        }
        
        # Hybrid adaptive search
        self.hybrid = HybridAdaptiveSearch(self.config)
        
        # Performance tracking
        self.strategy_performance = defaultdict(lambda: {
            'nodes_explored': 0,
            'coverage_achieved': 0.0,
            'time_spent': 0.0,
            'solutions_found': 0
        })
        
        self.current_strategy = None
        self.search_statistics = SearchStatistics()
        
    def select_initial_strategy(self) -> SearchStrategy:
        """Select initial strategy based on problem characteristics."""
        room_area = self.room.width * self.room.height
        total_panel_area = sum(p.area for p in self.panel_sizes)
        
        # Estimate problem difficulty
        coverage_ratio = total_panel_area / room_area
        num_panels = len(self.panel_sizes)
        
        # Simple problems - use beam search
        if num_panels < 10 and coverage_ratio < 0.8:
            return SearchStrategy.BEAM_SEARCH
        
        # Medium complexity - use best-first
        if num_panels < 50 and coverage_ratio < 0.9:
            return SearchStrategy.BEST_FIRST
        
        # Complex problems - use A* for optimality
        if coverage_ratio > 0.9:
            return SearchStrategy.A_STAR
        
        # Very complex - use iterative deepening
        return SearchStrategy.ITERATIVE_DEEPENING
    
    def adapt_strategy(self, current_stats: SearchStatistics) -> SearchStrategy:
        """
        Adapt strategy based on current search progress.
        Returns recommended strategy.
        """
        # Let hybrid adapter decide
        if self.config.strategy == SearchStrategy.HYBRID_ADAPTIVE:
            return self.hybrid.select_strategy(current_stats)
        
        # Manual adaptation based on progress
        if current_stats.nodes_explored > self.config.max_nodes * 0.8:
            # Running out of nodes - switch to greedy
            return SearchStrategy.BEAM_SEARCH
        
        if current_stats.best_coverage > 0.9:
            # Close to goal - use A* for optimality
            return SearchStrategy.A_STAR
        
        if current_stats.pruning_efficiency < 0.3:
            # Poor pruning - try different strategy
            if self.current_strategy == SearchStrategy.BEST_FIRST:
                return SearchStrategy.ITERATIVE_DEEPENING
            else:
                return SearchStrategy.BEST_FIRST
        
        # Continue with current strategy
        return self.current_strategy or self.select_initial_strategy()
    
    def update_performance(self, strategy: SearchStrategy, stats: SearchStatistics, 
                          time_elapsed: float):
        """Update performance metrics for strategy."""
        perf = self.strategy_performance[strategy]
        perf['nodes_explored'] += stats.nodes_explored
        perf['coverage_achieved'] = max(perf['coverage_achieved'], stats.best_coverage)
        perf['time_spent'] += time_elapsed
        perf['solutions_found'] += stats.solutions_found
    
    def get_best_performing_strategy(self) -> SearchStrategy:
        """Get strategy with best historical performance."""
        if not self.strategy_performance:
            return self.select_initial_strategy()
        
        best_strategy = None
        best_score = -1
        
        for strategy, perf in self.strategy_performance.items():
            if perf['nodes_explored'] == 0:
                continue
            
            # Score based on coverage per node explored
            efficiency = perf['coverage_achieved'] / (perf['nodes_explored'] / 1000.0)
            score = efficiency * perf['solutions_found']
            
            if score > best_score:
                best_score = score
                best_strategy = strategy
        
        return best_strategy or self.select_initial_strategy()
    
    def get_adaptation_report(self) -> Dict[str, Any]:
        """Get report on strategy adaptation performance."""
        return {
            'current_strategy': self.current_strategy,
            'strategies_used': list(self.strategy_performance.keys()),
            'total_switches': self.search_statistics.strategy_switches,
            'performance_by_strategy': dict(self.strategy_performance),
            'best_performing': self.get_best_performing_strategy(),
            'search_statistics': {
                'nodes_explored': self.search_statistics.nodes_explored,
                'best_coverage': self.search_statistics.best_coverage,
                'pruning_efficiency': self.search_statistics.pruning_efficiency
            }
        }


def create_search_strategy(strategy_type: SearchStrategy, 
                         config: Optional[SearchConfiguration] = None) -> Any:
    """
    Factory function to create search strategy instances.
    Returns appropriate strategy implementation.
    """
    config = config or SearchConfiguration()
    
    if strategy_type == SearchStrategy.BEST_FIRST:
        return BestFirstSearch(config)
    elif strategy_type == SearchStrategy.ITERATIVE_DEEPENING:
        return DepthFirstIterativeDeepening(config)
    elif strategy_type == SearchStrategy.BEAM_SEARCH:
        return BeamSearch(config)
    elif strategy_type == SearchStrategy.A_STAR:
        return AStarSearch(config)
    elif strategy_type == SearchStrategy.HYBRID_ADAPTIVE:
        return HybridAdaptiveSearch(config)
    else:
        # Default to best-first
        return BestFirstSearch(config)
#!/usr/bin/env python3
"""
dp_solver.py - Top-Down Dynamic Programming Solver
===============================================
Production-ready recursive DP solver with memoization for optimal panel placement.
Includes call optimization, stack overflow prevention, and comprehensive caching.
"""

import sys
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
from functools import lru_cache, wraps
from collections import defaultdict, deque
import hashlib
import weakref
from abc import ABC, abstractmethod

from models import Room, PanelSize, Point
from dp_state import DPState, DPStateFactory, DPTransitionFunction, DPTerminalDetector
from dp_grid import GridBasedDPOptimizer
from dp_decomposition import create_room_decomposer, AdaptivePartitioner, SubproblemMerger
from advanced_packing import PackingState, PanelPlacement


@dataclass
class DPSolutionNode:
    """
    Represents a solution node in the DP recursion tree.
    Contains both the state and computed optimal value.
    """
    state: DPState
    value: float
    solution: List[PanelPlacement] = field(default_factory=list)
    computation_time: float = 0.0
    children_explored: int = 0
    depth: int = 0
    is_cached: bool = False


class MemoizationCache:
    """
    Advanced memoization cache with LRU eviction and memory management.
    Optimized for DP state caching with collision detection.
    """
    
    def __init__(self, max_size: int = 100000, collision_detection: bool = True):
        self.max_size = max_size
        self.collision_detection = collision_detection
        self.cache: Dict[str, DPSolutionNode] = {}
        self.access_order = deque()
        self.hit_count = 0
        self.miss_count = 0
        self.collision_count = 0
        
    def get(self, state_key: str) -> Optional[DPSolutionNode]:
        """Retrieve cached solution if available."""
        if state_key in self.cache:
            node = self.cache[state_key]
            # Update access order for LRU
            if state_key in self.access_order:
                self.access_order.remove(state_key)
            self.access_order.append(state_key)
            self.hit_count += 1
            node.is_cached = True
            return node
        
        self.miss_count += 1
        return None
    
    def put(self, state_key: str, node: DPSolutionNode):
        """Cache solution with LRU eviction."""
        # Check for hash collisions if enabled
        if self.collision_detection and state_key in self.cache:
            existing = self.cache[state_key]
            if existing.state != node.state:
                self.collision_count += 1
                # Use longer hash to resolve collision
                state_key = self._extended_hash(node.state)
        
        # Evict least recently used if at capacity
        if len(self.cache) >= self.max_size:
            if self.access_order:
                oldest_key = self.access_order.popleft()
                if oldest_key in self.cache:
                    del self.cache[oldest_key]
        
        self.cache[state_key] = node
        self.access_order.append(state_key)
    
    def _extended_hash(self, state: DPState) -> str:
        """Generate extended hash to resolve collisions."""
        state_data = {
            'placements': [str(p) for p in sorted(state.placements, key=str)],
            'remaining': [str(p) for p in sorted(state.remaining_panels, key=str)],
            'depth': state.depth,
            'coverage': state.coverage
        }
        return hashlib.sha256(str(state_data).encode()).hexdigest()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0.0
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hit_count,
            'misses': self.miss_count,
            'hit_rate': hit_rate,
            'collisions': self.collision_count,
            'memory_usage': sys.getsizeof(self.cache)
        }
    
    def clear(self):
        """Clear all cached entries."""
        self.cache.clear()
        self.access_order.clear()
        self.hit_count = 0
        self.miss_count = 0
        self.collision_count = 0


class StackOverflowPrevention:
    """
    Prevents stack overflow in deep recursion through iterative conversion
    and call depth monitoring.
    """
    
    def __init__(self, max_recursion_depth: int = 500):
        self.max_recursion_depth = max_recursion_depth
        self.current_depth = 0
        self.call_stack: List[DPState] = []
        
    def check_depth(self) -> bool:
        """Check if recursion depth is safe."""
        return self.current_depth < self.max_recursion_depth
    
    def enter_call(self, state: DPState):
        """Register entry into recursive call."""
        self.current_depth += 1
        self.call_stack.append(state)
    
    def exit_call(self):
        """Register exit from recursive call."""
        self.current_depth -= 1
        if self.call_stack:
            self.call_stack.pop()
    
    def convert_to_iterative(self, initial_state: DPState, solver: 'TopDownDPSolver') -> DPSolutionNode:
        """Convert deep recursion to iterative processing."""
        work_stack = [(initial_state, [])]  # (state, path)
        completed = {}
        
        while work_stack:
            current_state, path = work_stack.pop()
            
            # Check if already computed
            state_key = current_state.canonical_hash
            if state_key in completed:
                continue
            
            # Check if terminal
            if solver.terminal_detector.is_terminal(current_state):
                node = DPSolutionNode(
                    state=current_state,
                    value=current_state.coverage,
                    solution=[p for p in current_state.placements],
                    depth=len(path)
                )
                completed[state_key] = node
                continue
            
            # Get successors
            successors = solver.transition_fn.get_successor_states(current_state)
            if not successors:
                # No valid moves - terminal
                node = DPSolutionNode(
                    state=current_state,
                    value=current_state.coverage,
                    solution=[p for p in current_state.placements],
                    depth=len(path)
                )
                completed[state_key] = node
                continue
            
            # Add successors to work stack
            for successor in successors:
                if successor.canonical_hash not in completed:
                    work_stack.append((successor, path + [current_state]))
        
        # Return best solution found
        if completed:
            best_key = max(completed.keys(), key=lambda k: completed[k].value)
            return completed[best_key]
        
        # Fallback
        return DPSolutionNode(state=initial_state, value=0.0, solution=[])


class CallOptimizer:
    """
    Optimizes recursive calls through intelligent ordering,
    pruning, and early termination strategies.
    """
    
    def __init__(self):
        self.call_count = 0
        self.pruned_count = 0
        self.early_termination_count = 0
        
    def order_successors(self, successors: List[DPState], current_best: float) -> List[DPState]:
        """Order successor states for optimal exploration."""
        # Sort by coverage descending (explore promising states first)
        return sorted(successors, key=lambda s: s.coverage, reverse=True)
    
    def should_prune(self, state: DPState, current_best: float, remaining_potential: float) -> bool:
        """Determine if branch should be pruned."""
        self.call_count += 1
        
        # Prune if can't possibly beat current best
        max_possible = state.coverage + remaining_potential
        if max_possible <= current_best:
            self.pruned_count += 1
            return True
        
        return False
    
    def check_early_termination(self, state: DPState, target_coverage: float) -> bool:
        """Check if we can terminate early due to good enough solution."""
        if state.coverage >= target_coverage:
            self.early_termination_count += 1
            return True
        return False
    
    def estimate_remaining_potential(self, state: DPState) -> float:
        """Estimate maximum additional coverage possible."""
        if not state.remaining_panels:
            return 0.0
        
        # Simple heuristic: sum of remaining panel areas
        remaining_area = sum(panel.area for panel in state.remaining_panels)
        rx, ry, rw, rh = state.room_bounds
        room_area = rw * rh
        
        # Maximum additional coverage (optimistic)
        return min(remaining_area / room_area, 1.0 - state.coverage)
    
    def stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        return {
            'total_calls': self.call_count,
            'pruned_calls': self.pruned_count,
            'early_terminations': self.early_termination_count,
            'prune_rate': self.pruned_count / max(1, self.call_count)
        }


class TopDownDPSolver:
    """
    Comprehensive top-down DP solver with memoization and optimization.
    Handles recursive panel placement optimization with stack overflow prevention.
    """
    
    def __init__(self, 
                 room: Room, 
                 panel_sizes: List[PanelSize],
                 target_coverage: float = 0.95,
                 max_time_seconds: float = 30.0,
                 use_decomposition: bool = True,
                 cache_size: int = 100000):
        
        self.room = room
        self.panel_sizes = panel_sizes
        self.target_coverage = target_coverage
        self.max_time_seconds = max_time_seconds
        self.use_decomposition = use_decomposition
        
        # Initialize DP components
        self.state_factory = DPStateFactory()
        self.transition_fn = DPTransitionFunction()
        self.terminal_detector = self.state_factory.create_terminal_detector(target_coverage)
        
        # Initialize optimization components
        self.cache = MemoizationCache(cache_size)
        self.stack_prevention = StackOverflowPrevention()
        self.call_optimizer = CallOptimizer()
        
        # Grid-based optimization (optional)
        self.grid_optimizer = None
        if len(panel_sizes) > 0:
            try:
                self.grid_optimizer = GridBasedDPOptimizer(room, panel_sizes)
            except Exception:
                pass  # Fall back to non-grid approach
        
        # Decomposition (optional)
        self.partitioner = None
        self.merger = None
        if use_decomposition:
            try:
                self.partitioner, self.merger, _ = create_room_decomposer(panel_sizes)
            except Exception:
                pass  # Fall back to non-decomposed approach
        
        # Solution tracking
        self.best_solution: Optional[DPSolutionNode] = None
        self.start_time: Optional[float] = None
        self.solutions_found = 0
        
    def solve(self) -> DPSolutionNode:
        """
        Main solve method with comprehensive optimization strategies.
        Returns optimal solution with performance statistics.
        """
        self.start_time = time.time()
        
        try:
            # Try decomposition first if available
            if self.partitioner and self.merger and self._should_decompose():
                solution = self._solve_with_decomposition()
                if solution and solution.value >= self.target_coverage:
                    return solution
            
            # Standard DP solve
            initial_state = self.state_factory.create_initial_state(self.room, self.panel_sizes)
            solution = self._solve_recursive(initial_state)
            
            return solution
            
        except Exception as e:
            # Fallback to basic solution
            print(f"DP solver error: {e}")
            return self._create_fallback_solution()
    
    def _solve_recursive(self, state: DPState) -> DPSolutionNode:
        """
        Main recursive DP solver with full optimization.
        """
        # Check time limit
        if self.start_time and time.time() - self.start_time > self.max_time_seconds:
            return self._create_timeout_solution(state)
        
        # Check cache first
        state_key = state.canonical_hash
        cached = self.cache.get(state_key)
        if cached:
            return cached
        
        # Check for stack overflow and convert to iterative if needed
        if not self.stack_prevention.check_depth():
            return self.stack_prevention.convert_to_iterative(state, self)
        
        self.stack_prevention.enter_call(state)
        
        try:
            # Base case - terminal state
            if self.terminal_detector.is_terminal(state):
                solution = DPSolutionNode(
                    state=state,
                    value=state.coverage,
                    solution=[p for p in state.placements],
                    depth=state.depth
                )
                self.cache.put(state_key, solution)
                return solution
            
            # Early termination if target reached
            if self.call_optimizer.check_early_termination(state, self.target_coverage):
                solution = DPSolutionNode(
                    state=state,
                    value=state.coverage,
                    solution=[p for p in state.placements],
                    depth=state.depth
                )
                self.cache.put(state_key, solution)
                return solution
            
            # Get successor states
            successors = self.transition_fn.get_successor_states(state)
            if not successors:
                # No valid moves
                solution = DPSolutionNode(
                    state=state,
                    value=state.coverage,
                    solution=[p for p in state.placements],
                    depth=state.depth
                )
                self.cache.put(state_key, solution)
                return solution
            
            # Order successors for optimal exploration
            current_best = self.best_solution.value if self.best_solution else 0.0
            ordered_successors = self.call_optimizer.order_successors(successors, current_best)
            
            best_child_solution = None
            best_value = state.coverage
            
            # Explore successors
            for successor in ordered_successors:
                # Pruning check
                remaining_potential = self.call_optimizer.estimate_remaining_potential(successor)
                if self.call_optimizer.should_prune(successor, current_best, remaining_potential):
                    continue
                
                # Recursive call
                child_solution = self._solve_recursive(successor)
                
                # Update best if improved
                if child_solution.value > best_value:
                    best_value = child_solution.value
                    best_child_solution = child_solution
                    
                    # Update global best
                    if not self.best_solution or best_value > self.best_solution.value:
                        self.best_solution = child_solution
                        self.solutions_found += 1
                        
                        # Early termination if target reached
                        if best_value >= self.target_coverage:
                            break
            
            # Create solution node
            if best_child_solution:
                solution = DPSolutionNode(
                    state=state,
                    value=best_child_solution.value,
                    solution=best_child_solution.solution,
                    depth=state.depth,
                    children_explored=len(ordered_successors)
                )
            else:
                solution = DPSolutionNode(
                    state=state,
                    value=state.coverage,
                    solution=[p for p in state.placements],
                    depth=state.depth
                )
            
            # Cache result
            self.cache.put(state_key, solution)
            return solution
            
        finally:
            self.stack_prevention.exit_call()
    
    def _should_decompose(self) -> bool:
        """Determine if room should be decomposed for solving."""
        if not self.partitioner or not self.merger:
            return False
        
        # Decompose large rooms or complex panel inventories
        room_area = self.room.width * self.room.height
        return (room_area > 200 or  # Large room
                len(self.panel_sizes) > 10 or  # Many panel types
                len([p for p in self.panel_sizes if p.area > room_area * 0.3]) > 0)  # Large panels
    
    def _solve_with_decomposition(self) -> Optional[DPSolutionNode]:
        """Solve using room decomposition strategy."""
        if not self.partitioner or not self.merger:
            return None
        
        try:
            # Partition room
            result = self.partitioner.partition(self.room)
            subregions = result.subregions
            
            if len(subregions) <= 1:
                return None  # No benefit from decomposition
            
            # Solve subproblems
            subproblem_solutions = []
            remaining_panels = set(self.panel_sizes)
            
            for subregion in subregions:
                if not remaining_panels:
                    break
                
                # Create subroom from subregion
                subroom = Room(
                    id=f"sub_{subregion.id}",
                    type=self.room.type,
                    position=Point(subregion.x, subregion.y),
                    width=subregion.width,
                    height=subregion.height,
                    boundary=[
                        Point(subregion.x, subregion.y),
                        Point(subregion.x + subregion.width, subregion.y),
                        Point(subregion.x + subregion.width, subregion.y + subregion.height),
                        Point(subregion.x, subregion.y + subregion.height)
                    ],
                    area=subregion.area
                )
                
                # Solve subproblem
                sub_solver = TopDownDPSolver(
                    room=subroom,
                    panel_sizes=list(remaining_panels),
                    target_coverage=self.target_coverage,
                    max_time_seconds=self.max_time_seconds / len(subregions),
                    use_decomposition=False,  # Avoid recursive decomposition
                    cache_size=self.cache.max_size // len(subregions)
                )
                
                sub_solution = sub_solver.solve()
                subproblem_solutions.append((subregion, sub_solution))
                
                # Update remaining panels (simplified - remove used panels)
                used_panels = {p.panel_size for p in sub_solution.solution}
                remaining_panels -= used_panels
            
            # Merge solutions
            merged_solution = self.merger.merge_solutions(subproblem_solutions, self.room)
            
            if merged_solution:
                # Convert to DPSolutionNode
                total_coverage = merged_solution.coverage
                all_placements = list(merged_solution.placements)
                
                return DPSolutionNode(
                    state=merged_solution,
                    value=total_coverage,
                    solution=all_placements,
                    depth=0
                )
            
        except Exception as e:
            print(f"Decomposition solving failed: {e}")
        
        return None
    
    def _create_timeout_solution(self, state: DPState) -> DPSolutionNode:
        """Create solution when timeout occurs."""
        return DPSolutionNode(
            state=state,
            value=state.coverage,
            solution=[p for p in state.placements],
            computation_time=self.max_time_seconds,
            depth=state.depth
        )
    
    def _create_fallback_solution(self) -> DPSolutionNode:
        """Create fallback solution when main solver fails."""
        initial_state = self.state_factory.create_initial_state(self.room, [])
        return DPSolutionNode(
            state=initial_state,
            value=0.0,
            solution=[],
            computation_time=0.0,
            depth=0
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive solver statistics."""
        total_time = time.time() - self.start_time if self.start_time else 0.0
        
        stats = {
            'solution_found': self.best_solution is not None,
            'best_coverage': self.best_solution.value if self.best_solution else 0.0,
            'target_coverage': self.target_coverage,
            'solutions_explored': self.solutions_found,
            'total_time': total_time,
            'cache_stats': self.cache.stats(),
            'optimization_stats': self.call_optimizer.stats(),
            'stack_depth_used': self.stack_prevention.current_depth
        }
        
        if self.best_solution:
            stats['solution_depth'] = self.best_solution.depth
            stats['placement_count'] = len(self.best_solution.solution)
            stats['children_explored'] = self.best_solution.children_explored
        
        return stats


def create_dp_solver(room: Room, 
                    panel_sizes: List[PanelSize], 
                    target_coverage: float = 0.95,
                    max_time: float = 30.0,
                    use_decomposition: bool = True) -> TopDownDPSolver:
    """
    Factory function to create configured DP solver.
    Returns ready-to-use top-down DP solver with all optimizations.
    """
    return TopDownDPSolver(
        room=room,
        panel_sizes=panel_sizes,
        target_coverage=target_coverage,
        max_time_seconds=max_time,
        use_decomposition=use_decomposition
    )
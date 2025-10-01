#!/usr/bin/env python3
"""
dp_parallel.py - Parallel DP Subproblem Processing
================================================
Production-ready parallel processing system for DP optimization including
work distribution, result aggregation, and dynamic load balancing.
"""

import time
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Callable, Union
from collections import deque, defaultdict
from enum import Enum
from abc import ABC, abstractmethod
import queue
import os
import numpy as np

from models import Room, PanelSize
from dp_state import DPState, DPStateFactory
from dp_solver import TopDownDPSolver, DPSolutionNode
from dp_bottom_up import BottomUpDPSolver, DPTableEntry
from advanced_packing import PanelPlacement


class ParallelStrategy(Enum):
    """Types of parallelization strategies."""
    THREAD_BASED = "threads"
    PROCESS_BASED = "processes"  
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"


class WorkItemPriority(Enum):
    """Priority levels for work items."""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4


@dataclass
class WorkItem:
    """
    Represents a single DP subproblem to be solved in parallel.
    Contains all information needed for independent processing.
    """
    item_id: str
    state: DPState
    remaining_time: float
    priority: WorkItemPriority = WorkItemPriority.NORMAL
    estimated_complexity: float = 1.0
    dependencies: Set[str] = field(default_factory=set)
    creation_time: float = field(default_factory=time.time)
    
    def __lt__(self, other: 'WorkItem') -> bool:
        """For priority queue ordering."""
        return self.priority.value < other.priority.value


@dataclass
class WorkResult:
    """
    Result from processing a work item.
    Contains solution and performance metadata.
    """
    item_id: str
    solution: Optional[DPSolutionNode]
    processing_time: float
    worker_id: str
    success: bool
    error_message: Optional[str] = None
    states_explored: int = 0
    coverage_achieved: float = 0.0


class WorkDistributor:
    """
    Intelligent work distribution system that balances load across workers
    based on worker capacity, work complexity, and system resources.
    """
    
    def __init__(self, num_workers: int, strategy: ParallelStrategy = ParallelStrategy.ADAPTIVE):
        self.num_workers = num_workers
        self.strategy = strategy
        self.work_queue = queue.PriorityQueue()
        self.completed_work: Dict[str, WorkResult] = {}
        
        # Load balancing state
        self.worker_loads: Dict[str, float] = {f"worker_{i}": 0.0 for i in range(num_workers)}
        self.worker_capabilities: Dict[str, float] = {f"worker_{i}": 1.0 for i in range(num_workers)}
        
        # Performance tracking
        self.distribution_count = 0
        self.total_work_items = 0
        self.avg_processing_time = 0.0
        
    def add_work_item(self, work_item: WorkItem):
        """Add work item to distribution queue."""
        # Adjust priority based on estimated complexity and remaining time
        adjusted_priority = self._calculate_adjusted_priority(work_item)
        work_item.priority = adjusted_priority
        
        self.work_queue.put((work_item.priority.value, time.time(), work_item))
        self.total_work_items += 1
        
    def get_work_for_worker(self, worker_id: str) -> Optional[WorkItem]:
        """Get next work item optimized for specific worker."""
        try:
            # Get highest priority work item
            priority, timestamp, work_item = self.work_queue.get_nowait()
            
            # Update worker load
            estimated_time = work_item.estimated_complexity * self.avg_processing_time or 1.0
            self.worker_loads[worker_id] += estimated_time
            
            self.distribution_count += 1
            return work_item
            
        except queue.Empty:
            return None
    
    def report_work_completion(self, result: WorkResult):
        """Report completion of work item and update load balancing."""
        self.completed_work[result.item_id] = result
        
        # Update worker capabilities based on performance
        worker_id = result.worker_id
        if worker_id in self.worker_capabilities:
            # Adjust capability based on processing speed
            expected_time = self.avg_processing_time or 1.0
            performance_ratio = expected_time / max(result.processing_time, 0.001)
            
            # Exponential moving average for capability adjustment
            alpha = 0.1
            self.worker_capabilities[worker_id] = (
                alpha * performance_ratio + 
                (1 - alpha) * self.worker_capabilities[worker_id]
            )
            
            # Update load (decrease)
            self.worker_loads[worker_id] = max(0.0, 
                self.worker_loads[worker_id] - result.processing_time
            )
        
        # Update average processing time
        if result.success:
            n = len([r for r in self.completed_work.values() if r.success])
            self.avg_processing_time = (
                (self.avg_processing_time * (n - 1) + result.processing_time) / n
            )
    
    def _calculate_adjusted_priority(self, work_item: WorkItem) -> WorkItemPriority:
        """Calculate adjusted priority based on multiple factors."""
        base_priority = work_item.priority.value
        
        # Adjust for time urgency
        if work_item.remaining_time < 5.0:  # Less than 5 seconds
            base_priority = min(1, base_priority - 1)  # Higher priority
        
        # Adjust for complexity (simpler tasks get higher priority for quick wins)
        if work_item.estimated_complexity < 0.5:
            base_priority = min(1, base_priority - 1)
        
        return WorkItemPriority(max(1, min(4, base_priority)))
    
    def get_load_balance_stats(self) -> Dict[str, Any]:
        """Get load balancing statistics."""
        if not self.worker_loads:
            return {'balanced': True, 'load_variance': 0.0}
        
        loads = list(self.worker_loads.values())
        capabilities = list(self.worker_capabilities.values())
        
        return {
            'total_work_distributed': self.distribution_count,
            'average_load': np.mean(loads),
            'load_variance': np.var(loads),
            'load_balance_ratio': min(loads) / max(loads) if max(loads) > 0 else 1.0,
            'worker_capabilities': dict(self.worker_capabilities),
            'completed_items': len(self.completed_work),
            'success_rate': sum(1 for r in self.completed_work.values() if r.success) / max(1, len(self.completed_work))
        }


class ParallelWorker:
    """
    Individual worker process/thread that solves DP subproblems.
    Handles communication with distributor and processes work items independently.
    """
    
    def __init__(self, worker_id: str, room: Room, panel_sizes: List[PanelSize], 
                 target_coverage: float = 0.95):
        self.worker_id = worker_id
        self.room = room
        self.panel_sizes = panel_sizes
        self.target_coverage = target_coverage
        
        # Create solvers for this worker
        self.top_down_solver = TopDownDPSolver(
            room=room, 
            panel_sizes=panel_sizes,
            target_coverage=target_coverage,
            max_time_seconds=10.0,  # Per-subproblem time limit
            use_decomposition=False  # Avoid nested parallelization
        )
        
        self.bottom_up_solver = BottomUpDPSolver(
            room=room,
            panel_sizes=panel_sizes, 
            target_coverage=target_coverage,
            max_time_seconds=10.0,
            max_memory_mb=100  # Limited memory per worker
        )
        
        # Worker statistics
        self.items_processed = 0
        self.total_processing_time = 0.0
        self.start_time = time.time()
        
    def process_work_item(self, work_item: WorkItem) -> WorkResult:
        """Process a single work item and return result."""
        start_time = time.time()
        
        try:
            # Choose solver strategy based on work item characteristics
            if self._should_use_top_down(work_item):
                solution = self._solve_top_down(work_item)
            else:
                solution = self._solve_bottom_up(work_item)
            
            processing_time = time.time() - start_time
            self.items_processed += 1
            self.total_processing_time += processing_time
            
            return WorkResult(
                item_id=work_item.item_id,
                solution=solution,
                processing_time=processing_time,
                worker_id=self.worker_id,
                success=True,
                states_explored=getattr(solution, 'depth', 0),
                coverage_achieved=solution.value if solution else 0.0
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return WorkResult(
                item_id=work_item.item_id,
                solution=None,
                processing_time=processing_time,
                worker_id=self.worker_id,
                success=False,
                error_message=str(e)
            )
    
    def _should_use_top_down(self, work_item: WorkItem) -> bool:
        """Decide which solver strategy to use."""
        # Use top-down for smaller state spaces or higher priority items
        state_complexity = len(work_item.state.remaining_panels)
        
        return (state_complexity <= 5 or 
                work_item.priority in [WorkItemPriority.CRITICAL, WorkItemPriority.HIGH] or
                work_item.remaining_time < 5.0)
    
    def _solve_top_down(self, work_item: WorkItem) -> Optional[DPSolutionNode]:
        """Solve using top-down approach."""
        # Create solver with remaining time budget
        solver = TopDownDPSolver(
            room=self.room,
            panel_sizes=list(work_item.state.remaining_panels),
            target_coverage=self.target_coverage,
            max_time_seconds=min(work_item.remaining_time, 10.0),
            use_decomposition=False
        )
        
        return solver.solve()
    
    def _solve_bottom_up(self, work_item: WorkItem) -> Optional[DPTableEntry]:
        """Solve using bottom-up approach."""
        # Create solver with remaining time budget
        solver = BottomUpDPSolver(
            room=self.room,
            panel_sizes=list(work_item.state.remaining_panels),
            target_coverage=self.target_coverage,
            max_time_seconds=min(work_item.remaining_time, 10.0),
            max_memory_mb=50  # Conservative memory limit for parallel workers
        )
        
        return solver.solve()
    
    def get_worker_stats(self) -> Dict[str, Any]:
        """Get worker performance statistics."""
        runtime = time.time() - self.start_time
        
        return {
            'worker_id': self.worker_id,
            'items_processed': self.items_processed,
            'total_processing_time': self.total_processing_time,
            'average_processing_time': self.total_processing_time / max(1, self.items_processed),
            'runtime': runtime,
            'throughput': self.items_processed / max(runtime, 0.001),
            'utilization': self.total_processing_time / max(runtime, 0.001)
        }


class ResultAggregator:
    """
    Aggregates results from parallel workers and determines optimal solution.
    Handles result merging, quality assessment, and final solution selection.
    """
    
    def __init__(self, target_coverage: float = 0.95):
        self.target_coverage = target_coverage
        self.results: Dict[str, WorkResult] = {}
        self.best_solution: Optional[Union[DPSolutionNode, DPTableEntry]] = None
        self.best_coverage = 0.0
        
        # Aggregation statistics
        self.solutions_received = 0
        self.successful_solutions = 0
        self.aggregation_time = 0.0
        
    def add_result(self, result: WorkResult):
        """Add result from parallel worker."""
        start_time = time.time()
        
        self.results[result.item_id] = result
        self.solutions_received += 1
        
        if result.success and result.solution:
            self.successful_solutions += 1
            
            # Update best solution if this is better
            coverage = result.coverage_achieved
            if coverage > self.best_coverage:
                self.best_coverage = coverage
                self.best_solution = result.solution
        
        self.aggregation_time += time.time() - start_time
    
    def aggregate_partial_solutions(self) -> Optional[Union[DPSolutionNode, DPTableEntry]]:
        """
        Aggregate partial solutions from different subproblems.
        Combines non-overlapping solutions for improved coverage.
        """
        if not self.results:
            return None
        
        successful_results = [r for r in self.results.values() 
                            if r.success and r.solution]
        
        if not successful_results:
            return None
        
        # Sort by coverage quality
        successful_results.sort(key=lambda r: r.coverage_achieved, reverse=True)
        
        # For now, return best single solution
        # In a full implementation, this would combine non-overlapping solutions
        if successful_results:
            return successful_results[0].solution
        
        return None
    
    def get_aggregation_stats(self) -> Dict[str, Any]:
        """Get result aggregation statistics."""
        processing_times = [r.processing_time for r in self.results.values() if r.success]
        coverages = [r.coverage_achieved for r in self.results.values() if r.success]
        
        return {
            'solutions_received': self.solutions_received,
            'successful_solutions': self.successful_solutions,
            'success_rate': self.successful_solutions / max(1, self.solutions_received),
            'best_coverage': self.best_coverage,
            'target_reached': self.best_coverage >= self.target_coverage,
            'aggregation_time': self.aggregation_time,
            'avg_processing_time': np.mean(processing_times) if processing_times else 0.0,
            'coverage_distribution': {
                'min': min(coverages) if coverages else 0.0,
                'max': max(coverages) if coverages else 0.0,
                'avg': np.mean(coverages) if coverages else 0.0,
                'std': np.std(coverages) if coverages else 0.0
            }
        }


class ParallelDPSolver:
    """
    Main parallel DP solver that coordinates work distribution, parallel processing,
    and result aggregation for optimal multi-core performance.
    """
    
    def __init__(self, room: Room, panel_sizes: List[PanelSize],
                 target_coverage: float = 0.95,
                 max_time_seconds: float = 30.0,
                 strategy: ParallelStrategy = ParallelStrategy.ADAPTIVE,
                 num_workers: Optional[int] = None):
        
        self.room = room
        self.panel_sizes = panel_sizes
        self.target_coverage = target_coverage
        self.max_time_seconds = max_time_seconds
        self.strategy = strategy
        
        # Determine optimal worker count
        self.num_workers = num_workers or self._determine_optimal_workers()
        
        # Initialize components
        self.distributor = WorkDistributor(self.num_workers, strategy)
        self.aggregator = ResultAggregator(target_coverage)
        
        # Performance tracking
        self.start_time: Optional[float] = None
        self.solve_time = 0.0
        self.parallelization_efficiency = 0.0
        
    def solve(self) -> Optional[Union[DPSolutionNode, DPTableEntry]]:
        """
        Main solve method using parallel processing.
        Distributes work across workers and aggregates results.
        """
        self.start_time = time.time()
        
        try:
            # Generate initial work items
            work_items = self._generate_work_items()
            
            if not work_items:
                return None
            
            # Add work items to distributor
            for item in work_items:
                self.distributor.add_work_item(item)
            
            # Process work in parallel
            if self.strategy == ParallelStrategy.THREAD_BASED:
                results = self._solve_with_threads()
            elif self.strategy == ParallelStrategy.PROCESS_BASED:
                results = self._solve_with_processes()
            else:  # ADAPTIVE or HYBRID
                results = self._solve_adaptive()
            
            # Aggregate results
            for result in results:
                self.aggregator.add_result(result)
                self.distributor.report_work_completion(result)
            
            # Calculate performance metrics
            self.solve_time = time.time() - self.start_time
            self._calculate_efficiency_metrics()
            
            # Return best solution
            return self.aggregator.aggregate_partial_solutions()
            
        except Exception as e:
            print(f"Parallel DP solver error: {e}")
            return None
    
    def _generate_work_items(self) -> List[WorkItem]:
        """Generate initial work items from problem decomposition."""
        state_factory = DPStateFactory()
        initial_state = state_factory.create_initial_state(self.room, self.panel_sizes)
        
        work_items = []
        
        # Create work items for different panel subsets
        for i in range(min(self.num_workers * 2, len(self.panel_sizes))):
            # Create different starting configurations
            if i == 0:
                # Full problem
                item = WorkItem(
                    item_id=f"full_problem",
                    state=initial_state,
                    remaining_time=self.max_time_seconds,
                    priority=WorkItemPriority.HIGH,
                    estimated_complexity=len(self.panel_sizes)
                )
            else:
                # Subset problems for parallel exploration
                subset_size = max(1, len(self.panel_sizes) - i)
                subset_panels = self.panel_sizes[:subset_size]
                subset_state = state_factory.create_initial_state(self.room, subset_panels)
                
                item = WorkItem(
                    item_id=f"subset_{i}",
                    state=subset_state,
                    remaining_time=self.max_time_seconds / self.num_workers,
                    priority=WorkItemPriority.NORMAL,
                    estimated_complexity=subset_size
                )
            
            work_items.append(item)
        
        return work_items
    
    def _solve_with_threads(self) -> List[WorkResult]:
        """Solve using thread-based parallelism."""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit work items
            futures = []
            
            while not self.distributor.work_queue.empty():
                for i in range(self.num_workers):
                    worker_id = f"thread_worker_{i}"
                    work_item = self.distributor.get_work_for_worker(worker_id)
                    
                    if work_item:
                        worker = ParallelWorker(worker_id, self.room, self.panel_sizes, self.target_coverage)
                        future = executor.submit(worker.process_work_item, work_item)
                        futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=self.max_time_seconds)
                    results.append(result)
                except Exception as e:
                    print(f"Worker error: {e}")
        
        return results
    
    def _solve_with_processes(self) -> List[WorkResult]:
        """Solve using process-based parallelism."""
        results = []
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit work items
            futures = []
            
            while not self.distributor.work_queue.empty():
                for i in range(self.num_workers):
                    worker_id = f"process_worker_{i}"
                    work_item = self.distributor.get_work_for_worker(worker_id)
                    
                    if work_item:
                        future = executor.submit(
                            _worker_process_function,
                            worker_id, 
                            self.room, 
                            self.panel_sizes,
                            self.target_coverage,
                            work_item
                        )
                        futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=self.max_time_seconds)
                    results.append(result)
                except Exception as e:
                    print(f"Process worker error: {e}")
        
        return results
    
    def _solve_adaptive(self) -> List[WorkResult]:
        """Adaptively choose between threads and processes based on workload."""
        # For CPU-intensive DP problems, prefer processes
        # For I/O or memory-shared problems, prefer threads
        
        work_complexity = sum(item.estimated_complexity for _, _, item in list(self.distributor.work_queue.queue))
        avg_complexity = work_complexity / max(1, len(list(self.distributor.work_queue.queue)))
        
        if avg_complexity > 3.0 or self.num_workers > 4:
            # High complexity or many workers - use processes
            return self._solve_with_processes()
        else:
            # Lower complexity - use threads for efficiency
            return self._solve_with_threads()
    
    def _determine_optimal_workers(self) -> int:
        """Determine optimal number of workers based on system resources."""
        try:
            cpu_count = os.cpu_count() or 4
        except:
            cpu_count = 4  # Default fallback
        
        # Conservative estimate: 1 worker per 2 CPU cores
        max_workers = max(1, cpu_count // 2)
        
        return min(max_workers, 8)  # Cap at 8 workers
    
    def _calculate_efficiency_metrics(self):
        """Calculate parallelization efficiency metrics."""
        if not self.aggregator.results:
            self.parallelization_efficiency = 0.0
            return
        
        # Estimate sequential time (sum of all processing times)
        sequential_time = sum(r.processing_time for r in self.aggregator.results.values())
        
        # Actual parallel time
        parallel_time = self.solve_time
        
        # Efficiency = Sequential_Time / (Parallel_Time × Number_of_Workers)
        if parallel_time > 0:
            self.parallelization_efficiency = sequential_time / (parallel_time * self.num_workers)
        else:
            self.parallelization_efficiency = 0.0
    
    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Get comprehensive parallel processing statistics."""
        stats = {
            'solver_config': {
                'num_workers': self.num_workers,
                'strategy': self.strategy.value,
                'target_coverage': self.target_coverage,
                'max_time': self.max_time_seconds
            },
            'performance': {
                'total_solve_time': self.solve_time,
                'parallelization_efficiency': self.parallelization_efficiency,
                'speedup_factor': min(self.parallelization_efficiency * self.num_workers, self.num_workers)
            }
        }
        
        # Add component statistics
        stats['work_distribution'] = self.distributor.get_load_balance_stats()
        stats['result_aggregation'] = self.aggregator.get_aggregation_stats()
        
        return stats


def _worker_process_function(worker_id: str, room: Room, panel_sizes: List[PanelSize], 
                           target_coverage: float, work_item: WorkItem) -> WorkResult:
    """Function for process-based parallelism (must be module-level for pickling)."""
    worker = ParallelWorker(worker_id, room, panel_sizes, target_coverage)
    return worker.process_work_item(work_item)


def create_parallel_solver(room: Room, 
                          panel_sizes: List[PanelSize],
                          target_coverage: float = 0.95,
                          max_time: float = 30.0,
                          strategy: ParallelStrategy = ParallelStrategy.ADAPTIVE,
                          num_workers: Optional[int] = None) -> ParallelDPSolver:
    """
    Factory function to create configured parallel DP solver.
    Returns ready-to-use parallel solver with optimal configuration.
    """
    return ParallelDPSolver(
        room=room,
        panel_sizes=panel_sizes,
        target_coverage=target_coverage,
        max_time_seconds=max_time,
        strategy=strategy,
        num_workers=num_workers
    )
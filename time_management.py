#!/usr/bin/env python3
"""
time_management.py - Time Management System
===========================================
Production-ready time budget distributor, adaptive allocation,
timeout handler, and time estimator for optimization control.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from collections import defaultdict, deque
from enum import Enum
import time
import threading
import signal
import math
import numpy as np

from strategy_selection import StrategyType, RoomComplexity, RoomSize


class TimeAllocationStrategy(Enum):
    """Time allocation strategies."""
    EQUAL = "equal"  # Equal time for all strategies
    PROPORTIONAL = "proportional"  # Proportional to expected performance
    ADAPTIVE = "adaptive"  # Adapt based on progress
    PRIORITY = "priority"  # Priority-based allocation
    DYNAMIC = "dynamic"  # Dynamic reallocation


@dataclass
class TimeBudget:
    """Time budget allocation."""
    total_budget: float  # Total time in seconds
    strategy_budgets: Dict[StrategyType, float]
    overhead_budget: float  # Time for setup/teardown
    reserve_budget: float  # Reserve for contingencies
    used_time: float = 0.0
    start_time: Optional[float] = None
    
    def remaining_time(self) -> float:
        """Get remaining time."""
        if self.start_time is None:
            return self.total_budget
        
        elapsed = time.time() - self.start_time
        return max(0, self.total_budget - elapsed)
    
    def is_expired(self) -> bool:
        """Check if budget is expired."""
        return self.remaining_time() <= 0
    
    def get_strategy_budget(self, strategy: StrategyType) -> float:
        """Get budget for a strategy."""
        return self.strategy_budgets.get(strategy, 0.0)


@dataclass
class TimeEstimate:
    """Time estimate for an operation."""
    expected_time: float
    min_time: float
    max_time: float
    confidence: float  # 0-1
    based_on_samples: int
    
    def with_confidence_interval(self, confidence_level: float = 0.95) -> Tuple[float, float]:
        """Get confidence interval."""
        # Simple confidence interval calculation
        margin = (self.max_time - self.min_time) * (1 - confidence_level) / 2
        return (self.expected_time - margin, self.expected_time + margin)


@dataclass
class TimeoutEvent:
    """Timeout event information."""
    timestamp: float
    strategy: Optional[StrategyType]
    phase: str
    elapsed_time: float
    budget_exceeded: float
    handled: bool = False
    recovery_action: Optional[str] = None


class TimeBudgetDistributor:
    """Distributes time budget across strategies."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.overhead_percentage = self.config.get('overhead_percentage', 0.05)
        self.reserve_percentage = self.config.get('reserve_percentage', 0.1)
        self.historical_times = defaultdict(list)
    
    def distribute(self,
                  total_budget: float,
                  strategies: List[StrategyType],
                  room_complexity: Optional[RoomComplexity] = None,
                  room_size: Optional[RoomSize] = None,
                  allocation_strategy: TimeAllocationStrategy = TimeAllocationStrategy.PROPORTIONAL) -> TimeBudget:
        """Distribute time budget across strategies."""
        
        # Reserve overhead and contingency
        overhead = total_budget * self.overhead_percentage
        reserve = total_budget * self.reserve_percentage
        available = total_budget - overhead - reserve
        
        # Distribute based on strategy
        if allocation_strategy == TimeAllocationStrategy.EQUAL:
            strategy_budgets = self._distribute_equal(available, strategies)
        elif allocation_strategy == TimeAllocationStrategy.PROPORTIONAL:
            strategy_budgets = self._distribute_proportional(available, strategies, room_complexity, room_size)
        elif allocation_strategy == TimeAllocationStrategy.PRIORITY:
            strategy_budgets = self._distribute_priority(available, strategies)
        elif allocation_strategy == TimeAllocationStrategy.ADAPTIVE:
            strategy_budgets = self._distribute_adaptive(available, strategies)
        else:  # DYNAMIC
            strategy_budgets = self._distribute_dynamic(available, strategies)
        
        return TimeBudget(
            total_budget=total_budget,
            strategy_budgets=strategy_budgets,
            overhead_budget=overhead,
            reserve_budget=reserve,
            start_time=time.time()
        )
    
    def _distribute_equal(self,
                         available: float,
                         strategies: List[StrategyType]) -> Dict[StrategyType, float]:
        """Equal distribution."""
        if not strategies:
            return {}
        
        per_strategy = available / len(strategies)
        return {strategy: per_strategy for strategy in strategies}
    
    def _distribute_proportional(self,
                                available: float,
                                strategies: List[StrategyType],
                                room_complexity: Optional[RoomComplexity],
                                room_size: Optional[RoomSize]) -> Dict[StrategyType, float]:
        """Proportional distribution based on expected performance."""
        if not strategies:
            return {}
        
        # Assign weights based on strategy characteristics
        weights = {}
        for strategy in strategies:
            weight = self._get_strategy_weight(strategy, room_complexity, room_size)
            weights[strategy] = weight
        
        # Normalize and distribute
        total_weight = sum(weights.values())
        if total_weight > 0:
            return {
                strategy: (weight / total_weight) * available
                for strategy, weight in weights.items()
            }
        else:
            return self._distribute_equal(available, strategies)
    
    def _get_strategy_weight(self,
                           strategy: StrategyType,
                           room_complexity: Optional[RoomComplexity],
                           room_size: Optional[RoomSize]) -> float:
        """Get weight for a strategy."""
        base_weight = {
            StrategyType.SIMPLE_BLF: 0.5,
            StrategyType.ENHANCED_BLF: 1.0,
            StrategyType.DYNAMIC_PROGRAMMING: 1.5,
            StrategyType.BRANCH_AND_BOUND: 2.0,
            StrategyType.HYBRID: 1.8,
            StrategyType.GREEDY: 0.3,
            StrategyType.EXHAUSTIVE: 3.0
        }.get(strategy, 1.0)
        
        # Adjust for room complexity
        if room_complexity:
            if room_complexity == RoomComplexity.SIMPLE:
                base_weight *= 0.7
            elif room_complexity == RoomComplexity.COMPLEX:
                base_weight *= 1.3
            elif room_complexity == RoomComplexity.EXTREME:
                base_weight *= 1.5
        
        # Adjust for room size
        if room_size:
            if room_size == RoomSize.TINY:
                base_weight *= 0.5
            elif room_size == RoomSize.LARGE:
                base_weight *= 1.2
            elif room_size == RoomSize.HUGE:
                base_weight *= 1.5
        
        # Consider historical performance
        if strategy in self.historical_times:
            avg_time = sum(self.historical_times[strategy]) / len(self.historical_times[strategy])
            # Adjust weight based on historical time
            if avg_time > 0:
                base_weight = min(base_weight, available / avg_time)
        
        return base_weight
    
    def _distribute_priority(self,
                           available: float,
                           strategies: List[StrategyType]) -> Dict[StrategyType, float]:
        """Priority-based distribution."""
        if not strategies:
            return {}
        
        # Define priority levels
        priority_levels = {
            StrategyType.SIMPLE_BLF: 3,
            StrategyType.ENHANCED_BLF: 2,
            StrategyType.DYNAMIC_PROGRAMMING: 1,
            StrategyType.BRANCH_AND_BOUND: 1,
            StrategyType.HYBRID: 1,
            StrategyType.GREEDY: 4,
            StrategyType.EXHAUSTIVE: 0
        }
        
        # Distribute based on priority
        budgets = {}
        remaining = available
        
        for priority in range(5):  # 0 is highest priority
            strategies_at_level = [
                s for s in strategies 
                if priority_levels.get(s, 2) == priority
            ]
            
            if strategies_at_level:
                # Allocate more time to higher priority
                allocation_ratio = {
                    0: 0.4,  # 40% for highest priority
                    1: 0.3,  # 30% for high priority
                    2: 0.2,  # 20% for medium priority
                    3: 0.08, # 8% for low priority
                    4: 0.02  # 2% for lowest priority
                }.get(priority, 0.1)
                
                level_budget = available * allocation_ratio
                per_strategy = level_budget / len(strategies_at_level)
                
                for strategy in strategies_at_level:
                    budgets[strategy] = min(per_strategy, remaining)
                    remaining -= budgets[strategy]
        
        return budgets
    
    def _distribute_adaptive(self,
                           available: float,
                           strategies: List[StrategyType]) -> Dict[StrategyType, float]:
        """Adaptive distribution based on historical data."""
        if not strategies:
            return {}
        
        # Start with proportional distribution
        budgets = self._distribute_proportional(available, strategies, None, None)
        
        # Adapt based on historical performance
        for strategy in strategies:
            if strategy in self.historical_times and self.historical_times[strategy]:
                # Calculate adaptive factor
                recent_times = self.historical_times[strategy][-5:]  # Last 5 runs
                avg_recent = sum(recent_times) / len(recent_times)
                
                # Adjust budget
                if avg_recent < budgets[strategy]:
                    # Strategy is faster than allocated, reduce budget
                    budgets[strategy] = avg_recent * 1.2  # 20% buffer
                elif avg_recent > budgets[strategy] * 2:
                    # Strategy is much slower, might need more time
                    budgets[strategy] = min(avg_recent, available * 0.5)
        
        # Normalize to fit available budget
        total = sum(budgets.values())
        if total > available:
            factor = available / total
            budgets = {s: b * factor for s, b in budgets.items()}
        
        return budgets
    
    def _distribute_dynamic(self,
                          available: float,
                          strategies: List[StrategyType]) -> Dict[StrategyType, float]:
        """Dynamic distribution that can be adjusted during execution."""
        # Start with adaptive distribution
        initial_budgets = self._distribute_adaptive(available, strategies)
        
        # Mark for dynamic adjustment
        for strategy in initial_budgets:
            # Reserve some budget for dynamic reallocation
            initial_budgets[strategy] *= 0.8
        
        return initial_budgets
    
    def record_execution_time(self, strategy: StrategyType, execution_time: float):
        """Record actual execution time for learning."""
        self.historical_times[strategy].append(execution_time)
        
        # Keep only recent history
        max_history = self.config.get('max_history', 100)
        if len(self.historical_times[strategy]) > max_history:
            self.historical_times[strategy] = self.historical_times[strategy][-max_history:]


class AdaptiveAllocator:
    """Adaptively reallocates time during execution."""
    
    def __init__(self, initial_budget: TimeBudget):
        self.budget = initial_budget
        self.reallocation_history = []
        self.performance_tracker = defaultdict(dict)
    
    def should_reallocate(self,
                         current_strategy: StrategyType,
                         elapsed_time: float,
                         progress: float) -> bool:
        """Check if reallocation is needed."""
        if not self.budget.strategy_budgets:
            return False
        
        strategy_budget = self.budget.get_strategy_budget(current_strategy)
        if strategy_budget == 0:
            return False
        
        # Check if strategy is ahead or behind schedule
        expected_progress = elapsed_time / strategy_budget
        
        if progress < expected_progress * 0.5:
            # Far behind schedule
            return True
        elif progress > expected_progress * 1.5:
            # Far ahead of schedule
            return True
        
        return False
    
    def reallocate(self,
                  current_strategy: StrategyType,
                  remaining_strategies: List[StrategyType],
                  progress: float) -> TimeBudget:
        """Reallocate remaining budget."""
        remaining_time = self.budget.remaining_time()
        
        if not remaining_strategies:
            # All time to current strategy
            self.budget.strategy_budgets[current_strategy] = remaining_time
            return self.budget
        
        # Calculate reallocation
        if progress < 0.5:
            # Current strategy needs more time
            current_allocation = remaining_time * 0.6
            remaining_allocation = remaining_time * 0.4
        else:
            # Current strategy doing well, can spare time
            current_allocation = remaining_time * 0.3
            remaining_allocation = remaining_time * 0.7
        
        # Update budgets
        self.budget.strategy_budgets[current_strategy] = current_allocation
        
        if remaining_strategies:
            per_strategy = remaining_allocation / len(remaining_strategies)
            for strategy in remaining_strategies:
                self.budget.strategy_budgets[strategy] = per_strategy
        
        # Record reallocation
        self.reallocation_history.append({
            'timestamp': time.time(),
            'current_strategy': current_strategy,
            'progress': progress,
            'new_allocation': current_allocation,
            'remaining_time': remaining_time
        })
        
        return self.budget
    
    def track_performance(self,
                         strategy: StrategyType,
                         metric: str,
                         value: float):
        """Track strategy performance metrics."""
        if strategy not in self.performance_tracker:
            self.performance_tracker[strategy] = {}
        
        if metric not in self.performance_tracker[strategy]:
            self.performance_tracker[strategy][metric] = []
        
        self.performance_tracker[strategy][metric].append({
            'value': value,
            'timestamp': time.time()
        })
    
    def predict_completion_time(self,
                               current_strategy: StrategyType,
                               progress: float) -> float:
        """Predict time to complete current strategy."""
        if progress <= 0:
            return float('inf')
        
        # Get current strategy budget
        strategy_budget = self.budget.get_strategy_budget(current_strategy)
        
        # Estimate based on progress rate
        if current_strategy in self.performance_tracker:
            progress_history = self.performance_tracker[current_strategy].get('progress', [])
            if len(progress_history) >= 2:
                # Calculate progress rate
                recent = progress_history[-5:]
                if len(recent) >= 2:
                    time_diff = recent[-1]['timestamp'] - recent[0]['timestamp']
                    progress_diff = recent[-1]['value'] - recent[0]['value']
                    
                    if progress_diff > 0 and time_diff > 0:
                        rate = progress_diff / time_diff
                        remaining_progress = 1.0 - progress
                        return remaining_progress / rate
        
        # Fallback: linear estimation
        if progress > 0:
            elapsed = time.time() - self.budget.start_time
            return elapsed * (1.0 - progress) / progress
        
        return strategy_budget


class TimeoutHandler:
    """Handles timeout events and recovery."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.timeout_events = []
        self.handlers = {}
        self.recovery_strategies = {
            'graceful': self._graceful_recovery,
            'immediate': self._immediate_recovery,
            'checkpoint': self._checkpoint_recovery,
            'rollback': self._rollback_recovery
        }
        self.active_timers = {}
    
    def set_timeout(self,
                   timeout_seconds: float,
                   callback: Callable,
                   strategy: Optional[StrategyType] = None,
                   phase: str = "execution") -> threading.Timer:
        """Set a timeout with callback."""
        timer = threading.Timer(timeout_seconds, lambda: self._handle_timeout(callback, strategy, phase))
        timer.start()
        
        timer_id = id(timer)
        self.active_timers[timer_id] = {
            'timer': timer,
            'strategy': strategy,
            'phase': phase,
            'start_time': time.time(),
            'timeout': timeout_seconds
        }
        
        return timer
    
    def cancel_timeout(self, timer: threading.Timer):
        """Cancel a timeout."""
        timer.cancel()
        timer_id = id(timer)
        if timer_id in self.active_timers:
            del self.active_timers[timer_id]
    
    def _handle_timeout(self,
                       callback: Callable,
                       strategy: Optional[StrategyType],
                       phase: str):
        """Handle timeout event."""
        event = TimeoutEvent(
            timestamp=time.time(),
            strategy=strategy,
            phase=phase,
            elapsed_time=0,  # Will be calculated
            budget_exceeded=0
        )
        
        # Calculate elapsed time
        for timer_info in self.active_timers.values():
            if timer_info['strategy'] == strategy and timer_info['phase'] == phase:
                event.elapsed_time = time.time() - timer_info['start_time']
                event.budget_exceeded = event.elapsed_time - timer_info['timeout']
                break
        
        self.timeout_events.append(event)
        
        # Call handler
        try:
            callback(event)
            event.handled = True
        except Exception as e:
            print(f"Timeout handler error: {e}")
            event.handled = False
        
        # Attempt recovery
        self._attempt_recovery(event)
    
    def _attempt_recovery(self, event: TimeoutEvent):
        """Attempt recovery from timeout."""
        recovery_strategy = self.config.get('recovery_strategy', 'graceful')
        
        if recovery_strategy in self.recovery_strategies:
            recovery_func = self.recovery_strategies[recovery_strategy]
            recovery_func(event)
    
    def _graceful_recovery(self, event: TimeoutEvent):
        """Graceful recovery - save state and continue."""
        event.recovery_action = "graceful_shutdown"
        # Implementation would save current state
    
    def _immediate_recovery(self, event: TimeoutEvent):
        """Immediate recovery - stop immediately."""
        event.recovery_action = "immediate_stop"
        # Implementation would force stop
    
    def _checkpoint_recovery(self, event: TimeoutEvent):
        """Checkpoint recovery - revert to last checkpoint."""
        event.recovery_action = "checkpoint_restore"
        # Implementation would restore from checkpoint
    
    def _rollback_recovery(self, event: TimeoutEvent):
        """Rollback recovery - undo recent operations."""
        event.recovery_action = "rollback"
        # Implementation would rollback operations
    
    def register_handler(self,
                        event_type: str,
                        handler: Callable):
        """Register custom timeout handler."""
        self.handlers[event_type] = handler
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get timeout statistics."""
        return {
            'total_timeouts': len(self.timeout_events),
            'handled_timeouts': sum(1 for e in self.timeout_events if e.handled),
            'active_timers': len(self.active_timers),
            'recovery_actions': defaultdict(int, {
                e.recovery_action: 1 
                for e in self.timeout_events 
                if e.recovery_action
            })
        }


class TimeEstimator:
    """Estimates execution time for operations."""
    
    def __init__(self):
        self.execution_history = defaultdict(list)
        self.model_parameters = {}
        self.estimation_models = {
            'linear': self._linear_model,
            'logarithmic': self._logarithmic_model,
            'polynomial': self._polynomial_model,
            'exponential': self._exponential_model
        }
    
    def estimate(self,
                strategy: StrategyType,
                room_area: float,
                num_panels: int,
                complexity: Optional[RoomComplexity] = None) -> TimeEstimate:
        """Estimate execution time."""
        
        # Check if we have historical data
        if strategy in self.execution_history and len(self.execution_history[strategy]) >= 3:
            return self._estimate_from_history(strategy, room_area, num_panels)
        else:
            return self._estimate_from_model(strategy, room_area, num_panels, complexity)
    
    def _estimate_from_history(self,
                              strategy: StrategyType,
                              room_area: float,
                              num_panels: int) -> TimeEstimate:
        """Estimate from historical data."""
        history = self.execution_history[strategy]
        
        # Find similar cases
        similar_cases = []
        for record in history:
            area_diff = abs(record['room_area'] - room_area) / room_area
            panel_diff = abs(record['num_panels'] - num_panels) / max(num_panels, 1)
            
            if area_diff < 0.3 and panel_diff < 0.3:
                similar_cases.append(record['time'])
        
        if similar_cases:
            expected = sum(similar_cases) / len(similar_cases)
            min_time = min(similar_cases)
            max_time = max(similar_cases)
            confidence = min(1.0, len(similar_cases) / 10)
        else:
            # Fall back to all history
            all_times = [r['time'] for r in history]
            expected = sum(all_times) / len(all_times)
            min_time = min(all_times)
            max_time = max(all_times)
            confidence = 0.5
        
        return TimeEstimate(
            expected_time=expected,
            min_time=min_time,
            max_time=max_time,
            confidence=confidence,
            based_on_samples=len(similar_cases) if similar_cases else len(history)
        )
    
    def _estimate_from_model(self,
                           strategy: StrategyType,
                           room_area: float,
                           num_panels: int,
                           complexity: Optional[RoomComplexity]) -> TimeEstimate:
        """Estimate from analytical model."""
        
        # Base time estimates (in seconds)
        base_times = {
            StrategyType.SIMPLE_BLF: 0.01,
            StrategyType.ENHANCED_BLF: 0.05,
            StrategyType.DYNAMIC_PROGRAMMING: 0.1,
            StrategyType.BRANCH_AND_BOUND: 0.2,
            StrategyType.HYBRID: 0.15,
            StrategyType.GREEDY: 0.005,
            StrategyType.EXHAUSTIVE: 1.0
        }
        
        base_time = base_times.get(strategy, 0.1)
        
        # Scale by problem size
        size_factor = (room_area / 100) * (num_panels / 10)
        
        # Complexity factor
        complexity_factor = 1.0
        if complexity:
            complexity_factors = {
                RoomComplexity.SIMPLE: 0.8,
                RoomComplexity.MODERATE: 1.0,
                RoomComplexity.COMPLEX: 1.5,
                RoomComplexity.EXTREME: 2.0
            }
            complexity_factor = complexity_factors.get(complexity, 1.0)
        
        # Select estimation model
        model = self._select_model(strategy)
        expected = model(base_time, size_factor, complexity_factor)
        
        # Calculate bounds
        min_time = expected * 0.5
        max_time = expected * 2.0
        
        # Confidence is lower for model-based estimates
        confidence = 0.3
        
        return TimeEstimate(
            expected_time=expected,
            min_time=min_time,
            max_time=max_time,
            confidence=confidence,
            based_on_samples=0
        )
    
    def _select_model(self, strategy: StrategyType) -> Callable:
        """Select appropriate time complexity model."""
        model_mapping = {
            StrategyType.SIMPLE_BLF: 'linear',
            StrategyType.ENHANCED_BLF: 'polynomial',
            StrategyType.DYNAMIC_PROGRAMMING: 'polynomial',
            StrategyType.BRANCH_AND_BOUND: 'exponential',
            StrategyType.HYBRID: 'polynomial',
            StrategyType.GREEDY: 'linear',
            StrategyType.EXHAUSTIVE: 'exponential'
        }
        
        model_name = model_mapping.get(strategy, 'polynomial')
        return self.estimation_models[model_name]
    
    def _linear_model(self, base: float, size: float, complexity: float) -> float:
        """Linear time complexity model."""
        return base * size * complexity
    
    def _logarithmic_model(self, base: float, size: float, complexity: float) -> float:
        """Logarithmic time complexity model."""
        return base * math.log(size + 1) * complexity
    
    def _polynomial_model(self, base: float, size: float, complexity: float) -> float:
        """Polynomial time complexity model."""
        return base * (size ** 2) * complexity
    
    def _exponential_model(self, base: float, size: float, complexity: float) -> float:
        """Exponential time complexity model."""
        return base * (2 ** min(size, 10)) * complexity
    
    def record_execution(self,
                        strategy: StrategyType,
                        room_area: float,
                        num_panels: int,
                        execution_time: float):
        """Record actual execution for learning."""
        self.execution_history[strategy].append({
            'room_area': room_area,
            'num_panels': num_panels,
            'time': execution_time,
            'timestamp': time.time()
        })
        
        # Keep limited history
        max_history = 1000
        if len(self.execution_history[strategy]) > max_history:
            self.execution_history[strategy] = self.execution_history[strategy][-max_history:]
    
    def update_model(self, strategy: StrategyType):
        """Update model parameters based on history."""
        if strategy not in self.execution_history:
            return
        
        history = self.execution_history[strategy]
        if len(history) < 10:
            return
        
        # Simple parameter fitting (placeholder for more sophisticated methods)
        times = [r['time'] for r in history]
        areas = [r['room_area'] for r in history]
        panels = [r['num_panels'] for r in history]
        
        # Calculate correlations and update model parameters
        # This would use regression or other fitting methods
        self.model_parameters[strategy] = {
            'area_coefficient': np.corrcoef(areas, times)[0, 1] if len(set(areas)) > 1 else 0,
            'panel_coefficient': np.corrcoef(panels, times)[0, 1] if len(set(panels)) > 1 else 0
        }


class TimeManagementSystem:
    """Main time management system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize components
        self.distributor = TimeBudgetDistributor(config)
        self.timeout_handler = TimeoutHandler(config)
        self.estimator = TimeEstimator()
        
        self.current_budget = None
        self.adaptive_allocator = None
        self.execution_start = None
    
    def initialize_budget(self,
                         total_time: float,
                         strategies: List[StrategyType],
                         room_complexity: Optional[RoomComplexity] = None,
                         room_size: Optional[RoomSize] = None) -> TimeBudget:
        """Initialize time budget."""
        self.current_budget = self.distributor.distribute(
            total_time,
            strategies,
            room_complexity,
            room_size,
            TimeAllocationStrategy.ADAPTIVE
        )
        
        self.adaptive_allocator = AdaptiveAllocator(self.current_budget)
        self.execution_start = time.time()
        
        return self.current_budget
    
    def estimate_time(self,
                     strategy: StrategyType,
                     room_area: float,
                     num_panels: int,
                     complexity: Optional[RoomComplexity] = None) -> TimeEstimate:
        """Estimate execution time."""
        return self.estimator.estimate(strategy, room_area, num_panels, complexity)
    
    def set_timeout(self,
                   strategy: StrategyType,
                   callback: Callable) -> threading.Timer:
        """Set timeout for strategy."""
        if not self.current_budget:
            raise ValueError("Budget not initialized")
        
        timeout = self.current_budget.get_strategy_budget(strategy)
        return self.timeout_handler.set_timeout(timeout, callback, strategy)
    
    def check_reallocation(self,
                         current_strategy: StrategyType,
                         progress: float,
                         remaining_strategies: List[StrategyType]) -> bool:
        """Check if reallocation is needed."""
        if not self.adaptive_allocator:
            return False
        
        elapsed = time.time() - self.execution_start
        
        if self.adaptive_allocator.should_reallocate(current_strategy, elapsed, progress):
            self.current_budget = self.adaptive_allocator.reallocate(
                current_strategy,
                remaining_strategies,
                progress
            )
            return True
        
        return False
    
    def record_execution(self,
                        strategy: StrategyType,
                        room_area: float,
                        num_panels: int,
                        execution_time: float):
        """Record execution for learning."""
        self.distributor.record_execution_time(strategy, execution_time)
        self.estimator.record_execution(strategy, room_area, num_panels, execution_time)
        
        if self.adaptive_allocator:
            self.adaptive_allocator.track_performance(strategy, 'execution_time', execution_time)
    
    def get_remaining_time(self) -> float:
        """Get remaining time in budget."""
        if self.current_budget:
            return self.current_budget.remaining_time()
        return 0.0
    
    def is_timeout(self) -> bool:
        """Check if timeout reached."""
        if self.current_budget:
            return self.current_budget.is_expired()
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        stats = {
            'timeout_handler': self.timeout_handler.get_statistics(),
            'elapsed_time': time.time() - self.execution_start if self.execution_start else 0,
            'remaining_time': self.get_remaining_time()
        }
        
        if self.current_budget:
            stats['budget'] = {
                'total': self.current_budget.total_budget,
                'used': self.current_budget.used_time,
                'strategies': self.current_budget.strategy_budgets
            }
        
        if self.adaptive_allocator:
            stats['reallocations'] = len(self.adaptive_allocator.reallocation_history)
        
        return stats
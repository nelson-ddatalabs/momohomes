#!/usr/bin/env python3
"""
progress_tracking.py - Progress Tracking System
===============================================
Production-ready coverage monitor, convergence detector,
metric collector, and progress reporter for optimization monitoring.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from collections import deque, defaultdict
from enum import Enum
import time
import math
import numpy as np
import json

from advanced_packing import PackingState, PanelPlacement
from strategy_selection import StrategyType


class ConvergenceStatus(Enum):
    """Convergence status types."""
    IMPROVING = "improving"
    PLATEAUED = "plateaued"
    CONVERGED = "converged"
    DIVERGING = "diverging"
    OSCILLATING = "oscillating"
    STAGNANT = "stagnant"


class MetricType(Enum):
    """Types of metrics to track."""
    COVERAGE = "coverage"
    PANELS_PLACED = "panels_placed"
    WASTE = "waste"
    ITERATION_TIME = "iteration_time"
    MEMORY_USAGE = "memory_usage"
    SOLUTION_QUALITY = "solution_quality"
    GAP_SIZE = "gap_size"
    FRAGMENTATION = "fragmentation"


@dataclass
class ProgressSnapshot:
    """Snapshot of optimization progress."""
    timestamp: float
    iteration: int
    strategy: StrategyType
    coverage: float
    panels_placed: int
    convergence_status: ConvergenceStatus
    improvement_rate: float
    estimated_completion: Optional[float]
    metrics: Dict[MetricType, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConvergenceInfo:
    """Information about convergence."""
    status: ConvergenceStatus
    iterations_since_improvement: int
    best_coverage: float
    improvement_threshold: float
    plateau_duration: float
    oscillation_detected: bool
    convergence_confidence: float


@dataclass
class MetricSummary:
    """Summary of collected metrics."""
    metric_type: MetricType
    current_value: float
    best_value: float
    worst_value: float
    average_value: float
    std_deviation: float
    trend: str  # "improving", "worsening", "stable"
    samples: int


class CoverageMonitor:
    """Monitors coverage progress."""
    
    def __init__(self, target_coverage: float = 0.95):
        self.target_coverage = target_coverage
        self.coverage_history = deque(maxlen=1000)
        self.best_coverage = 0.0
        self.best_state = None
        self.milestones = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
        self.milestones_reached = set()
        self.coverage_by_strategy = defaultdict(list)
    
    def update(self, state: PackingState, strategy: StrategyType) -> float:
        """Update coverage monitoring."""
        # Calculate coverage
        if hasattr(state, 'coverage_ratio'):
            coverage = state.coverage_ratio
        else:
            coverage = self._calculate_coverage(state)
        
        # Record history
        self.coverage_history.append({
            'coverage': coverage,
            'timestamp': time.time(),
            'panels': len(state.placed_panels),
            'strategy': strategy
        })
        
        # Track by strategy
        self.coverage_by_strategy[strategy].append(coverage)
        
        # Update best
        if coverage > self.best_coverage:
            self.best_coverage = coverage
            self.best_state = state
        
        # Check milestones
        for milestone in self.milestones:
            if coverage >= milestone and milestone not in self.milestones_reached:
                self.milestones_reached.add(milestone)
                self._notify_milestone(milestone)
        
        return coverage
    
    def _calculate_coverage(self, state: PackingState) -> float:
        """Calculate coverage from state."""
        if not state.placed_panels:
            return 0.0
        
        total_area = sum(
            p.panel_size.width * p.panel_size.height
            for p in state.placed_panels
        )
        
        # Assuming room dimensions are available
        room_area = 1000.0  # Placeholder
        
        return total_area / room_area
    
    def _notify_milestone(self, milestone: float):
        """Notify when milestone is reached."""
        print(f"Coverage milestone reached: {milestone:.1%}")
    
    def get_improvement_rate(self, window: int = 10) -> float:
        """Calculate recent improvement rate."""
        if len(self.coverage_history) < 2:
            return 0.0
        
        recent = list(self.coverage_history)[-window:]
        if len(recent) < 2:
            return 0.0
        
        time_diff = recent[-1]['timestamp'] - recent[0]['timestamp']
        coverage_diff = recent[-1]['coverage'] - recent[0]['coverage']
        
        if time_diff > 0:
            return coverage_diff / time_diff
        return 0.0
    
    def get_distance_to_target(self) -> float:
        """Get distance to target coverage."""
        return max(0, self.target_coverage - self.best_coverage)
    
    def estimate_completion_time(self) -> Optional[float]:
        """Estimate time to reach target coverage."""
        rate = self.get_improvement_rate()
        
        if rate <= 0:
            return None
        
        distance = self.get_distance_to_target()
        if distance <= 0:
            return 0.0
        
        return distance / rate
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get coverage statistics."""
        if not self.coverage_history:
            return {}
        
        coverages = [h['coverage'] for h in self.coverage_history]
        
        return {
            'current': coverages[-1] if coverages else 0,
            'best': self.best_coverage,
            'average': sum(coverages) / len(coverages),
            'std_dev': np.std(coverages) if len(coverages) > 1 else 0,
            'improvement_rate': self.get_improvement_rate(),
            'distance_to_target': self.get_distance_to_target(),
            'milestones_reached': list(self.milestones_reached),
            'estimated_completion': self.estimate_completion_time()
        }


class ConvergenceDetector:
    """Detects convergence patterns."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.plateau_threshold = self.config.get('plateau_threshold', 0.001)
        self.plateau_window = self.config.get('plateau_window', 20)
        self.oscillation_threshold = self.config.get('oscillation_threshold', 0.05)
        self.convergence_threshold = self.config.get('convergence_threshold', 0.0001)
        
        self.value_history = deque(maxlen=100)
        self.gradient_history = deque(maxlen=50)
        self.last_improvement_iteration = 0
        self.best_value = -float('inf')
    
    def update(self, value: float, iteration: int) -> ConvergenceInfo:
        """Update convergence detection."""
        self.value_history.append(value)
        
        # Calculate gradient
        if len(self.value_history) >= 2:
            gradient = value - self.value_history[-2]
            self.gradient_history.append(gradient)
        
        # Update best value
        if value > self.best_value:
            self.best_value = value
            self.last_improvement_iteration = iteration
        
        # Detect convergence status
        status = self._detect_status()
        
        # Calculate convergence confidence
        confidence = self._calculate_confidence()
        
        return ConvergenceInfo(
            status=status,
            iterations_since_improvement=iteration - self.last_improvement_iteration,
            best_coverage=self.best_value,
            improvement_threshold=self.plateau_threshold,
            plateau_duration=self._get_plateau_duration(),
            oscillation_detected=self._detect_oscillation(),
            convergence_confidence=confidence
        )
    
    def _detect_status(self) -> ConvergenceStatus:
        """Detect current convergence status."""
        if len(self.value_history) < 3:
            return ConvergenceStatus.IMPROVING
        
        recent_values = list(self.value_history)[-self.plateau_window:]
        
        # Check for improvement
        if recent_values[-1] > recent_values[0] + self.plateau_threshold:
            return ConvergenceStatus.IMPROVING
        
        # Check for plateau
        if max(recent_values) - min(recent_values) < self.plateau_threshold:
            return ConvergenceStatus.PLATEAUED
        
        # Check for convergence
        if self._is_converged():
            return ConvergenceStatus.CONVERGED
        
        # Check for oscillation
        if self._detect_oscillation():
            return ConvergenceStatus.OSCILLATING
        
        # Check for divergence
        if recent_values[-1] < recent_values[0] - self.plateau_threshold:
            return ConvergenceStatus.DIVERGING
        
        return ConvergenceStatus.STAGNANT
    
    def _is_converged(self) -> bool:
        """Check if converged."""
        if len(self.gradient_history) < 10:
            return False
        
        recent_gradients = list(self.gradient_history)[-10:]
        avg_gradient = abs(sum(recent_gradients) / len(recent_gradients))
        
        return avg_gradient < self.convergence_threshold
    
    def _detect_oscillation(self) -> bool:
        """Detect oscillation pattern."""
        if len(self.gradient_history) < 6:
            return False
        
        recent = list(self.gradient_history)[-6:]
        
        # Count sign changes
        sign_changes = 0
        for i in range(1, len(recent)):
            if recent[i] * recent[i-1] < 0:
                sign_changes += 1
        
        return sign_changes >= 3
    
    def _get_plateau_duration(self) -> float:
        """Get duration of current plateau."""
        if len(self.value_history) < 2:
            return 0.0
        
        plateau_start = len(self.value_history) - 1
        current_value = self.value_history[-1]
        
        for i in range(len(self.value_history) - 2, -1, -1):
            if abs(self.value_history[i] - current_value) > self.plateau_threshold:
                break
            plateau_start = i
        
        return len(self.value_history) - plateau_start
    
    def _calculate_confidence(self) -> float:
        """Calculate convergence confidence."""
        if len(self.value_history) < 10:
            return 0.0
        
        # Multiple factors for confidence
        factors = []
        
        # Factor 1: Gradient trend
        if len(self.gradient_history) >= 5:
            recent_gradients = list(self.gradient_history)[-5:]
            avg_gradient = abs(sum(recent_gradients) / len(recent_gradients))
            gradient_factor = 1.0 - min(1.0, avg_gradient / self.plateau_threshold)
            factors.append(gradient_factor)
        
        # Factor 2: Value stability
        recent_values = list(self.value_history)[-10:]
        value_range = max(recent_values) - min(recent_values)
        stability_factor = 1.0 - min(1.0, value_range / 0.1)
        factors.append(stability_factor)
        
        # Factor 3: Time since improvement
        time_factor = min(1.0, self._get_plateau_duration() / 50)
        factors.append(time_factor)
        
        return sum(factors) / len(factors) if factors else 0.0


class MetricCollector:
    """Collects and aggregates metrics."""
    
    def __init__(self):
        self.metrics = defaultdict(lambda: deque(maxlen=1000))
        self.metric_summaries = {}
        self.collection_interval = 1.0
        self.last_collection = {}
        self.aggregators = {
            MetricType.COVERAGE: self._aggregate_standard,
            MetricType.PANELS_PLACED: self._aggregate_sum,
            MetricType.WASTE: self._aggregate_standard,
            MetricType.ITERATION_TIME: self._aggregate_average,
            MetricType.MEMORY_USAGE: self._aggregate_max,
            MetricType.SOLUTION_QUALITY: self._aggregate_standard,
            MetricType.GAP_SIZE: self._aggregate_min,
            MetricType.FRAGMENTATION: self._aggregate_standard
        }
    
    def collect(self, metric_type: MetricType, value: float, timestamp: Optional[float] = None):
        """Collect a metric value."""
        if timestamp is None:
            timestamp = time.time()
        
        # Check collection interval
        if metric_type in self.last_collection:
            if timestamp - self.last_collection[metric_type] < self.collection_interval:
                return
        
        self.last_collection[metric_type] = timestamp
        
        # Store metric
        self.metrics[metric_type].append({
            'value': value,
            'timestamp': timestamp
        })
        
        # Update summary
        self._update_summary(metric_type)
    
    def _update_summary(self, metric_type: MetricType):
        """Update metric summary."""
        if metric_type not in self.metrics or not self.metrics[metric_type]:
            return
        
        values = [m['value'] for m in self.metrics[metric_type]]
        
        # Calculate trend
        trend = "stable"
        if len(values) >= 10:
            recent = values[-10:]
            older = values[-20:-10] if len(values) >= 20 else values[:10]
            
            recent_avg = sum(recent) / len(recent)
            older_avg = sum(older) / len(older)
            
            if recent_avg > older_avg * 1.05:
                trend = "improving"
            elif recent_avg < older_avg * 0.95:
                trend = "worsening"
        
        self.metric_summaries[metric_type] = MetricSummary(
            metric_type=metric_type,
            current_value=values[-1],
            best_value=max(values) if metric_type != MetricType.WASTE else min(values),
            worst_value=min(values) if metric_type != MetricType.WASTE else max(values),
            average_value=sum(values) / len(values),
            std_deviation=np.std(values) if len(values) > 1 else 0,
            trend=trend,
            samples=len(values)
        )
    
    def _aggregate_standard(self, values: List[float]) -> float:
        """Standard aggregation (average)."""
        return sum(values) / len(values) if values else 0
    
    def _aggregate_sum(self, values: List[float]) -> float:
        """Sum aggregation."""
        return sum(values)
    
    def _aggregate_average(self, values: List[float]) -> float:
        """Average aggregation."""
        return sum(values) / len(values) if values else 0
    
    def _aggregate_max(self, values: List[float]) -> float:
        """Maximum aggregation."""
        return max(values) if values else 0
    
    def _aggregate_min(self, values: List[float]) -> float:
        """Minimum aggregation."""
        return min(values) if values else 0
    
    def get_metric(self, metric_type: MetricType) -> Optional[MetricSummary]:
        """Get metric summary."""
        return self.metric_summaries.get(metric_type)
    
    def get_recent_values(self, metric_type: MetricType, count: int = 10) -> List[float]:
        """Get recent metric values."""
        if metric_type not in self.metrics:
            return []
        
        recent = list(self.metrics[metric_type])[-count:]
        return [m['value'] for m in recent]
    
    def get_time_series(self, metric_type: MetricType) -> List[Tuple[float, float]]:
        """Get time series data for metric."""
        if metric_type not in self.metrics:
            return []
        
        return [(m['timestamp'], m['value']) for m in self.metrics[metric_type]]
    
    def clear_metrics(self, metric_type: Optional[MetricType] = None):
        """Clear collected metrics."""
        if metric_type:
            self.metrics[metric_type].clear()
            if metric_type in self.metric_summaries:
                del self.metric_summaries[metric_type]
        else:
            self.metrics.clear()
            self.metric_summaries.clear()
    
    def export_metrics(self) -> Dict[str, Any]:
        """Export all metrics."""
        return {
            metric_type.value: {
                'summary': {
                    'current': summary.current_value,
                    'best': summary.best_value,
                    'worst': summary.worst_value,
                    'average': summary.average_value,
                    'std_dev': summary.std_deviation,
                    'trend': summary.trend,
                    'samples': summary.samples
                },
                'recent': self.get_recent_values(metric_type)
            }
            for metric_type, summary in self.metric_summaries.items()
        }


class ProgressReporter:
    """Reports optimization progress."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.report_interval = self.config.get('report_interval', 10)
        self.verbose = self.config.get('verbose', True)
        self.output_format = self.config.get('format', 'text')  # text, json, csv
        
        self.iteration_count = 0
        self.last_report_time = time.time()
        self.start_time = time.time()
        self.report_history = []
        self.listeners = []
    
    def report(self, snapshot: ProgressSnapshot):
        """Generate and deliver progress report."""
        self.iteration_count += 1
        
        # Check if should report
        if not self._should_report():
            return
        
        # Generate report
        report = self._generate_report(snapshot)
        
        # Store history
        self.report_history.append(report)
        
        # Output report
        self._output_report(report)
        
        # Notify listeners
        self._notify_listeners(report)
        
        self.last_report_time = time.time()
    
    def _should_report(self) -> bool:
        """Check if should generate report."""
        if self.iteration_count % self.report_interval == 0:
            return True
        
        if time.time() - self.last_report_time > 5.0:  # Time-based reporting
            return True
        
        return False
    
    def _generate_report(self, snapshot: ProgressSnapshot) -> Dict[str, Any]:
        """Generate progress report."""
        elapsed = time.time() - self.start_time
        
        report = {
            'timestamp': time.time(),
            'iteration': snapshot.iteration,
            'elapsed_time': elapsed,
            'strategy': snapshot.strategy.value,
            'coverage': snapshot.coverage,
            'panels_placed': snapshot.panels_placed,
            'convergence': snapshot.convergence_status.value,
            'improvement_rate': snapshot.improvement_rate,
            'estimated_completion': snapshot.estimated_completion,
            'metrics': {
                k.value: v for k, v in snapshot.metrics.items()
            }
        }
        
        # Add derived metrics
        report['iterations_per_second'] = self.iteration_count / elapsed if elapsed > 0 else 0
        report['coverage_per_minute'] = (snapshot.coverage * 60) / elapsed if elapsed > 0 else 0
        
        return report
    
    def _output_report(self, report: Dict[str, Any]):
        """Output report in specified format."""
        if not self.verbose:
            return
        
        if self.output_format == 'json':
            print(json.dumps(report, indent=2))
        elif self.output_format == 'csv':
            # CSV format
            if self.iteration_count == 1:
                print(','.join(report.keys()))
            values = [str(v) for v in report.values()]
            print(','.join(values))
        else:  # text
            print(f"\n{'='*60}")
            print(f"Progress Report - Iteration {report['iteration']}")
            print(f"{'='*60}")
            print(f"Time Elapsed: {report['elapsed_time']:.2f}s")
            print(f"Strategy: {report['strategy']}")
            print(f"Coverage: {report['coverage']:.2%}")
            print(f"Panels Placed: {report['panels_placed']}")
            print(f"Convergence: {report['convergence']}")
            print(f"Improvement Rate: {report['improvement_rate']:.4f}/s")
            if report['estimated_completion']:
                print(f"Est. Completion: {report['estimated_completion']:.1f}s")
            print(f"Iterations/sec: {report['iterations_per_second']:.2f}")
            print(f"{'='*60}")
    
    def _notify_listeners(self, report: Dict[str, Any]):
        """Notify report listeners."""
        for listener in self.listeners:
            try:
                listener(report)
            except Exception as e:
                print(f"Report listener error: {e}")
    
    def add_listener(self, listener: Callable):
        """Add report listener."""
        self.listeners.append(listener)
    
    def final_report(self, best_state: PackingState, coverage: float):
        """Generate final report."""
        elapsed = time.time() - self.start_time
        
        report = {
            'status': 'completed',
            'total_time': elapsed,
            'total_iterations': self.iteration_count,
            'final_coverage': coverage,
            'panels_placed': len(best_state.placed_panels) if best_state else 0,
            'average_iteration_time': elapsed / self.iteration_count if self.iteration_count > 0 else 0,
            'reports_generated': len(self.report_history)
        }
        
        print(f"\n{'='*60}")
        print("FINAL REPORT")
        print(f"{'='*60}")
        print(f"Total Time: {report['total_time']:.2f}s")
        print(f"Total Iterations: {report['total_iterations']}")
        print(f"Final Coverage: {report['final_coverage']:.2%}")
        print(f"Panels Placed: {report['panels_placed']}")
        print(f"Avg Iteration Time: {report['average_iteration_time']:.4f}s")
        print(f"{'='*60}\n")
        
        return report


class ProgressTrackingSystem:
    """Main progress tracking system."""
    
    def __init__(self, 
                target_coverage: float = 0.95,
                config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.target_coverage = target_coverage
        
        # Initialize components
        self.coverage_monitor = CoverageMonitor(target_coverage)
        self.convergence_detector = ConvergenceDetector(config)
        self.metric_collector = MetricCollector()
        self.progress_reporter = ProgressReporter(config)
        
        self.iteration = 0
        self.current_strategy = None
        self.snapshots = deque(maxlen=1000)
    
    def update(self,
              state: PackingState,
              strategy: StrategyType,
              metrics: Optional[Dict[MetricType, float]] = None) -> ProgressSnapshot:
        """Update progress tracking."""
        self.iteration += 1
        self.current_strategy = strategy
        
        # Update coverage
        coverage = self.coverage_monitor.update(state, strategy)
        
        # Detect convergence
        convergence_info = self.convergence_detector.update(coverage, self.iteration)
        
        # Collect metrics
        self.metric_collector.collect(MetricType.COVERAGE, coverage)
        self.metric_collector.collect(MetricType.PANELS_PLACED, len(state.placed_panels))
        
        if metrics:
            for metric_type, value in metrics.items():
                self.metric_collector.collect(metric_type, value)
        
        # Create snapshot
        snapshot = ProgressSnapshot(
            timestamp=time.time(),
            iteration=self.iteration,
            strategy=strategy,
            coverage=coverage,
            panels_placed=len(state.placed_panels),
            convergence_status=convergence_info.status,
            improvement_rate=self.coverage_monitor.get_improvement_rate(),
            estimated_completion=self.coverage_monitor.estimate_completion_time(),
            metrics=metrics or {}
        )
        
        self.snapshots.append(snapshot)
        
        # Report progress
        self.progress_reporter.report(snapshot)
        
        return snapshot
    
    def is_converged(self) -> bool:
        """Check if optimization has converged."""
        if not self.snapshots:
            return False
        
        latest = self.snapshots[-1]
        return latest.convergence_status in [
            ConvergenceStatus.CONVERGED,
            ConvergenceStatus.PLATEAUED
        ]
    
    def should_terminate(self) -> bool:
        """Check if should terminate optimization."""
        # Check target reached
        if self.coverage_monitor.best_coverage >= self.target_coverage:
            return True
        
        # Check convergence with no hope of improvement
        if self.is_converged():
            latest = self.snapshots[-1]
            if latest.convergence_status == ConvergenceStatus.CONVERGED:
                if self.convergence_detector.best_value < self.target_coverage * 0.9:
                    return True
        
        return False
    
    def get_best_state(self) -> Optional[PackingState]:
        """Get best state found."""
        return self.coverage_monitor.best_state
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            'iteration': self.iteration,
            'coverage': self.coverage_monitor.get_statistics(),
            'convergence': {
                'status': self.snapshots[-1].convergence_status.value if self.snapshots else None,
                'confidence': self.convergence_detector._calculate_confidence()
            },
            'metrics': self.metric_collector.export_metrics(),
            'progress': {
                'target_coverage': self.target_coverage,
                'best_coverage': self.coverage_monitor.best_coverage,
                'distance_to_target': self.coverage_monitor.get_distance_to_target(),
                'estimated_completion': self.coverage_monitor.estimate_completion_time()
            }
        }
    
    def finalize(self):
        """Finalize tracking and generate final report."""
        if self.coverage_monitor.best_state:
            self.progress_reporter.final_report(
                self.coverage_monitor.best_state,
                self.coverage_monitor.best_coverage
            )
        
        return self.get_statistics()
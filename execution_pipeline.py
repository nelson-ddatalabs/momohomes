#!/usr/bin/env python3
"""
execution_pipeline.py - Execution Pipeline System
=================================================
Production-ready strategy sequencer, result validator, transition handler,
and pipeline state manager for orchestrating optimization algorithms.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from collections import deque
from enum import Enum
import time
import threading
import queue
import traceback

from models import Room, PanelSize
from advanced_packing import PackingState, PanelPlacement
from strategy_selection import StrategyType


class PipelineState(Enum):
    """Pipeline execution states."""
    IDLE = "idle"
    INITIALIZING = "initializing"
    RUNNING = "running"
    TRANSITIONING = "transitioning"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TransitionType(Enum):
    """Types of strategy transitions."""
    INITIAL = "initial"  # First strategy
    FALLBACK = "fallback"  # Failed, trying next
    IMPROVEMENT = "improvement"  # Seeking better result
    TIMEOUT = "timeout"  # Time limit reached
    COMPLETION = "completion"  # Target achieved
    FORCED = "forced"  # Manual transition


@dataclass
class ExecutionContext:
    """Context for strategy execution."""
    room: Room
    panels: List[PanelSize]
    strategy: StrategyType
    config: Dict[str, Any]
    time_budget: float
    target_coverage: float
    current_state: Optional[PackingState] = None
    best_state: Optional[PackingState] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """Result from strategy execution."""
    strategy: StrategyType
    success: bool
    coverage: float
    time_taken: float
    iterations: int
    final_state: Optional[PackingState]
    validation_passed: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result from solution validation."""
    is_valid: bool
    coverage: float
    violations: List[str]
    warnings: List[str]
    metrics: Dict[str, float]


@dataclass
class TransitionDecision:
    """Decision about strategy transition."""
    should_transition: bool
    transition_type: TransitionType
    next_strategy: Optional[StrategyType]
    reasoning: str


class StrategySequencer:
    """Sequences strategy execution."""
    
    def __init__(self, strategies: List[StrategyType]):
        self.strategies = strategies
        self.current_index = 0
        self.execution_history = []
        self.transition_history = []
    
    def get_current_strategy(self) -> Optional[StrategyType]:
        """Get current strategy."""
        if 0 <= self.current_index < len(self.strategies):
            return self.strategies[self.current_index]
        return None
    
    def get_next_strategy(self) -> Optional[StrategyType]:
        """Get next strategy in sequence."""
        if self.current_index + 1 < len(self.strategies):
            return self.strategies[self.current_index + 1]
        return None
    
    def advance(self) -> bool:
        """Advance to next strategy."""
        if self.current_index < len(self.strategies) - 1:
            self.current_index += 1
            return True
        return False
    
    def reset(self):
        """Reset sequencer."""
        self.current_index = 0
        self.execution_history.clear()
        self.transition_history.clear()
    
    def record_execution(self, result: ExecutionResult):
        """Record execution result."""
        self.execution_history.append({
            'strategy': result.strategy,
            'success': result.success,
            'coverage': result.coverage,
            'time': result.time_taken,
            'timestamp': time.time()
        })
    
    def record_transition(self, decision: TransitionDecision):
        """Record transition decision."""
        self.transition_history.append({
            'type': decision.transition_type,
            'from_strategy': self.get_current_strategy(),
            'to_strategy': decision.next_strategy,
            'reasoning': decision.reasoning,
            'timestamp': time.time()
        })
    
    def should_continue(self, context: ExecutionContext) -> bool:
        """Check if sequencing should continue."""
        # Check if we have more strategies
        if self.current_index >= len(self.strategies) - 1:
            return False
        
        # Check time budget
        total_time = sum(h['time'] for h in self.execution_history)
        if total_time >= context.time_budget:
            return False
        
        # Check if target achieved
        if context.best_state and context.best_state.coverage_ratio >= context.target_coverage:
            return False
        
        return True


class ResultValidator:
    """Validates optimization results."""
    
    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance
        self.validation_rules = [
            self._validate_no_overlap,
            self._validate_boundaries,
            self._validate_coverage,
            self._validate_panel_counts,
            self._validate_structural_integrity
        ]
    
    def validate(self, 
                state: PackingState,
                room: Room,
                panels: List[PanelSize]) -> ValidationResult:
        """Validate a packing solution."""
        violations = []
        warnings = []
        metrics = {}
        
        # Run validation rules
        for rule in self.validation_rules:
            rule_violations, rule_warnings, rule_metrics = rule(state, room, panels)
            violations.extend(rule_violations)
            warnings.extend(rule_warnings)
            metrics.update(rule_metrics)
        
        # Calculate coverage
        placed_area = sum(
            p.panel_size.width * p.panel_size.height 
            for p in state.placed_panels
        )
        room_area = room.width * room.height
        coverage = placed_area / room_area if room_area > 0 else 0
        
        metrics['coverage'] = coverage
        metrics['panels_placed'] = len(state.placed_panels)
        metrics['utilization'] = coverage
        
        return ValidationResult(
            is_valid=len(violations) == 0,
            coverage=coverage,
            violations=violations,
            warnings=warnings,
            metrics=metrics
        )
    
    def _validate_no_overlap(self,
                            state: PackingState,
                            room: Room,
                            panels: List[PanelSize]) -> Tuple[List[str], List[str], Dict[str, float]]:
        """Validate no panels overlap."""
        violations = []
        warnings = []
        metrics = {}
        
        placements = state.placed_panels
        overlap_count = 0
        
        for i, p1 in enumerate(placements):
            for p2 in placements[i+1:]:
                if self._panels_overlap(p1, p2):
                    overlap_count += 1
                    violations.append(
                        f"Overlap between panels at {p1.position} and {p2.position}"
                    )
        
        metrics['overlap_count'] = overlap_count
        
        return violations, warnings, metrics
    
    def _panels_overlap(self, p1: PanelPlacement, p2: PanelPlacement) -> bool:
        """Check if two panels overlap."""
        x1, y1 = p1.position
        x2, y2 = p2.position
        
        return not (
            x1 + p1.panel_size.width <= x2 + self.tolerance or
            x2 + p2.panel_size.width <= x1 + self.tolerance or
            y1 + p1.panel_size.height <= y2 + self.tolerance or
            y2 + p2.panel_size.height <= y1 + self.tolerance
        )
    
    def _validate_boundaries(self,
                            state: PackingState,
                            room: Room,
                            panels: List[PanelSize]) -> Tuple[List[str], List[str], Dict[str, float]]:
        """Validate panels are within room boundaries."""
        violations = []
        warnings = []
        metrics = {}
        
        out_of_bounds = 0
        
        for p in state.placed_panels:
            x, y = p.position
            
            if x < -self.tolerance or y < -self.tolerance:
                violations.append(f"Panel at {p.position} outside room (negative coords)")
                out_of_bounds += 1
            elif (x + p.panel_size.width > room.width + self.tolerance or
                  y + p.panel_size.height > room.height + self.tolerance):
                violations.append(f"Panel at {p.position} exceeds room boundaries")
                out_of_bounds += 1
        
        metrics['out_of_bounds'] = out_of_bounds
        
        return violations, warnings, metrics
    
    def _validate_coverage(self,
                          state: PackingState,
                          room: Room,
                          panels: List[PanelSize]) -> Tuple[List[str], List[str], Dict[str, float]]:
        """Validate coverage metrics."""
        violations = []
        warnings = []
        metrics = {}
        
        # Calculate actual vs claimed coverage
        actual_area = sum(
            p.panel_size.width * p.panel_size.height 
            for p in state.placed_panels
        )
        room_area = room.width * room.height
        actual_coverage = actual_area / room_area if room_area > 0 else 0
        
        # Check consistency with state's coverage
        if hasattr(state, 'coverage_ratio'):
            if abs(state.coverage_ratio - actual_coverage) > 0.01:
                warnings.append(
                    f"Coverage mismatch: state={state.coverage_ratio:.3f}, "
                    f"actual={actual_coverage:.3f}"
                )
        
        metrics['actual_coverage'] = actual_coverage
        metrics['coverage_gap'] = 1.0 - actual_coverage
        
        return violations, warnings, metrics
    
    def _validate_panel_counts(self,
                              state: PackingState,
                              room: Room,
                              panels: List[PanelSize]) -> Tuple[List[str], List[str], Dict[str, float]]:
        """Validate panel quantity constraints."""
        violations = []
        warnings = []
        metrics = {}
        
        # Count used panels by type
        used_counts = {}
        for p in state.placed_panels:
            panel_id = p.panel_size.id
            used_counts[panel_id] = used_counts.get(panel_id, 0) + 1
        
        # Check against available quantities
        for panel in panels:
            used = used_counts.get(panel.id, 0)
            available = getattr(panel, 'quantity', float('inf'))
            
            if used > available:
                violations.append(
                    f"Panel {panel.id}: used {used} > available {available}"
                )
        
        metrics['total_panels_used'] = len(state.placed_panels)
        metrics['unique_panel_types'] = len(used_counts)
        
        return violations, warnings, metrics
    
    def _validate_structural_integrity(self,
                                      state: PackingState,
                                      room: Room,
                                      panels: List[PanelSize]) -> Tuple[List[str], List[str], Dict[str, float]]:
        """Validate structural integrity of placement."""
        violations = []
        warnings = []
        metrics = {}
        
        # Check for floating panels (simplified check)
        supported_panels = 0
        for p in state.placed_panels:
            x, y = p.position
            # Panel is supported if at floor level or has support below
            if y <= self.tolerance:  # On floor
                supported_panels += 1
            else:
                # Check if supported by another panel (simplified)
                has_support = False
                for other in state.placed_panels:
                    if other == p:
                        continue
                    ox, oy = other.position
                    # Check if other panel is below and overlaps horizontally
                    if (oy + other.panel_size.height >= y - self.tolerance and
                        oy < y and
                        not (x + p.panel_size.width <= ox or
                             ox + other.panel_size.width <= x)):
                        has_support = True
                        break
                
                if has_support:
                    supported_panels += 1
                else:
                    warnings.append(f"Panel at {p.position} may lack structural support")
        
        metrics['supported_panels'] = supported_panels
        metrics['support_ratio'] = supported_panels / len(state.placed_panels) if state.placed_panels else 1.0
        
        return violations, warnings, metrics


class TransitionHandler:
    """Handles transitions between strategies."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.min_improvement = self.config.get('min_improvement', 0.05)
        self.patience = self.config.get('patience', 3)
        self.no_improvement_count = 0
        self.last_best_coverage = 0.0
    
    def decide_transition(self,
                         current_result: ExecutionResult,
                         context: ExecutionContext,
                         sequencer: StrategySequencer) -> TransitionDecision:
        """Decide whether to transition to another strategy."""
        
        # Check if current strategy failed
        if not current_result.success:
            return self._handle_failure(current_result, context, sequencer)
        
        # Check if target achieved
        if current_result.coverage >= context.target_coverage:
            return TransitionDecision(
                should_transition=False,
                transition_type=TransitionType.COMPLETION,
                next_strategy=None,
                reasoning=f"Target coverage {context.target_coverage:.1%} achieved"
            )
        
        # Check for improvement
        improvement = current_result.coverage - self.last_best_coverage
        if improvement >= self.min_improvement:
            self.last_best_coverage = current_result.coverage
            self.no_improvement_count = 0
            
            # Continue with current strategy if improving
            return TransitionDecision(
                should_transition=False,
                transition_type=TransitionType.IMPROVEMENT,
                next_strategy=None,
                reasoning=f"Strategy improving (gained {improvement:.1%})"
            )
        else:
            self.no_improvement_count += 1
        
        # Check patience
        if self.no_improvement_count >= self.patience:
            return self._handle_stagnation(current_result, context, sequencer)
        
        # Continue with current strategy
        return TransitionDecision(
            should_transition=False,
            transition_type=None,
            next_strategy=None,
            reasoning="Continuing with current strategy"
        )
    
    def _handle_failure(self,
                       result: ExecutionResult,
                       context: ExecutionContext,
                       sequencer: StrategySequencer) -> TransitionDecision:
        """Handle strategy failure."""
        next_strategy = sequencer.get_next_strategy()
        
        if next_strategy:
            return TransitionDecision(
                should_transition=True,
                transition_type=TransitionType.FALLBACK,
                next_strategy=next_strategy,
                reasoning=f"Strategy {result.strategy.value} failed: {result.error_message}"
            )
        else:
            return TransitionDecision(
                should_transition=False,
                transition_type=TransitionType.COMPLETION,
                next_strategy=None,
                reasoning="No more fallback strategies available"
            )
    
    def _handle_stagnation(self,
                          result: ExecutionResult,
                          context: ExecutionContext,
                          sequencer: StrategySequencer) -> TransitionDecision:
        """Handle stagnating performance."""
        next_strategy = sequencer.get_next_strategy()
        
        if next_strategy:
            self.no_improvement_count = 0
            return TransitionDecision(
                should_transition=True,
                transition_type=TransitionType.IMPROVEMENT,
                next_strategy=next_strategy,
                reasoning=f"No improvement for {self.patience} iterations"
            )
        else:
            return TransitionDecision(
                should_transition=False,
                transition_type=None,
                next_strategy=None,
                reasoning="No alternative strategies available"
            )
    
    def handle_timeout(self,
                      context: ExecutionContext,
                      sequencer: StrategySequencer) -> TransitionDecision:
        """Handle timeout situation."""
        return TransitionDecision(
            should_transition=False,
            transition_type=TransitionType.TIMEOUT,
            next_strategy=None,
            reasoning=f"Time budget of {context.time_budget}s exceeded"
        )
    
    def reset(self):
        """Reset transition handler state."""
        self.no_improvement_count = 0
        self.last_best_coverage = 0.0


class PipelineStateManager:
    """Manages overall pipeline state."""
    
    def __init__(self):
        self.state = PipelineState.IDLE
        self.state_history = []
        self.state_lock = threading.Lock()
        self.listeners = []
        self.metadata = {}
        self.start_time = None
        self.end_time = None
    
    def transition_to(self, new_state: PipelineState, metadata: Optional[Dict[str, Any]] = None):
        """Transition to new state."""
        with self.state_lock:
            old_state = self.state
            self.state = new_state
            
            # Record transition
            self.state_history.append({
                'from': old_state,
                'to': new_state,
                'timestamp': time.time(),
                'metadata': metadata or {}
            })
            
            # Update timing
            if new_state == PipelineState.RUNNING and self.start_time is None:
                self.start_time = time.time()
            elif new_state in [PipelineState.COMPLETED, PipelineState.FAILED, PipelineState.CANCELLED]:
                self.end_time = time.time()
            
            # Notify listeners
            self._notify_listeners(old_state, new_state, metadata)
    
    def get_state(self) -> PipelineState:
        """Get current state."""
        with self.state_lock:
            return self.state
    
    def is_terminal(self) -> bool:
        """Check if in terminal state."""
        return self.state in [
            PipelineState.COMPLETED,
            PipelineState.FAILED,
            PipelineState.CANCELLED
        ]
    
    def can_transition_to(self, target_state: PipelineState) -> bool:
        """Check if transition to target state is valid."""
        current = self.state
        
        # Define valid transitions
        valid_transitions = {
            PipelineState.IDLE: [PipelineState.INITIALIZING],
            PipelineState.INITIALIZING: [PipelineState.RUNNING, PipelineState.FAILED],
            PipelineState.RUNNING: [
                PipelineState.TRANSITIONING,
                PipelineState.VALIDATING,
                PipelineState.COMPLETED,
                PipelineState.FAILED,
                PipelineState.CANCELLED
            ],
            PipelineState.TRANSITIONING: [PipelineState.RUNNING, PipelineState.FAILED],
            PipelineState.VALIDATING: [PipelineState.COMPLETED, PipelineState.FAILED],
            PipelineState.COMPLETED: [],
            PipelineState.FAILED: [],
            PipelineState.CANCELLED: []
        }
        
        return target_state in valid_transitions.get(current, [])
    
    def add_listener(self, listener: Callable):
        """Add state change listener."""
        self.listeners.append(listener)
    
    def _notify_listeners(self,
                         old_state: PipelineState,
                         new_state: PipelineState,
                         metadata: Optional[Dict[str, Any]]):
        """Notify listeners of state change."""
        for listener in self.listeners:
            try:
                listener(old_state, new_state, metadata)
            except Exception as e:
                # Log but don't fail
                print(f"Listener error: {e}")
    
    def get_elapsed_time(self) -> float:
        """Get elapsed execution time."""
        if self.start_time is None:
            return 0.0
        
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            'current_state': self.state.value,
            'elapsed_time': self.get_elapsed_time(),
            'state_transitions': len(self.state_history),
            'is_terminal': self.is_terminal(),
            'metadata': self.metadata
        }


class ExecutionPipeline:
    """Main execution pipeline orchestrator."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize components
        self.sequencer = None
        self.validator = ResultValidator(
            tolerance=self.config.get('validation_tolerance', 1e-6)
        )
        self.transition_handler = TransitionHandler(config)
        self.state_manager = PipelineStateManager()
        
        # Execution state
        self.current_context = None
        self.execution_thread = None
        self.stop_requested = False
        
        # Results
        self.results = []
        self.best_result = None
    
    def initialize(self,
                  room: Room,
                  panels: List[PanelSize],
                  strategies: List[StrategyType],
                  time_budget: float = 300.0,
                  target_coverage: float = 0.95) -> bool:
        """Initialize pipeline for execution."""
        try:
            self.state_manager.transition_to(PipelineState.INITIALIZING)
            
            # Create sequencer
            self.sequencer = StrategySequencer(strategies)
            
            # Create context
            self.current_context = ExecutionContext(
                room=room,
                panels=panels,
                strategy=strategies[0] if strategies else None,
                config=self.config,
                time_budget=time_budget,
                target_coverage=target_coverage
            )
            
            # Reset components
            self.transition_handler.reset()
            self.results.clear()
            self.best_result = None
            self.stop_requested = False
            
            return True
            
        except Exception as e:
            self.state_manager.transition_to(
                PipelineState.FAILED,
                {'error': str(e)}
            )
            return False
    
    def execute_async(self) -> threading.Thread:
        """Execute pipeline asynchronously."""
        self.execution_thread = threading.Thread(target=self._execute)
        self.execution_thread.start()
        return self.execution_thread
    
    def execute_sync(self) -> Optional[ExecutionResult]:
        """Execute pipeline synchronously."""
        return self._execute()
    
    def _execute(self) -> Optional[ExecutionResult]:
        """Main execution loop."""
        if not self.current_context or not self.sequencer:
            return None
        
        self.state_manager.transition_to(PipelineState.RUNNING)
        
        try:
            while not self.stop_requested and not self.state_manager.is_terminal():
                # Get current strategy
                strategy = self.sequencer.get_current_strategy()
                if not strategy:
                    break
                
                # Update context
                self.current_context.strategy = strategy
                
                # Execute strategy (placeholder - would call actual optimizer)
                result = self._execute_strategy(strategy, self.current_context)
                
                # Validate result
                if result.final_state:
                    validation = self.validator.validate(
                        result.final_state,
                        self.current_context.room,
                        self.current_context.panels
                    )
                    result.validation_passed = validation.is_valid
                
                # Record result
                self.results.append(result)
                self.sequencer.record_execution(result)
                
                # Update best result
                if result.validation_passed:
                    if not self.best_result or result.coverage > self.best_result.coverage:
                        self.best_result = result
                        self.current_context.best_state = result.final_state
                
                # Check transition
                self.state_manager.transition_to(PipelineState.TRANSITIONING)
                decision = self.transition_handler.decide_transition(
                    result,
                    self.current_context,
                    self.sequencer
                )
                
                if decision.should_transition:
                    self.sequencer.record_transition(decision)
                    if decision.next_strategy:
                        self.sequencer.advance()
                    else:
                        break
                elif decision.transition_type == TransitionType.COMPLETION:
                    break
                
                self.state_manager.transition_to(PipelineState.RUNNING)
                
                # Check time budget
                if self.state_manager.get_elapsed_time() >= self.current_context.time_budget:
                    decision = self.transition_handler.handle_timeout(
                        self.current_context,
                        self.sequencer
                    )
                    self.sequencer.record_transition(decision)
                    break
            
            # Final validation
            if self.best_result:
                self.state_manager.transition_to(PipelineState.VALIDATING)
                final_validation = self.validator.validate(
                    self.best_result.final_state,
                    self.current_context.room,
                    self.current_context.panels
                )
                
                if final_validation.is_valid:
                    self.state_manager.transition_to(
                        PipelineState.COMPLETED,
                        {'coverage': self.best_result.coverage}
                    )
                else:
                    self.state_manager.transition_to(
                        PipelineState.FAILED,
                        {'reason': 'Final validation failed', 'violations': final_validation.violations}
                    )
            else:
                self.state_manager.transition_to(
                    PipelineState.FAILED,
                    {'reason': 'No valid solution found'}
                )
            
            return self.best_result
            
        except Exception as e:
            self.state_manager.transition_to(
                PipelineState.FAILED,
                {'error': str(e), 'traceback': traceback.format_exc()}
            )
            return None
    
    def _execute_strategy(self,
                         strategy: StrategyType,
                         context: ExecutionContext) -> ExecutionResult:
        """Execute a single strategy (placeholder)."""
        # This would call the actual optimizer
        # For now, return a mock result
        import random
        
        time.sleep(0.1)  # Simulate execution
        
        coverage = random.uniform(0.7, 0.98)
        success = random.random() > 0.2
        
        # Create mock state
        mock_state = PackingState()
        mock_state.coverage_ratio = coverage
        mock_state.placed_panels = []
        
        return ExecutionResult(
            strategy=strategy,
            success=success,
            coverage=coverage,
            time_taken=0.1,
            iterations=random.randint(10, 100),
            final_state=mock_state if success else None,
            validation_passed=False,
            error_message=None if success else "Mock failure"
        )
    
    def cancel(self):
        """Cancel pipeline execution."""
        self.stop_requested = True
        if self.state_manager.get_state() == PipelineState.RUNNING:
            self.state_manager.transition_to(PipelineState.CANCELLED)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics."""
        return {
            'state': self.state_manager.get_statistics(),
            'results': len(self.results),
            'best_coverage': self.best_result.coverage if self.best_result else 0.0,
            'strategies_tried': len(set(r.strategy for r in self.results)),
            'total_time': self.state_manager.get_elapsed_time(),
            'validation_passes': sum(1 for r in self.results if r.validation_passed),
            'execution_history': self.sequencer.execution_history if self.sequencer else [],
            'transition_history': self.sequencer.transition_history if self.sequencer else []
        }
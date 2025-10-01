from typing import List, Dict, Optional, Set, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from datetime import datetime
import logging
import hashlib
import random
import traceback
import sys
from decimal import Decimal, ROUND_HALF_UP
from collections import defaultdict
import json
import pickle

from core import PackingState, Panel, Room, Position, PlacedPanel
from algorithm_interface import OptimizerConfig, OptimizerResult

logger = logging.getLogger(__name__)


class StabilityIssueType(Enum):
    FLOATING_POINT_ERROR = "floating_point_error"
    TIE_BREAKING = "tie_breaking"
    NON_DETERMINISTIC = "non_deterministic"
    OVERFLOW = "overflow"
    UNDERFLOW = "underflow"
    DIVISION_BY_ZERO = "division_by_zero"
    INVALID_STATE = "invalid_state"
    MEMORY_ERROR = "memory_error"


@dataclass
class StabilityIssue:
    issue_type: StabilityIssueType
    location: str
    description: str
    severity: str  # "low", "medium", "high", "critical"
    resolution: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ErrorContext:
    function_name: str
    input_data: Dict[str, Any]
    error_type: str
    error_message: str
    stack_trace: str
    recovery_action: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ReproducibilityConfig:
    seed: int
    precision_digits: int
    tie_breaking_rule: str
    deterministic_mode: bool
    error_handling_mode: str  # "strict", "recover", "fallback"


class NumericalStability:
    """Ensures numerical stability in calculations"""
    
    def __init__(self, epsilon: float = 1e-10, precision: int = 10):
        self.epsilon = epsilon
        self.precision = precision
        self.overflow_threshold = 1e100
        self.underflow_threshold = 1e-100
    
    def safe_divide(self, numerator: float, denominator: float,
                   default: float = 0.0) -> float:
        """Safe division with zero check"""
        if abs(denominator) < self.epsilon:
            logger.debug(f"Division by near-zero: {denominator}")
            return default
        
        result = numerator / denominator
        
        # Check for overflow/underflow
        if abs(result) > self.overflow_threshold:
            logger.warning(f"Overflow in division: {result}")
            return np.sign(result) * self.overflow_threshold
        
        if 0 < abs(result) < self.underflow_threshold:
            logger.debug(f"Underflow in division: {result}")
            return 0.0
        
        return result
    
    def safe_sqrt(self, value: float) -> float:
        """Safe square root with negative check"""
        if value < -self.epsilon:
            logger.warning(f"Square root of negative number: {value}")
            return 0.0
        
        # Handle small negative values due to floating point errors
        if value < 0:
            value = 0.0
        
        return np.sqrt(value)
    
    def safe_log(self, value: float, base: float = np.e) -> float:
        """Safe logarithm with positive check"""
        if value <= 0:
            logger.debug(f"Log of non-positive number: {value}")
            return -float('inf')
        
        if value < self.underflow_threshold:
            return np.log(self.underflow_threshold) / np.log(base)
        
        return np.log(value) / np.log(base)
    
    def round_to_precision(self, value: float) -> float:
        """Round to specified precision to avoid floating point errors"""
        return round(value, self.precision)
    
    def compare_floats(self, a: float, b: float) -> int:
        """Compare floats with epsilon tolerance"""
        if abs(a - b) < self.epsilon:
            return 0  # Equal
        elif a < b:
            return -1  # a < b
        else:
            return 1  # a > b
    
    def stabilize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Stabilize matrix for numerical operations"""
        # Check for NaN or Inf
        if np.any(np.isnan(matrix)) or np.any(np.isinf(matrix)):
            logger.warning("Matrix contains NaN or Inf values")
            matrix = np.nan_to_num(matrix, nan=0.0, posinf=self.overflow_threshold,
                                  neginf=-self.overflow_threshold)
        
        # Check condition number for numerical stability
        try:
            cond = np.linalg.cond(matrix)
            if cond > 1e10:
                logger.warning(f"Ill-conditioned matrix: condition number = {cond}")
                # Add small regularization
                matrix = matrix + np.eye(matrix.shape[0]) * self.epsilon
        except:
            pass
        
        return matrix
    
    def stable_sum(self, values: List[float]) -> float:
        """Kahan summation for improved precision"""
        if not values:
            return 0.0
        
        total = 0.0
        compensation = 0.0
        
        for value in values:
            y = value - compensation
            t = total + y
            compensation = (t - total) - y
            total = t
        
        return total
    
    def stable_mean(self, values: List[float]) -> float:
        """Numerically stable mean calculation"""
        if not values:
            return 0.0
        
        # Use Welford's algorithm for numerical stability
        mean = 0.0
        for i, value in enumerate(values):
            delta = value - mean
            mean += delta / (i + 1)
        
        return mean
    
    def stable_variance(self, values: List[float]) -> float:
        """Numerically stable variance calculation"""
        if len(values) < 2:
            return 0.0
        
        # Welford's algorithm
        mean = 0.0
        m2 = 0.0
        
        for i, value in enumerate(values):
            delta = value - mean
            mean += delta / (i + 1)
            delta2 = value - mean
            m2 += delta * delta2
        
        return m2 / (len(values) - 1)
    
    def clip_value(self, value: float, min_val: float = None,
                  max_val: float = None) -> float:
        """Clip value to prevent overflow/underflow"""
        if min_val is not None and value < min_val:
            return min_val
        if max_val is not None and value > max_val:
            return max_val
        return value


class DeterministicTieBreaker:
    """Ensures deterministic tie-breaking in algorithms"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)
        self.tie_count = 0
        self.tie_history = []
    
    def break_tie(self, items: List[Any], key_func: Callable[[Any], float],
                 secondary_key: Optional[Callable[[Any], Any]] = None) -> Any:
        """Break ties deterministically"""
        if not items:
            return None
        
        # Find best value
        best_value = key_func(items[0])
        tied_items = []
        
        for item in items:
            value = key_func(item)
            if abs(value - best_value) < 1e-10:  # Floating point equality
                tied_items.append(item)
            elif value > best_value:
                best_value = value
                tied_items = [item]
        
        self.tie_count += 1
        
        if len(tied_items) == 1:
            return tied_items[0]
        
        # Use secondary key if provided
        if secondary_key:
            tied_items.sort(key=secondary_key)
            return tied_items[0]
        
        # Use hash-based deterministic selection
        return self._deterministic_select(tied_items)
    
    def _deterministic_select(self, items: List[Any]) -> Any:
        """Select item deterministically using hash"""
        # Create stable hash for each item
        hashes = []
        for item in items:
            item_str = self._item_to_string(item)
            item_hash = hashlib.md5(item_str.encode()).hexdigest()
            hashes.append((item_hash, item))
        
        # Sort by hash and return first
        hashes.sort(key=lambda x: x[0])
        
        selected = hashes[0][1]
        self.tie_history.append({
            'items_count': len(items),
            'selected_hash': hashes[0][0],
            'timestamp': datetime.now()
        })
        
        return selected
    
    def _item_to_string(self, item: Any) -> str:
        """Convert item to stable string representation"""
        if hasattr(item, '__dict__'):
            # Sort dictionary keys for stability
            d = {k: v for k, v in sorted(item.__dict__.items())}
            return json.dumps(d, default=str, sort_keys=True)
        else:
            return str(item)
    
    def sort_deterministic(self, items: List[Any],
                          key_func: Callable[[Any], float]) -> List[Any]:
        """Sort with deterministic tie-breaking"""
        # Create decorated list with original indices
        decorated = [(key_func(item), i, item) for i, item in enumerate(items)]
        
        # Sort by value, then by original index for ties
        decorated.sort(key=lambda x: (x[0], x[1]))
        
        # Extract sorted items
        return [item for _, _, item in decorated]
    
    def choose_deterministic(self, items: List[Any],
                            weights: Optional[List[float]] = None) -> Any:
        """Choose item deterministically with optional weights"""
        if not items:
            return None
        
        if weights:
            # Weighted selection
            if len(weights) != len(items):
                raise ValueError("Weights must match items length")
            
            # Normalize weights
            total = sum(weights)
            if total <= 0:
                # All zero weights, choose first
                return items[0]
            
            normalized = [w / total for w in weights]
            
            # Use deterministic random choice
            rand_val = self.rng.random()
            cumsum = 0.0
            
            for item, weight in zip(items, normalized):
                cumsum += weight
                if rand_val <= cumsum:
                    return item
            
            return items[-1]
        else:
            # Uniform selection
            return items[self.rng.randint(0, len(items) - 1)]


class ReproducibilityManager:
    """Ensures reproducible results"""
    
    def __init__(self, config: ReproducibilityConfig):
        self.config = config
        self.execution_trace = []
        self.checkpoints = []
        
        # Set global seeds
        random.seed(config.seed)
        np.random.seed(config.seed)
        
        # Create deterministic RNG
        self.rng = random.Random(config.seed)
    
    def checkpoint(self, name: str, state: Any):
        """Create reproducibility checkpoint"""
        checkpoint = {
            'name': name,
            'timestamp': datetime.now(),
            'state_hash': self._compute_state_hash(state),
            'state_snapshot': self._create_snapshot(state)
        }
        
        self.checkpoints.append(checkpoint)
        
        logger.debug(f"Checkpoint '{name}': hash={checkpoint['state_hash'][:8]}")
    
    def _compute_state_hash(self, state: Any) -> str:
        """Compute deterministic hash of state"""
        try:
            # Convert to bytes
            if hasattr(state, '__dict__'):
                state_dict = {k: v for k, v in sorted(state.__dict__.items())}
                state_bytes = pickle.dumps(state_dict)
            else:
                state_bytes = pickle.dumps(state)
            
            # Compute hash
            return hashlib.sha256(state_bytes).hexdigest()
        except:
            # Fallback to string representation
            state_str = str(state)
            return hashlib.sha256(state_str.encode()).hexdigest()
    
    def _create_snapshot(self, state: Any) -> Any:
        """Create snapshot of state for reproducibility"""
        try:
            # Deep copy for mutable objects
            import copy
            return copy.deepcopy(state)
        except:
            # Fallback to simple copy
            return state
    
    def trace_execution(self, function_name: str, inputs: Dict[str, Any],
                       output: Any):
        """Trace function execution for reproducibility"""
        trace_entry = {
            'function': function_name,
            'input_hash': self._compute_state_hash(inputs),
            'output_hash': self._compute_state_hash(output),
            'timestamp': datetime.now()
        }
        
        self.execution_trace.append(trace_entry)
    
    def verify_reproducibility(self, other_trace: List[Dict]) -> bool:
        """Verify execution trace matches another"""
        if len(self.execution_trace) != len(other_trace):
            logger.warning(f"Trace length mismatch: {len(self.execution_trace)} vs {len(other_trace)}")
            return False
        
        for i, (our, their) in enumerate(zip(self.execution_trace, other_trace)):
            if our['function'] != their['function']:
                logger.warning(f"Function mismatch at step {i}: {our['function']} vs {their['function']}")
                return False
            
            if our['input_hash'] != their['input_hash']:
                logger.warning(f"Input hash mismatch at step {i}")
                return False
            
            if our['output_hash'] != their['output_hash']:
                logger.warning(f"Output hash mismatch at step {i}")
                return False
        
        return True
    
    def make_deterministic(self, func: Callable) -> Callable:
        """Decorator to make function deterministic"""
        import functools
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Reset RNG state
            self.rng.seed(self.config.seed)
            
            # Round floating point inputs
            processed_args = self._process_inputs(args)
            processed_kwargs = self._process_inputs(kwargs)
            
            # Execute function
            result = func(*processed_args, **processed_kwargs)
            
            # Process output
            processed_result = self._process_output(result)
            
            # Trace execution
            self.trace_execution(
                func.__name__,
                {'args': processed_args, 'kwargs': processed_kwargs},
                processed_result
            )
            
            return processed_result
        
        return wrapper
    
    def _process_inputs(self, inputs: Any) -> Any:
        """Process inputs for determinism"""
        if isinstance(inputs, float):
            # Round floats to specified precision
            return round(inputs, self.config.precision_digits)
        elif isinstance(inputs, (list, tuple)):
            return type(inputs)(self._process_inputs(item) for item in inputs)
        elif isinstance(inputs, dict):
            return {k: self._process_inputs(v) for k, v in sorted(inputs.items())}
        else:
            return inputs
    
    def _process_output(self, output: Any) -> Any:
        """Process output for determinism"""
        return self._process_inputs(output)  # Same processing as inputs
    
    def save_trace(self, filepath: str):
        """Save execution trace to file"""
        trace_data = {
            'config': {
                'seed': self.config.seed,
                'precision_digits': self.config.precision_digits,
                'tie_breaking_rule': self.config.tie_breaking_rule,
                'deterministic_mode': self.config.deterministic_mode
            },
            'trace': self.execution_trace,
            'checkpoints': [
                {
                    'name': cp['name'],
                    'state_hash': cp['state_hash'],
                    'timestamp': cp['timestamp'].isoformat()
                }
                for cp in self.checkpoints
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(trace_data, f, indent=2, default=str)
    
    def load_trace(self, filepath: str) -> List[Dict]:
        """Load execution trace from file"""
        with open(filepath, 'r') as f:
            trace_data = json.load(f)
        
        return trace_data['trace']


class ErrorRecovery:
    """Handles error recovery and fallback strategies"""
    
    def __init__(self):
        self.error_history = []
        self.recovery_strategies = {}
        self.fallback_functions = {}
        self.max_retries = 3
    
    def register_recovery(self, error_type: type,
                        recovery_func: Callable[[Exception, Any], Any]):
        """Register recovery strategy for error type"""
        self.recovery_strategies[error_type.__name__] = recovery_func
    
    def register_fallback(self, function_name: str,
                        fallback_func: Callable):
        """Register fallback function"""
        self.fallback_functions[function_name] = fallback_func
    
    def with_recovery(self, func: Callable) -> Callable:
        """Decorator to add error recovery to function"""
        import functools
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            last_error = None
            
            while retries < self.max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    
                    # Log error
                    self._log_error(func.__name__, e, args, kwargs)
                    
                    # Try recovery strategy
                    recovery_func = self.recovery_strategies.get(type(e).__name__)
                    if recovery_func:
                        try:
                            result = recovery_func(e, {'args': args, 'kwargs': kwargs})
                            if result is not None:
                                return result
                        except:
                            pass
                    
                    retries += 1
                    
                    # Exponential backoff
                    if retries < self.max_retries:
                        import time
                        time.sleep(0.1 * (2 ** retries))
            
            # Try fallback function
            fallback = self.fallback_functions.get(func.__name__)
            if fallback:
                logger.warning(f"Using fallback for {func.__name__}")
                try:
                    return fallback(*args, **kwargs)
                except Exception as fallback_error:
                    self._log_error(f"{func.__name__}_fallback", fallback_error, args, kwargs)
            
            # Re-raise last error
            raise last_error
        
        return wrapper
    
    def _log_error(self, function_name: str, error: Exception,
                  args: tuple, kwargs: dict):
        """Log error context"""
        context = ErrorContext(
            function_name=function_name,
            input_data={'args': str(args)[:200], 'kwargs': str(kwargs)[:200]},
            error_type=type(error).__name__,
            error_message=str(error),
            stack_trace=traceback.format_exc(),
            recovery_action="attempting_recovery"
        )
        
        self.error_history.append(context)
        
        logger.error(f"Error in {function_name}: {error}")
        logger.debug(f"Stack trace: {context.stack_trace}")
    
    def safe_execute(self, func: Callable, *args,
                    default=None, **kwargs) -> Any:
        """Execute function safely with default return"""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self._log_error(func.__name__, e, args, kwargs)
            return default
    
    def create_safe_wrapper(self, func: Callable,
                          default=None) -> Callable:
        """Create safe wrapper for function"""
        import functools
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.safe_execute(func, *args, default=default, **kwargs)
        
        return wrapper
    
    def recover_state(self, corrupted_state: Any,
                     validation_func: Callable[[Any], bool]) -> Any:
        """Attempt to recover corrupted state"""
        # Try to fix common issues
        recovery_attempts = [
            self._fix_nan_values,
            self._fix_infinite_values,
            self._fix_negative_dimensions,
            self._fix_out_of_bounds
        ]
        
        for recovery_func in recovery_attempts:
            try:
                recovered = recovery_func(corrupted_state)
                if validation_func(recovered):
                    logger.info(f"State recovered using {recovery_func.__name__}")
                    return recovered
            except:
                continue
        
        logger.error("Failed to recover state")
        return None
    
    def _fix_nan_values(self, state: Any) -> Any:
        """Fix NaN values in state"""
        if hasattr(state, '__dict__'):
            for key, value in state.__dict__.items():
                if isinstance(value, float) and np.isnan(value):
                    state.__dict__[key] = 0.0
                elif isinstance(value, np.ndarray):
                    state.__dict__[key] = np.nan_to_num(value)
        
        return state
    
    def _fix_infinite_values(self, state: Any) -> Any:
        """Fix infinite values in state"""
        if hasattr(state, '__dict__'):
            for key, value in state.__dict__.items():
                if isinstance(value, float) and np.isinf(value):
                    state.__dict__[key] = np.sign(value) * 1e100
                elif isinstance(value, np.ndarray):
                    state.__dict__[key] = np.clip(value, -1e100, 1e100)
        
        return state
    
    def _fix_negative_dimensions(self, state: Any) -> Any:
        """Fix negative dimensions"""
        if hasattr(state, 'width') and state.width < 0:
            state.width = abs(state.width)
        if hasattr(state, 'height') and state.height < 0:
            state.height = abs(state.height)
        
        return state
    
    def _fix_out_of_bounds(self, state: Any) -> Any:
        """Fix out of bounds positions"""
        if hasattr(state, 'position') and hasattr(state, 'room'):
            if state.position.x < 0:
                state.position.x = 0
            if state.position.y < 0:
                state.position.y = 0
            if state.position.x > state.room.width:
                state.position.x = state.room.width
            if state.position.y > state.room.height:
                state.position.y = state.room.height
        
        return state
    
    def get_error_report(self) -> str:
        """Generate error report"""
        report = []
        report.append("=" * 60)
        report.append("ERROR RECOVERY REPORT")
        report.append("=" * 60)
        report.append(f"\nTotal errors: {len(self.error_history)}")
        
        # Group by error type
        error_counts = defaultdict(int)
        for error in self.error_history:
            error_counts[error.error_type] += 1
        
        report.append("\nErrors by type:")
        for error_type, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
            report.append(f"  {error_type}: {count}")
        
        # Recent errors
        report.append("\nRecent errors:")
        for error in self.error_history[-5:]:
            report.append(f"  [{error.timestamp.strftime('%H:%M:%S')}] {error.function_name}: {error.error_message}")
        
        return "\n".join(report)


class StabilityManager:
    """Main stability management system"""
    
    def __init__(self):
        self.numerical_stability = NumericalStability()
        self.tie_breaker = DeterministicTieBreaker()
        self.reproducibility = None
        self.error_recovery = ErrorRecovery()
        self.stability_issues = []
    
    def initialize(self, config: Optional[ReproducibilityConfig] = None):
        """Initialize stability systems"""
        if config is None:
            config = ReproducibilityConfig(
                seed=42,
                precision_digits=10,
                tie_breaking_rule="hash",
                deterministic_mode=True,
                error_handling_mode="recover"
            )
        
        self.reproducibility = ReproducibilityManager(config)
        
        # Register common recovery strategies
        self._register_recovery_strategies()
        
        logger.info("Stability manager initialized")
    
    def _register_recovery_strategies(self):
        """Register common recovery strategies"""
        
        def recover_from_division_by_zero(error: Exception, context: Dict) -> float:
            logger.debug("Recovering from division by zero")
            return 0.0
        
        self.error_recovery.register_recovery(ZeroDivisionError, recover_from_division_by_zero)
        
        def recover_from_value_error(error: Exception, context: Dict) -> Any:
            logger.debug(f"Recovering from value error: {error}")
            return None
        
        self.error_recovery.register_recovery(ValueError, recover_from_value_error)
        
        def recover_from_index_error(error: Exception, context: Dict) -> Any:
            logger.debug(f"Recovering from index error: {error}")
            args = context.get('args', ())
            if args and hasattr(args[0], '__len__'):
                return args[0][0] if len(args[0]) > 0 else None
            return None
        
        self.error_recovery.register_recovery(IndexError, recover_from_index_error)
    
    def make_stable(self, func: Callable) -> Callable:
        """Make function numerically stable and deterministic"""
        # Apply all stability improvements
        func = self.error_recovery.with_recovery(func)
        
        if self.reproducibility:
            func = self.reproducibility.make_deterministic(func)
        
        return func
    
    def check_stability(self, state: PackingState) -> List[StabilityIssue]:
        """Check for stability issues in state"""
        issues = []
        
        # Check for NaN/Inf values
        for i, panel in enumerate(state.placed_panels):
            if np.isnan(panel.position.x) or np.isnan(panel.position.y):
                issues.append(StabilityIssue(
                    issue_type=StabilityIssueType.FLOATING_POINT_ERROR,
                    location=f"panel_{i}_position",
                    description="NaN value in panel position",
                    severity="critical",
                    resolution="Reset position to (0, 0)"
                ))
            
            if np.isinf(panel.position.x) or np.isinf(panel.position.y):
                issues.append(StabilityIssue(
                    issue_type=StabilityIssueType.OVERFLOW,
                    location=f"panel_{i}_position",
                    description="Infinite value in panel position",
                    severity="high",
                    resolution="Clip to room boundaries"
                ))
        
        # Check for invalid state
        if len(state.placed_panels) > 10000:
            issues.append(StabilityIssue(
                issue_type=StabilityIssueType.INVALID_STATE,
                location="placed_panels",
                description="Excessive number of placed panels",
                severity="high",
                resolution="Limit panel count or use batching"
            ))
        
        self.stability_issues.extend(issues)
        return issues
    
    def fix_stability_issues(self, state: PackingState,
                           issues: List[StabilityIssue]) -> PackingState:
        """Fix detected stability issues"""
        for issue in issues:
            if issue.issue_type == StabilityIssueType.FLOATING_POINT_ERROR:
                # Fix NaN values
                for panel in state.placed_panels:
                    if np.isnan(panel.position.x):
                        panel.position.x = 0.0
                    if np.isnan(panel.position.y):
                        panel.position.y = 0.0
            
            elif issue.issue_type == StabilityIssueType.OVERFLOW:
                # Fix infinite values
                for panel in state.placed_panels:
                    if np.isinf(panel.position.x):
                        panel.position.x = state.room.width if panel.position.x > 0 else 0
                    if np.isinf(panel.position.y):
                        panel.position.y = state.room.height if panel.position.y > 0 else 0
        
        return state
    
    def ensure_determinism(self, items: List[Any],
                         key_func: Callable[[Any], float]) -> List[Any]:
        """Ensure deterministic ordering of items"""
        return self.tie_breaker.sort_deterministic(items, key_func)
    
    def get_stability_report(self) -> str:
        """Generate stability report"""
        report = []
        report.append("=" * 60)
        report.append("STABILITY REPORT")
        report.append("=" * 60)
        
        # Numerical stability
        report.append("\nNumerical Stability:")
        report.append(f"  Epsilon: {self.numerical_stability.epsilon}")
        report.append(f"  Precision: {self.numerical_stability.precision}")
        
        # Tie breaking
        report.append("\nTie Breaking:")
        report.append(f"  Total ties: {self.tie_breaker.tie_count}")
        report.append(f"  Recent ties: {len(self.tie_breaker.tie_history[-10:])}")
        
        # Reproducibility
        if self.reproducibility:
            report.append("\nReproducibility:")
            report.append(f"  Seed: {self.reproducibility.config.seed}")
            report.append(f"  Checkpoints: {len(self.reproducibility.checkpoints)}")
            report.append(f"  Trace length: {len(self.reproducibility.execution_trace)}")
        
        # Error recovery
        report.append("\n" + self.error_recovery.get_error_report())
        
        # Stability issues
        if self.stability_issues:
            report.append("\nStability Issues:")
            issue_counts = defaultdict(int)
            for issue in self.stability_issues:
                issue_counts[issue.issue_type.value] += 1
            
            for issue_type, count in issue_counts.items():
                report.append(f"  {issue_type}: {count}")
        
        return "\n".join(report)
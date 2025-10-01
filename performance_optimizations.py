from typing import List, Dict, Optional, Set, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from datetime import datetime
import logging
import time
import functools
import pickle
import json
from collections import defaultdict
import hashlib

from core import PackingState, Panel, Room, Position, PlacedPanel
from spatial_index import SpatialIndex, OccupancyGrid
from algorithm_interface import OptimizerConfig, OptimizerResult

logger = logging.getLogger(__name__)


class OptimizationType(Enum):
    FAST_PATH = "fast_path"
    ALGORITHM_ACCELERATION = "algorithm_acceleration"
    CACHE_OPTIMIZATION = "cache_optimization"
    VECTORIZATION = "vectorization"
    PARALLEL_EXECUTION = "parallel_execution"
    MEMORY_OPTIMIZATION = "memory_optimization"


@dataclass
class FastPathCondition:
    name: str
    check_function: Callable[[Any], bool]
    description: str
    estimated_speedup: float


@dataclass
class FastPathImplementation:
    condition: FastPathCondition
    implementation: Callable
    fallback: Callable
    hit_count: int = 0
    miss_count: int = 0
    total_time_saved: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0


@dataclass
class AlgorithmAcceleration:
    name: str
    original_algorithm: str
    optimization_type: OptimizationType
    speedup_factor: float
    implementation: Callable
    applicable_conditions: List[str]


@dataclass
class PerformanceBaseline:
    timestamp: datetime
    function_name: str
    input_characteristics: Dict[str, Any]
    execution_time: float
    memory_usage: float
    result_quality: float


@dataclass
class RegressionReport:
    detected: bool
    function_name: str
    baseline_time: float
    current_time: float
    regression_factor: float
    confidence: float
    likely_cause: str


class FastPathOptimizer:
    """Implements fast paths for common cases"""
    
    def __init__(self):
        self.fast_paths = {}
        self.execution_stats = defaultdict(lambda: {'hits': 0, 'misses': 0, 'time_saved': 0})
    
    def register_fast_path(self, function_name: str, 
                          condition: FastPathCondition,
                          fast_implementation: Callable,
                          original_implementation: Callable):
        """Register a fast path for a function"""
        
        fast_path = FastPathImplementation(
            condition=condition,
            implementation=fast_implementation,
            fallback=original_implementation
        )
        
        if function_name not in self.fast_paths:
            self.fast_paths[function_name] = []
        
        self.fast_paths[function_name].append(fast_path)
        
        logger.debug(f"Registered fast path for {function_name}: {condition.name}")
    
    def create_optimized_function(self, function_name: str,
                                original_func: Callable) -> Callable:
        """Create optimized version of function with fast paths"""
        
        fast_paths = self.fast_paths.get(function_name, [])
        
        @functools.wraps(original_func)
        def optimized_func(*args, **kwargs):
            # Check fast path conditions
            for fast_path in fast_paths:
                if fast_path.condition.check_function(args):
                    start_time = time.perf_counter()
                    
                    try:
                        # Use fast path
                        result = fast_path.implementation(*args, **kwargs)
                        fast_path.hit_count += 1
                        
                        # Measure time saved
                        end_time = time.perf_counter()
                        fast_time = end_time - start_time
                        
                        # Estimate time for original (simplified)
                        estimated_original_time = fast_time * fast_path.condition.estimated_speedup
                        fast_path.total_time_saved += max(0, estimated_original_time - fast_time)
                        
                        return result
                    except:
                        # Fall back to original if fast path fails
                        fast_path.miss_count += 1
                        pass
            
            # No fast path applicable, use original
            return original_func(*args, **kwargs)
        
        return optimized_func
    
    def implement_common_fast_paths(self):
        """Implement common fast path optimizations"""
        
        # Fast path for small rooms
        def small_room_check(args) -> bool:
            if len(args) > 0 and hasattr(args[0], 'width') and hasattr(args[0], 'height'):
                room = args[0]
                return room.width * room.height < 10000
            return False
        
        small_room_condition = FastPathCondition(
            name="small_room",
            check_function=small_room_check,
            description="Room area < 10000",
            estimated_speedup=5.0
        )
        
        # Fast path for few panels
        def few_panels_check(args) -> bool:
            if len(args) > 1 and isinstance(args[1], list):
                panels = args[1]
                return len(panels) < 10
            return False
        
        few_panels_condition = FastPathCondition(
            name="few_panels",
            check_function=few_panels_check,
            description="Less than 10 panels",
            estimated_speedup=3.0
        )
        
        # Fast path for uniform panels
        def uniform_panels_check(args) -> bool:
            if len(args) > 1 and isinstance(args[1], list) and len(args[1]) > 0:
                panels = args[1]
                if all(hasattr(p, 'width') and hasattr(p, 'height') for p in panels):
                    first_size = (panels[0].width, panels[0].height)
                    return all((p.width, p.height) == first_size for p in panels)
            return False
        
        uniform_panels_condition = FastPathCondition(
            name="uniform_panels",
            check_function=uniform_panels_check,
            description="All panels same size",
            estimated_speedup=2.0
        )
        
        return [small_room_condition, few_panels_condition, uniform_panels_condition]
    
    def create_spatial_fast_paths(self) -> Dict[str, Callable]:
        """Create fast paths for spatial operations"""
        
        fast_paths = {}
        
        # Fast collision detection for aligned rectangles
        def fast_aligned_collision(rect1: Tuple[float, float, float, float],
                                  rect2: Tuple[float, float, float, float]) -> bool:
            """Fast collision detection for axis-aligned rectangles"""
            return not (rect1[2] <= rect2[0] or rect2[2] <= rect1[0] or
                       rect1[3] <= rect2[1] or rect2[3] <= rect1[1])
        
        fast_paths['collision_detection'] = fast_aligned_collision
        
        # Fast point-in-rectangle test
        def fast_point_in_rect(point: Tuple[float, float],
                              rect: Tuple[float, float, float, float]) -> bool:
            """Fast point-in-rectangle test"""
            return rect[0] <= point[0] <= rect[2] and rect[1] <= point[1] <= rect[3]
        
        fast_paths['point_in_rect'] = fast_point_in_rect
        
        # Fast area calculation
        def fast_rect_area(rect: Tuple[float, float, float, float]) -> float:
            """Fast rectangle area calculation"""
            return (rect[2] - rect[0]) * (rect[3] - rect[1])
        
        fast_paths['rect_area'] = fast_rect_area
        
        return fast_paths
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get fast path usage statistics"""
        stats = {}
        
        for func_name, paths in self.fast_paths.items():
            func_stats = {
                'total_hits': sum(p.hit_count for p in paths),
                'total_misses': sum(p.miss_count for p in paths),
                'total_time_saved': sum(p.total_time_saved for p in paths),
                'paths': []
            }
            
            for path in paths:
                func_stats['paths'].append({
                    'condition': path.condition.name,
                    'hit_rate': path.hit_rate,
                    'hits': path.hit_count,
                    'misses': path.miss_count,
                    'time_saved': path.total_time_saved
                })
            
            stats[func_name] = func_stats
        
        return stats


class AlgorithmAccelerator:
    """Implements algorithm-specific accelerations"""
    
    def __init__(self):
        self.accelerations = {}
        self.vectorized_operations = {}
        self.cached_results = {}
    
    def register_acceleration(self, acceleration: AlgorithmAcceleration):
        """Register an algorithm acceleration"""
        self.accelerations[acceleration.name] = acceleration
        logger.debug(f"Registered acceleration: {acceleration.name} "
                    f"(speedup: {acceleration.speedup_factor}x)")
    
    def implement_sorting_accelerations(self):
        """Implement accelerated sorting algorithms"""
        
        # Radix sort for integer values
        def radix_sort_panels(panels: List[Panel], key_func: Callable[[Panel], int]) -> List[Panel]:
            """Radix sort for integer panel attributes"""
            if not panels:
                return panels
            
            # Get max value
            max_val = max(key_func(p) for p in panels)
            
            # Radix sort
            exp = 1
            output = panels.copy()
            
            while max_val // exp > 0:
                counting_sort_by_digit(output, exp, key_func)
                exp *= 10
            
            return output
        
        def counting_sort_by_digit(arr: List[Panel], exp: int, key_func: Callable):
            """Helper for radix sort"""
            n = len(arr)
            output = [None] * n
            count = [0] * 10
            
            for panel in arr:
                index = (key_func(panel) // exp) % 10
                count[index] += 1
            
            for i in range(1, 10):
                count[i] += count[i - 1]
            
            for i in range(n - 1, -1, -1):
                index = (key_func(arr[i]) // exp) % 10
                output[count[index] - 1] = arr[i]
                count[index] -= 1
            
            for i in range(n):
                arr[i] = output[i]
        
        # Register radix sort acceleration
        self.register_acceleration(AlgorithmAcceleration(
            name="radix_sort_panels",
            original_algorithm="quicksort",
            optimization_type=OptimizationType.ALGORITHM_ACCELERATION,
            speedup_factor=2.0,
            implementation=radix_sort_panels,
            applicable_conditions=["integer_keys", "large_dataset"]
        ))
        
        # Bucket sort for uniform distribution
        def bucket_sort_panels(panels: List[Panel], key_func: Callable[[Panel], float]) -> List[Panel]:
            """Bucket sort for uniformly distributed values"""
            if not panels:
                return panels
            
            n = len(panels)
            buckets = [[] for _ in range(n)]
            
            # Find range
            values = [key_func(p) for p in panels]
            min_val, max_val = min(values), max(values)
            
            if min_val == max_val:
                return panels
            
            # Distribute to buckets
            for panel, value in zip(panels, values):
                index = int((value - min_val) / (max_val - min_val) * (n - 1))
                buckets[index].append(panel)
            
            # Sort each bucket and concatenate
            result = []
            for bucket in buckets:
                if bucket:
                    bucket.sort(key=key_func)
                    result.extend(bucket)
            
            return result
        
        self.register_acceleration(AlgorithmAcceleration(
            name="bucket_sort_panels",
            original_algorithm="quicksort",
            optimization_type=OptimizationType.ALGORITHM_ACCELERATION,
            speedup_factor=1.5,
            implementation=bucket_sort_panels,
            applicable_conditions=["uniform_distribution", "floating_point_keys"]
        ))
    
    def implement_search_accelerations(self):
        """Implement accelerated search algorithms"""
        
        # Binary search with interpolation
        def interpolation_search(arr: List[float], target: float) -> int:
            """Interpolation search for uniformly distributed data"""
            left, right = 0, len(arr) - 1
            
            while left <= right and arr[left] <= target <= arr[right]:
                if left == right:
                    return left if arr[left] == target else -1
                
                # Interpolation formula
                pos = left + int((target - arr[left]) * (right - left) / 
                               (arr[right] - arr[left]))
                
                if arr[pos] == target:
                    return pos
                elif arr[pos] < target:
                    left = pos + 1
                else:
                    right = pos - 1
            
            return -1
        
        self.register_acceleration(AlgorithmAcceleration(
            name="interpolation_search",
            original_algorithm="binary_search",
            optimization_type=OptimizationType.ALGORITHM_ACCELERATION,
            speedup_factor=1.3,
            implementation=interpolation_search,
            applicable_conditions=["sorted_array", "uniform_distribution"]
        ))
    
    def implement_spatial_accelerations(self):
        """Implement spatial algorithm accelerations"""
        
        # Sweep line algorithm for rectangle intersections
        def sweep_line_intersections(rectangles: List[Tuple[float, float, float, float]]) -> List[Tuple[int, int]]:
            """Find all intersecting rectangle pairs using sweep line"""
            events = []
            
            # Create events
            for i, rect in enumerate(rectangles):
                events.append((rect[0], 'start', i, rect))
                events.append((rect[2], 'end', i, rect))
            
            events.sort()
            
            active = set()
            intersections = []
            
            for x, event_type, rect_id, rect in events:
                if event_type == 'start':
                    # Check intersections with active rectangles
                    for active_id in active:
                        active_rect = rectangles[active_id]
                        # Check y-overlap
                        if not (rect[3] <= active_rect[1] or active_rect[3] <= rect[1]):
                            intersections.append((min(rect_id, active_id), max(rect_id, active_id)))
                    
                    active.add(rect_id)
                else:
                    active.discard(rect_id)
            
            return list(set(intersections))
        
        self.register_acceleration(AlgorithmAcceleration(
            name="sweep_line_intersections",
            original_algorithm="brute_force_intersections",
            optimization_type=OptimizationType.ALGORITHM_ACCELERATION,
            speedup_factor=10.0,
            implementation=sweep_line_intersections,
            applicable_conditions=["many_rectangles", "find_all_intersections"]
        ))
    
    def vectorize_operations(self):
        """Implement vectorized operations using NumPy"""
        
        # Vectorized distance calculations
        def vectorized_distances(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
            """Calculate pairwise Euclidean distances using vectorization"""
            # points1: (n, 2), points2: (m, 2)
            # result: (n, m) distance matrix
            diff = points1[:, np.newaxis, :] - points2[np.newaxis, :, :]
            return np.sqrt(np.sum(diff ** 2, axis=2))
        
        self.vectorized_operations['pairwise_distances'] = vectorized_distances
        
        # Vectorized collision detection
        def vectorized_collision_check(rects1: np.ndarray, rects2: np.ndarray) -> np.ndarray:
            """Check collisions between rectangle sets"""
            # rects: (n, 4) array of [x1, y1, x2, y2]
            x1_1, y1_1, x2_1, y2_1 = rects1.T
            x1_2, y1_2, x2_2, y2_2 = rects2.T
            
            # Vectorized collision check
            no_x_overlap = (x2_1[:, np.newaxis] <= x1_2) | (x2_2 <= x1_1[:, np.newaxis])
            no_y_overlap = (y2_1[:, np.newaxis] <= y1_2) | (y2_2 <= y1_1[:, np.newaxis])
            
            collisions = ~(no_x_overlap | no_y_overlap)
            return collisions
        
        self.vectorized_operations['collision_check'] = vectorized_collision_check
        
        # Vectorized area calculations
        def vectorized_areas(rects: np.ndarray) -> np.ndarray:
            """Calculate areas of multiple rectangles"""
            return (rects[:, 2] - rects[:, 0]) * (rects[:, 3] - rects[:, 1])
        
        self.vectorized_operations['rect_areas'] = vectorized_areas
    
    def apply_acceleration(self, algorithm_name: str, *args, **kwargs) -> Any:
        """Apply acceleration if available"""
        
        if algorithm_name in self.accelerations:
            acceleration = self.accelerations[algorithm_name]
            return acceleration.implementation(*args, **kwargs)
        
        raise ValueError(f"No acceleration found for {algorithm_name}")


class PerformanceRegressionDetector:
    """Detects performance regressions in code"""
    
    def __init__(self, baseline_file: str = "performance_baseline.pkl"):
        self.baseline_file = baseline_file
        self.baselines = self._load_baselines()
        self.current_measurements = []
        self.regression_threshold = 1.2  # 20% slower is regression
        self.confidence_threshold = 0.8
    
    def _load_baselines(self) -> Dict[str, List[PerformanceBaseline]]:
        """Load performance baselines from file"""
        try:
            with open(self.baseline_file, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return defaultdict(list)
    
    def save_baselines(self):
        """Save baselines to file"""
        with open(self.baseline_file, 'wb') as f:
            pickle.dump(dict(self.baselines), f)
    
    def record_baseline(self, function_name: str,
                       input_characteristics: Dict[str, Any],
                       execution_time: float,
                       memory_usage: float = 0,
                       result_quality: float = 0):
        """Record a performance baseline"""
        
        baseline = PerformanceBaseline(
            timestamp=datetime.now(),
            function_name=function_name,
            input_characteristics=input_characteristics,
            execution_time=execution_time,
            memory_usage=memory_usage,
            result_quality=result_quality
        )
        
        self.baselines[function_name].append(baseline)
        
        # Keep only recent baselines (last 100)
        if len(self.baselines[function_name]) > 100:
            self.baselines[function_name] = self.baselines[function_name][-100:]
    
    def check_regression(self, function_name: str,
                        current_time: float,
                        input_characteristics: Dict[str, Any]) -> Optional[RegressionReport]:
        """Check if current performance is a regression"""
        
        if function_name not in self.baselines:
            return None
        
        # Find similar baselines
        similar_baselines = self._find_similar_baselines(
            function_name, input_characteristics
        )
        
        if not similar_baselines:
            return None
        
        # Calculate baseline statistics
        baseline_times = [b.execution_time for b in similar_baselines]
        baseline_mean = np.mean(baseline_times)
        baseline_std = np.std(baseline_times)
        
        # Check for regression
        if current_time > baseline_mean * self.regression_threshold:
            # Calculate confidence
            if baseline_std > 0:
                z_score = (current_time - baseline_mean) / baseline_std
                confidence = min(1.0, abs(z_score) / 3)
            else:
                confidence = 1.0 if current_time > baseline_mean * 1.5 else 0.5
            
            if confidence >= self.confidence_threshold:
                return RegressionReport(
                    detected=True,
                    function_name=function_name,
                    baseline_time=baseline_mean,
                    current_time=current_time,
                    regression_factor=current_time / baseline_mean,
                    confidence=confidence,
                    likely_cause=self._identify_likely_cause(
                        function_name, current_time, baseline_mean
                    )
                )
        
        return None
    
    def _find_similar_baselines(self, function_name: str,
                               input_characteristics: Dict[str, Any]) -> List[PerformanceBaseline]:
        """Find baselines with similar input characteristics"""
        
        similar = []
        all_baselines = self.baselines.get(function_name, [])
        
        for baseline in all_baselines:
            if self._characteristics_similar(baseline.input_characteristics, input_characteristics):
                similar.append(baseline)
        
        return similar
    
    def _characteristics_similar(self, char1: Dict[str, Any],
                                char2: Dict[str, Any]) -> bool:
        """Check if two input characteristics are similar"""
        
        # Simple similarity check
        for key in set(char1.keys()) | set(char2.keys()):
            if key not in char1 or key not in char2:
                continue
            
            val1, val2 = char1[key], char2[key]
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Numeric values should be within 20%
                if val1 == 0 and val2 == 0:
                    continue
                if val1 == 0 or val2 == 0:
                    return False
                ratio = max(val1, val2) / min(val1, val2)
                if ratio > 1.2:
                    return False
            elif val1 != val2:
                return False
        
        return True
    
    def _identify_likely_cause(self, function_name: str,
                              current_time: float,
                              baseline_time: float) -> str:
        """Try to identify likely cause of regression"""
        
        regression_ratio = current_time / baseline_time
        
        if regression_ratio > 10:
            return "Algorithmic complexity increase (possible O(n²) or worse)"
        elif regression_ratio > 5:
            return "Significant algorithmic change or nested loops"
        elif regression_ratio > 2:
            return "Added computation or inefficient data structure"
        elif regression_ratio > 1.5:
            return "Cache misses or memory allocation overhead"
        else:
            return "Minor inefficiency or system load"
    
    def monitor_function(self, func: Callable) -> Callable:
        """Decorator to monitor function performance"""
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract input characteristics
            input_chars = {
                'arg_count': len(args),
                'kwarg_count': len(kwargs)
            }
            
            # Add size information if available
            if args and hasattr(args[0], '__len__'):
                input_chars['input_size'] = len(args[0])
            
            # Measure execution
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            
            execution_time = end_time - start_time
            
            # Check for regression
            regression = self.check_regression(func.__name__, execution_time, input_chars)
            
            if regression and regression.detected:
                logger.warning(f"Performance regression detected in {func.__name__}: "
                             f"{regression.regression_factor:.2f}x slower "
                             f"(confidence: {regression.confidence:.2f})")
                logger.warning(f"Likely cause: {regression.likely_cause}")
            
            # Record for future baselines
            self.current_measurements.append({
                'function': func.__name__,
                'time': execution_time,
                'characteristics': input_chars
            })
            
            return result
        
        return wrapper
    
    def generate_regression_report(self) -> str:
        """Generate detailed regression report"""
        
        report = []
        report.append("=" * 60)
        report.append("PERFORMANCE REGRESSION REPORT")
        report.append("=" * 60)
        
        # Check all recent measurements
        regressions_found = []
        
        for measurement in self.current_measurements:
            regression = self.check_regression(
                measurement['function'],
                measurement['time'],
                measurement['characteristics']
            )
            
            if regression and regression.detected:
                regressions_found.append(regression)
        
        if not regressions_found:
            report.append("\nNo performance regressions detected.")
        else:
            report.append(f"\nFound {len(regressions_found)} regressions:\n")
            
            for reg in regressions_found:
                report.append(f"Function: {reg.function_name}")
                report.append(f"  Regression: {reg.regression_factor:.2f}x slower")
                report.append(f"  Baseline: {reg.baseline_time:.3f}s")
                report.append(f"  Current: {reg.current_time:.3f}s")
                report.append(f"  Confidence: {reg.confidence:.2f}")
                report.append(f"  Likely cause: {reg.likely_cause}")
                report.append("")
        
        return "\n".join(report)


class PerformanceOptimizer:
    """Main performance optimization system"""
    
    def __init__(self):
        self.fast_path_optimizer = FastPathOptimizer()
        self.algorithm_accelerator = AlgorithmAccelerator()
        self.regression_detector = PerformanceRegressionDetector()
        self.optimization_history = []
    
    def initialize_optimizations(self):
        """Initialize all optimization strategies"""
        
        # Set up fast paths
        self.fast_path_optimizer.implement_common_fast_paths()
        
        # Set up algorithm accelerations
        self.algorithm_accelerator.implement_sorting_accelerations()
        self.algorithm_accelerator.implement_search_accelerations()
        self.algorithm_accelerator.implement_spatial_accelerations()
        self.algorithm_accelerator.vectorize_operations()
        
        logger.info("Performance optimizations initialized")
    
    def optimize_function(self, func: Callable, optimization_level: int = 2) -> Callable:
        """Apply optimizations to a function"""
        
        optimized = func
        
        # Level 1: Basic optimizations
        if optimization_level >= 1:
            # Add performance monitoring
            optimized = self.regression_detector.monitor_function(optimized)
        
        # Level 2: Fast paths
        if optimization_level >= 2:
            # Try to add fast paths
            if func.__name__ in self.fast_path_optimizer.fast_paths:
                optimized = self.fast_path_optimizer.create_optimized_function(
                    func.__name__, optimized
                )
        
        # Level 3: Aggressive optimizations
        if optimization_level >= 3:
            # Add caching
            optimized = self._add_caching(optimized)
        
        return optimized
    
    def _add_caching(self, func: Callable, max_cache_size: int = 128) -> Callable:
        """Add LRU caching to function"""
        from functools import lru_cache
        
        @lru_cache(maxsize=max_cache_size)
        @functools.wraps(func)
        def cached_func(*args, **kwargs):
            # Convert unhashable arguments to hashable
            hashable_args = []
            for arg in args:
                if isinstance(arg, (list, dict)):
                    hashable_args.append(str(arg))
                else:
                    hashable_args.append(arg)
            
            return func(*tuple(hashable_args), **kwargs)
        
        return cached_func
    
    def benchmark_optimization(self, original_func: Callable,
                             optimized_func: Callable,
                             test_inputs: List[Tuple]) -> Dict[str, Any]:
        """Benchmark optimization effectiveness"""
        
        results = {
            'original_times': [],
            'optimized_times': [],
            'speedups': [],
            'average_speedup': 0
        }
        
        for inputs in test_inputs:
            # Time original
            start = time.perf_counter()
            original_func(*inputs)
            original_time = time.perf_counter() - start
            
            # Time optimized
            start = time.perf_counter()
            optimized_func(*inputs)
            optimized_time = time.perf_counter() - start
            
            results['original_times'].append(original_time)
            results['optimized_times'].append(optimized_time)
            
            speedup = original_time / optimized_time if optimized_time > 0 else 1.0
            results['speedups'].append(speedup)
        
        results['average_speedup'] = np.mean(results['speedups'])
        results['speedup_std'] = np.std(results['speedups'])
        
        return results
    
    def get_optimization_report(self) -> str:
        """Generate optimization report"""
        
        report = []
        report.append("=" * 60)
        report.append("PERFORMANCE OPTIMIZATION REPORT")
        report.append("=" * 60)
        
        # Fast path statistics
        report.append("\nFAST PATH STATISTICS:")
        fast_path_stats = self.fast_path_optimizer.get_statistics()
        
        for func_name, stats in fast_path_stats.items():
            report.append(f"\n{func_name}:")
            report.append(f"  Total hits: {stats['total_hits']}")
            report.append(f"  Total misses: {stats['total_misses']}")
            report.append(f"  Time saved: {stats['total_time_saved']:.3f}s")
            
            for path in stats['paths']:
                report.append(f"  - {path['condition']}: {path['hit_rate']:.1%} hit rate")
        
        # Algorithm accelerations
        report.append("\nALGORITHM ACCELERATIONS:")
        for name, accel in self.algorithm_accelerator.accelerations.items():
            report.append(f"  {name}: {accel.speedup_factor:.1f}x speedup")
        
        # Regression detection
        report.append("\n" + self.regression_detector.generate_regression_report())
        
        return "\n".join(report)
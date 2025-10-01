from typing import List, Dict, Optional, Set, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from datetime import datetime
import logging
import time
import cProfile
import pstats
import io
import traceback
import sys
from collections import defaultdict, deque
import inspect
import functools

from core import PackingState, Panel, Room, Position, PlacedPanel
from algorithm_interface import OptimizerConfig, OptimizerResult

logger = logging.getLogger(__name__)


class ComplexityClass(Enum):
    CONSTANT = "O(1)"
    LOGARITHMIC = "O(log n)"
    LINEAR = "O(n)"
    LINEARITHMIC = "O(n log n)"
    QUADRATIC = "O(n^2)"
    CUBIC = "O(n^3)"
    EXPONENTIAL = "O(2^n)"
    FACTORIAL = "O(n!)"


class OpportunityType(Enum):
    CACHE_MISS = "cache_miss"
    REDUNDANT_COMPUTATION = "redundant_computation"
    INEFFICIENT_LOOP = "inefficient_loop"
    MEMORY_ALLOCATION = "memory_allocation"
    ALGORITHMIC = "algorithmic"
    DATA_STRUCTURE = "data_structure"
    PARALLELIZATION = "parallelization"
    EARLY_TERMINATION = "early_termination"


@dataclass
class FunctionProfile:
    name: str
    module: str
    total_time: float
    call_count: int
    cumulative_time: float
    avg_time_per_call: float
    percent_time: float
    callers: List[str] = field(default_factory=list)
    callees: List[str] = field(default_factory=list)
    
    @property
    def is_hot_path(self) -> bool:
        return self.percent_time > 5.0 or self.call_count > 1000


@dataclass
class HotPath:
    functions: List[FunctionProfile]
    total_time: float
    call_frequency: int
    depth: int
    description: str
    
    @property
    def impact_score(self) -> float:
        return self.total_time * np.log1p(self.call_frequency)


@dataclass
class ComplexityMeasurement:
    function_name: str
    input_sizes: List[int]
    execution_times: List[float]
    estimated_complexity: ComplexityClass
    confidence: float
    growth_rate: float


@dataclass
class OptimizationOpportunity:
    type: OpportunityType
    location: str
    description: str
    estimated_speedup: float
    implementation_effort: str  # "low", "medium", "high"
    priority: int
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BottleneckAnalysisResult:
    profile_data: List[FunctionProfile]
    hot_paths: List[HotPath]
    complexity_measurements: List[ComplexityMeasurement]
    opportunities: List[OptimizationOpportunity]
    total_execution_time: float
    analysis_time: float
    recommendations: List[str]


class ExecutionProfiler:
    """Profiles code execution to identify performance bottlenecks"""
    
    def __init__(self):
        self.profiler = None
        self.profile_data = {}
        self.call_stack = []
        self.timing_data = defaultdict(list)
    
    def start_profiling(self):
        """Start profiling execution"""
        self.profiler = cProfile.Profile()
        self.profiler.enable()
    
    def stop_profiling(self) -> List[FunctionProfile]:
        """Stop profiling and return results"""
        if not self.profiler:
            return []
        
        self.profiler.disable()
        
        # Get statistics
        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s)
        ps.strip_dirs()
        ps.sort_stats('cumulative')
        
        # Parse statistics
        profiles = self._parse_stats(ps)
        
        self.profiler = None
        return profiles
    
    def _parse_stats(self, stats: pstats.Stats) -> List[FunctionProfile]:
        """Parse profiler statistics into FunctionProfile objects"""
        profiles = []
        
        total_time = sum(stats.stats[key][2] for key in stats.stats)
        
        for func_key, func_stats in stats.stats.items():
            filename, line_num, func_name = func_key
            cc, nc, tt, ct, callers = func_stats
            
            # Calculate percentage
            percent = (tt / total_time * 100) if total_time > 0 else 0
            
            # Get caller information
            caller_list = []
            for caller_key in callers:
                caller_file, caller_line, caller_name = caller_key
                caller_list.append(caller_name)
            
            profile = FunctionProfile(
                name=func_name,
                module=filename,
                total_time=tt,
                call_count=nc,
                cumulative_time=ct,
                avg_time_per_call=tt/nc if nc > 0 else 0,
                percent_time=percent,
                callers=caller_list[:5]  # Limit to top 5 callers
            )
            
            profiles.append(profile)
        
        # Sort by total time
        profiles.sort(key=lambda p: p.total_time, reverse=True)
        
        return profiles
    
    def profile_function(self, func: Callable, *args, **kwargs) -> Tuple[Any, FunctionProfile]:
        """Profile a single function execution"""
        profiler = cProfile.Profile()
        profiler.enable()
        
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
        finally:
            end_time = time.perf_counter()
            profiler.disable()
        
        # Parse profile
        stats = pstats.Stats(profiler)
        profiles = self._parse_stats(stats)
        
        # Find the main function profile
        func_name = func.__name__
        main_profile = next((p for p in profiles if p.name == func_name), profiles[0] if profiles else None)
        
        if main_profile:
            main_profile.total_time = end_time - start_time
        
        return result, main_profile
    
    def time_function(self, func: Callable) -> Callable:
        """Decorator to time function execution"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            
            self.timing_data[func.__name__].append(end - start)
            
            return result
        
        return wrapper
    
    def get_timing_statistics(self, func_name: str) -> Dict[str, float]:
        """Get timing statistics for a function"""
        times = self.timing_data.get(func_name, [])
        
        if not times:
            return {}
        
        return {
            'count': len(times),
            'total': sum(times),
            'mean': np.mean(times),
            'median': np.median(times),
            'std': np.std(times),
            'min': min(times),
            'max': max(times)
        }


class HotPathIdentifier:
    """Identifies hot paths in code execution"""
    
    def __init__(self):
        self.call_graph = defaultdict(set)
        self.execution_counts = defaultdict(int)
        self.path_times = defaultdict(float)
    
    def identify_hot_paths(self, profiles: List[FunctionProfile],
                          threshold: float = 0.1) -> List[HotPath]:
        """Identify hot execution paths"""
        hot_paths = []
        
        # Build call graph
        self._build_call_graph(profiles)
        
        # Find paths that consume significant time
        total_time = sum(p.total_time for p in profiles)
        threshold_time = total_time * threshold
        
        # Use DFS to find hot paths
        visited = set()
        for profile in profiles:
            if profile.total_time >= threshold_time and profile.name not in visited:
                path = self._trace_hot_path(profile, profiles, visited)
                if path:
                    hot_paths.append(path)
        
        # Sort by impact score
        hot_paths.sort(key=lambda p: p.impact_score, reverse=True)
        
        return hot_paths
    
    def _build_call_graph(self, profiles: List[FunctionProfile]):
        """Build call graph from profiles"""
        for profile in profiles:
            for caller in profile.callers:
                self.call_graph[caller].add(profile.name)
            
            self.execution_counts[profile.name] = profile.call_count
            self.path_times[profile.name] = profile.total_time
    
    def _trace_hot_path(self, start_profile: FunctionProfile,
                       all_profiles: List[FunctionProfile],
                       visited: Set[str]) -> Optional[HotPath]:
        """Trace a hot execution path"""
        path_functions = []
        current = start_profile
        total_time = 0
        total_calls = 0
        
        # Follow the hottest path
        while current and current.name not in visited:
            visited.add(current.name)
            path_functions.append(current)
            total_time += current.total_time
            total_calls += current.call_count
            
            # Find hottest callee
            hottest_callee = None
            max_time = 0
            
            for profile in all_profiles:
                if current.name in profile.callers and profile.total_time > max_time:
                    max_time = profile.total_time
                    hottest_callee = profile
            
            current = hottest_callee
        
        if len(path_functions) < 2:
            return None
        
        return HotPath(
            functions=path_functions,
            total_time=total_time,
            call_frequency=min(p.call_count for p in path_functions),
            depth=len(path_functions),
            description=f"{path_functions[0].name} -> ... -> {path_functions[-1].name}"
        )
    
    def analyze_call_patterns(self, profiles: List[FunctionProfile]) -> Dict[str, Any]:
        """Analyze function call patterns"""
        patterns = {
            'recursive_functions': [],
            'high_frequency_calls': [],
            'deep_call_chains': [],
            'bottleneck_functions': []
        }
        
        # Find recursive functions
        for profile in profiles:
            if profile.name in profile.callers:
                patterns['recursive_functions'].append(profile.name)
        
        # Find high frequency calls
        for profile in profiles:
            if profile.call_count > 1000:
                patterns['high_frequency_calls'].append({
                    'function': profile.name,
                    'count': profile.call_count,
                    'avg_time': profile.avg_time_per_call
                })
        
        # Find functions that are bottlenecks
        total_time = sum(p.total_time for p in profiles)
        for profile in profiles:
            if profile.percent_time > 10:
                patterns['bottleneck_functions'].append({
                    'function': profile.name,
                    'percent': profile.percent_time,
                    'total_time': profile.total_time
                })
        
        return patterns


class ComplexityAnalyzer:
    """Analyzes algorithmic complexity of functions"""
    
    def __init__(self):
        self.measurements = {}
        self.complexity_cache = {}
    
    def analyze_complexity(self, func: Callable,
                          input_generator: Callable[[int], Any],
                          sizes: List[int] = None) -> ComplexityMeasurement:
        """Analyze complexity of a function"""
        
        if sizes is None:
            sizes = [10, 20, 50, 100, 200, 500, 1000]
        
        func_name = func.__name__
        times = []
        
        # Measure execution time for different input sizes
        for size in sizes:
            input_data = input_generator(size)
            
            # Multiple runs for accuracy
            run_times = []
            for _ in range(3):
                start = time.perf_counter()
                func(input_data)
                end = time.perf_counter()
                run_times.append(end - start)
            
            avg_time = np.mean(run_times)
            times.append(avg_time)
        
        # Estimate complexity
        complexity, confidence, growth_rate = self._estimate_complexity(sizes, times)
        
        measurement = ComplexityMeasurement(
            function_name=func_name,
            input_sizes=sizes,
            execution_times=times,
            estimated_complexity=complexity,
            confidence=confidence,
            growth_rate=growth_rate
        )
        
        self.measurements[func_name] = measurement
        return measurement
    
    def _estimate_complexity(self, sizes: List[int], 
                           times: List[float]) -> Tuple[ComplexityClass, float, float]:
        """Estimate complexity class from measurements"""
        
        if len(sizes) < 2:
            return ComplexityClass.LINEAR, 0.0, 0.0
        
        # Normalize data
        sizes = np.array(sizes, dtype=float)
        times = np.array(times, dtype=float)
        
        # Avoid division by zero
        times = np.maximum(times, 1e-10)
        
        # Test different complexity models
        models = {
            ComplexityClass.CONSTANT: lambda n: np.ones_like(n),
            ComplexityClass.LOGARITHMIC: lambda n: np.log2(np.maximum(n, 1)),
            ComplexityClass.LINEAR: lambda n: n,
            ComplexityClass.LINEARITHMIC: lambda n: n * np.log2(np.maximum(n, 1)),
            ComplexityClass.QUADRATIC: lambda n: n ** 2,
            ComplexityClass.CUBIC: lambda n: n ** 3,
            ComplexityClass.EXPONENTIAL: lambda n: np.minimum(2 ** n, 1e10)
        }
        
        best_fit = ComplexityClass.LINEAR
        best_r2 = -float('inf')
        best_growth = 0.0
        
        for complexity, model_func in models.items():
            try:
                # Fit model
                X = model_func(sizes).reshape(-1, 1)
                
                # Use least squares to fit
                coeffs = np.linalg.lstsq(X, times, rcond=None)[0]
                
                # Calculate R^2
                predicted = X @ coeffs
                ss_res = np.sum((times - predicted) ** 2)
                ss_tot = np.sum((times - np.mean(times)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                if r2 > best_r2:
                    best_r2 = r2
                    best_fit = complexity
                    best_growth = coeffs[0] if len(coeffs) > 0 else 0
            except:
                continue
        
        # Convert R^2 to confidence (0-1 scale)
        confidence = max(0, min(1, best_r2))
        
        return best_fit, confidence, best_growth
    
    def detect_nested_loops(self, source_code: str) -> List[Dict[str, Any]]:
        """Detect nested loops in source code"""
        nested_loops = []
        
        lines = source_code.split('\n')
        loop_stack = []
        
        for i, line in enumerate(lines):
            indent = len(line) - len(line.lstrip())
            
            # Detect loop start
            if 'for ' in line or 'while ' in line:
                loop_info = {
                    'line': i + 1,
                    'type': 'for' if 'for ' in line else 'while',
                    'indent': indent,
                    'code': line.strip()
                }
                
                # Check if nested
                if loop_stack and indent > loop_stack[-1]['indent']:
                    nested_loops.append({
                        'outer': loop_stack[-1],
                        'inner': loop_info,
                        'depth': len(loop_stack) + 1
                    })
                
                loop_stack.append(loop_info)
            
            # Pop loops that have ended (based on indentation)
            while loop_stack and indent <= loop_stack[-1]['indent']:
                loop_stack.pop()
        
        return nested_loops


class OpportunityDetector:
    """Detects optimization opportunities"""
    
    def __init__(self):
        self.opportunities = []
        self.cache_analysis = {}
        self.redundancy_patterns = {}
    
    def detect_opportunities(self, profiles: List[FunctionProfile],
                           hot_paths: List[HotPath],
                           complexity_measurements: List[ComplexityMeasurement]) -> List[OptimizationOpportunity]:
        """Detect optimization opportunities"""
        opportunities = []
        
        # Check for cache optimization opportunities
        cache_opps = self._detect_cache_opportunities(profiles)
        opportunities.extend(cache_opps)
        
        # Check for redundant computations
        redundancy_opps = self._detect_redundancy(profiles, hot_paths)
        opportunities.extend(redundancy_opps)
        
        # Check for algorithmic improvements
        algo_opps = self._detect_algorithmic_opportunities(complexity_measurements)
        opportunities.extend(algo_opps)
        
        # Check for parallelization opportunities
        parallel_opps = self._detect_parallelization_opportunities(profiles)
        opportunities.extend(parallel_opps)
        
        # Check for early termination opportunities
        early_term_opps = self._detect_early_termination(hot_paths)
        opportunities.extend(early_term_opps)
        
        # Sort by priority
        opportunities.sort(key=lambda o: o.priority, reverse=True)
        
        self.opportunities = opportunities
        return opportunities
    
    def _detect_cache_opportunities(self, profiles: List[FunctionProfile]) -> List[OptimizationOpportunity]:
        """Detect caching opportunities"""
        opportunities = []
        
        for profile in profiles:
            # High call count with consistent computation
            if profile.call_count > 100 and profile.avg_time_per_call > 0.001:
                estimated_speedup = min(10.0, profile.call_count / 100)
                
                opportunities.append(OptimizationOpportunity(
                    type=OpportunityType.CACHE_MISS,
                    location=f"{profile.module}:{profile.name}",
                    description=f"Function {profile.name} called {profile.call_count} times, candidate for memoization",
                    estimated_speedup=estimated_speedup,
                    implementation_effort="low",
                    priority=int(estimated_speedup * 10),
                    details={
                        'call_count': profile.call_count,
                        'avg_time': profile.avg_time_per_call,
                        'total_time': profile.total_time
                    }
                ))
        
        return opportunities
    
    def _detect_redundancy(self, profiles: List[FunctionProfile],
                          hot_paths: List[HotPath]) -> List[OptimizationOpportunity]:
        """Detect redundant computations"""
        opportunities = []
        
        # Look for functions called multiple times in hot paths
        for path in hot_paths:
            call_counts = {}
            for func in path.functions:
                call_counts[func.name] = call_counts.get(func.name, 0) + 1
            
            for func_name, count in call_counts.items():
                if count > 1:
                    func_profile = next((f for f in path.functions if f.name == func_name), None)
                    if func_profile:
                        opportunities.append(OptimizationOpportunity(
                            type=OpportunityType.REDUNDANT_COMPUTATION,
                            location=func_name,
                            description=f"Function {func_name} called {count} times in hot path",
                            estimated_speedup=count - 1,
                            implementation_effort="medium",
                            priority=int((count - 1) * func_profile.percent_time),
                            details={'redundant_calls': count - 1}
                        ))
        
        return opportunities
    
    def _detect_algorithmic_opportunities(self, 
                                         measurements: List[ComplexityMeasurement]) -> List[OptimizationOpportunity]:
        """Detect algorithmic improvement opportunities"""
        opportunities = []
        
        for measurement in measurements:
            # Flag high complexity algorithms
            if measurement.estimated_complexity in [ComplexityClass.QUADRATIC,
                                                   ComplexityClass.CUBIC,
                                                   ComplexityClass.EXPONENTIAL]:
                
                # Estimate potential speedup
                if measurement.estimated_complexity == ComplexityClass.QUADRATIC:
                    potential_speedup = 10.0
                    effort = "medium"
                elif measurement.estimated_complexity == ComplexityClass.CUBIC:
                    potential_speedup = 100.0
                    effort = "high"
                else:
                    potential_speedup = 1000.0
                    effort = "high"
                
                opportunities.append(OptimizationOpportunity(
                    type=OpportunityType.ALGORITHMIC,
                    location=measurement.function_name,
                    description=f"Function has {measurement.estimated_complexity.value} complexity",
                    estimated_speedup=potential_speedup,
                    implementation_effort=effort,
                    priority=int(potential_speedup),
                    details={
                        'current_complexity': measurement.estimated_complexity.value,
                        'confidence': measurement.confidence
                    }
                ))
        
        return opportunities
    
    def _detect_parallelization_opportunities(self, 
                                             profiles: List[FunctionProfile]) -> List[OptimizationOpportunity]:
        """Detect parallelization opportunities"""
        opportunities = []
        
        # Look for CPU-intensive functions that could be parallelized
        for profile in profiles:
            if profile.total_time > 0.1 and profile.call_count > 1:
                # Check if function name suggests it's parallelizable
                parallelizable_keywords = ['map', 'filter', 'reduce', 'process', 'compute', 'calculate']
                
                if any(keyword in profile.name.lower() for keyword in parallelizable_keywords):
                    opportunities.append(OptimizationOpportunity(
                        type=OpportunityType.PARALLELIZATION,
                        location=profile.name,
                        description=f"Function {profile.name} is CPU-intensive and may benefit from parallelization",
                        estimated_speedup=4.0,  # Assume 4-core speedup
                        implementation_effort="medium",
                        priority=int(profile.percent_time * 4),
                        details={
                            'total_time': profile.total_time,
                            'call_count': profile.call_count
                        }
                    ))
        
        return opportunities
    
    def _detect_early_termination(self, hot_paths: List[HotPath]) -> List[OptimizationOpportunity]:
        """Detect early termination opportunities"""
        opportunities = []
        
        for path in hot_paths:
            # Look for paths with high depth that could terminate early
            if path.depth > 5:
                opportunities.append(OptimizationOpportunity(
                    type=OpportunityType.EARLY_TERMINATION,
                    location=path.description,
                    description=f"Deep call chain ({path.depth} levels) may benefit from early termination",
                    estimated_speedup=path.depth / 5,
                    implementation_effort="low",
                    priority=int(path.impact_score / 10),
                    details={
                        'depth': path.depth,
                        'total_time': path.total_time
                    }
                ))
        
        return opportunities


class BottleneckAnalyzer:
    """Main bottleneck analysis system"""
    
    def __init__(self):
        self.profiler = ExecutionProfiler()
        self.hot_path_identifier = HotPathIdentifier()
        self.complexity_analyzer = ComplexityAnalyzer()
        self.opportunity_detector = OpportunityDetector()
    
    def analyze(self, target_function: Callable,
               *args, **kwargs) -> BottleneckAnalysisResult:
        """Perform comprehensive bottleneck analysis"""
        
        analysis_start = time.time()
        
        # Profile execution
        self.profiler.start_profiling()
        
        start_time = time.perf_counter()
        try:
            result = target_function(*args, **kwargs)
        finally:
            end_time = time.perf_counter()
            profiles = self.profiler.stop_profiling()
        
        total_execution_time = end_time - start_time
        
        # Identify hot paths
        hot_paths = self.hot_path_identifier.identify_hot_paths(profiles)
        
        # Analyze call patterns
        patterns = self.hot_path_identifier.analyze_call_patterns(profiles)
        
        # Analyze complexity (for top functions)
        complexity_measurements = []
        for profile in profiles[:10]:  # Top 10 functions
            # Skip built-in functions
            if not profile.module.startswith('<'):
                # Create simple input generator
                def input_gen(size):
                    return list(range(size))
                
                # Note: In production, would need actual function references
                # This is a simplified version
                complexity_measurements.append(ComplexityMeasurement(
                    function_name=profile.name,
                    input_sizes=[10, 50, 100],
                    execution_times=[profile.avg_time_per_call * i for i in [10, 50, 100]],
                    estimated_complexity=self._estimate_complexity_from_profile(profile),
                    confidence=0.7,
                    growth_rate=1.0
                ))
        
        # Detect optimization opportunities
        opportunities = self.opportunity_detector.detect_opportunities(
            profiles, hot_paths, complexity_measurements
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            profiles, hot_paths, opportunities, patterns
        )
        
        analysis_time = time.time() - analysis_start
        
        return BottleneckAnalysisResult(
            profile_data=profiles,
            hot_paths=hot_paths,
            complexity_measurements=complexity_measurements,
            opportunities=opportunities,
            total_execution_time=total_execution_time,
            analysis_time=analysis_time,
            recommendations=recommendations
        )
    
    def _estimate_complexity_from_profile(self, profile: FunctionProfile) -> ComplexityClass:
        """Estimate complexity from profile data"""
        # Simple heuristic based on function characteristics
        if 'sort' in profile.name.lower():
            return ComplexityClass.LINEARITHMIC
        elif 'search' in profile.name.lower():
            if 'binary' in profile.name.lower():
                return ComplexityClass.LOGARITHMIC
            else:
                return ComplexityClass.LINEAR
        elif profile.call_count > 1000:
            return ComplexityClass.LINEAR
        else:
            return ComplexityClass.CONSTANT
    
    def _generate_recommendations(self, profiles: List[FunctionProfile],
                                 hot_paths: List[HotPath],
                                 opportunities: List[OptimizationOpportunity],
                                 patterns: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Top bottleneck recommendation
        if profiles:
            top_bottleneck = profiles[0]
            recommendations.append(
                f"Focus optimization on {top_bottleneck.name} - consuming {top_bottleneck.percent_time:.1f}% of execution time"
            )
        
        # Hot path recommendation
        if hot_paths:
            hottest_path = hot_paths[0]
            recommendations.append(
                f"Optimize hot path: {hottest_path.description} (impact score: {hottest_path.impact_score:.1f})"
            )
        
        # Caching recommendation
        cache_opportunities = [o for o in opportunities if o.type == OpportunityType.CACHE_MISS]
        if cache_opportunities:
            recommendations.append(
                f"Implement caching for {len(cache_opportunities)} functions to reduce redundant computations"
            )
        
        # Algorithmic improvement
        algo_opportunities = [o for o in opportunities if o.type == OpportunityType.ALGORITHMIC]
        if algo_opportunities:
            recommendations.append(
                f"Consider algorithmic improvements for {len(algo_opportunities)} high-complexity functions"
            )
        
        # Parallelization
        parallel_opportunities = [o for o in opportunities if o.type == OpportunityType.PARALLELIZATION]
        if parallel_opportunities:
            recommendations.append(
                f"Parallelize {len(parallel_opportunities)} CPU-intensive functions for multi-core speedup"
            )
        
        # Recursive function warning
        if patterns.get('recursive_functions'):
            recommendations.append(
                f"Review recursive functions: {', '.join(patterns['recursive_functions'])} for stack overflow risk"
            )
        
        return recommendations
    
    def generate_report(self, analysis_result: BottleneckAnalysisResult) -> str:
        """Generate detailed analysis report"""
        report = []
        report.append("=" * 80)
        report.append("BOTTLENECK ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"\nTotal Execution Time: {analysis_result.total_execution_time:.3f}s")
        report.append(f"Analysis Time: {analysis_result.analysis_time:.3f}s")
        
        # Top Functions
        report.append("\n" + "-" * 40)
        report.append("TOP TIME-CONSUMING FUNCTIONS")
        report.append("-" * 40)
        for i, profile in enumerate(analysis_result.profile_data[:10], 1):
            report.append(f"{i}. {profile.name}")
            report.append(f"   Time: {profile.total_time:.3f}s ({profile.percent_time:.1f}%)")
            report.append(f"   Calls: {profile.call_count}")
            report.append(f"   Avg/Call: {profile.avg_time_per_call*1000:.3f}ms")
        
        # Hot Paths
        report.append("\n" + "-" * 40)
        report.append("HOT EXECUTION PATHS")
        report.append("-" * 40)
        for i, path in enumerate(analysis_result.hot_paths[:5], 1):
            report.append(f"{i}. {path.description}")
            report.append(f"   Impact Score: {path.impact_score:.1f}")
            report.append(f"   Total Time: {path.total_time:.3f}s")
            report.append(f"   Depth: {path.depth}")
        
        # Optimization Opportunities
        report.append("\n" + "-" * 40)
        report.append("OPTIMIZATION OPPORTUNITIES")
        report.append("-" * 40)
        for i, opp in enumerate(analysis_result.opportunities[:10], 1):
            report.append(f"{i}. [{opp.type.value}] {opp.location}")
            report.append(f"   {opp.description}")
            report.append(f"   Estimated Speedup: {opp.estimated_speedup:.1f}x")
            report.append(f"   Effort: {opp.implementation_effort}")
            report.append(f"   Priority: {opp.priority}")
        
        # Recommendations
        report.append("\n" + "-" * 40)
        report.append("RECOMMENDATIONS")
        report.append("-" * 40)
        for i, rec in enumerate(analysis_result.recommendations, 1):
            report.append(f"{i}. {rec}")
        
        return "\n".join(report)
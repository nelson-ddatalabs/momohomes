from typing import List, Dict, Optional, Set, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from datetime import datetime
import logging
import itertools
import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import pickle

from core import PackingState, Panel, Room, Position, PlacedPanel
from algorithm_interface import OptimizerConfig, OptimizerResult

logger = logging.getLogger(__name__)


class ParameterType(Enum):
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    CATEGORICAL = "categorical"
    CONTINUOUS = "continuous"


@dataclass
class ParameterDefinition:
    name: str
    type: ParameterType
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    choices: Optional[List[Any]] = None
    default_value: Any = None
    description: str = ""
    affects_quality: bool = True
    affects_speed: bool = False
    scale: str = "linear"  # linear, log, exponential


@dataclass
class ParameterConfiguration:
    parameters: Dict[str, Any]
    score: Optional[float] = None
    coverage: Optional[float] = None
    execution_time: Optional[float] = None
    iterations: Optional[int] = None
    
    def __hash__(self):
        return hash(tuple(sorted(self.parameters.items())))


@dataclass
class TuningResult:
    best_config: ParameterConfiguration
    all_configs: List[ParameterConfiguration]
    sensitivity_analysis: Dict[str, float]
    parameter_importance: Dict[str, float]
    convergence_history: List[float]
    tuning_time: float
    evaluations: int


@dataclass
class SensitivityPoint:
    parameter: str
    value: Any
    score: float
    coverage: float
    time: float


class ParameterGridSearch:
    """Performs grid search over parameter space"""
    
    def __init__(self, max_evaluations: int = 100):
        self.max_evaluations = max_evaluations
        self.evaluation_count = 0
        self.best_score = -float('inf')
        self.best_config = None
    
    def search(self, parameters: List[ParameterDefinition],
              evaluate_fn: Callable[[Dict[str, Any]], float],
              resolution: int = 10) -> List[ParameterConfiguration]:
        """Perform grid search over parameter space"""
        
        # Generate grid points
        grid_points = self._generate_grid(parameters, resolution)
        
        # Limit to max evaluations
        if len(grid_points) > self.max_evaluations:
            # Random sampling if grid is too large
            import random
            grid_points = random.sample(grid_points, self.max_evaluations)
        
        configurations = []
        
        for point in grid_points:
            # Evaluate configuration
            score = evaluate_fn(point)
            
            config = ParameterConfiguration(
                parameters=point,
                score=score
            )
            configurations.append(config)
            
            # Track best
            if score > self.best_score:
                self.best_score = score
                self.best_config = config
            
            self.evaluation_count += 1
            
            if self.evaluation_count >= self.max_evaluations:
                break
        
        logger.info(f"Grid search completed: {self.evaluation_count} evaluations, "
                   f"best score: {self.best_score:.3f}")
        
        return configurations
    
    def _generate_grid(self, parameters: List[ParameterDefinition],
                       resolution: int) -> List[Dict[str, Any]]:
        """Generate grid points for parameters"""
        param_values = {}
        
        for param in parameters:
            if param.type == ParameterType.BOOLEAN:
                param_values[param.name] = [True, False]
            
            elif param.type == ParameterType.CATEGORICAL:
                param_values[param.name] = param.choices
            
            elif param.type in [ParameterType.INTEGER, ParameterType.FLOAT, ParameterType.CONTINUOUS]:
                if param.min_value is not None and param.max_value is not None:
                    if param.scale == "log":
                        values = np.logspace(
                            np.log10(max(param.min_value, 1e-10)),
                            np.log10(param.max_value),
                            resolution
                        )
                    elif param.scale == "exponential":
                        values = np.exp(np.linspace(
                            np.log(max(param.min_value, 1e-10)),
                            np.log(param.max_value),
                            resolution
                        ))
                    else:  # linear
                        values = np.linspace(param.min_value, param.max_value, resolution)
                    
                    if param.type == ParameterType.INTEGER:
                        values = np.unique(values.astype(int))
                    
                    param_values[param.name] = values.tolist()
                else:
                    param_values[param.name] = [param.default_value]
        
        # Generate all combinations
        keys = list(param_values.keys())
        values = list(param_values.values())
        
        grid_points = []
        for combination in itertools.product(*values):
            point = dict(zip(keys, combination))
            grid_points.append(point)
        
        return grid_points
    
    def search_adaptive(self, parameters: List[ParameterDefinition],
                       evaluate_fn: Callable[[Dict[str, Any]], float],
                       initial_resolution: int = 5) -> List[ParameterConfiguration]:
        """Adaptive grid search that refines around promising regions"""
        
        # Initial coarse search
        coarse_configs = self.search(parameters, evaluate_fn, initial_resolution)
        
        # Find top configurations
        sorted_configs = sorted(coarse_configs, key=lambda c: c.score or 0, reverse=True)
        top_configs = sorted_configs[:max(1, len(sorted_configs) // 10)]
        
        # Refine around top configurations
        refined_configs = []
        
        for top_config in top_configs:
            # Define refined search space around this configuration
            refined_params = self._create_refined_parameters(parameters, top_config.parameters)
            
            # Search refined space
            refined = self.search(refined_params, evaluate_fn, initial_resolution * 2)
            refined_configs.extend(refined)
        
        # Combine all configurations
        all_configs = coarse_configs + refined_configs
        
        # Remove duplicates
        unique_configs = {}
        for config in all_configs:
            key = hash(config)
            if key not in unique_configs or config.score > unique_configs[key].score:
                unique_configs[key] = config
        
        return list(unique_configs.values())
    
    def _create_refined_parameters(self, parameters: List[ParameterDefinition],
                                  center_values: Dict[str, Any]) -> List[ParameterDefinition]:
        """Create refined parameter definitions around center values"""
        refined = []
        
        for param in parameters:
            if param.type in [ParameterType.FLOAT, ParameterType.CONTINUOUS]:
                # Refine continuous parameters
                center = center_values.get(param.name, param.default_value)
                range_size = (param.max_value - param.min_value) * 0.2  # 20% range
                
                refined_param = ParameterDefinition(
                    name=param.name,
                    type=param.type,
                    min_value=max(param.min_value, center - range_size / 2),
                    max_value=min(param.max_value, center + range_size / 2),
                    default_value=center,
                    description=param.description,
                    affects_quality=param.affects_quality,
                    affects_speed=param.affects_speed,
                    scale=param.scale
                )
                refined.append(refined_param)
            
            elif param.type == ParameterType.INTEGER:
                # Refine integer parameters
                center = center_values.get(param.name, param.default_value)
                range_size = max(2, int((param.max_value - param.min_value) * 0.2))
                
                refined_param = ParameterDefinition(
                    name=param.name,
                    type=param.type,
                    min_value=max(param.min_value, center - range_size // 2),
                    max_value=min(param.max_value, center + range_size // 2),
                    default_value=center,
                    description=param.description,
                    affects_quality=param.affects_quality,
                    affects_speed=param.affects_speed
                )
                refined.append(refined_param)
            
            else:
                # Keep categorical and boolean parameters as is
                refined.append(param)
        
        return refined


class SensitivityAnalyzer:
    """Analyzes parameter sensitivity and importance"""
    
    def __init__(self):
        self.sensitivity_data = []
        self.baseline_score = None
        self.baseline_config = None
    
    def analyze(self, parameters: List[ParameterDefinition],
               evaluate_fn: Callable[[Dict[str, Any]], float],
               baseline_config: Dict[str, Any],
               samples_per_param: int = 10) -> Dict[str, float]:
        """Analyze sensitivity of each parameter"""
        
        # Evaluate baseline
        self.baseline_score = evaluate_fn(baseline_config)
        self.baseline_config = baseline_config
        
        sensitivities = {}
        
        for param in parameters:
            # Sample parameter values
            param_sensitivities = []
            
            if param.type == ParameterType.BOOLEAN:
                values = [True, False]
            elif param.type == ParameterType.CATEGORICAL:
                values = param.choices
            else:
                values = self._sample_parameter_values(param, samples_per_param)
            
            for value in values:
                # Create modified configuration
                test_config = baseline_config.copy()
                test_config[param.name] = value
                
                # Evaluate
                score = evaluate_fn(test_config)
                
                # Record sensitivity
                sensitivity = abs(score - self.baseline_score)
                param_sensitivities.append(sensitivity)
                
                self.sensitivity_data.append(SensitivityPoint(
                    parameter=param.name,
                    value=value,
                    score=score,
                    coverage=0.0,  # To be filled if needed
                    time=0.0  # To be filled if needed
                ))
            
            # Calculate overall sensitivity for parameter
            sensitivities[param.name] = np.mean(param_sensitivities)
        
        # Normalize sensitivities
        total_sensitivity = sum(sensitivities.values())
        if total_sensitivity > 0:
            for param_name in sensitivities:
                sensitivities[param_name] /= total_sensitivity
        
        logger.info(f"Sensitivity analysis complete: {len(sensitivities)} parameters analyzed")
        
        return sensitivities
    
    def _sample_parameter_values(self, param: ParameterDefinition,
                                num_samples: int) -> List[Any]:
        """Sample values for a parameter"""
        if param.min_value is None or param.max_value is None:
            return [param.default_value]
        
        if param.scale == "log":
            values = np.logspace(
                np.log10(max(param.min_value, 1e-10)),
                np.log10(param.max_value),
                num_samples
            )
        elif param.scale == "exponential":
            values = np.exp(np.linspace(
                np.log(max(param.min_value, 1e-10)),
                np.log(param.max_value),
                num_samples
            ))
        else:  # linear
            values = np.linspace(param.min_value, param.max_value, num_samples)
        
        if param.type == ParameterType.INTEGER:
            values = np.unique(values.astype(int))
        
        return values.tolist()
    
    def analyze_interactions(self, parameters: List[ParameterDefinition],
                           evaluate_fn: Callable[[Dict[str, Any]], float],
                           baseline_config: Dict[str, Any]) -> Dict[Tuple[str, str], float]:
        """Analyze parameter interactions"""
        interactions = {}
        
        # Test pairwise interactions
        for i, param1 in enumerate(parameters):
            for j, param2 in enumerate(parameters[i+1:], i+1):
                interaction_strength = self._measure_interaction(
                    param1, param2, evaluate_fn, baseline_config
                )
                interactions[(param1.name, param2.name)] = interaction_strength
        
        return interactions
    
    def _measure_interaction(self, param1: ParameterDefinition,
                           param2: ParameterDefinition,
                           evaluate_fn: Callable[[Dict[str, Any]], float],
                           baseline_config: Dict[str, Any]) -> float:
        """Measure interaction strength between two parameters"""
        
        # Sample values for both parameters
        values1 = self._sample_parameter_values(param1, 3)
        values2 = self._sample_parameter_values(param2, 3)
        
        # Measure individual effects
        effects1 = []
        for v1 in values1:
            config = baseline_config.copy()
            config[param1.name] = v1
            score = evaluate_fn(config)
            effects1.append(score - self.baseline_score)
        
        effects2 = []
        for v2 in values2:
            config = baseline_config.copy()
            config[param2.name] = v2
            score = evaluate_fn(config)
            effects2.append(score - self.baseline_score)
        
        # Measure combined effects
        combined_effects = []
        for v1 in values1:
            for v2 in values2:
                config = baseline_config.copy()
                config[param1.name] = v1
                config[param2.name] = v2
                score = evaluate_fn(config)
                
                # Expected additive effect
                expected = self.baseline_score + np.mean(effects1) + np.mean(effects2)
                
                # Actual effect
                actual = score
                
                # Interaction is the difference
                interaction = abs(actual - expected)
                combined_effects.append(interaction)
        
        return np.mean(combined_effects)


class AdaptiveAdjustment:
    """Adaptively adjusts parameters during optimization"""
    
    def __init__(self):
        self.adjustment_history = []
        self.performance_history = []
        self.current_config = None
        self.adjustment_rate = 0.1
        self.momentum = 0.9
        self.velocity = {}
    
    def initialize(self, initial_config: Dict[str, Any],
                  parameters: List[ParameterDefinition]):
        """Initialize adaptive adjustment"""
        self.current_config = initial_config.copy()
        self.velocity = {param.name: 0.0 for param in parameters}
    
    def adjust(self, performance: float,
              parameters: List[ParameterDefinition]) -> Dict[str, Any]:
        """Adjust parameters based on performance"""
        
        self.performance_history.append(performance)
        
        # Calculate performance gradient
        if len(self.performance_history) > 1:
            gradient = performance - self.performance_history[-2]
        else:
            gradient = 0.0
        
        # Adjust each parameter
        new_config = self.current_config.copy()
        
        for param in parameters:
            if not param.affects_quality:
                continue
            
            # Calculate adjustment
            if param.type in [ParameterType.FLOAT, ParameterType.CONTINUOUS]:
                # Use gradient ascent with momentum
                adjustment = self._calculate_adjustment(param, gradient)
                
                # Update velocity with momentum
                self.velocity[param.name] = (self.momentum * self.velocity[param.name] +
                                            (1 - self.momentum) * adjustment)
                
                # Apply adjustment
                old_value = self.current_config[param.name]
                new_value = old_value + self.velocity[param.name]
                
                # Clip to bounds
                if param.min_value is not None:
                    new_value = max(param.min_value, new_value)
                if param.max_value is not None:
                    new_value = min(param.max_value, new_value)
                
                new_config[param.name] = new_value
            
            elif param.type == ParameterType.INTEGER:
                # Similar to float but round to integer
                adjustment = self._calculate_adjustment(param, gradient)
                self.velocity[param.name] = (self.momentum * self.velocity[param.name] +
                                            (1 - self.momentum) * adjustment)
                
                old_value = self.current_config[param.name]
                new_value = int(round(old_value + self.velocity[param.name]))
                
                # Clip to bounds
                if param.min_value is not None:
                    new_value = max(int(param.min_value), new_value)
                if param.max_value is not None:
                    new_value = min(int(param.max_value), new_value)
                
                new_config[param.name] = new_value
        
        # Record adjustment
        self.adjustment_history.append({
            'config': new_config,
            'performance': performance,
            'gradient': gradient
        })
        
        self.current_config = new_config
        return new_config
    
    def _calculate_adjustment(self, param: ParameterDefinition,
                             gradient: float) -> float:
        """Calculate parameter adjustment based on gradient"""
        
        # Base adjustment proportional to gradient
        adjustment = gradient * self.adjustment_rate
        
        # Scale by parameter range
        if param.min_value is not None and param.max_value is not None:
            range_size = param.max_value - param.min_value
            adjustment *= range_size
        
        # Apply logarithmic scaling if specified
        if param.scale == "log":
            current_value = self.current_config.get(param.name, param.default_value)
            if current_value > 0:
                adjustment *= np.log10(current_value + 1)
        
        return adjustment
    
    def adapt_learning_rate(self):
        """Adapt learning rate based on performance history"""
        if len(self.performance_history) < 10:
            return
        
        # Check recent performance trend
        recent = self.performance_history[-10:]
        trend = np.polyfit(range(len(recent)), recent, 1)[0]
        
        if trend > 0:
            # Performance improving, increase learning rate
            self.adjustment_rate = min(0.5, self.adjustment_rate * 1.1)
        else:
            # Performance degrading, decrease learning rate
            self.adjustment_rate = max(0.01, self.adjustment_rate * 0.9)


class HyperparameterTuner:
    """Main hyperparameter tuning system"""
    
    def __init__(self):
        self.grid_search = ParameterGridSearch()
        self.sensitivity_analyzer = SensitivityAnalyzer()
        self.adaptive_adjustment = AdaptiveAdjustment()
        self.tuning_cache = {}
        self.parameter_definitions = []
    
    def define_parameters(self) -> List[ParameterDefinition]:
        """Define all tunable parameters"""
        parameters = [
            # BLF parameters
            ParameterDefinition(
                name="blf_max_iterations",
                type=ParameterType.INTEGER,
                min_value=100,
                max_value=10000,
                default_value=1000,
                description="Maximum iterations for BLF",
                affects_quality=True,
                affects_speed=True
            ),
            ParameterDefinition(
                name="blf_backtrack_threshold",
                type=ParameterType.FLOAT,
                min_value=0.001,
                max_value=0.1,
                default_value=0.01,
                description="Threshold for triggering backtracking",
                affects_quality=True
            ),
            ParameterDefinition(
                name="blf_lookahead_depth",
                type=ParameterType.INTEGER,
                min_value=1,
                max_value=10,
                default_value=3,
                description="Lookahead depth for placement",
                affects_quality=True,
                affects_speed=True
            ),
            
            # Dynamic Programming parameters
            ParameterDefinition(
                name="dp_grid_resolution",
                type=ParameterType.INTEGER,
                min_value=5,
                max_value=50,
                default_value=10,
                description="Grid resolution for DP",
                affects_quality=True,
                affects_speed=True
            ),
            ParameterDefinition(
                name="dp_pruning_threshold",
                type=ParameterType.FLOAT,
                min_value=0.8,
                max_value=1.0,
                default_value=0.95,
                description="Pruning threshold for dominated states",
                affects_quality=True,
                affects_speed=True
            ),
            
            # Branch & Bound parameters
            ParameterDefinition(
                name="bb_branching_factor",
                type=ParameterType.INTEGER,
                min_value=2,
                max_value=10,
                default_value=4,
                description="Branching factor for B&B",
                affects_quality=True,
                affects_speed=True
            ),
            ParameterDefinition(
                name="bb_bound_tightness",
                type=ParameterType.FLOAT,
                min_value=0.5,
                max_value=1.0,
                default_value=0.8,
                description="Tightness of bounds",
                affects_quality=True,
                affects_speed=True
            ),
            
            # General parameters
            ParameterDefinition(
                name="time_limit",
                type=ParameterType.FLOAT,
                min_value=0.1,
                max_value=60.0,
                default_value=5.0,
                description="Time limit for optimization",
                affects_speed=True,
                scale="log"
            ),
            ParameterDefinition(
                name="min_coverage_target",
                type=ParameterType.FLOAT,
                min_value=0.8,
                max_value=0.99,
                default_value=0.95,
                description="Minimum coverage target",
                affects_quality=True
            ),
            ParameterDefinition(
                name="use_caching",
                type=ParameterType.BOOLEAN,
                default_value=True,
                description="Enable caching",
                affects_speed=True
            ),
            ParameterDefinition(
                name="parallel_threads",
                type=ParameterType.INTEGER,
                min_value=1,
                max_value=16,
                default_value=4,
                description="Number of parallel threads",
                affects_speed=True
            )
        ]
        
        self.parameter_definitions = parameters
        return parameters
    
    def tune(self, room: Room, panels: List[Panel],
            optimize_fn: Callable[[Room, List[Panel], Dict[str, Any]], OptimizerResult],
            method: str = "grid",
            time_limit: float = 300.0) -> TuningResult:
        """Tune hyperparameters for given problem"""
        import time
        start_time = time.time()
        
        # Define evaluation function
        def evaluate_config(config: Dict[str, Any]) -> float:
            try:
                result = optimize_fn(room, panels, config)
                
                # Score based on coverage and time
                coverage_weight = 0.7
                time_weight = 0.3
                
                coverage_score = result.coverage if result.coverage else 0.0
                time_score = 1.0 / (1.0 + result.execution_time) if result.execution_time else 0.0
                
                score = coverage_weight * coverage_score + time_weight * time_score
                
                return score
            except Exception as e:
                logger.error(f"Error evaluating config: {e}")
                return 0.0
        
        # Get parameter definitions
        if not self.parameter_definitions:
            self.parameter_definitions = self.define_parameters()
        
        # Choose tuning method
        if method == "grid":
            configs = self.grid_search.search(
                self.parameter_definitions,
                evaluate_config,
                resolution=5
            )
        elif method == "adaptive_grid":
            configs = self.grid_search.search_adaptive(
                self.parameter_definitions,
                evaluate_config,
                initial_resolution=3
            )
        elif method == "bayesian":
            configs = self._bayesian_optimization(
                self.parameter_definitions,
                evaluate_config,
                time_limit
            )
        else:
            raise ValueError(f"Unknown tuning method: {method}")
        
        # Sensitivity analysis on best config
        best_config = max(configs, key=lambda c: c.score or 0)
        sensitivity = self.sensitivity_analyzer.analyze(
            self.parameter_definitions,
            evaluate_config,
            best_config.parameters,
            samples_per_param=5
        )
        
        # Calculate parameter importance
        importance = self._calculate_importance(configs, self.parameter_definitions)
        
        # Convergence history
        convergence = [c.score for c in configs if c.score is not None]
        
        tuning_time = time.time() - start_time
        
        logger.info(f"Tuning complete: {len(configs)} evaluations in {tuning_time:.1f}s, "
                   f"best score: {best_config.score:.3f}")
        
        return TuningResult(
            best_config=best_config,
            all_configs=configs,
            sensitivity_analysis=sensitivity,
            parameter_importance=importance,
            convergence_history=convergence,
            tuning_time=tuning_time,
            evaluations=len(configs)
        )
    
    def _bayesian_optimization(self, parameters: List[ParameterDefinition],
                              evaluate_fn: Callable[[Dict[str, Any]], float],
                              time_limit: float) -> List[ParameterConfiguration]:
        """Bayesian optimization (simplified version)"""
        import time
        start_time = time.time()
        
        configs = []
        
        # Initial random sampling
        n_initial = min(20, len(parameters) * 2)
        for _ in range(n_initial):
            config_dict = {}
            for param in parameters:
                if param.type == ParameterType.BOOLEAN:
                    config_dict[param.name] = np.random.choice([True, False])
                elif param.type == ParameterType.CATEGORICAL:
                    config_dict[param.name] = np.random.choice(param.choices)
                elif param.type == ParameterType.INTEGER:
                    config_dict[param.name] = np.random.randint(
                        param.min_value, param.max_value + 1
                    )
                else:
                    config_dict[param.name] = np.random.uniform(
                        param.min_value, param.max_value
                    )
            
            score = evaluate_fn(config_dict)
            configs.append(ParameterConfiguration(
                parameters=config_dict,
                score=score
            ))
        
        # Adaptive sampling based on performance
        while time.time() - start_time < time_limit:
            # Find most promising region
            sorted_configs = sorted(configs, key=lambda c: c.score or 0, reverse=True)
            best_configs = sorted_configs[:max(1, len(sorted_configs) // 5)]
            
            # Sample near best configurations
            new_config = self._sample_near_best(best_configs, parameters)
            score = evaluate_fn(new_config)
            
            configs.append(ParameterConfiguration(
                parameters=new_config,
                score=score
            ))
        
        return configs
    
    def _sample_near_best(self, best_configs: List[ParameterConfiguration],
                         parameters: List[ParameterDefinition]) -> Dict[str, Any]:
        """Sample new configuration near best ones"""
        # Choose a random best config as center
        center = np.random.choice(best_configs)
        
        new_config = {}
        for param in parameters:
            if param.type == ParameterType.BOOLEAN:
                # Small chance to flip
                if np.random.random() < 0.2:
                    new_config[param.name] = not center.parameters[param.name]
                else:
                    new_config[param.name] = center.parameters[param.name]
            
            elif param.type == ParameterType.CATEGORICAL:
                # Small chance to change
                if np.random.random() < 0.2:
                    new_config[param.name] = np.random.choice(param.choices)
                else:
                    new_config[param.name] = center.parameters[param.name]
            
            else:
                # Add Gaussian noise
                center_value = center.parameters[param.name]
                noise_scale = (param.max_value - param.min_value) * 0.1
                new_value = center_value + np.random.normal(0, noise_scale)
                
                # Clip to bounds
                new_value = max(param.min_value, min(param.max_value, new_value))
                
                if param.type == ParameterType.INTEGER:
                    new_value = int(round(new_value))
                
                new_config[param.name] = new_value
        
        return new_config
    
    def _calculate_importance(self, configs: List[ParameterConfiguration],
                            parameters: List[ParameterDefinition]) -> Dict[str, float]:
        """Calculate parameter importance from configurations"""
        if not configs:
            return {}
        
        importance = {}
        
        for param in parameters:
            # Calculate variance in parameter values
            values = [c.parameters[param.name] for c in configs]
            
            if param.type in [ParameterType.FLOAT, ParameterType.CONTINUOUS, ParameterType.INTEGER]:
                # Calculate correlation with score
                scores = [c.score for c in configs if c.score is not None]
                if scores and len(set(values)) > 1:
                    correlation = abs(np.corrcoef(values[:len(scores)], scores)[0, 1])
                    importance[param.name] = correlation if not np.isnan(correlation) else 0.0
                else:
                    importance[param.name] = 0.0
            else:
                # For categorical/boolean, use variance in scores for each value
                value_scores = {}
                for config in configs:
                    if config.score is not None:
                        value = config.parameters[param.name]
                        if value not in value_scores:
                            value_scores[value] = []
                        value_scores[value].append(config.score)
                
                if value_scores:
                    mean_scores = [np.mean(scores) for scores in value_scores.values()]
                    importance[param.name] = np.std(mean_scores) if len(mean_scores) > 1 else 0.0
                else:
                    importance[param.name] = 0.0
        
        # Normalize
        total = sum(importance.values())
        if total > 0:
            for param_name in importance:
                importance[param_name] /= total
        
        return importance
    
    def save_tuning_results(self, result: TuningResult, filepath: str):
        """Save tuning results to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(result, f)
        
        # Also save human-readable version
        json_filepath = filepath.replace('.pkl', '.json')
        json_data = {
            'best_config': result.best_config.parameters,
            'best_score': result.best_config.score,
            'sensitivity': result.sensitivity_analysis,
            'importance': result.parameter_importance,
            'evaluations': result.evaluations,
            'tuning_time': result.tuning_time
        }
        
        with open(json_filepath, 'w') as f:
            json.dump(json_data, f, indent=2)
    
    def load_tuning_results(self, filepath: str) -> TuningResult:
        """Load tuning results from file"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
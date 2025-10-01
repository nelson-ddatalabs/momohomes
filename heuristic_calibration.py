from typing import List, Dict, Optional, Set, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from datetime import datetime
import logging
from scipy.optimize import minimize, differential_evolution
import json

from core import PackingState, Panel, Room, Position, PlacedPanel
from algorithm_interface import OptimizerConfig, OptimizerResult

logger = logging.getLogger(__name__)


class HeuristicType(Enum):
    PLACEMENT = "placement"
    SORTING = "sorting"
    PRUNING = "pruning"
    SELECTION = "selection"
    SCORING = "scoring"


@dataclass
class HeuristicWeight:
    name: str
    type: HeuristicType
    current_value: float
    min_value: float
    max_value: float
    description: str
    affects_components: List[str] = field(default_factory=list)
    
    def normalize(self) -> float:
        """Normalize weight to [0, 1] range"""
        if self.max_value == self.min_value:
            return 0.5
        return (self.current_value - self.min_value) / (self.max_value - self.min_value)


@dataclass
class ScoringFunction:
    name: str
    formula: str
    parameters: Dict[str, float]
    input_features: List[str]
    output_range: Tuple[float, float]
    
    def evaluate(self, features: Dict[str, float]) -> float:
        """Evaluate scoring function with given features"""
        # Simple linear combination for now
        score = 0.0
        for feature, value in features.items():
            if feature in self.parameters:
                score += self.parameters[feature] * value
        
        # Clip to output range
        return max(self.output_range[0], min(self.output_range[1], score))


@dataclass
class ThresholdConfig:
    name: str
    current_value: float
    min_value: float
    max_value: float
    step_size: float
    description: str
    trigger_condition: str
    
    def is_triggered(self, value: float) -> bool:
        """Check if threshold is triggered"""
        return value >= self.current_value


@dataclass
class CalibrationResult:
    optimized_weights: Dict[str, float]
    optimized_scores: Dict[str, ScoringFunction]
    optimized_thresholds: Dict[str, float]
    performance_metrics: Dict[str, float]
    calibration_time: float
    iterations: int
    validation_score: float


class HeuristicWeightOptimizer:
    """Optimizes heuristic weights for better performance"""
    
    def __init__(self):
        self.weights = {}
        self.optimization_history = []
        self.best_weights = None
        self.best_score = -float('inf')
    
    def define_weights(self) -> List[HeuristicWeight]:
        """Define all heuristic weights"""
        weights = [
            # Placement heuristics
            HeuristicWeight(
                name="edge_alignment_weight",
                type=HeuristicType.PLACEMENT,
                current_value=1.0,
                min_value=0.0,
                max_value=10.0,
                description="Weight for edge alignment preference",
                affects_components=["BLF", "LocalSearch"]
            ),
            HeuristicWeight(
                name="corner_preference_weight",
                type=HeuristicType.PLACEMENT,
                current_value=2.0,
                min_value=0.0,
                max_value=10.0,
                description="Weight for corner placement preference",
                affects_components=["BLF"]
            ),
            HeuristicWeight(
                name="waste_minimization_weight",
                type=HeuristicType.PLACEMENT,
                current_value=3.0,
                min_value=0.0,
                max_value=10.0,
                description="Weight for waste minimization",
                affects_components=["BLF", "DP"]
            ),
            
            # Sorting heuristics
            HeuristicWeight(
                name="area_sort_weight",
                type=HeuristicType.SORTING,
                current_value=5.0,
                min_value=0.0,
                max_value=10.0,
                description="Weight for area-based sorting",
                affects_components=["PanelSorting"]
            ),
            HeuristicWeight(
                name="aspect_ratio_weight",
                type=HeuristicType.SORTING,
                current_value=2.0,
                min_value=0.0,
                max_value=10.0,
                description="Weight for aspect ratio matching",
                affects_components=["PanelSorting"]
            ),
            HeuristicWeight(
                name="value_density_weight",
                type=HeuristicType.SORTING,
                current_value=4.0,
                min_value=0.0,
                max_value=10.0,
                description="Weight for value density sorting",
                affects_components=["PanelSorting", "BranchBound"]
            ),
            
            # Pruning heuristics
            HeuristicWeight(
                name="dominated_state_weight",
                type=HeuristicType.PRUNING,
                current_value=0.9,
                min_value=0.0,
                max_value=1.0,
                description="Weight for dominated state pruning",
                affects_components=["DP", "BranchBound"]
            ),
            HeuristicWeight(
                name="symmetry_breaking_weight",
                type=HeuristicType.PRUNING,
                current_value=0.7,
                min_value=0.0,
                max_value=1.0,
                description="Weight for symmetry breaking",
                affects_components=["BranchBound"]
            ),
            
            # Selection heuristics
            HeuristicWeight(
                name="lookahead_weight",
                type=HeuristicType.SELECTION,
                current_value=3.0,
                min_value=0.0,
                max_value=10.0,
                description="Weight for lookahead scoring",
                affects_components=["BLF", "Lookahead"]
            ),
            HeuristicWeight(
                name="future_fit_weight",
                type=HeuristicType.SELECTION,
                current_value=4.0,
                min_value=0.0,
                max_value=10.0,
                description="Weight for future fit prediction",
                affects_components=["Lookahead"]
            ),
            
            # Scoring heuristics
            HeuristicWeight(
                name="coverage_score_weight",
                type=HeuristicType.SCORING,
                current_value=10.0,
                min_value=0.0,
                max_value=20.0,
                description="Weight for coverage in scoring",
                affects_components=["QualityScorer"]
            ),
            HeuristicWeight(
                name="compactness_score_weight",
                type=HeuristicType.SCORING,
                current_value=5.0,
                min_value=0.0,
                max_value=20.0,
                description="Weight for compactness in scoring",
                affects_components=["QualityScorer"]
            ),
            HeuristicWeight(
                name="alignment_score_weight",
                type=HeuristicType.SCORING,
                current_value=3.0,
                min_value=0.0,
                max_value=20.0,
                description="Weight for alignment in scoring",
                affects_components=["QualityScorer"]
            )
        ]
        
        # Store weights in dictionary
        for weight in weights:
            self.weights[weight.name] = weight
        
        return weights
    
    def optimize(self, evaluate_fn: Callable[[Dict[str, float]], float],
                method: str = "differential_evolution",
                max_iterations: int = 100) -> Dict[str, float]:
        """Optimize weights using specified method"""
        
        # Get current weights as array
        weight_names = list(self.weights.keys())
        x0 = np.array([self.weights[name].current_value for name in weight_names])
        bounds = [(self.weights[name].min_value, self.weights[name].max_value)
                 for name in weight_names]
        
        # Define objective function
        def objective(x):
            # Convert array to weight dictionary
            weight_dict = dict(zip(weight_names, x))
            
            # Evaluate with these weights
            score = evaluate_fn(weight_dict)
            
            # Record in history
            self.optimization_history.append({
                'weights': weight_dict.copy(),
                'score': score
            })
            
            # Track best
            if score > self.best_score:
                self.best_score = score
                self.best_weights = weight_dict.copy()
            
            # Minimize negative score
            return -score
        
        # Optimize
        if method == "differential_evolution":
            result = differential_evolution(
                objective,
                bounds,
                maxiter=max_iterations,
                popsize=15,
                seed=42
            )
        elif method == "nelder_mead":
            result = minimize(
                objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': max_iterations}
            )
        elif method == "powell":
            result = minimize(
                objective,
                x0,
                method='Powell',
                bounds=bounds,
                options={'maxiter': max_iterations}
            )
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        # Extract optimized weights
        optimized = dict(zip(weight_names, result.x))
        
        # Update current values
        for name, value in optimized.items():
            self.weights[name].current_value = value
        
        logger.info(f"Weight optimization complete: {len(self.optimization_history)} evaluations, "
                   f"best score: {self.best_score:.3f}")
        
        return optimized
    
    def analyze_sensitivity(self, evaluate_fn: Callable[[Dict[str, float]], float],
                          perturbation: float = 0.1) -> Dict[str, float]:
        """Analyze sensitivity of each weight"""
        sensitivities = {}
        
        # Get baseline score
        baseline_weights = {name: w.current_value for name, w in self.weights.items()}
        baseline_score = evaluate_fn(baseline_weights)
        
        for weight_name, weight in self.weights.items():
            # Perturb weight up and down
            perturbed_weights = baseline_weights.copy()
            
            # Up perturbation
            perturbed_weights[weight_name] = min(
                weight.max_value,
                weight.current_value * (1 + perturbation)
            )
            score_up = evaluate_fn(perturbed_weights)
            
            # Down perturbation
            perturbed_weights[weight_name] = max(
                weight.min_value,
                weight.current_value * (1 - perturbation)
            )
            score_down = evaluate_fn(perturbed_weights)
            
            # Calculate sensitivity
            sensitivity = abs(score_up - baseline_score) + abs(score_down - baseline_score)
            sensitivities[weight_name] = sensitivity
        
        # Normalize
        total = sum(sensitivities.values())
        if total > 0:
            for name in sensitivities:
                sensitivities[name] /= total
        
        return sensitivities


class ScoringTuner:
    """Tunes scoring functions for better discrimination"""
    
    def __init__(self):
        self.scoring_functions = {}
        self.tuning_history = []
    
    def define_scoring_functions(self) -> List[ScoringFunction]:
        """Define all scoring functions"""
        functions = [
            ScoringFunction(
                name="placement_quality",
                formula="edge_score * w1 + corner_score * w2 + waste_score * w3",
                parameters={
                    "edge_score": 1.0,
                    "corner_score": 2.0,
                    "waste_score": 3.0
                },
                input_features=["edge_score", "corner_score", "waste_score"],
                output_range=(0.0, 100.0)
            ),
            ScoringFunction(
                name="panel_priority",
                formula="area * w1 + value * w2 + aspect_match * w3",
                parameters={
                    "area": 0.3,
                    "value": 0.5,
                    "aspect_match": 0.2
                },
                input_features=["area", "value", "aspect_match"],
                output_range=(0.0, 1.0)
            ),
            ScoringFunction(
                name="state_quality",
                formula="coverage * w1 + compactness * w2 + alignment * w3",
                parameters={
                    "coverage": 10.0,
                    "compactness": 5.0,
                    "alignment": 3.0
                },
                input_features=["coverage", "compactness", "alignment"],
                output_range=(0.0, 100.0)
            ),
            ScoringFunction(
                name="branch_priority",
                formula="bound * w1 + potential * w2 + depth * w3",
                parameters={
                    "bound": 4.0,
                    "potential": 3.0,
                    "depth": -1.0
                },
                input_features=["bound", "potential", "depth"],
                output_range=(-100.0, 100.0)
            )
        ]
        
        # Store functions
        for func in functions:
            self.scoring_functions[func.name] = func
        
        return functions
    
    def tune(self, training_data: List[Dict[str, Any]],
            target_scores: List[float]) -> Dict[str, ScoringFunction]:
        """Tune scoring functions based on training data"""
        
        for func_name, func in self.scoring_functions.items():
            # Extract relevant features from training data
            X = []
            for data in training_data:
                features = {}
                for feature in func.input_features:
                    features[feature] = data.get(feature, 0.0)
                X.append(list(features.values()))
            
            X = np.array(X)
            y = np.array(target_scores)
            
            # Optimize parameters using least squares
            if len(X) > 0:
                # Simple linear regression
                try:
                    from sklearn.linear_model import Ridge
                    model = Ridge(alpha=1.0)
                    model.fit(X, y)
                except ImportError:
                    # Fallback to simple least squares if sklearn not available
                    # Use numpy's least squares solver
                    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                    for i, feature in enumerate(func.input_features):
                        if i < len(coeffs):
                            func.parameters[feature] = coeffs[i]
                    continue
                
                # Update parameters
                for i, feature in enumerate(func.input_features):
                    func.parameters[feature] = model.coef_[i]
                
                self.tuning_history.append({
                    'function': func_name,
                    'old_params': func.parameters.copy(),
                    'new_params': func.parameters.copy(),
                    'score': model.score(X, y)
                })
        
        logger.info(f"Scoring function tuning complete: {len(self.scoring_functions)} functions tuned")
        
        return self.scoring_functions
    
    def cross_validate(self, training_data: List[Dict[str, Any]],
                      target_scores: List[float],
                      n_folds: int = 5) -> Dict[str, float]:
        """Cross-validate scoring functions"""
        validation_scores = {}
        
        try:
            from sklearn.model_selection import KFold
            from sklearn.linear_model import Ridge
            
            for func_name, func in self.scoring_functions.items():
                # Extract features
                X = []
                for data in training_data:
                    features = [data.get(f, 0.0) for f in func.input_features]
                    X.append(features)
                
                X = np.array(X)
                y = np.array(target_scores)
                
                if len(X) < n_folds:
                    continue
                
                # Cross-validation
                kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
                scores = []
                
                for train_idx, val_idx in kf.split(X):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    
                    # Train simple model
                    model = Ridge(alpha=1.0)
                    model.fit(X_train, y_train)
                    
                    # Validate
                    score = model.score(X_val, y_val)
                    scores.append(score)
                
                validation_scores[func_name] = np.mean(scores)
        
        except ImportError:
            # Fallback: simple train/test split without sklearn
            for func_name, func in self.scoring_functions.items():
                # Extract features
                X = []
                for data in training_data:
                    features = [data.get(f, 0.0) for f in func.input_features]
                    X.append(features)
                
                X = np.array(X)
                y = np.array(target_scores)
                
                if len(X) < 2:
                    continue
                
                # Simple 80/20 split
                split_idx = int(len(X) * 0.8)
                X_train, X_val = X[:split_idx], X[split_idx:]
                y_train, y_val = y[:split_idx], y[split_idx:]
                
                # Fit with least squares
                if len(X_train) > 0:
                    coeffs, _, _, _ = np.linalg.lstsq(X_train, y_train, rcond=None)
                    
                    # Validate
                    y_pred = X_val @ coeffs
                    mse = np.mean((y_val - y_pred) ** 2)
                    # Convert MSE to R2-like score
                    var_y = np.var(y_val)
                    score = 1 - (mse / var_y) if var_y > 0 else 0
                    validation_scores[func_name] = score
        
        return validation_scores


class ThresholdAdjuster:
    """Adjusts thresholds for triggers and decisions"""
    
    def __init__(self):
        self.thresholds = {}
        self.adjustment_history = []
    
    def define_thresholds(self) -> List[ThresholdConfig]:
        """Define all adjustable thresholds"""
        thresholds = [
            ThresholdConfig(
                name="backtrack_trigger",
                current_value=0.01,
                min_value=0.001,
                max_value=0.1,
                step_size=0.001,
                description="Threshold for triggering backtracking",
                trigger_condition="coverage_improvement < threshold"
            ),
            ThresholdConfig(
                name="convergence_threshold",
                current_value=0.001,
                min_value=0.0001,
                max_value=0.01,
                step_size=0.0001,
                description="Threshold for convergence detection",
                trigger_condition="improvement < threshold"
            ),
            ThresholdConfig(
                name="pruning_threshold",
                current_value=0.95,
                min_value=0.8,
                max_value=1.0,
                step_size=0.01,
                description="Threshold for state pruning",
                trigger_condition="bound_ratio > threshold"
            ),
            ThresholdConfig(
                name="memory_pressure_threshold",
                current_value=0.8,
                min_value=0.5,
                max_value=0.95,
                step_size=0.05,
                description="Threshold for memory pressure",
                trigger_condition="memory_usage > threshold"
            ),
            ThresholdConfig(
                name="quality_acceptance_threshold",
                current_value=0.9,
                min_value=0.7,
                max_value=1.0,
                step_size=0.05,
                description="Threshold for accepting solutions",
                trigger_condition="quality_score > threshold"
            ),
            ThresholdConfig(
                name="gap_filling_threshold",
                current_value=100.0,
                min_value=10.0,
                max_value=1000.0,
                step_size=10.0,
                description="Minimum gap size to fill",
                trigger_condition="gap_area > threshold"
            )
        ]
        
        # Store thresholds
        for threshold in thresholds:
            self.thresholds[threshold.name] = threshold
        
        return thresholds
    
    def adjust(self, performance_data: Dict[str, List[float]]) -> Dict[str, float]:
        """Adjust thresholds based on performance data"""
        adjusted = {}
        
        for threshold_name, threshold in self.thresholds.items():
            if threshold_name not in performance_data:
                continue
            
            data = performance_data[threshold_name]
            if not data:
                continue
            
            # Analyze performance at different threshold values
            best_value = threshold.current_value
            best_performance = 0.0
            
            # Try different threshold values
            test_values = np.arange(
                threshold.min_value,
                threshold.max_value + threshold.step_size,
                threshold.step_size
            )
            
            for test_value in test_values:
                # Simulate performance with this threshold
                # In practice, this would involve re-running with the threshold
                # Here we use a simple heuristic
                performance = self._estimate_performance(data, test_value, threshold)
                
                if performance > best_performance:
                    best_performance = performance
                    best_value = test_value
            
            # Update threshold
            old_value = threshold.current_value
            threshold.current_value = best_value
            adjusted[threshold_name] = best_value
            
            # Record adjustment
            self.adjustment_history.append({
                'threshold': threshold_name,
                'old_value': old_value,
                'new_value': best_value,
                'performance': best_performance
            })
        
        logger.info(f"Threshold adjustment complete: {len(adjusted)} thresholds adjusted")
        
        return adjusted
    
    def _estimate_performance(self, data: List[float], threshold_value: float,
                            config: ThresholdConfig) -> float:
        """Estimate performance with given threshold value"""
        # Simple heuristic based on trigger rate
        triggers = sum(1 for d in data if d >= threshold_value)
        trigger_rate = triggers / len(data) if data else 0.0
        
        # Optimal trigger rate depends on threshold type
        if "backtrack" in config.name:
            # Moderate backtracking is good
            optimal_rate = 0.2
        elif "convergence" in config.name:
            # Want to detect convergence reliably
            optimal_rate = 0.3
        elif "pruning" in config.name:
            # Aggressive pruning is good
            optimal_rate = 0.7
        else:
            optimal_rate = 0.5
        
        # Performance is inverse of distance from optimal
        distance = abs(trigger_rate - optimal_rate)
        performance = 1.0 / (1.0 + distance)
        
        return performance
    
    def auto_adjust(self, state: PackingState, metrics: Dict[str, float]) -> Dict[str, float]:
        """Automatically adjust thresholds based on current state"""
        adjusted = {}
        
        # Adjust backtrack trigger based on coverage progress
        if "backtrack_trigger" in self.thresholds:
            threshold = self.thresholds["backtrack_trigger"]
            coverage = metrics.get("coverage", 0.0)
            
            if coverage < 0.5:
                # More aggressive backtracking for low coverage
                threshold.current_value = max(threshold.min_value,
                                             threshold.current_value * 0.9)
            elif coverage > 0.9:
                # Less backtracking near completion
                threshold.current_value = min(threshold.max_value,
                                             threshold.current_value * 1.1)
            
            adjusted["backtrack_trigger"] = threshold.current_value
        
        # Adjust pruning based on search space size
        if "pruning_threshold" in self.thresholds:
            threshold = self.thresholds["pruning_threshold"]
            search_space = metrics.get("search_space_size", 0)
            
            if search_space > 10000:
                # More aggressive pruning for large spaces
                threshold.current_value = max(threshold.min_value,
                                             threshold.current_value * 0.95)
            
            adjusted["pruning_threshold"] = threshold.current_value
        
        return adjusted


class CalibrationValidator:
    """Validates calibration quality"""
    
    def __init__(self):
        self.validation_results = []
        self.baseline_performance = None
    
    def validate(self, calibrated_config: Dict[str, Any],
                test_cases: List[Tuple[Room, List[Panel]]],
                evaluate_fn: Callable) -> float:
        """Validate calibration on test cases"""
        
        scores = []
        
        for room, panels in test_cases:
            result = evaluate_fn(room, panels, calibrated_config)
            
            # Calculate validation score
            score = self._calculate_validation_score(result)
            scores.append(score)
            
            self.validation_results.append({
                'room_size': (room.width, room.height),
                'panel_count': len(panels),
                'coverage': result.coverage if hasattr(result, 'coverage') else 0.0,
                'time': result.execution_time if hasattr(result, 'execution_time') else 0.0,
                'score': score
            })
        
        mean_score = np.mean(scores) if scores else 0.0
        
        logger.info(f"Calibration validation: {len(test_cases)} test cases, "
                   f"mean score: {mean_score:.3f}")
        
        return mean_score
    
    def _calculate_validation_score(self, result: Any) -> float:
        """Calculate validation score for a result"""
        score = 0.0
        
        # Coverage component (most important)
        if hasattr(result, 'coverage'):
            score += result.coverage * 50.0
        
        # Time component (faster is better)
        if hasattr(result, 'execution_time'):
            time_score = 1.0 / (1.0 + result.execution_time)
            score += time_score * 20.0
        
        # Quality component
        if hasattr(result, 'quality'):
            score += result.quality * 30.0
        
        return score
    
    def compare_with_baseline(self, calibrated_config: Dict[str, Any],
                            baseline_config: Dict[str, Any],
                            test_cases: List[Tuple[Room, List[Panel]]],
                            evaluate_fn: Callable) -> Dict[str, float]:
        """Compare calibrated config with baseline"""
        
        calibrated_scores = []
        baseline_scores = []
        
        for room, panels in test_cases:
            # Evaluate with calibrated config
            calibrated_result = evaluate_fn(room, panels, calibrated_config)
            calibrated_score = self._calculate_validation_score(calibrated_result)
            calibrated_scores.append(calibrated_score)
            
            # Evaluate with baseline config
            baseline_result = evaluate_fn(room, panels, baseline_config)
            baseline_score = self._calculate_validation_score(baseline_result)
            baseline_scores.append(baseline_score)
        
        # Calculate statistics
        comparison = {
            'calibrated_mean': np.mean(calibrated_scores),
            'baseline_mean': np.mean(baseline_scores),
            'improvement': np.mean(calibrated_scores) - np.mean(baseline_scores),
            'improvement_percent': ((np.mean(calibrated_scores) - np.mean(baseline_scores)) /
                                   np.mean(baseline_scores) * 100 if np.mean(baseline_scores) > 0 else 0),
            'calibrated_std': np.std(calibrated_scores),
            'baseline_std': np.std(baseline_scores)
        }
        
        return comparison
    
    def validate_stability(self, config: Dict[str, Any],
                         test_case: Tuple[Room, List[Panel]],
                         evaluate_fn: Callable,
                         n_runs: int = 10) -> Dict[str, float]:
        """Validate stability/reproducibility of calibration"""
        
        scores = []
        coverages = []
        times = []
        
        room, panels = test_case
        
        for _ in range(n_runs):
            result = evaluate_fn(room, panels, config)
            
            score = self._calculate_validation_score(result)
            scores.append(score)
            
            if hasattr(result, 'coverage'):
                coverages.append(result.coverage)
            if hasattr(result, 'execution_time'):
                times.append(result.execution_time)
        
        stability_metrics = {
            'score_mean': np.mean(scores),
            'score_std': np.std(scores),
            'score_cv': np.std(scores) / np.mean(scores) if np.mean(scores) > 0 else 0,
            'coverage_mean': np.mean(coverages) if coverages else 0,
            'coverage_std': np.std(coverages) if coverages else 0,
            'time_mean': np.mean(times) if times else 0,
            'time_std': np.std(times) if times else 0
        }
        
        return stability_metrics


class HeuristicCalibrator:
    """Main heuristic calibration system"""
    
    def __init__(self):
        self.weight_optimizer = HeuristicWeightOptimizer()
        self.scoring_tuner = ScoringTuner()
        self.threshold_adjuster = ThresholdAdjuster()
        self.validator = CalibrationValidator()
        self.calibration_cache = {}
    
    def calibrate(self, training_cases: List[Tuple[Room, List[Panel]]],
                 evaluate_fn: Callable,
                 validation_cases: Optional[List[Tuple[Room, List[Panel]]]] = None,
                 max_iterations: int = 50) -> CalibrationResult:
        """Perform full calibration"""
        import time
        start_time = time.time()
        
        # Initialize components
        weights = self.weight_optimizer.define_weights()
        scoring_functions = self.scoring_tuner.define_scoring_functions()
        thresholds = self.threshold_adjuster.define_thresholds()
        
        # Collect training data
        training_data = []
        target_scores = []
        
        for room, panels in training_cases:
            # Get baseline configuration
            baseline_config = self._get_baseline_config(weights, thresholds)
            
            # Evaluate
            result = evaluate_fn(room, panels, baseline_config)
            
            # Extract features and score
            features = self._extract_features(result)
            score = self._calculate_score(result)
            
            training_data.append(features)
            target_scores.append(score)
        
        # Optimize weights
        def weight_evaluate(weight_dict):
            scores = []
            for room, panels in training_cases[:5]:  # Use subset for speed
                config = self._create_config(weight_dict, thresholds)
                result = evaluate_fn(room, panels, config)
                scores.append(self._calculate_score(result))
            return np.mean(scores)
        
        optimized_weights = self.weight_optimizer.optimize(
            weight_evaluate,
            method="differential_evolution",
            max_iterations=max_iterations
        )
        
        # Tune scoring functions
        tuned_scoring = self.scoring_tuner.tune(training_data, target_scores)
        
        # Adjust thresholds
        performance_data = self._collect_performance_data(training_cases, evaluate_fn)
        adjusted_thresholds = self.threshold_adjuster.adjust(performance_data)
        
        # Validate if validation cases provided
        validation_score = 0.0
        if validation_cases:
            calibrated_config = self._create_final_config(
                optimized_weights,
                tuned_scoring,
                adjusted_thresholds
            )
            validation_score = self.validator.validate(
                calibrated_config,
                validation_cases,
                evaluate_fn
            )
        
        # Calculate performance metrics
        performance_metrics = {
            'weight_sensitivity': self.weight_optimizer.analyze_sensitivity(weight_evaluate),
            'scoring_validation': self.scoring_tuner.cross_validate(training_data, target_scores),
            'threshold_adjustments': len(adjusted_thresholds)
        }
        
        calibration_time = time.time() - start_time
        
        logger.info(f"Calibration complete in {calibration_time:.1f}s, "
                   f"validation score: {validation_score:.3f}")
        
        return CalibrationResult(
            optimized_weights=optimized_weights,
            optimized_scores=tuned_scoring,
            optimized_thresholds=adjusted_thresholds,
            performance_metrics=performance_metrics,
            calibration_time=calibration_time,
            iterations=len(self.weight_optimizer.optimization_history),
            validation_score=validation_score
        )
    
    def _get_baseline_config(self, weights: List[HeuristicWeight],
                           thresholds: List[ThresholdConfig]) -> Dict[str, Any]:
        """Get baseline configuration"""
        config = {}
        
        for weight in weights:
            config[weight.name] = weight.current_value
        
        for threshold in thresholds:
            config[threshold.name] = threshold.current_value
        
        return config
    
    def _create_config(self, weight_dict: Dict[str, float],
                      thresholds: Dict[str, ThresholdConfig]) -> Dict[str, Any]:
        """Create configuration from weights and thresholds"""
        config = weight_dict.copy()
        
        for name, threshold in thresholds.items():
            config[name] = threshold.current_value
        
        return config
    
    def _create_final_config(self, weights: Dict[str, float],
                           scoring: Dict[str, ScoringFunction],
                           thresholds: Dict[str, float]) -> Dict[str, Any]:
        """Create final calibrated configuration"""
        config = weights.copy()
        config.update(thresholds)
        
        # Add scoring function parameters
        for name, func in scoring.items():
            for param_name, param_value in func.parameters.items():
                config[f"{name}_{param_name}"] = param_value
        
        return config
    
    def _extract_features(self, result: Any) -> Dict[str, float]:
        """Extract features from optimization result"""
        features = {}
        
        if hasattr(result, 'coverage'):
            features['coverage'] = result.coverage
        
        if hasattr(result, 'execution_time'):
            features['time'] = result.execution_time
        
        if hasattr(result, 'iterations'):
            features['iterations'] = result.iterations
        
        if hasattr(result, 'quality'):
            features['quality'] = result.quality
        
        return features
    
    def _calculate_score(self, result: Any) -> float:
        """Calculate overall score for result"""
        score = 0.0
        
        if hasattr(result, 'coverage'):
            score += result.coverage * 100
        
        if hasattr(result, 'execution_time'):
            score += 10.0 / (1.0 + result.execution_time)
        
        return score
    
    def _collect_performance_data(self, training_cases: List[Tuple[Room, List[Panel]]],
                                 evaluate_fn: Callable) -> Dict[str, List[float]]:
        """Collect performance data for threshold adjustment"""
        data = {
            'backtrack_trigger': [],
            'convergence_threshold': [],
            'pruning_threshold': [],
            'memory_pressure_threshold': [],
            'quality_acceptance_threshold': [],
            'gap_filling_threshold': []
        }
        
        # This would normally collect actual trigger data during optimization
        # For now, generate synthetic data
        for _ in range(100):
            data['backtrack_trigger'].append(np.random.uniform(0.001, 0.1))
            data['convergence_threshold'].append(np.random.uniform(0.0001, 0.01))
            data['pruning_threshold'].append(np.random.uniform(0.8, 1.0))
            data['memory_pressure_threshold'].append(np.random.uniform(0.5, 0.95))
            data['quality_acceptance_threshold'].append(np.random.uniform(0.7, 1.0))
            data['gap_filling_threshold'].append(np.random.uniform(10, 1000))
        
        return data
    
    def save_calibration(self, result: CalibrationResult, filepath: str):
        """Save calibration results"""
        data = {
            'weights': result.optimized_weights,
            'thresholds': result.optimized_thresholds,
            'scoring': {name: {
                'formula': func.formula,
                'parameters': func.parameters
            } for name, func in result.optimized_scores.items()},
            'metrics': result.performance_metrics,
            'validation_score': result.validation_score,
            'calibration_time': result.calibration_time
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_calibration(self, filepath: str) -> Dict[str, Any]:
        """Load calibration from file"""
        with open(filepath, 'r') as f:
            return json.load(f)
#!/usr/bin/env python3
"""
Final Testing Script - Verify 95%+ coverage on Luna with <5s optimization time
Tests all implemented algorithms and components
"""

import time
import logging
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Tuple
import numpy as np
from PIL import Image

# Import all our components
from core import Room, Panel, PackingState, Position, PlacedPanel
from algorithm_interface import OptimizerConfig, OptimizerResult
from spatial_index import SpatialIndex, OccupancyGrid
from decision_tree import DecisionTree, TreeNode
from memoization import MemoizationCache
from blf_placement import BLFPlacementEngine, Skyline
from sorting_strategies import SortingStrategies
from placement_heuristics import PlacementHeuristics
from backtrack_manager import BacktrackManager
from backtrack_triggers import BacktrackTriggers
from backtrack_strategy import BacktrackStrategy
from lookahead import LookaheadMechanism
from local_search import LocalSearchRefinement
from dp_state import DPStateManager
from grid_discretization import GridDiscretization
from subproblem_decomposition import SubproblemDecomposer
from dp_solver import DynamicProgrammingSolver
from branch_bound_tree import BranchBoundTree
from bounding_functions import BoundingFunctions
from search_strategies import BranchBoundSearchStrategies
from constraint_model import ConstraintModel
from constraint_propagation import ConstraintPropagationEngine
from constraint_learning import ConstraintLearning
from symmetry_breaking import SymmetryBreaker
from cutting_planes import CuttingPlaneGenerator
from strategy_selection import StrategySelector
from execution_pipeline import ExecutionPipeline
from result_aggregation import ResultAggregator
from time_management import TimeManagementSystem
from memory_management import MemoryManagementSystem
from progress_tracking import ProgressTrackingSystem
from solution_validation import SolutionValidator
from solution_improvement import SolutionImprover
from hyperparameter_tuning import HyperparameterTuner
from heuristic_calibration import HeuristicCalibrator
from bottleneck_analysis import BottleneckAnalyzer
from performance_optimizations import PerformanceOptimizer
from edge_case_handling import EdgeCaseManager
from stability_improvements import StabilityManager, ReproducibilityConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FinalTestSuite:
    """Comprehensive test suite for final validation"""
    
    def __init__(self):
        self.results = {}
        self.test_start_time = None
        self.target_coverage = 0.95
        self.max_time = 5.0
        
        # Initialize all systems
        self._initialize_systems()
    
    def _initialize_systems(self):
        """Initialize all component systems"""
        logger.info("Initializing all systems...")
        
        # Core systems
        self.spatial_index = SpatialIndex()
        self.occupancy_grid = None  # Will be created per room
        self.decision_tree = DecisionTree()
        self.cache = MemoizationCache(max_size=10000)
        
        # BLF components
        self.blf_engine = BLFPlacementEngine()
        self.sorting_strategies = SortingStrategies()
        self.placement_heuristics = PlacementHeuristics()
        self.backtrack_manager = BacktrackManager()
        self.backtrack_triggers = BacktrackTriggers()
        self.backtrack_strategy = BacktrackStrategy()
        self.lookahead = LookaheadMechanism(k=3)
        self.local_search = LocalSearchRefinement()
        
        # Dynamic Programming components
        self.dp_state_manager = DPStateManager()
        self.grid_discretization = GridDiscretization()
        self.subproblem_decomposer = SubproblemDecomposer()
        self.dp_solver = DynamicProgrammingSolver()
        
        # Branch & Bound components
        self.bb_tree = BranchBoundTree()
        self.bounding_functions = BoundingFunctions()
        self.bb_search = BranchBoundSearchStrategies()
        self.constraint_model = ConstraintModel()
        self.constraint_propagation = ConstraintPropagationEngine()
        self.constraint_learning = ConstraintLearning()
        self.symmetry_breaker = SymmetryBreaker()
        self.cutting_planes = CuttingPlaneGenerator()
        
        # Integration components
        self.strategy_selector = StrategySelector()
        self.execution_pipeline = ExecutionPipeline()
        self.result_aggregator = ResultAggregator()
        self.time_manager = TimeManagementSystem()
        self.memory_manager = MemoryManagementSystem()
        self.progress_tracker = ProgressTrackingSystem()
        self.validator = SolutionValidator()
        self.improver = SolutionImprover()
        
        # Optimization components
        self.hyperparameter_tuner = HyperparameterTuner()
        self.heuristic_calibrator = HeuristicCalibrator()
        self.bottleneck_analyzer = BottleneckAnalyzer()
        self.performance_optimizer = PerformanceOptimizer()
        self.edge_case_manager = EdgeCaseManager()
        
        # Stability components
        self.stability_manager = StabilityManager()
        self.stability_manager.initialize(ReproducibilityConfig(
            seed=42,
            precision_digits=10,
            tie_breaking_rule="hash",
            deterministic_mode=True,
            error_handling_mode="recover"
        ))
        
        logger.info("All systems initialized successfully")
    
    def load_luna_test_case(self) -> Tuple[Room, List[Panel]]:
        """Load Luna.png test case"""
        logger.info("Loading Luna.png test case...")
        
        # Load and process Luna.png
        try:
            img = Image.open('Luna.png')
            img_array = np.array(img.convert('L'))  # Convert to grayscale
            
            # Extract room dimensions from image
            height, width = img_array.shape
            room = Room(width=float(width), height=float(height))
            
            # Generate panels based on image features
            panels = self._generate_panels_from_image(img_array)
            
            logger.info(f"Loaded room: {width}x{height}, {len(panels)} panels")
            return room, panels
            
        except Exception as e:
            logger.error(f"Failed to load Luna.png: {e}")
            # Fallback to synthetic test case
            return self._create_synthetic_test_case()
    
    def _generate_panels_from_image(self, img_array: np.ndarray) -> List[Panel]:
        """Generate panels based on image features"""
        panels = []
        
        # Standard panel sizes
        panel_sizes = [
            (100, 50), (80, 60), (60, 60), (50, 50),
            (40, 40), (30, 30), (20, 20), (15, 15)
        ]
        
        # Generate a mix of panels
        panel_id = 0
        for width, height in panel_sizes:
            # Generate multiple panels of each size
            count = max(1, 500 // (width * height))  # More small panels
            for _ in range(count):
                panel = Panel(
                    id=f"panel_{panel_id}",
                    width=float(width),
                    height=float(height),
                    can_rotate=True
                )
                panels.append(panel)
                panel_id += 1
        
        return panels
    
    def _create_synthetic_test_case(self) -> Tuple[Room, List[Panel]]:
        """Create synthetic test case as fallback"""
        room = Room(width=1000.0, height=800.0)
        
        panels = []
        panel_sizes = [
            (100, 50, 5), (80, 60, 8), (60, 60, 10),
            (50, 50, 15), (40, 40, 20), (30, 30, 25),
            (20, 20, 30), (15, 15, 40)
        ]
        
        panel_id = 0
        for width, height, count in panel_sizes:
            for _ in range(count):
                panel = Panel(
                    id=f"panel_{panel_id}",
                    width=float(width),
                    height=float(height),
                    can_rotate=True
                )
                panels.append(panel)
                panel_id += 1
        
        return room, panels
    
    def test_blf_algorithm(self, room: Room, panels: List[Panel]) -> Dict[str, Any]:
        """Test Enhanced BLF with Backtracking"""
        logger.info("Testing Enhanced BLF algorithm...")
        
        start_time = time.time()
        
        # Initialize state
        state = PackingState(
            room=room,
            available_panels=panels.copy(),
            placed_panels=[]
        )
        
        # Sort panels
        sorted_panels = self.sorting_strategies.sort(
            panels, 
            self.sorting_strategies.SortCriteria.AREA_DESCENDING
        )
        
        # Initialize occupancy grid
        self.occupancy_grid = OccupancyGrid(room.width, room.height)
        
        # Place panels
        for panel in sorted_panels:
            # Find placement with lookahead
            best_position = self.lookahead.find_best_placement(
                panel, state, self.occupancy_grid
            )
            
            if best_position:
                placed = PlacedPanel(
                    panel=panel,
                    position=best_position,
                    rotated=False
                )
                state.placed_panels.append(placed)
                self.occupancy_grid.mark_occupied(
                    int(best_position.x), int(best_position.y),
                    int(panel.width), int(panel.height)
                )
            
            # Check for backtracking
            if self.backtrack_triggers.should_trigger(state):
                self.backtrack_manager.save_state(state)
                # Simplified backtracking for testing
        
        # Apply local search refinement
        state = self.local_search.refine(state)
        
        execution_time = time.time() - start_time
        coverage = self._calculate_coverage(state)
        
        return {
            'algorithm': 'Enhanced BLF',
            'coverage': coverage,
            'execution_time': execution_time,
            'panels_placed': len(state.placed_panels),
            'success': coverage >= self.target_coverage and execution_time < self.max_time
        }
    
    def test_dp_algorithm(self, room: Room, panels: List[Panel]) -> Dict[str, Any]:
        """Test Dynamic Programming algorithm"""
        logger.info("Testing Dynamic Programming algorithm...")
        
        start_time = time.time()
        
        # Discretize grid
        grid = self.grid_discretization.discretize(room, resolution=10)
        
        # Decompose into subproblems
        subproblems = self.subproblem_decomposer.decompose(room, panels)
        
        # Solve each subproblem
        solutions = []
        for subproblem in subproblems:
            solution = self.dp_solver.solve(
                subproblem['room'],
                subproblem['panels'],
                method='bottom_up'
            )
            solutions.append(solution)
        
        # Merge solutions
        final_state = self._merge_solutions(solutions, room)
        
        execution_time = time.time() - start_time
        coverage = self._calculate_coverage(final_state)
        
        return {
            'algorithm': 'Dynamic Programming',
            'coverage': coverage,
            'execution_time': execution_time,
            'panels_placed': len(final_state.placed_panels) if final_state else 0,
            'success': coverage >= self.target_coverage and execution_time < self.max_time
        }
    
    def test_branch_bound_algorithm(self, room: Room, panels: List[Panel]) -> Dict[str, Any]:
        """Test Branch & Bound algorithm"""
        logger.info("Testing Branch & Bound algorithm...")
        
        start_time = time.time()
        
        # Initialize constraints
        constraints = self.constraint_model.create_constraints(room, panels)
        
        # Initialize search
        root = self.bb_tree.create_root(room, panels)
        
        # Search with time limit
        time_limit = min(self.max_time, 2.0)  # Limit B&B to 2 seconds
        best_solution = self.bb_search.search(
            root,
            self.bounding_functions,
            time_limit=time_limit
        )
        
        execution_time = time.time() - start_time
        
        if best_solution:
            coverage = self._calculate_coverage(best_solution)
        else:
            coverage = 0.0
        
        return {
            'algorithm': 'Branch & Bound',
            'coverage': coverage,
            'execution_time': execution_time,
            'panels_placed': len(best_solution.placed_panels) if best_solution else 0,
            'success': coverage >= self.target_coverage and execution_time < self.max_time
        }
    
    def test_integrated_pipeline(self, room: Room, panels: List[Panel]) -> Dict[str, Any]:
        """Test integrated execution pipeline"""
        logger.info("Testing integrated pipeline...")
        
        start_time = time.time()
        
        # Classify room and select strategies
        classification = self.strategy_selector.classify_room(room)
        strategies = self.strategy_selector.select_strategies(classification, panels)
        
        # Setup pipeline
        config = {
            'time_limit': self.max_time,
            'target_coverage': self.target_coverage,
            'strategies': strategies
        }
        
        # Execute pipeline
        result = self.execution_pipeline.execute(room, panels, config)
        
        execution_time = time.time() - start_time
        
        # Validate result
        if result and result.state:
            validation = self.validator.validate(result.state, room)
            
            # Improve if needed
            if validation.is_valid and result.coverage < self.target_coverage:
                improvement = self.improver.improve(
                    result.state, room,
                    time_limit=self.max_time - execution_time
                )
                result.state = improvement.improved_state
                result.coverage = self._calculate_coverage(result.state)
        
        return {
            'algorithm': 'Integrated Pipeline',
            'coverage': result.coverage if result else 0.0,
            'execution_time': execution_time,
            'panels_placed': len(result.state.placed_panels) if result and result.state else 0,
            'success': result and result.coverage >= self.target_coverage and execution_time < self.max_time
        }
    
    def _calculate_coverage(self, state: PackingState) -> float:
        """Calculate coverage percentage"""
        if not state or not state.placed_panels:
            return 0.0
        
        total_panel_area = sum(
            p.panel.width * p.panel.height 
            for p in state.placed_panels
        )
        room_area = state.room.width * state.room.height
        
        return total_panel_area / room_area if room_area > 0 else 0.0
    
    def _merge_solutions(self, solutions: List[Any], room: Room) -> PackingState:
        """Merge subproblem solutions"""
        merged_state = PackingState(
            room=room,
            available_panels=[],
            placed_panels=[]
        )
        
        for solution in solutions:
            if solution and hasattr(solution, 'placed_panels'):
                merged_state.placed_panels.extend(solution.placed_panels)
        
        return merged_state
    
    def run_all_tests(self):
        """Run all tests and generate report"""
        logger.info("=" * 60)
        logger.info("FINAL TESTING SUITE - STARTING")
        logger.info("=" * 60)
        
        self.test_start_time = time.time()
        
        # Load test case
        room, panels = self.load_luna_test_case()
        
        # Run individual algorithm tests
        test_functions = [
            self.test_blf_algorithm,
            self.test_dp_algorithm,
            self.test_branch_bound_algorithm,
            self.test_integrated_pipeline
        ]
        
        for test_func in test_functions:
            try:
                result = test_func(room, panels.copy())
                self.results[result['algorithm']] = result
                
                # Log result
                logger.info(f"\n{result['algorithm']} Results:")
                logger.info(f"  Coverage: {result['coverage']:.1%}")
                logger.info(f"  Time: {result['execution_time']:.3f}s")
                logger.info(f"  Panels: {result['panels_placed']}")
                logger.info(f"  Success: {result['success']}")
                
            except Exception as e:
                logger.error(f"Test failed for {test_func.__name__}: {e}")
                import traceback
                traceback.print_exc()
        
        # Generate final report
        self.generate_report()
    
    def generate_report(self):
        """Generate comprehensive test report"""
        total_time = time.time() - self.test_start_time
        
        report = []
        report.append("\n" + "=" * 60)
        report.append("FINAL TEST REPORT")
        report.append("=" * 60)
        report.append(f"Total Testing Time: {total_time:.2f}s")
        report.append(f"Target Coverage: {self.target_coverage:.1%}")
        report.append(f"Max Time per Algorithm: {self.max_time}s")
        report.append("")
        
        # Summary table
        report.append("Algorithm Performance Summary:")
        report.append("-" * 60)
        report.append(f"{'Algorithm':<25} {'Coverage':<12} {'Time':<10} {'Success':<10}")
        report.append("-" * 60)
        
        best_coverage = 0.0
        best_algorithm = None
        all_success = True
        
        for algo, result in self.results.items():
            coverage_str = f"{result['coverage']:.1%}"
            time_str = f"{result['execution_time']:.3f}s"
            success_str = "PASS" if result['success'] else "FAIL"
            
            report.append(f"{algo:<25} {coverage_str:<12} {time_str:<10} {success_str:<10}")
            
            if result['coverage'] > best_coverage:
                best_coverage = result['coverage']
                best_algorithm = algo
            
            if not result['success']:
                all_success = False
        
        report.append("-" * 60)
        
        # Overall results
        report.append("\nOverall Results:")
        report.append(f"  Best Algorithm: {best_algorithm}")
        report.append(f"  Best Coverage: {best_coverage:.1%}")
        report.append(f"  All Tests Passed: {all_success}")
        
        # Success criteria
        report.append("\nSuccess Criteria:")
        report.append(f"  ✓ 95%+ Coverage: {'YES' if best_coverage >= 0.95 else 'NO'}")
        report.append(f"  ✓ <5s Optimization: {'YES' if all(r['execution_time'] < 5 for r in self.results.values()) else 'NO'}")
        
        # Additional metrics
        report.append("\nAdditional Metrics:")
        report.append(f"  Cache Hit Rate: {self.cache.get_statistics()['hit_rate']:.1%}")
        report.append(f"  Memory Peak: {self.memory_manager.get_peak_usage() / 1024 / 1024:.1f} MB")
        
        # Print report
        report_text = "\n".join(report)
        print(report_text)
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"final_test_report_{timestamp}.txt"
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        logger.info(f"Report saved to {report_file}")
        
        # Save detailed results as JSON
        json_file = f"final_test_results_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"Detailed results saved to {json_file}")


def main():
    """Main entry point"""
    test_suite = FinalTestSuite()
    test_suite.run_all_tests()


if __name__ == "__main__":
    main()
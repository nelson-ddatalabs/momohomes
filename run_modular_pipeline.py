#!/usr/bin/env python3
"""
Run Modular Cassette Optimization Pipeline
===========================================
This script runs the new modular pipeline system on floor plans.
Can work with both image input (with edge detection) or direct polygon input.
"""

import sys
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any
import logging

# Import pipeline components
from optimization_pipeline import OptimizationPipeline, ValidationStage
from edge_cleaner_stage import EdgeCleaner
from corner_placer_stage import CornerPlacer
from perimeter_tracer_stage import PerimeterTracer
from concentric_filler_stage import ConcentricFiller
from gap_filler_stage import GapFiller
from mathematical_verifier_stage import MathematicalVerifier
from intelligent_backtracker_stage import IntelligentBacktracker
from coverage_analyzer_stage import CoverageAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_optimization_pipeline(config: Dict[str, Any] = None) -> OptimizationPipeline:
    """
    Create the complete optimization pipeline with configurable parameters.

    Args:
        config: Optional configuration dictionary

    Returns:
        Configured optimization pipeline
    """
    if config is None:
        config = {}

    pipeline = OptimizationPipeline()

    # Add all stages with configurable parameters
    pipeline.add_stage(ValidationStage())

    pipeline.add_stage(EdgeCleaner(
        min_edge_length=config.get('min_edge_length', 2.0),
        collinear_tolerance=config.get('collinear_tolerance', 0.01)
    ))

    pipeline.add_stage(CornerPlacer(
        min_angle=config.get('min_corner_angle', 30.0)
    ))

    pipeline.add_stage(PerimeterTracer(
        micro_adjustment=config.get('micro_adjustment', 0.125),
        max_gap=config.get('max_gap', 0.5)
    ))

    pipeline.add_stage(ConcentricFiller(
        initial_offset=config.get('initial_offset', 8.0),
        layer_spacing=config.get('layer_spacing', 6.0)
    ))

    pipeline.add_stage(GapFiller(
        grid_resolution=config.get('gap_grid_resolution', 0.5),
        min_gap_area=config.get('min_gap_area', 4.0)
    ))

    if config.get('enable_backtracking', True):
        pipeline.add_stage(IntelligentBacktracker(
            max_iterations=config.get('backtrack_iterations', 5),
            min_improvement=config.get('min_improvement', 0.5)
        ))

    if config.get('enable_verification', True):
        pipeline.add_stage(MathematicalVerifier(
            grid_resolution=config.get('verification_grid', 0.25)
        ))

    if config.get('enable_analysis', True):
        pipeline.add_stage(CoverageAnalyzer(
            target_coverage=config.get('target_coverage', 94.0)
        ))

    return pipeline


def load_polygon_from_file(filepath: str) -> List[Tuple[float, float]]:
    """
    Load polygon from various file formats.

    Args:
        filepath: Path to polygon file

    Returns:
        List of (x, y) vertices
    """
    path = Path(filepath)

    if path.suffix == '.json':
        with open(filepath, 'r') as f:
            data = json.load(f)

            # Check various possible keys
            if 'polygon' in data:
                return [(p[0], p[1]) for p in data['polygon']]
            elif 'vertices' in data:
                return [(p[0], p[1]) for p in data['vertices']]
            elif isinstance(data, list):
                return [(p[0], p[1]) for p in data]

    elif path.suffix in ['.txt', '.csv']:
        polygon = []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.replace(',', ' ').split()
                    if len(parts) >= 2:
                        polygon.append((float(parts[0]), float(parts[1])))
        return polygon

    elif path.suffix in ['.png', '.jpg', '.jpeg']:
        # For images, try to use the cardinal edge detection system
        try:
            from cassette_layout_system_cardinal import CassetteLayoutSystemCardinal

            print(f"Processing image file: {filepath}")
            print("This will require manual edge measurement input...")

            output_dir = f"output_{path.stem}"
            system = CassetteLayoutSystemCardinal(output_dir=output_dir)
            result = system.process_floor_plan(filepath)

            if result['success'] and 'polygon' in result:
                return result['polygon']
            else:
                print(f"Error: Could not extract polygon from image")
                return None

        except ImportError:
            print("Error: Image processing requires cassette_layout_system_cardinal.py")
            return None

    raise ValueError(f"Unsupported file format: {path.suffix}")


def load_test_polygon(name: str) -> List[Tuple[float, float]]:
    """
    Load predefined test polygons.

    Args:
        name: Name of test polygon (bungalow, luna, rectangle, l-shape)

    Returns:
        List of vertices
    """
    test_polygons = {
        'bungalow': [
            (0, 37), (23.5, 37), (23.5, 30.5), (42, 30.5),
            (42, 6.5), (23.5, 6.5), (23.5, 0), (0, 0),
            (0, 15.5), (0, 37)
        ],
        'rectangle': [
            (0, 0), (40, 0), (40, 30), (0, 30)
        ],
        'l-shape': [
            (0, 0), (30, 0), (30, 20), (15, 20),
            (15, 35), (0, 35)
        ],
        'luna': [
            (0, 50), (78, 50), (78, 0), (8.5, 0),
            (8.5, 37), (0, 37)
        ]
    }

    if name.lower() in test_polygons:
        return test_polygons[name.lower()]

    # Try to load from results file
    results_path = Path(f"output_{name}-Conditioned/results_cardinal.json")
    if results_path.exists():
        with open(results_path, 'r') as f:
            data = json.load(f)
            if 'polygon' in data:
                return [(p[0], p[1]) for p in data['polygon']]

    raise ValueError(f"Unknown test polygon: {name}")


def save_results(results: Dict[str, Any], output_path: str, polygon: List[Tuple[float, float]] = None):
    """Save optimization results to file."""
    # Add polygon to results if not present
    if polygon and 'polygon' not in results:
        results['polygon'] = polygon

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_path}")


def print_results(results: Dict[str, Any]):
    """Print formatted results."""
    print("\n" + "=" * 70)
    print("OPTIMIZATION RESULTS")
    print("=" * 70)

    print(f"Total cassettes: {results['num_cassettes']}")
    print(f"Total area: {results['total_area']:.1f} sq ft")
    print(f"Covered area: {results['covered_area']:.1f} sq ft")
    print(f"Gap area: {results['gap_area']:.1f} sq ft")
    print(f"Coverage: {results['coverage_percent']:.1f}%")
    print(f"Meets 94% requirement: {results['meets_requirement']}")
    print(f"Total weight: {results['total_weight']:.1f} lbs")

    print("\n" + "=" * 70)
    print("STAGE-BY-STAGE PROGRESSION")
    print("=" * 70)

    for stage_name, stage_data in results['stage_results'].items():
        if 'coverage_after' in stage_data:
            print(f"{stage_name:25} {stage_data.get('coverage_after', 0):6.1f}% "
                  f"(+{stage_data.get('coverage_gain', 0):5.1f}%)")

    print("\n" + "=" * 70)
    print("CASSETTE SIZE DISTRIBUTION")
    print("=" * 70)

    for size, count in sorted(results['size_distribution'].items()):
        print(f"  {size:5}: {count:3} cassettes")

    # Check for analysis results
    if 'CoverageAnalyzer' in results.get('stage_results', {}):
        analyzer_meta = results['stage_results']['CoverageAnalyzer'].get('metadata', {})
        if 'coverage_analysis' in analyzer_meta:
            analysis = analyzer_meta['coverage_analysis']

            print("\n" + "=" * 70)
            print("COVERAGE ANALYSIS")
            print("=" * 70)

            print(f"Theoretical maximum: {analysis['theoretical_maximum']:.1f}%")

            if analysis['recommendations']:
                print("\nRecommendations:")
                for rec in analysis['recommendations']:
                    print(f"  {rec}")


def main():
    """Main entry point."""
    print("\n" + "=" * 70)
    print("MODULAR CASSETTE OPTIMIZATION PIPELINE")
    print("=" * 70)

    # Parse arguments
    if len(sys.argv) > 1:
        input_arg = sys.argv[1]

        # Check if it's a test polygon name
        if input_arg.lower() in ['bungalow', 'luna', 'rectangle', 'l-shape']:
            print(f"\nLoading test polygon: {input_arg}")
            polygon = load_test_polygon(input_arg)
            output_name = f"{input_arg.lower()}_modular"

        # Check if it's a file path
        elif Path(input_arg).exists():
            print(f"\nLoading polygon from file: {input_arg}")
            polygon = load_polygon_from_file(input_arg)
            if polygon is None:
                print("Error: Could not load polygon from file")
                return
            output_name = Path(input_arg).stem + "_modular"

        else:
            print(f"Error: Unknown input: {input_arg}")
            print("\nUsage:")
            print("  python run_modular_pipeline.py <test_name>")
            print("  python run_modular_pipeline.py <path_to_file>")
            print("\nTest names: bungalow, luna, rectangle, l-shape")
            print("File formats: .json, .txt, .csv, .png, .jpg")
            return

    else:
        print("\nAvailable options:")
        print("1. Test polygons: bungalow, luna, rectangle, l-shape")
        print("2. Load from file (.json, .txt, .csv, .png)")
        print("3. Enter coordinates manually")

        choice = input("\nEnter choice (1-3): ").strip()

        if choice == '1':
            name = input("Enter test name (bungalow/luna/rectangle/l-shape): ").strip()
            polygon = load_test_polygon(name)
            output_name = f"{name}_modular"

        elif choice == '2':
            filepath = input("Enter file path: ").strip()
            polygon = load_polygon_from_file(filepath)
            if polygon is None:
                print("Error: Could not load polygon")
                return
            output_name = Path(filepath).stem + "_modular"

        elif choice == '3':
            print("\nEnter vertices (x y) one per line. Empty line to finish:")
            polygon = []
            while True:
                line = input(f"Vertex {len(polygon) + 1}: ").strip()
                if not line:
                    break
                parts = line.replace(',', ' ').split()
                if len(parts) >= 2:
                    polygon.append((float(parts[0]), float(parts[1])))
            output_name = "custom_modular"

        else:
            print("Invalid choice")
            return

    if not polygon or len(polygon) < 3:
        print("Error: Invalid polygon (need at least 3 vertices)")
        return

    print(f"\nPolygon loaded: {len(polygon)} vertices")
    print(f"Area: {calculate_polygon_area(polygon):.1f} sq ft")

    # Configuration options
    config = {}

    if '--fast' in sys.argv:
        print("\nUsing fast mode (no backtracking or analysis)")
        config['enable_backtracking'] = False
        config['enable_analysis'] = False

    if '--no-verify' in sys.argv:
        config['enable_verification'] = False

    # Create and run pipeline
    print("\nRunning optimization pipeline...")
    print("-" * 70)

    pipeline = create_optimization_pipeline(config)
    results = pipeline.optimize(polygon)

    # Print and save results
    print_results(results)

    # Save results with polygon data
    output_path = f"{output_name}_results.json"
    save_results(results, output_path, polygon)

    # Create visualization automatically
    try:
        from visualize_results import visualize_optimization_results

        print("\nGenerating visualization...")
        visualize_optimization_results(
            polygon=polygon,
            cassettes=results['cassettes'],
            output_name=output_name
        )

    except Exception as e:
        print(f"Visualization skipped: {e}")


def calculate_polygon_area(polygon: List[Tuple[float, float]]) -> float:
    """Calculate polygon area using shoelace formula."""
    n = len(polygon)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]
    return abs(area) / 2.0


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Run Backtracking Cassette Optimizer
====================================
Optimal cassette placement using backtracking search with pruning.
Guarantees best solution within time constraints.
"""

import sys
import json
from pathlib import Path
from backtracking_optimizer import BacktrackingOptimizer


def get_polygon_from_name(name: str):
    """Get pre-defined polygon by name"""
    polygons = {
        'luna': [
            (0, 0),
            (21.0, 0),
            (21.0, 8.5),
            (14.0, 8.5),
            (14.0, 21.5),
            (29.0, 21.5),
            (29.0, 43.5),
            (8.0, 43.5),
            (8.0, 38.0),
            (0, 38.0)
        ],
        'bungalow': [
            (0.0, 43.5),
            (8.0, 43.5),
            (8.0, 38.0),
            (14.0, 38.0),
            (14.0, 43.5),
            (29.0, 43.5),
            (29.0, 21.5),
            (21.0, 21.5),
            (21.0, 0.0),
            (0.0, 0.0)
        ],
        'umbra': [
            (0.0, 28.0),
            (55.5, 28.0),
            (55.5, 12.0),
            (16.0, 12.0),
            (16.0, 0.0),
            (0.0, 0.0)
        ],
        'rectangle': [
            (0, 0),
            (40, 0),
            (40, 30),
            (0, 30)
        ]
    }

    name_lower = name.lower()
    for key in polygons:
        if key in name_lower:
            return polygons[key]

    return None


def load_polygon_from_file(filepath: Path):
    """Load polygon from JSON file"""
    if filepath.suffix == '.json':
        with open(filepath, 'r') as f:
            data = json.load(f)
            if 'polygon' in data:
                return data['polygon']
    return None


def main():
    """Main entry point for backtracking optimizer"""

    # Get input from command line
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        print("\nUsage: python run_backtracking_optimizer.py <polygon_name_or_file> [output_dir] [max_depth]")
        print("\nSupported inputs:")
        print("  - Pre-defined polygons: luna, bungalow, umbra, rectangle")
        print("  - JSON files with 'polygon' field")
        print("\nOptional arguments:")
        print("  - output_dir: Directory for results (default: output_backtrack_<name>)")
        print("  - max_depth: Maximum search depth (default: 15)")
        input_path = input("\nEnter polygon name or file path: ").strip()

    # Parse output directory
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    else:
        output_dir = f"output_backtrack_{Path(input_path).stem}"

    # Parse max depth
    if len(sys.argv) > 3:
        try:
            max_depth = int(sys.argv[3])
        except ValueError:
            print(f"Invalid max_depth: {sys.argv[3]}, using default 15")
            max_depth = 15
    else:
        max_depth = 15

    # Get polygon
    polygon = None

    # Try as pre-defined name first
    polygon = get_polygon_from_name(input_path)

    # Try as file path
    if polygon is None:
        filepath = Path(input_path)
        if filepath.exists():
            polygon = load_polygon_from_file(filepath)

    if polygon is None:
        print(f"Error: Could not load polygon from '{input_path}'")
        print("Please provide a valid polygon name (luna, bungalow, umbra) or JSON file")
        return

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("BACKTRACKING CASSETTE OPTIMIZER")
    print("="*70)
    print(f"Output directory: {output_dir}")
    print(f"Max depth: {max_depth}")

    # Run backtracking optimization
    optimizer = BacktrackingOptimizer(polygon)
    result = optimizer.optimize(max_depth=max_depth)

    # Display results
    print("\n" + "="*70)
    print("OPTIMIZATION RESULTS")
    print("="*70)
    print(f"Coverage: {result['coverage_percent']:.1f}%")
    print(f"Cassettes: {result['num_cassettes']} units")
    print(f"Total Area: {result['total_area']:.1f} sq ft")
    print(f"Covered Area: {result['covered_area']:.1f} sq ft")
    print(f"Gap Area: {result['gap_area']:.1f} sq ft")
    print(f"Total Weight: {result['total_weight']:.0f} lbs")
    print()
    print("Search Statistics:")
    stats = result['search_stats']
    print(f"  Nodes explored: {stats['nodes_explored']}")
    print(f"  Nodes pruned: {stats['nodes_pruned']}")
    print(f"  Search time: {stats['search_time']:.1f}s")
    print()
    print("Cassette Size Distribution:")
    for size, count in sorted(result['size_distribution'].items()):
        w, h = map(int, size.split('x'))
        area = w * h
        print(f"  {size:7} : {count:3} cassettes ({area:3.0f} sq ft each)")

    # Save results
    results_file = output_path / 'results_backtrack.json'
    with open(results_file, 'w') as f:
        json.dump({
            'cassettes': result['cassettes'],
            'statistics': {
                'coverage_percent': result['coverage_percent'],
                'num_cassettes': result['num_cassettes'],
                'total_area': result['total_area'],
                'covered_area': result['covered_area'],
                'gap_area': result['gap_area'],
                'total_weight': result['total_weight'],
                'size_distribution': result['size_distribution']
            },
            'search_stats': result['search_stats'],
            'polygon': polygon
        }, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    # Generate visualization if available
    try:
        from hundred_percent_visualizer import create_simple_visualization

        vis_path = output_path / 'cassette_layout_backtrack.png'

        # Format statistics for visualization
        vis_stats = {
            'coverage': result['coverage_percent'],
            'total_area': result['total_area'],
            'covered': result['covered_area'],
            'gap_area': result['gap_area'],
            'cassettes': result['num_cassettes'],
            'total_weight': result['total_weight']
        }

        # Extract floor plan name
        floor_plan_name = Path(input_path).stem.replace('_', ' ').replace('-', ' ').upper()

        create_simple_visualization(
            cassettes=result['cassettes'],
            polygon=polygon,
            statistics=vis_stats,
            output_path=str(vis_path),
            floor_plan_name=floor_plan_name
        )
        print(f"Visualization saved to: {vis_path}")
    except Exception as e:
        print(f"Note: Could not generate visualization: {e}")

    print("="*70)


if __name__ == "__main__":
    main()

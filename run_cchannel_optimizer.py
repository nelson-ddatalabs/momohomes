#!/usr/bin/env python3
"""
Run C-Channel Cassette Optimizer
=================================
Ultra-smart optimizer with perimeter C-channel (1.5" - 18")
Primary goal: Maximize cassette coverage with structural C-channel perimeter
"""

import sys
import json
from pathlib import Path
from cassette_optimizer_with_cchannel import CChannelOptimizer

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
    """Main entry point for C-channel optimizer"""

    # Get input from command line
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        print("\nUsage: python run_cchannel_optimizer.py <polygon_name_or_file> [output_dir]")
        print("\nSupported inputs:")
        print("  • Pre-defined polygons: luna, bungalow, umbra, rectangle")
        print("  • JSON files with 'polygon' field")
        input_path = input("\nEnter polygon name or file path: ").strip()

    # Parse output directory
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    else:
        output_dir = f"output_hundred_{Path(input_path).stem}"

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
    print("C-CHANNEL CASSETTE OPTIMIZER")
    print("="*70)
    print(f"Output directory: {output_dir}")

    # Run C-channel optimization
    optimizer = CChannelOptimizer(polygon)
    result = optimizer.optimize()

    # Display results
    stats = result['statistics']
    print("\n" + "="*70)
    print("OPTIMIZATION RESULTS")
    print("="*70)
    print(f"Total Area: {stats['total_area']:.1f} sq ft")
    print(f"Coverage: {stats['coverage_percent']:.1f}%")
    print()
    print(f"Cassettes: {stats['cassette_count']} units")
    print(f"Cassette Area: {stats['cassette_area']} sq ft ({stats['cassette_percent']:.1f}%)")
    print(f"Total Weight: {stats['total_weight']:.0f} lbs")
    print()
    print(f"C-Channel Area: {stats['cchannel_area']:.1f} sq ft ({stats['cchannel_percent']:.1f}%)")
    print(f"C-Channel Widths:")
    for direction in ['N', 'S', 'E', 'W']:
        width = stats['cchannel_widths_inches'][direction]
        print(f"  {direction}: {width:.1f}\"")
    print()
    print("Cassette Size Distribution:")
    for size, count in stats['cassette_counts'].items():
        w, h = map(int, size.split('x'))
        area = w * h
        print(f"  {size:7} : {count:3} cassettes ({area:3.0f} sq ft each)")

    # Save results
    results_file = output_path / 'results_hundred_cchannel.json'
    with open(results_file, 'w') as f:
        json.dump({
            'statistics': stats,
            'cchannel_widths': {k: v * 12.0 for k, v in result['cchannel_widths'].items()},
            'polygon': polygon,
            'cassette_count': stats['cassette_count']
        }, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    # Generate visualization if available
    try:
        from hundred_percent_visualizer import create_simple_visualization

        vis_path = output_path / 'cassette_layout_cchannel.png'

        # Format statistics for visualization
        vis_stats = {
            'coverage': stats['coverage_percent'],
            'total_area': stats['total_area'],
            'covered': stats['cassette_area'] + stats['cchannel_area'],
            'gap_area': stats['total_area'] - stats['cassette_area'] - stats['cchannel_area'],
            'cassettes': stats['cassette_count'],
            'total_weight': stats['total_weight'],
            'cchannel_area': stats['cchannel_area'],
            'cchannel_widths': stats['cchannel_widths_inches']
        }

        # Extract floor plan name
        floor_plan_name = Path(input_path).stem.replace('_', ' ').replace('-', ' ').upper()

        create_simple_visualization(
            cassettes=result['cassettes'],
            polygon=result['original_polygon'],
            statistics=vis_stats,
            output_path=str(vis_path),
            floor_plan_name=floor_plan_name,
            inset_polygon=result['inset_polygon']
        )
        print(f"Visualization saved to: {vis_path}")
    except Exception as e:
        print(f"Note: Could not generate visualization: {e}")

    print("="*70)

if __name__ == "__main__":
    main()

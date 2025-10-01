#!/usr/bin/env python3
"""
Run 100% Coverage Cassette Optimizer
=====================================
Primary goal: 100% coverage
Secondary goal: Minimize cassette count
"""

import sys
import json
from pathlib import Path
from hundred_percent_optimizer import HundredPercentOptimizer
from hundred_percent_visualizer import create_simple_visualization

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
    """Load polygon from JSON file or process floor plan image"""

    # Handle JSON files
    if filepath.suffix == '.json':
        with open(filepath, 'r') as f:
            data = json.load(f)
            if 'polygon' in data:
                return data['polygon']

    # Handle image files (floor plans)
    elif filepath.suffix.lower() in ['.png', '.jpg', '.jpeg']:
        try:
            from cassette_layout_system_cardinal import CassetteLayoutSystemCardinal

            print(f"\nProcessing floor plan image: {filepath}")
            print("This will require manual edge measurement input...")
            print("-" * 70)

            # Create output directory for intermediate results
            temp_output_dir = f"output_{filepath.stem}_temp"

            # Process floor plan with measurement collection
            system = CassetteLayoutSystemCardinal(output_dir=temp_output_dir)
            result = system.process_floor_plan(str(filepath))

            if result['success']:
                # Try to get polygon from result
                if 'polygon' in result:
                    print(f"\nPolygon successfully extracted from floor plan")
                    return result['polygon']

                # If polygon not in direct result, try loading from saved JSON
                results_file = Path(temp_output_dir) / 'results_cardinal.json'
                if results_file.exists():
                    with open(results_file, 'r') as f:
                        saved_data = json.load(f)
                        if 'polygon' in saved_data:
                            print(f"\nPolygon loaded from processed results")
                            return saved_data['polygon']

                print(f"Error: Could not extract polygon from floor plan results")
            else:
                print(f"Error processing floor plan: {result.get('error', 'Unknown error')}")

        except ImportError as e:
            print(f"Error: Could not import floor plan processing system: {e}")
        except Exception as e:
            print(f"Error processing floor plan image: {e}")

    return None

def main():
    """Main entry point for 100% coverage optimizer"""

    # Get input from command line
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        print("\nUsage: python run_hundred_percent.py <polygon_name_or_file> [output_dir]")
        print("\nSupported inputs:")
        print("  • Pre-defined polygons: luna, bungalow, umbra, rectangle")
        print("  • Floor plan images: .png, .jpg, .jpeg (will ask for measurements)")
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
        print("Please provide a valid polygon name or JSON file")
        return

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("100% COVERAGE CASSETTE OPTIMIZER")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir}")

    # Run optimization
    optimizer = HundredPercentOptimizer(polygon)
    results = optimizer.optimize()

    # Print results
    print(f"\n{'='*70}")
    print("OPTIMIZATION RESULTS")
    print(f"{'='*70}")
    print(f"Coverage: {results['coverage_percent']:.1f}%")
    print(f"Cassettes: {results['num_cassettes']}")
    print(f"Total area: {results['total_area']:.1f} sq ft")
    print(f"Covered area: {results['covered_area']:.1f} sq ft")
    print(f"Gap area: {results['gap_area']:.1f} sq ft")
    print(f"Total weight: {results['total_weight']:.0f} lbs")
    print(f"Average cassette: {results['avg_cassette_area']:.1f} sq ft")

    if results['meets_requirement']:
        print(f"\nACHIEVED 100% COVERAGE!")
    else:
        print(f"\nCoverage: {results['coverage_percent']:.1f}% (Target: 100%)")

    print("\nSize distribution:")
    for size, count in sorted(results['size_distribution'].items()):
        w, h = size.split('x')
        area = float(w) * float(h)
        print(f"  {size:8s}: {count:3d} cassettes ({area:5.1f} sq ft each)")

    # Save results
    results_file = output_path / 'results_hundred.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # Generate visualization
    try:
        vis_path = output_path / 'cassette_layout_hundred.png'

        # Prepare statistics for visualization
        statistics = {
            'coverage': results['coverage_percent'],
            'total_area': results['total_area'],
            'covered': results['covered_area'],
            'gap_area': results['gap_area'],
            'cassettes': results['num_cassettes'],
            'total_weight': results['total_weight']
        }

        # Extract floor plan name from input path
        floor_plan_name = Path(input_path).stem.replace('_', ' ').replace('-', ' ')

        create_simple_visualization(
            cassettes=results['cassettes'],
            polygon=results['polygon'],
            statistics=statistics,
            output_path=str(vis_path),
            floor_plan_name=floor_plan_name
        )
        print(f"Visualization saved to: {vis_path}")
    except Exception as e:
        print(f"Could not generate visualization: {e}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Run Cassette Layout System on Your Floor Plan
==============================================
This script runs the cassette optimization on your floor plan.
You can choose between the original system or the new modular pipeline.
"""

import sys
from pathlib import Path
from cassette_layout_system_cardinal import CassetteLayoutSystemCardinal

def main():
    # Check for optimizer flags
    use_modular = '--modular' in sys.argv or '-m' in sys.argv
    use_hundred = '--hundred' in sys.argv or '-h100' in sys.argv

    # Special handling for --hundred with floor plan images
    if use_hundred and len(sys.argv) > 1:
        potential_file = sys.argv[1]
        # Check if it's a floor plan image that needs measurement collection
        if Path(potential_file).exists() and Path(potential_file).suffix.lower() in ['.png', '.jpg', '.jpeg']:
            print("\nProcessing floor plan for 100% COVERAGE OPTIMIZER...")
            print("First, we need to collect measurements...")

            # Process the floor plan to get measurements and polygon
            floor_plan_path = potential_file
            output_dir = f"output_{Path(floor_plan_path).stem}_hundred"

            # Use the regular cardinal system to collect measurements
            system = CassetteLayoutSystemCardinal(output_dir=output_dir)
            result = system.process_floor_plan(floor_plan_path)

            if result['success']:
                # Load the polygon from saved results
                results_file = Path(output_dir) / 'results_cardinal.json'
                if results_file.exists():
                    import json
                    with open(results_file, 'r') as f:
                        saved_data = json.load(f)
                        if 'polygon' in saved_data:
                            # Now run 100% optimizer with the extracted polygon
                            print("\nNow running 100% coverage optimizer on extracted polygon...")
                            from hundred_percent_optimizer import HundredPercentOptimizer
                            from hundred_percent_visualizer import create_simple_visualization

                            polygon = saved_data['polygon']
                            optimizer = HundredPercentOptimizer(polygon)
                            results = optimizer.optimize()

                            # Print results
                            print("\n" + "="*70)
                            print("100% COVERAGE OPTIMIZATION RESULTS")
                            print("="*70)
                            print(f"Coverage: {results['coverage_percent']:.1f}%")
                            print(f"Cassettes: {results['num_cassettes']}")
                            print(f"Total area: {results['total_area']:.1f} sq ft")
                            print(f"Covered area: {results['covered_area']:.1f} sq ft")
                            print(f"Gap area: {results['gap_area']:.1f} sq ft")

                            if results['meets_requirement']:
                                print("\nACHIEVED 100% COVERAGE!")
                            else:
                                print(f"\nCoverage: {results['coverage_percent']:.1f}% (Target: 100%)")

                            # Save results
                            results_hundred_file = Path(output_dir) / 'results_hundred.json'
                            with open(results_hundred_file, 'w') as f:
                                json.dump(results, f, indent=2)
                            print(f"\nResults saved to: {results_hundred_file}")

                            # Generate visualization
                            try:
                                vis_path = Path(output_dir) / 'cassette_layout_hundred.png'
                                statistics = {
                                    'coverage': results['coverage_percent'],
                                    'total_area': results['total_area'],
                                    'covered': results['covered_area'],
                                    'gap_area': results['gap_area'],
                                    'cassettes': results['num_cassettes'],
                                    'total_weight': results['total_weight']
                                }
                                # Extract floor plan name from path
                                floor_plan_name = Path(floor_plan_path).stem.replace('_', ' ').replace('-', ' ')
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

                            return
                        else:
                            print("Error: No polygon found in results")
                            return
                else:
                    print(f"Error: Results file not found: {results_file}")
                    return
            else:
                print(f"Error processing floor plan: {result.get('error', 'Unknown error')}")
                return

    # For non-image inputs with --hundred, use the regular flow
    if use_hundred:
        print("\nUsing 100% COVERAGE OPTIMIZER...")
        from run_hundred_percent import main as run_hundred
        # Remove the flag from argv before calling
        sys.argv = [arg for arg in sys.argv if arg not in ['--hundred', '-h100']]
        run_hundred()
        return
    elif use_modular:
        print("\nUsing NEW MODULAR PIPELINE...")
        from run_modular_pipeline import main as run_modular
        # Remove the flag from argv before calling
        sys.argv = [arg for arg in sys.argv if arg not in ['--modular', '-m']]
        run_modular()
        return

    # Get floor plan path from command line or use default
    if len(sys.argv) > 1:
        floor_plan_path = sys.argv[1]
    else:
        print("\nUsage: python run_cassette_system.py <path_to_floor_plan> [options]")
        print("\nOptions:")
        print("  --modular, -m    Use the new modular optimization pipeline")
        print("  --hundred, -h100 Use the 100% coverage optimizer")
        print("\nExample: python run_cassette_system.py floorplans/Luna-Conditioned.png")
        print("Example: python run_cassette_system.py bungalow --modular")
        print("Example: python run_cassette_system.py umbra --hundred")
        floor_plan_path = input("\nEnter path to your floor plan image: ").strip()

    # Check if file exists
    if not Path(floor_plan_path).exists():
        print(f"Error: File not found: {floor_plan_path}")
        return

    # Create output directory name based on input file
    input_name = Path(floor_plan_path).stem
    output_dir = f"output_{input_name}"

    print(f"\nProcessing: {floor_plan_path}")
    print(f"Output will be saved to: {output_dir}/")
    print("-" * 70)

    # Initialize system
    system = CassetteLayoutSystemCardinal(output_dir=output_dir)

    # Process floor plan (will ask for measurements interactively)
    result = system.process_floor_plan(floor_plan_path)

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    if result['success']:
        print(f"✓ Processing successful")
        print(f"  Floor area: {result['area']:.1f} sq ft")
        print(f"  Perimeter: {result['perimeter']:.1f} ft")
        print(f"  Number of cassettes: {result['num_cassettes']}")
        print(f"  Coverage: {result['coverage_percent']:.1f}%")
        print(f"  Total weight: {result['total_weight']:.0f} lbs")

        if result['is_closed']:
            print(f"\n✓ Perfect polygon closure achieved!")
            print(f"  Closure error: {result['closure_error']:.6f} feet")
        else:
            print(f"\n⚠ Polygon closure error: {result['closure_error']:.2f} feet")
            print("  (For perfect closure, ensure ΣE=ΣW and ΣN=ΣS)")

        if result['meets_requirement']:
            print(f"\n✓ Meets 94% coverage requirement")
        else:
            print(f"\n✗ Below 94% coverage requirement")

        print(f"\n📁 Output files:")
        print(f"  • Binary image: {output_dir}/binary.png")
        print(f"  • Edge detection: {output_dir}/cardinal_edges.png")
        print(f"  • Numbered edges: {output_dir}/numbered_cardinal_edges.png")
        print(f"  • Final layout: {output_dir}/cassette_layout_cardinal.png")
        print(f"  • Results JSON: {output_dir}/results_cardinal.json")
    else:
        print(f"✗ Processing failed: {result['error']}")

if __name__ == "__main__":
    main()
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
    use_cchannel = '--cchannel' in sys.argv or '-cc' in sys.argv
    use_backtrack = '--backtrack' in sys.argv or '--bt' in sys.argv
    use_fill = '--fill' in sys.argv

    # Special handling for --fill with floor plan images
    if use_fill and len(sys.argv) > 1:
        potential_file = sys.argv[1]
        if Path(potential_file).exists() and Path(potential_file).suffix.lower() in ['.png', '.jpg', '.jpeg']:
            print("\nProcessing floor plan for PER-CASSETTE C-CHANNEL OPTIMIZER...")
            print("First, we need to collect measurements...")

            floor_plan_path = potential_file
            output_dir = f"output_{Path(floor_plan_path).stem}_fill"

            # Use cardinal system to collect measurements
            system = CassetteLayoutSystemCardinal(output_dir=output_dir)
            result = system.process_floor_plan(floor_plan_path)

            if result['success']:
                # Load polygon from saved results
                results_file = Path(output_dir) / 'results_cardinal.json'
                if results_file.exists():
                    import json
                    with open(results_file, 'r') as f:
                        saved_data = json.load(f)
                        if 'polygon' in saved_data:
                            print("\nNow running Gap Redistribution C-Channel Optimizer...")
                            from gap_redistribution_optimizer import GapRedistributionOptimizer

                            polygon = saved_data['polygon']
                            optimizer = GapRedistributionOptimizer(polygon)
                            fill_result = optimizer.optimize()

                            stats = fill_result['statistics']

                            # Print results
                            print("\n" + "="*70)
                            print("GAP REDISTRIBUTION C-CHANNEL OPTIMIZATION RESULTS")
                            print("="*70)
                            print(f"Total Area: {stats['total_area']:.1f} sq ft")
                            print(f"Coverage: {stats['coverage_percent']:.2f}%")
                            print()
                            print(f"Cassettes: {stats['cassette_count']} units")
                            print(f"Cassette Area: {stats['cassette_area']:.1f} sq ft")
                            print(f"C-Channel Area: {stats['cchannel_area']:.1f} sq ft")
                            print(f"C-Channel Count: {len(fill_result['c_channels'])}")
                            print("="*70)

                            # Save results
                            fill_results_file = Path(output_dir) / 'results_fill.json'
                            with open(fill_results_file, 'w') as f:
                                json.dump({
                                    'cassettes': fill_result['cassettes'],
                                    'c_channels': fill_result['c_channels'],
                                    'c_channels_inches': fill_result['c_channels_inches'],
                                    'statistics': stats,
                                    'polygon': polygon
                                }, f, indent=2)

                            # Generate visualization
                            try:
                                from hundred_percent_visualizer import create_simple_visualization

                                vis_path = Path(output_dir) / 'cassette_layout_fill.png'
                                vis_stats = {
                                    'coverage': stats['coverage_percent'],
                                    'total_area': stats['total_area'],
                                    'covered': stats['cassette_area'] + stats['cchannel_area'],
                                    'cassettes': stats['cassette_count'],
                                    'per_cassette_cchannel': True,  # Flag for visualizer
                                    'cchannel_widths_per_cassette': fill_result['c_channels_inches'],
                                    'cchannel_area': stats['cchannel_area'],  # Add for legend display
                                    'cchannel_geometries': fill_result.get('cchannel_geometries', [])  # NEW: Actual geometries
                                }

                                floor_plan_name = Path(floor_plan_path).stem.replace('_', ' ').replace('-', ' ')

                                create_simple_visualization(
                                    cassettes=fill_result['cassettes'],
                                    polygon=polygon,
                                    statistics=vis_stats,
                                    output_path=str(vis_path),
                                    floor_plan_name=floor_plan_name
                                )
                                print(f"\nVisualization saved to: {vis_path}")
                            except Exception as e:
                                print(f"\nWarning: Could not generate visualization: {e}")

                            print(f"\nResults saved to: {fill_results_file}")
                            return

            print("\nError: Failed to process floor plan")
            return

    # Special handling for --backtrack with floor plan images
    if use_backtrack and len(sys.argv) > 1:
        potential_file = sys.argv[1]
        if Path(potential_file).exists() and Path(potential_file).suffix.lower() in ['.png', '.jpg', '.jpeg']:
            print("\nProcessing floor plan for BACKTRACKING OPTIMIZER...")
            print("First, we need to collect measurements...")

            floor_plan_path = potential_file
            output_dir = f"output_{Path(floor_plan_path).stem}_backtrack"

            # Use cardinal system to collect measurements
            system = CassetteLayoutSystemCardinal(output_dir=output_dir)
            result = system.process_floor_plan(floor_plan_path)

            if result['success']:
                # Load polygon from saved results
                results_file = Path(output_dir) / 'results_cardinal.json'
                if results_file.exists():
                    import json
                    with open(results_file, 'r') as f:
                        saved_data = json.load(f)
                        if 'polygon' in saved_data:
                            print("\nNow running JOINT backtracking optimizer (cassettes + C-channel)...")
                            from joint_backtracking_optimizer import JointBacktrackingOptimizer

                            polygon = saved_data['polygon']
                            optimizer = JointBacktrackingOptimizer(polygon)
                            bt_result = optimizer.optimize(max_time=120, max_depth=10)

                            if not bt_result:
                                print("\nError: Optimization failed")
                                return

                            stats = bt_result['statistics']
                            search = bt_result['search_stats']

                            # Print results
                            print("\n" + "="*70)
                            print("JOINT BACKTRACKING OPTIMIZATION RESULTS")
                            print("="*70)
                            print(f"Total Area: {stats['total_area']:.1f} sq ft")
                            print(f"Coverage: {stats['coverage_percent']:.1f}%")
                            print()
                            print(f"Cassettes: {stats['cassette_count']} units")
                            print(f"Cassette Area: {stats['cassette_area']:.1f} sq ft ({stats['cassette_percent']:.1f}%)")
                            print(f"Total Weight: {stats['total_weight']:.0f} lbs")
                            print()
                            print(f"C-Channel Area: {stats['cchannel_area']:.1f} sq ft ({stats['cchannel_percent']:.1f}%)")
                            print(f"C-Channel Widths:")
                            for direction in ['N', 'S', 'E', 'W']:
                                width = stats['cchannel_widths_inches'][direction]
                                print(f"  {direction}: {width:.1f}\"")
                            print()
                            print("Search Statistics:")
                            print(f"  Total time: {search['total_time']:.1f}s")
                            print(f"  C-channel configs explored: {search['configs_explored']}")
                            print(f"  C-channel configs pruned: {search['configs_pruned']}")
                            print(f"  Cassette nodes explored: {search['cassette_nodes_explored']}")
                            print(f"  Cassette nodes pruned: {search['cassette_nodes_pruned']}")
                            print(f"  Best config: {search['best_config']}")
                            print()
                            print("Cassette Size Distribution:")
                            for size, count in sorted(stats['size_distribution'].items()):
                                w, h = map(int, size.split('x'))
                                area = w * h
                                print(f"  {size:7} : {count:3} cassettes ({area:3.0f} sq ft each)")

                            # Save backtracking results
                            bt_results_file = Path(output_dir) / 'results_backtrack.json'
                            with open(bt_results_file, 'w') as f:
                                json.dump({
                                    'cassettes': bt_result['cassettes'],
                                    'statistics': stats,
                                    'search_stats': search,
                                    'cchannel_widths': {k: v * 12.0 for k, v in bt_result['cchannel_widths'].items()},
                                    'polygon': polygon,
                                    'inset_polygon': bt_result['inset_polygon']
                                }, f, indent=2)

                            # Generate visualization
                            try:
                                from hundred_percent_visualizer import create_simple_visualization

                                vis_path = Path(output_dir) / 'cassette_layout_backtrack.png'
                                vis_stats = {
                                    'coverage': stats['coverage_percent'],
                                    'total_area': stats['total_area'],
                                    'covered': stats['cassette_area'] + stats['cchannel_area'],
                                    'gap_area': stats['gap_area'],
                                    'cassettes': stats['cassette_count'],
                                    'total_weight': stats['total_weight'],
                                    'cchannel_area': stats['cchannel_area'],
                                    'cchannel_widths': stats['cchannel_widths_inches']
                                }

                                floor_plan_name = Path(floor_plan_path).stem.replace('_', ' ').replace('-', ' ')

                                create_simple_visualization(
                                    cassettes=bt_result['cassettes'],
                                    polygon=bt_result['original_polygon'],
                                    statistics=vis_stats,
                                    output_path=str(vis_path),
                                    floor_plan_name=floor_plan_name,
                                    inset_polygon=bt_result['inset_polygon']
                                )
                                print(f"\nVisualization saved to: {vis_path}")
                            except Exception as e:
                                print(f"\nWarning: Could not generate visualization: {e}")

                            print(f"\nBacktracking results saved to: {bt_results_file}")
                            return

            print("\nError: Failed to process floor plan")
            return

        else:
            # Not an image file, delegate to run_backtracking_optimizer
            import run_backtracking_optimizer
            run_backtracking_optimizer.main()
            return

    # Special handling for --cchannel with floor plan images
    if use_cchannel and len(sys.argv) > 1:
        potential_file = sys.argv[1]
        # Check if it's a floor plan image that needs measurement collection
        if Path(potential_file).exists() and Path(potential_file).suffix.lower() in ['.png', '.jpg', '.jpeg']:
            print("\nProcessing floor plan for C-CHANNEL OPTIMIZER...")
            print("First, we need to collect measurements...")

            # Process the floor plan to get measurements and polygon
            floor_plan_path = potential_file
            output_dir = f"output_{Path(floor_plan_path).stem}_cchannel"

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
                            # Now run C-channel optimizer with the extracted polygon
                            print("\nNow running C-channel optimizer on extracted polygon...")
                            from cassette_optimizer_with_cchannel import CChannelOptimizer

                            polygon = saved_data['polygon']
                            optimizer = CChannelOptimizer(polygon)
                            cc_result = optimizer.optimize()
                            stats = cc_result['statistics']

                            # Print results
                            print("\n" + "="*70)
                            print("C-CHANNEL OPTIMIZATION RESULTS")
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

                            # Save results
                            results_cchannel_file = Path(output_dir) / 'results_hundred_cchannel.json'
                            with open(results_cchannel_file, 'w') as f:
                                json.dump({
                                    'statistics': stats,
                                    'cchannel_widths': {k: v * 12.0 for k, v in cc_result['cchannel_widths'].items()},
                                    'polygon': polygon,
                                    'cassette_count': stats['cassette_count']
                                }, f, indent=2)
                            print(f"\nResults saved to: {results_cchannel_file}")

                            # Generate visualization
                            try:
                                from hundred_percent_visualizer import create_simple_visualization

                                vis_path = Path(output_dir) / 'cassette_layout_cchannel.png'
                                statistics = {
                                    'coverage': stats['coverage_percent'],
                                    'total_area': stats['total_area'],
                                    'covered': stats['cassette_area'] + stats['cchannel_area'],
                                    'gap_area': stats['total_area'] - stats['cassette_area'] - stats['cchannel_area'],
                                    'cassettes': stats['cassette_count'],
                                    'total_weight': stats['total_weight'],
                                    'cchannel_area': stats['cchannel_area'],
                                    'cchannel_widths': stats['cchannel_widths_inches']
                                }
                                # Extract floor plan name from path
                                floor_plan_name = Path(floor_plan_path).stem.replace('_', ' ').replace('-', ' ')
                                create_simple_visualization(
                                    cassettes=cc_result['cassettes'],
                                    polygon=cc_result['original_polygon'],
                                    statistics=statistics,
                                    output_path=str(vis_path),
                                    floor_plan_name=floor_plan_name,
                                    inset_polygon=cc_result['inset_polygon']
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

    # For non-image inputs with --cchannel, use C-channel optimizer
    if use_cchannel:
        print("\nUsing C-CHANNEL OPTIMIZER (Ultra-Smart with Perimeter C-Channel)...")
        from run_cchannel_optimizer import main as run_cchannel
        # Remove the flag from argv before calling
        sys.argv = [arg for arg in sys.argv if arg not in ['--cchannel', '-cc']]
        run_cchannel()
        return
    # For non-image inputs with --hundred, use the regular flow
    elif use_hundred:
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
        print("  --modular, -m       Use the new modular optimization pipeline")
        print("  --hundred, -h100    Use the 100% coverage optimizer")
        print("  --cchannel, -cc     Use the C-channel optimizer (ultra-smart + perimeter C-channel)")
        print("  --backtrack, --bt   Use the joint backtracking optimizer (optimizes cassettes + C-channel widths)")
        print("\nExample: python run_cassette_system.py floorplans/Luna-Conditioned.png")
        print("Example: python run_cassette_system.py bungalow --modular")
        print("Example: python run_cassette_system.py umbra --hundred")
        print("Example: python run_cassette_system.py umbra --cchannel")
        print("Example: python run_cassette_system.py umbra --bt")
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
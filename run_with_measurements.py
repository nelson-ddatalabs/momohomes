#!/usr/bin/env python3
"""
Run Cassette System with Pre-defined Measurements
==================================================
Use this if you already know your measurements.
"""

import sys
from pathlib import Path
from cassette_layout_system_cardinal import CassetteLayoutSystemCardinal

def main():
    # Get floor plan path
    if len(sys.argv) > 1:
        floor_plan_path = sys.argv[1]
    else:
        print("\nUsage: python run_with_measurements.py <path_to_floor_plan>")
        floor_plan_path = input("\nEnter path to your floor plan image: ").strip()

    if not Path(floor_plan_path).exists():
        print(f"Error: File not found: {floor_plan_path}")
        return

    # EDIT THESE MEASUREMENTS FOR YOUR FLOOR PLAN
    # The system will show you numbered edges - measure them clockwise
    # For perfect closure: Sum of E must equal sum of W, sum of N must equal sum of S

    measurements = {
        # Example for a simple rectangle:
        # 0: 40,   # N - left edge going up
        # 1: 78,   # E - top edge going right
        # 2: 40,   # S - right edge going down
        # 3: 78,   # W - bottom edge going left
        # Set remaining to 0 if not needed
        # 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0

        # Your measurements (will be shown after edge detection):
        0: 0,   # Edge 0 direction and length
        1: 0,   # Edge 1 direction and length
        2: 0,   # Edge 2 direction and length
        3: 0,   # Edge 3 direction and length
        4: 0,   # Edge 4 direction and length
        5: 0,   # Edge 5 direction and length
        6: 0,   # Edge 6 direction and length
        7: 0,   # Edge 7 direction and length
        8: 0,   # Edge 8 direction and length
        9: 0,   # Edge 9 direction and length
        10: 0,  # Edge 10 direction and length
        11: 0   # Edge 11 direction and length
    }

    # Check if measurements are set
    if all(v == 0 for v in measurements.values()):
        print("\n⚠️  No measurements defined!")
        print("Please edit this file and set your measurements in the 'measurements' dictionary.")
        print("\nTo see which edges to measure:")
        print("1. Run: python run_cassette_system.py " + floor_plan_path)
        print("2. Look at the numbered_cardinal_edges.png to see edge numbers and directions")
        print("3. Edit this file with your measurements")
        print("4. Run this script again")
        return

    input_name = Path(floor_plan_path).stem
    output_dir = f"output_{input_name}"

    print(f"\nProcessing: {floor_plan_path}")
    print(f"Using pre-defined measurements")
    print("-" * 70)

    # Initialize and run
    system = CassetteLayoutSystemCardinal(output_dir=output_dir)
    result = system.process_floor_plan(floor_plan_path, measurements)

    # Print results
    if result['success']:
        print(f"\n✓ Processing successful")
        print(f"  Coverage: {result['coverage_percent']:.1f}%")
        print(f"  Cassettes: {result['num_cassettes']}")
        print(f"  Output: {result['output_visualization']}")

        # Check closure
        E_total = measurements[1] + measurements[5] + measurements[9]  # Adjust based on your edges
        W_total = measurements[3] + measurements[7] + measurements[11]  # Adjust based on your edges
        N_total = measurements[0] + measurements[4] + measurements[8]   # Adjust based on your edges
        S_total = measurements[2] + measurements[6] + measurements[10]  # Adjust based on your edges

        print(f"\nBalance check:")
        print(f"  E total: {E_total}, W total: {W_total} {'✓' if E_total == W_total else '✗'}")
        print(f"  N total: {N_total}, S total: {S_total} {'✓' if N_total == S_total else '✗'}")
    else:
        print(f"✗ Failed: {result['error']}")

if __name__ == "__main__":
    main()
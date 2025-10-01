#!/usr/bin/env python3
"""
Test Perfect Closure with Balanced Measurements
================================================
Tests the system with measurements that balance perfectly (ΣE=ΣW, ΣN=ΣS).
These measurements should result in 0.0000 ft closure error.
"""

from cassette_layout_system_cardinal import CassetteLayoutSystemCardinal
import os

def test_perfect_closure():
    """Test with measurements that should give perfect closure."""

    # Balanced measurements that MUST close perfectly
    # ΣN = 50 + 4 = 54 ft
    # ΣS = 16 + 9 + 15 + 14 = 54 ft
    # ΣE = 78 ft
    # ΣW = 7.5 + 15 + 8 + 32.5 + 15 = 78 ft
    balanced_measurements = {
        0: 50.0,   # N
        1: 78.0,   # E
        2: 16.0,   # S
        3: 7.5,    # W
        4: 9.0,    # S
        5: 15.0,   # W
        6: 4.0,    # N
        7: 8.0,    # W
        8: 15.0,   # S
        9: 32.5,   # W
        10: 14.0,  # S
        11: 15.0   # W
    }

    # Initialize system
    system = CassetteLayoutSystemCardinal(output_dir="test_perfect_closure_output")

    # Find Luna floor plan
    luna_path = "/Users/nelsondsouza/Dropbox (Personal)/Mac/Documents/products/momohomes/floorplans/Luna-Conditioned.png"

    if not os.path.exists(luna_path):
        luna_path = "floorplans/Luna-Conditioned.png"

    if not os.path.exists(luna_path):
        print("❌ Luna floor plan not found")
        return False

    print("="*70)
    print("TESTING PERFECT CLOSURE WITH BALANCED MEASUREMENTS")
    print("="*70)

    # Verify balance
    print("\n1. MEASUREMENT BALANCE CHECK:")
    print("-" * 40)
    north_sum = balanced_measurements[0] + balanced_measurements[6]
    south_sum = balanced_measurements[2] + balanced_measurements[4] + balanced_measurements[8] + balanced_measurements[10]
    east_sum = balanced_measurements[1]
    west_sum = balanced_measurements[3] + balanced_measurements[5] + balanced_measurements[7] + balanced_measurements[9] + balanced_measurements[11]

    print(f"  North total: {north_sum:.1f} ft")
    print(f"  South total: {south_sum:.1f} ft")
    print(f"  East total:  {east_sum:.1f} ft")
    print(f"  West total:  {west_sum:.1f} ft")
    print(f"  N-S Balance: {abs(north_sum - south_sum):.4f} ft (should be 0)")
    print(f"  E-W Balance: {abs(east_sum - west_sum):.4f} ft (should be 0)")

    # Process with balanced measurements
    print("\n2. PROCESSING WITH CARDINAL SYSTEM:")
    print("-" * 40)
    result = system.process_floor_plan(luna_path, balanced_measurements)

    # Check results
    print("\n3. RESULTS:")
    print("-" * 40)

    if result['success']:
        print(f"✓ Processing successful")
        print(f"  Area: {result['area']:.1f} sq ft")
        print(f"  Perimeter: {result['perimeter']:.1f} ft")
        print(f"  Cassettes: {result['num_cassettes']}")
        print(f"  Coverage: {result['coverage_percent']:.1f}%")

        # THE KEY TEST: Closure should be perfect with balanced measurements
        print(f"\n4. CLOSURE CHECK (Measurement-based):")
        print("-" * 40)

        if result['is_closed'] and result['closure_error'] < 0.001:
            print(f"✅✅✅ PERFECT CLOSURE ACHIEVED!")
            print(f"  Closure error: {result['closure_error']:.6f} feet")
            print(f"  Status: Polygon is perfectly closed")
            print("\nThis confirms the system is now using measurement-based polygon!")
            return True
        else:
            print(f"❌ UNEXPECTED: Closure error with balanced measurements!")
            print(f"  Closure error: {result['closure_error']:.2f} feet")
            print(f"  Status: {'Closed' if result['is_closed'] else 'Not closed'}")
            print("\nThis suggests the system may still be using pixel-based checking.")
            return False

    else:
        print(f"❌ Processing failed: {result['error']}")
        return False

if __name__ == "__main__":
    success = test_perfect_closure()

    print("\n" + "="*70)
    if success:
        print("✅ TEST PASSED: System correctly uses measurement-based polygon")
        print("The closure error is essentially 0 with balanced measurements.")
    else:
        print("❌ TEST FAILED: System may still have issues")
        print("Check if pixel-based closure is still being used somewhere.")
    print("="*70)
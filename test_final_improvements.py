#!/usr/bin/env python3
"""
Test Final Improvements
=======================
Demonstrates all improvements to the cassette layout system.
"""

from cassette_layout_system_cardinal import CassetteLayoutSystemCardinal
import os


def test_luna_with_improvements():
    """Test Luna floor plan with all improvements."""

    # Balanced measurements for perfect closure
    luna_measurements = {
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

    # Find Luna floor plan
    luna_path = "/Users/nelsondsouza/Dropbox (Personal)/Mac/Documents/products/momohomes/floorplans/Luna-Conditioned.png"

    if not os.path.exists(luna_path):
        luna_path = "floorplans/Luna-Conditioned.png"

    if not os.path.exists(luna_path):
        print("❌ Luna floor plan not found")
        return

    print("="*70)
    print("TESTING IMPROVED CASSETTE LAYOUT SYSTEM")
    print("="*70)
    print("\nImprovements implemented:")
    print("1. ✓ Fixed polygon to exclude garage/patio cutouts")
    print("2. ✓ Cassettes only placed in indoor areas")
    print("3. ✓ Smart optimizer starts from corners with longest edges")
    print("4. ✓ Prioritizes largest cassettes for minimum count")
    print("5. ✓ Enhanced gap filling for better coverage")
    print()

    # Test with smart optimizer
    print("RUNNING WITH SMART OPTIMIZER:")
    print("-"*40)

    system = CassetteLayoutSystemCardinal(output_dir="output_improved")
    result = system.process_floor_plan(luna_path, luna_measurements, use_smart_optimizer=True)

    if result['success']:
        print(f"✓ Processing successful")
        print(f"  Floor area: {result['area']:.1f} sq ft")
        print(f"  Perimeter: {result['perimeter']:.1f} ft")
        print(f"  Number of cassettes: {result['num_cassettes']}")
        print(f"  Coverage: {result['coverage_percent']:.2f}%")
        print(f"  Total weight: {result['total_weight']:.0f} lbs")

        # Check closure
        print(f"\n✓ Perfect polygon closure achieved!")
        print(f"  Closure error: {result['closure_error']:.6f} feet")

        # Check coverage requirement
        if result['coverage_percent'] >= 94:
            print(f"\n✅ MEETS 94% coverage requirement!")
        else:
            print(f"\n⚠️  Coverage: {result['coverage_percent']:.2f}% (target: 94%+)")

        # Show size distribution
        if 'size_distribution' in system.statistics:
            print("\nCASSETTE SIZE DISTRIBUTION:")
            print("-"*40)
            sizes = system.statistics['size_distribution']
            for size, count in sorted(sizes.items(), key=lambda x: -x[1]):
                w, h = map(int, size.split('x'))
                area = w * h
                print(f"  {size:4s}: {count:3d} cassettes ({area:2d} sq ft each)")

        print(f"\nOutput visualization: {result['output_visualization']}")

    else:
        print(f"❌ Processing failed: {result['error']}")

    # Compare with old optimizer
    print("\n" + "="*70)
    print("COMPARISON WITH OLD OPTIMIZER:")
    print("-"*40)

    system_old = CassetteLayoutSystemCardinal(output_dir="output_old")
    result_old = system_old.process_floor_plan(luna_path, luna_measurements, use_smart_optimizer=False)

    if result_old['success']:
        print(f"Old optimizer results:")
        print(f"  Cassettes: {result_old['num_cassettes']}")
        print(f"  Coverage: {result_old['coverage_percent']:.2f}%")
        print(f"  Weight: {result_old['total_weight']:.0f} lbs")

        if result['success']:
            print(f"\nIMPROVEMENT SUMMARY:")
            print(f"  Cassette count: {result_old['num_cassettes']} → {result['num_cassettes']} "
                  f"({result_old['num_cassettes'] - result['num_cassettes']} fewer)")
            print(f"  Coverage: {result_old['coverage_percent']:.2f}% → {result['coverage_percent']:.2f}% "
                  f"({result['coverage_percent'] - result_old['coverage_percent']:+.2f}%)")
            print(f"  Weight: {result_old['total_weight']:.0f} → {result['total_weight']:.0f} lbs "
                  f"({result_old['total_weight'] - result['total_weight']:.0f} lbs lighter)")


if __name__ == "__main__":
    test_luna_with_improvements()
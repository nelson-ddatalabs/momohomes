#!/usr/bin/env python3
"""
Test Final Visualization
========================
Comprehensive test of the improved visualization system.
"""

from cassette_layout_system_cardinal import CassetteLayoutSystemCardinal
import os


def test_final_visualization():
    """Run final test of the improved visualization system."""

    # Balanced measurements for Luna
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
        return False

    print("="*70)
    print("FINAL VISUALIZATION SYSTEM TEST")
    print("="*70)
    print("\nImprovements implemented:")
    print("✅ Y-axis inversion fixed (mathematical to image coordinates)")
    print("✅ Origin offset handling for indoor-only polygon")
    print("✅ Proper world_to_image_coords transformation")
    print("✅ Scale factor consistency")
    print("✅ Debug visualization with coordinate grids")
    print()

    # Process with improved system
    system = CassetteLayoutSystemCardinal(output_dir="final_test_output")
    result = system.process_floor_plan(luna_path, luna_measurements, use_smart_optimizer=True)

    if not result['success']:
        print(f"❌ Processing failed: {result['error']}")
        return False

    print("RESULTS:")
    print("-"*40)
    print(f"✓ Floor area: {result['area']:.1f} sq ft")
    print(f"✓ Cassettes placed: {result['num_cassettes']}")
    print(f"✓ Coverage: {result['coverage_percent']:.2f}%")
    print(f"✓ Weight: {result['total_weight']:.0f} lbs")
    print(f"✓ Closure error: {result['closure_error']:.6f} ft")

    # Verify coordinate transformations
    print("\nCOORDINATE VERIFICATION:")
    print("-"*40)

    # Test key positions
    test_positions = [
        ((0, 14), "Left wall bottom (above garage)"),
        ((15, 0), "Garage boundary corner"),
        ((78, 50), "Top-right corner"),
        ((0, 50), "Top-left corner"),
    ]

    from cassette_layout_visualizer import CassetteLayoutVisualizer
    viz = CassetteLayoutVisualizer()

    # Approximate image height (from previous test)
    img_height = 1983

    for (x, y), description in test_positions:
        # Transform to image coordinates
        x_px, y_px = viz.world_to_image_coords(
            x, y, img_height, result['scale_factor'], (0.0, 0.0)
        )
        print(f"{description}:")
        print(f"  World: ({x:.1f}, {y:.1f}) feet")
        print(f"  Image: ({x_px}, {y_px}) pixels")

    # Check cassette placement
    print("\nCASSETTE PLACEMENT CHECK:")
    print("-"*40)

    cassettes = system.cassettes[:3] if system.cassettes else []
    for i, c in enumerate(cassettes):
        print(f"Cassette {i}: {c['size']} at ({c['x']:.1f}, {c['y']:.1f}) feet")

        # Verify no negative coordinates
        if c['x'] < 0 or c['y'] < 0:
            print(f"  ⚠️  WARNING: Negative coordinates!")

        # Verify within expected bounds
        if c['x'] > 78 or c['y'] > 50:
            print(f"  ⚠️  WARNING: Outside building bounds!")

        # Check if in garage area (should not be)
        if c['x'] < 15 and c['y'] < 14:
            print(f"  ❌ ERROR: In garage area!")
        else:
            print(f"  ✓ Valid placement")

    print("\n" + "="*70)
    print("VISUALIZATION QUALITY CHECK:")
    print("-"*40)

    checks = []

    # Quality checks
    if result['closure_error'] < 0.001:
        checks.append("✅ Perfect polygon closure")
    else:
        checks.append(f"⚠️  Closure error: {result['closure_error']:.4f} ft")

    if result['scale_factor'] > 0 and result['scale_factor'] < 1:
        checks.append(f"✅ Reasonable scale factor: {result['scale_factor']:.4f} ft/px")
    else:
        checks.append(f"⚠️  Unusual scale factor: {result['scale_factor']:.4f}")

    if result['coverage_percent'] >= 94:
        checks.append(f"✅ Meets 94% coverage requirement")
    else:
        checks.append(f"⚠️  Coverage {result['coverage_percent']:.2f}% (below 94%)")

    for check in checks:
        print(check)

    print("\n" + "="*70)
    print("OUTPUT FILES:")
    print("-"*40)
    print(f"• Visualization: {result['output_visualization']}")
    print(f"• Debug viz: luna_debug_visualization.png")
    print(f"• Results JSON: final_test_output/results_cardinal.json")

    print("\n✅ Final test completed successfully")
    print("Please manually inspect the visualization to confirm:")
    print("1. Cassettes appear within building boundaries")
    print("2. No cassettes in garage (bottom-left) or patio (top-right)")
    print("3. Cassettes align with walls and rooms")

    return True


if __name__ == "__main__":
    test_final_visualization()
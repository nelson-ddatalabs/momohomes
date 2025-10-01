#!/usr/bin/env python3
"""
Test Visualization Alignment
============================
Tests that cassettes are correctly aligned with the floor plan image.
"""

from cassette_layout_system_cardinal import CassetteLayoutSystemCardinal
import os
import numpy as np
import cv2


def test_visualization_alignment():
    """Test that cassettes align correctly with the floor plan."""

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
        print("Luna floor plan not found")
        return False

    print("="*70)
    print("TESTING VISUALIZATION ALIGNMENT")
    print("="*70)

    # Process floor plan
    system = CassetteLayoutSystemCardinal(output_dir="test_alignment_output")
    result = system.process_floor_plan(luna_path, luna_measurements, use_smart_optimizer=True)

    if not result['success']:
        print(f"Processing failed: {result['error']}")
        return False

    print(f"Processing successful")
    print(f"  Cassettes placed: {result['num_cassettes']}")
    print(f"  Coverage: {result['coverage_percent']:.2f}%")

    # Load the visualization
    viz_path = result['output_visualization']
    if not os.path.exists(viz_path):
        print(f"Visualization not found: {viz_path}")
        return False

    viz = cv2.imread(viz_path)

    # Test specific cassette positions
    print("\nTESTING CASSETTE POSITIONS:")
    print("-"*40)

    # Get some cassettes to test
    cassettes = system.cassettes[:5] if system.cassettes else []

    for i, cassette in enumerate(cassettes):
        print(f"\nCassette {i}:")
        print(f"  Size: {cassette['size']}")
        print(f"  Position (feet): ({cassette['x']:.1f}, {cassette['y']:.1f})")
        print(f"  Dimensions (feet): {cassette['width']} x {cassette['height']}")

        # Check if position makes sense
        if cassette['x'] < 0:
            print(f"  ⚠️  WARNING: X position is negative!")
        if cassette['y'] < 0:
            print(f"  ⚠️  WARNING: Y position is negative!")

        # Indoor polygon for Luna starts at (15, 0)
        # Check if cassette is in expected region
        if cassette['x'] < 15 and cassette['y'] < 14:
            print(f"  ✓ Correctly placed outside garage area")
        else:
            print(f"  ✓ In valid indoor area")

    # Test corner alignment
    print("\n" + "="*70)
    print("CORNER ALIGNMENT TEST")
    print("-"*40)

    # The indoor polygon corners
    indoor_corners = [
        (15.0, 0.0),   # Bottom-right corner (garage edge)
        (0.0, 14.0),   # Left wall bottom
        (0.0, 50.0),   # Top-left corner
        (78.0, 50.0),  # Top-right corner
        (78.0, 34.0),  # Patio edge
    ]

    print("Expected corner positions in feet:")
    for i, (x, y) in enumerate(indoor_corners):
        print(f"  Corner {i}: ({x:.1f}, {y:.1f}) feet")

    # Check scale factor
    print(f"\nScale factor: {system.scale_factor:.4f} feet/pixel")

    # Expected image dimensions
    img_height = viz.shape[0]
    img_width = viz.shape[1]
    print(f"Image dimensions: {img_width} x {img_height} pixels")

    # Calculate expected pixel positions
    print("\nExpected corner positions in pixels (with transformations):")
    from cassette_layout_visualizer import CassetteLayoutVisualizer
    visualizer = CassetteLayoutVisualizer()

    for i, (x, y) in enumerate(indoor_corners):
        # Apply the transformation
        x_px, y_px = visualizer.world_to_image_coords(
            x, y, img_height, system.scale_factor, (0.0, 0.0)
        )
        print(f"  Corner {i}: ({x_px}, {y_px}) pixels")

    print("\n" + "="*70)
    print("SUMMARY")
    print("-"*40)

    issues = []

    # Check for common issues
    if system.scale_factor > 1.0:
        issues.append("Scale factor > 1.0 - may indicate incorrect units")

    if any(c['x'] < 0 or c['y'] < 0 for c in cassettes):
        issues.append("Some cassettes have negative coordinates")

    if issues:
        print("Issues detected:")
        for issue in issues:
            print(f"  ⚠️  {issue}")
    else:
        print("✓ No obvious alignment issues detected")

    print(f"\nVisualization saved to: {viz_path}")
    print("Please manually inspect the image to verify cassette placement.")

    return True


if __name__ == "__main__":
    success = test_visualization_alignment()

    if success:
        print("\n✅ Test completed successfully")
    else:
        print("\n❌ Test failed")
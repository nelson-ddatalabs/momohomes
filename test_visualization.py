#!/usr/bin/env python3
"""
Test Visualization Improvements
================================
Test script to verify the improved cassette layout visualization.
"""

import cv2
import numpy as np
from cassette_layout_visualizer import CassetteLayoutVisualizer


def create_test_visualization():
    """Create a test visualization to verify improvements."""

    # Create a blank floor plan image
    width, height = 1200, 900
    floor_plan = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Draw a simple floor outline
    polygon_pixels = [
        (100, 100),
        (1100, 100),
        (1100, 700),
        (100, 700)
    ]
    cv2.polylines(floor_plan, [np.array(polygon_pixels)], True, (0, 0, 0), 2)

    # Save temporary floor plan
    temp_floor_plan = "temp_floor_plan.png"
    cv2.imwrite(temp_floor_plan, floor_plan)

    # Create test cassettes with various sizes
    test_cassettes = [
        {'x': 5, 'y': 5, 'width': 6, 'height': 8, 'size': '6x8'},
        {'x': 12, 'y': 5, 'width': 6, 'height': 8, 'size': '6x8'},
        {'x': 19, 'y': 5, 'width': 5, 'height': 8, 'size': '5x8'},
        {'x': 5, 'y': 14, 'width': 6, 'height': 6, 'size': '6x6'},
        {'x': 12, 'y': 14, 'width': 4, 'height': 8, 'size': '4x8'},
        {'x': 17, 'y': 14, 'width': 5, 'height': 6, 'size': '5x6'},
        {'x': 23, 'y': 14, 'width': 4, 'height': 6, 'size': '4x6'},
        {'x': 5, 'y': 21, 'width': 4, 'height': 4, 'size': '4x4'},
        {'x': 10, 'y': 21, 'width': 3, 'height': 4, 'size': '3x4'},
        {'x': 14, 'y': 21, 'width': 2, 'height': 8, 'size': '2x8'},
        {'x': 17, 'y': 21, 'width': 2, 'height': 6, 'size': '2x6'},
        {'x': 20, 'y': 21, 'width': 2, 'height': 4, 'size': '2x4'},
    ]

    # Calculate statistics
    total_area = 40 * 30  # Example area
    covered_area = sum(c['width'] * c['height'] for c in test_cassettes)
    statistics = {
        'total_area': total_area,
        'covered_area': covered_area,
        'gap_area': total_area - covered_area,
        'coverage_percent': (covered_area / total_area) * 100,
        'gap_percent': ((total_area - covered_area) / total_area) * 100,
        'num_cassettes': len(test_cassettes),
        'total_weight': covered_area * 10.4  # 10.4 lbs per sq ft
    }

    # Create visualizer and generate image
    visualizer = CassetteLayoutVisualizer()

    # Scale factor (feet per pixel)
    scale_factor = 40.0 / 1000.0  # 40 feet = 1000 pixels

    # Create visualization
    result = visualizer.create_visualization(
        floor_plan_path=temp_floor_plan,
        cassettes=test_cassettes,
        scale_factor=scale_factor,
        polygon_pixels=polygon_pixels,
        statistics=statistics,
        output_path="test_visualization_output.png",
        origin_offset=(2.5, 2.5)  # Small offset for margin
    )

    print("Test visualization created successfully!")
    print(f"Output saved to: test_visualization_output.png")
    print(f"\nStatistics:")
    print(f"  Coverage: {statistics['coverage_percent']:.1f}%")
    print(f"  Cassettes: {statistics['num_cassettes']}")
    print(f"  Total Weight: {statistics['total_weight']:.0f} lbs")

    # Clean up
    import os
    os.remove(temp_floor_plan)

    return result


if __name__ == "__main__":
    create_test_visualization()
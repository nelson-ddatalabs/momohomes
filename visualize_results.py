#!/usr/bin/env python3
"""
Visualize Optimization Results
===============================
Creates clean visualization from optimization results.
"""

import cv2
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from cassette_layout_visualizer import CassetteLayoutVisualizer


def create_simple_floor_plan(polygon: List[Tuple[float, float]],
                           output_path: str = "floor_plan.png",
                           img_size: Tuple[int, int] = (1200, 900)) -> Tuple[str, float, List]:
    """
    Create a simple floor plan image from polygon coordinates.

    Args:
        polygon: List of (x, y) vertices in feet
        output_path: Path to save floor plan image
        img_size: Image size (width, height) in pixels

    Returns:
        Tuple of (image_path, scale_factor, polygon_pixels)
    """
    width, height = img_size

    # Create white background
    floor_plan = np.ones((height, width, 3), dtype=np.uint8) * 245  # Light gray

    # Calculate bounding box and scale
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    building_width = max_x - min_x
    building_height = max_y - min_y

    # Calculate scale with margin
    margin = 100  # pixels
    scale_x = (width - 2 * margin) / building_width
    scale_y = (height - 2 * margin) / building_height
    scale = min(scale_x, scale_y)

    # Convert polygon to pixel coordinates
    polygon_pixels = []
    for x, y in polygon:
        px = int((x - min_x) * scale + margin)
        py = int(height - ((y - min_y) * scale + margin))  # Flip Y
        polygon_pixels.append((px, py))

    # Draw building outline
    pts = np.array(polygon_pixels, dtype=np.int32)
    cv2.fillPoly(floor_plan, [pts], (255, 255, 255))  # White fill
    cv2.polylines(floor_plan, [pts], True, (0, 0, 0), 3)  # Black border

    # Add grid lines for reference
    grid_spacing = 50  # pixels
    for x in range(0, width, grid_spacing):
        cv2.line(floor_plan, (x, 0), (x, height), (230, 230, 230), 1)
    for y in range(0, height, grid_spacing):
        cv2.line(floor_plan, (0, y), (width, y), (230, 230, 230), 1)

    # Save floor plan
    cv2.imwrite(output_path, floor_plan)

    # Calculate scale factor (feet per pixel)
    scale_factor = 1.0 / scale

    return output_path, scale_factor, polygon_pixels


def visualize_optimization_results(results_file: str = None,
                                  polygon: List[Tuple[float, float]] = None,
                                  cassettes: List[Dict] = None,
                                  output_name: str = "cassette_layout"):
    """
    Create visualization from optimization results.

    Args:
        results_file: Path to JSON results file from optimization
        polygon: Optional polygon coordinates (if not in results)
        cassettes: Optional cassette list (if not in results)
        output_name: Base name for output files
    """
    # Load results if file provided
    if results_file:
        with open(results_file, 'r') as f:
            results = json.load(f)

        # Extract data from results
        if not cassettes:
            cassettes = results.get('cassettes', [])

        # Try to get polygon from results
        if not polygon and 'polygon' in results:
            polygon = [(p[0], p[1]) for p in results['polygon']]

        statistics = {
            'total_area': results.get('total_area', 0),
            'covered_area': results.get('covered_area', 0),
            'gap_area': results.get('gap_area', 0),
            'coverage_percent': results.get('coverage_percent', 0),
            'gap_percent': results.get('gap_percent', 0),
            'num_cassettes': results.get('num_cassettes', 0),
            'total_weight': results.get('total_weight', 0)
        }
    else:
        # Calculate statistics from provided data
        if not polygon or not cassettes:
            print("Error: Need either results file or both polygon and cassettes")
            return

        total_area = calculate_polygon_area(polygon)
        covered_area = sum(c.get('area', c['width'] * c['height']) for c in cassettes)

        statistics = {
            'total_area': total_area,
            'covered_area': covered_area,
            'gap_area': total_area - covered_area,
            'coverage_percent': (covered_area / total_area * 100) if total_area > 0 else 0,
            'gap_percent': ((total_area - covered_area) / total_area * 100) if total_area > 0 else 0,
            'num_cassettes': len(cassettes),
            'total_weight': covered_area * 10.4
        }

    if not polygon:
        print("Error: No polygon data available")
        return

    # Create simple floor plan
    print("Creating floor plan image...")
    floor_plan_path, scale_factor, polygon_pixels = create_simple_floor_plan(
        polygon,
        output_path=f"{output_name}_floor.png"
    )

    # Calculate origin offset
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    min_x, min_y = min(xs), min(ys)

    # Create visualization
    print("Creating cassette visualization...")
    visualizer = CassetteLayoutVisualizer()

    visualization = visualizer.create_visualization(
        floor_plan_path=floor_plan_path,
        cassettes=cassettes,
        scale_factor=scale_factor,
        polygon_pixels=polygon_pixels,
        statistics=statistics,
        output_path=f"{output_name}_final.png",
        origin_offset=(-min_x, -min_y)
    )

    print(f"\nVisualization complete!")
    print(f"  Floor plan: {output_name}_floor.png")
    print(f"  Final layout: {output_name}_final.png")
    print(f"\nStatistics:")
    print(f"  Coverage: {statistics['coverage_percent']:.1f}%")
    print(f"  Cassettes: {statistics['num_cassettes']}")
    print(f"  Total area: {statistics['total_area']:.1f} sq ft")
    print(f"  Gap area: {statistics['gap_area']:.1f} sq ft")

    return visualization


def calculate_polygon_area(polygon: List[Tuple[float, float]]) -> float:
    """Calculate polygon area using shoelace formula."""
    n = len(polygon)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]
    return abs(area) / 2.0


def main():
    """Main entry point."""
    import sys

    if len(sys.argv) > 1:
        results_file = sys.argv[1]

        # Check if file exists
        if not Path(results_file).exists():
            print(f"Error: File not found: {results_file}")
            return

        output_name = Path(results_file).stem.replace('_results', '') + '_visual'
        visualize_optimization_results(results_file=results_file, output_name=output_name)

    else:
        print("Usage: python visualize_results.py <results.json>")
        print("\nExample: python visualize_results.py bungalow_modular_results.json")
        print("\nTesting with default rectangle...")

        # Test with simple rectangle
        test_polygon = [(0, 0), (40, 0), (40, 30), (0, 30)]
        test_cassettes = [
            {'x': 0, 'y': 0, 'width': 6, 'height': 8, 'size': '6x8'},
            {'x': 6, 'y': 0, 'width': 6, 'height': 8, 'size': '6x8'},
            {'x': 12, 'y': 0, 'width': 5, 'height': 8, 'size': '5x8'},
            {'x': 17, 'y': 0, 'width': 4, 'height': 8, 'size': '4x8'},
            {'x': 0, 'y': 8, 'width': 6, 'height': 6, 'size': '6x6'},
            {'x': 6, 'y': 8, 'width': 5, 'height': 6, 'size': '5x6'},
            {'x': 11, 'y': 8, 'width': 4, 'height': 6, 'size': '4x6'},
        ]

        visualize_optimization_results(
            polygon=test_polygon,
            cassettes=test_cassettes,
            output_name="test_visual"
        )


if __name__ == "__main__":
    main()
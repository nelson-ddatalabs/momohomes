#!/usr/bin/env python3
"""
Debug Visualization
===================
Creates a debug visualization showing both coordinate systems.
"""

import cv2
import numpy as np
from typing import List, Tuple


def create_debug_visualization(image_path: str, polygon_feet: List[Tuple[float, float]],
                             scale_factor: float, origin_offset: Tuple[float, float],
                             output_path: str = "debug_visualization.png"):
    """
    Create debug visualization with coordinate system overlays.

    Args:
        image_path: Path to floor plan image
        polygon_feet: Polygon vertices in feet
        scale_factor: Feet per pixel
        origin_offset: Origin offset in feet
        output_path: Path to save debug visualization
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Cannot load image: {image_path}")
        return

    h, w = img.shape[:2]
    print(f"Image dimensions: {w} x {h} pixels")

    # Create overlay
    overlay = img.copy()

    # Draw grid (every 10 feet)
    grid_spacing = 10  # feet
    grid_pixels = int(grid_spacing / scale_factor)

    # Vertical lines
    for x_feet in range(0, 80, grid_spacing):
        x_pixels = int((x_feet + origin_offset[0]) / scale_factor)
        cv2.line(overlay, (x_pixels, 0), (x_pixels, h), (0, 255, 0), 1)
        # Label
        cv2.putText(overlay, f"{x_feet}ft", (x_pixels + 5, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Horizontal lines (with Y-axis inversion)
    for y_feet in range(0, 60, grid_spacing):
        y_pixels = int(h - (y_feet / scale_factor))
        cv2.line(overlay, (0, y_pixels), (w, y_pixels), (0, 255, 0), 1)
        # Label
        cv2.putText(overlay, f"{y_feet}ft", (10, y_pixels - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Draw polygon
    if polygon_feet:
        points = []
        for x_feet, y_feet in polygon_feet:
            x_adjusted = x_feet + origin_offset[0]
            y_adjusted = y_feet + origin_offset[1]
            x_px = int(x_adjusted / scale_factor)
            y_px = int(h - (y_adjusted / scale_factor))
            points.append((x_px, y_px))

        # Draw polygon
        points = np.array(points, np.int32)
        cv2.polylines(overlay, [points], True, (255, 0, 0), 2)

        # Mark vertices
        for i, (x_px, y_px) in enumerate(points):
            cv2.circle(overlay, (x_px, y_px), 5, (0, 0, 255), -1)
            cv2.putText(overlay, f"V{i}", (x_px + 10, y_px),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Draw coordinate axes
    # X-axis (red)
    cv2.arrowedLine(overlay, (50, h-50), (200, h-50), (0, 0, 255), 2)
    cv2.putText(overlay, "X (feet)", (210, h-50),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

    # Y-axis (blue) - inverted
    cv2.arrowedLine(overlay, (50, h-50), (50, h-200), (255, 0, 0), 2)
    cv2.putText(overlay, "Y (feet)", (55, h-210),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)

    # Add legend
    legend_y = 100
    cv2.rectangle(overlay, (20, 20), (400, legend_y + 20), (255, 255, 255), -1)
    cv2.rectangle(overlay, (20, 20), (400, legend_y + 20), (0, 0, 0), 2)

    cv2.putText(overlay, "DEBUG VISUALIZATION", (30, 45),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    info = [
        f"Scale: {scale_factor:.4f} ft/pixel",
        f"Origin offset: ({origin_offset[0]:.1f}, {origin_offset[1]:.1f}) ft",
        f"Image: {w} x {h} pixels",
        "Green grid: 10 ft spacing",
        "Blue line: Polygon boundary",
        "Red dots: Vertices"
    ]

    y_pos = 70
    for line in info:
        cv2.putText(overlay, line, (30, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        y_pos += 20

    # Blend with original
    result = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)

    # Save
    cv2.imwrite(output_path, result)
    print(f"Debug visualization saved to: {output_path}")


def test_debug_visualization():
    """Test the debug visualization with Luna floor plan."""

    from fix_polygon_for_indoor_only import get_corrected_luna_polygon

    # Get indoor polygon
    polygon = get_corrected_luna_polygon()

    # Luna parameters
    scale_factor = 0.0394  # From test output
    origin_offset = (0.0, 0.0)  # For full image coordinates

    # Find Luna floor plan
    luna_path = "/Users/nelsondsouza/Dropbox (Personal)/Mac/Documents/products/momohomes/floorplans/Luna-Conditioned.png"

    if not os.path.exists(luna_path):
        luna_path = "floorplans/Luna-Conditioned.png"

    if not os.path.exists(luna_path):
        print("Luna floor plan not found")
        return

    # Create debug visualization
    create_debug_visualization(
        luna_path,
        polygon,
        scale_factor,
        origin_offset,
        "luna_debug_visualization.png"
    )

    print("\nDebug visualization shows:")
    print("- Green grid lines every 10 feet")
    print("- Blue polygon boundary")
    print("- Red vertex markers")
    print("- Coordinate axes")
    print("\nCheck if polygon aligns with indoor areas.")


if __name__ == "__main__":
    import os
    test_debug_visualization()
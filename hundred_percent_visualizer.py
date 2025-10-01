"""
Simple visualizer for 100% coverage optimizer
Creates visualization without requiring floor plan image
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def create_simple_visualization(cassettes: List[Dict], polygon: List[Tuple[float, float]],
                               statistics: Dict, output_path: str, floor_plan_name: str = None):
    """Create a simple visualization of cassette layout"""

    # Calculate bounds
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # Image dimensions
    margin = 100
    scale = 20  # pixels per foot
    width = int((max_x - min_x) * scale) + 2 * margin
    height = int((max_y - min_y) * scale) + 2 * margin

    # Create white background
    image = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Draw polygon
    polygon_points = []
    for x, y in polygon:
        px = int((x - min_x) * scale + margin)
        py = height - int((y - min_y) * scale + margin)  # Flip Y axis
        polygon_points.append([px, py])

    polygon_points = np.array(polygon_points, np.int32)
    cv2.fillPoly(image, [polygon_points], (240, 240, 240))
    cv2.polylines(image, [polygon_points], True, (100, 100, 100), 2)

    # Count cassette sizes
    size_counts = {}
    for cassette in cassettes:
        size = cassette.get('size', '')
        if size:
            size_counts[size] = size_counts.get(size, 0) + 1

    # Assign colors to each unique size
    color_palette = [
        (255, 180, 180),  # Light red
        (180, 255, 180),  # Light green
        (180, 180, 255),  # Light blue
        (255, 255, 180),  # Light yellow
        (255, 180, 255),  # Light magenta
        (180, 255, 255),  # Light cyan
        (255, 210, 180),  # Light orange
        (210, 180, 255),  # Light purple
        (180, 255, 210),  # Light teal
    ]

    # Create size-to-color mapping
    size_colors = {}
    sorted_sizes = sorted(size_counts.keys())
    for i, size in enumerate(sorted_sizes):
        size_colors[size] = color_palette[i % len(color_palette)]

    # Draw cassettes with consistent colors per size
    for cassette in cassettes:
        x1 = int((cassette['x'] - min_x) * scale + margin)
        y1 = height - int((cassette['y'] + cassette['height'] - min_y) * scale + margin)
        x2 = int((cassette['x'] + cassette['width'] - min_x) * scale + margin)
        y2 = height - int((cassette['y'] - min_y) * scale + margin)

        size = cassette.get('size', '')
        color = size_colors.get(size, (200, 200, 200))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(image, (x1, y1), (x2, y2), (50, 50, 50), 1)

        # Add size label in center of cassette
        if size:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            text_size = cv2.getTextSize(size, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0]
            cv2.putText(image, size, (cx - text_size[0]//2, cy + text_size[1]//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)

    # Add title with floor plan name
    if floor_plan_name:
        title = f"{floor_plan_name.upper()} CASSETTE PLAN"
    else:
        title = "CASSETTE FLOOR PLAN"
    cv2.putText(image, title, (20, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    # Add legend (bottom left)
    legend_x = 20
    legend_y = height - 20 - (len(sorted_sizes) * 30 + 40)  # Calculate from bottom

    # Legend background
    legend_width = 200
    legend_height = len(sorted_sizes) * 30 + 40
    cv2.rectangle(image, (legend_x - 5, legend_y - 5),
                 (legend_x + legend_width, legend_y + legend_height),
                 (250, 250, 250), -1)
    cv2.rectangle(image, (legend_x - 5, legend_y - 5),
                 (legend_x + legend_width, legend_y + legend_height),
                 (100, 100, 100), 1)

    # Legend title
    cv2.putText(image, "CASSETTE SIZES", (legend_x + 5, legend_y + 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Legend entries
    for i, size in enumerate(sorted_sizes):
        y_pos = legend_y + 40 + i * 30

        # Color box
        color = size_colors[size]
        cv2.rectangle(image, (legend_x + 5, y_pos - 15),
                     (legend_x + 25, y_pos + 5), color, -1)
        cv2.rectangle(image, (legend_x + 5, y_pos - 15),
                     (legend_x + 25, y_pos + 5), (50, 50, 50), 1)

        # Size and count text
        text = f"{size}: {size_counts[size]} units"
        cv2.putText(image, text, (legend_x + 35, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Add statistics (bottom left, above legend)
    stats_x = 20
    stats_y = legend_y - 120  # Position above legend

    # Statistics background
    cv2.rectangle(image, (stats_x - 5, stats_y - 25),
                 (stats_x + 250, stats_y + 75),
                 (250, 250, 250), -1)
    cv2.rectangle(image, (stats_x - 5, stats_y - 25),
                 (stats_x + 250, stats_y + 75),
                 (100, 100, 100), 1)

    # Statistics text
    cv2.putText(image, "STATISTICS", (stats_x + 5, stats_y - 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(image, f"Coverage: {statistics['coverage']:.1f}%", (stats_x + 5, stats_y + 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(image, f"Cassettes: {statistics['cassettes']}", (stats_x + 5, stats_y + 40),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(image, f"Total Area: {statistics['total_area']:.0f} sq ft", (stats_x + 5, stats_y + 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Save image
    cv2.imwrite(output_path, image)
    logger.info(f"Visualization saved to {output_path}")
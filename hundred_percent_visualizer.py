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
                               statistics: Dict, output_path: str, floor_plan_name: str = None,
                               inset_polygon: List[Tuple[float, float]] = None):
    """Create a simple visualization of cassette layout

    Args:
        cassettes: List of cassette dictionaries
        polygon: Original outer polygon
        statistics: Statistics dictionary
        output_path: Path to save visualization
        floor_plan_name: Name of floor plan
        inset_polygon: Optional inset polygon for C-channel visualization
    """

    # Calculate bounds
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # Image dimensions
    margin = 100
    scale = 20  # pixels per foot
    floor_plan_width = int((max_x - min_x) * scale) + 2 * margin
    floor_plan_height = int((max_y - min_y) * scale) + 2 * margin

    # Add extra space below for legend (300 pixels) and ensure minimum width for legend
    legend_space = 300
    min_width_for_legend = floor_plan_width + 50  # Ensure at least 50px padding on right
    width = max(floor_plan_width, min_width_for_legend)
    height = floor_plan_height + legend_space

    # Create white background
    image = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Helper function to convert coordinates
    def to_pixel_coords(x, y):
        px = int((x - min_x) * scale + margin)
        py = floor_plan_height - int((y - min_y) * scale + margin)  # Flip Y axis
        return px, py

    # Draw polygon areas
    polygon_points = np.array([to_pixel_coords(x, y) for x, y in polygon], np.int32)

    if inset_polygon:
        # C-channel visualization mode
        inset_points = np.array([to_pixel_coords(x, y) for x, y in inset_polygon], np.int32)

        # Fill original polygon with white (base)
        cv2.fillPoly(image, [polygon_points], (255, 255, 255))

        # Fill inset polygon with light gray (gap areas)
        cv2.fillPoly(image, [inset_points], (230, 230, 230))

        # Draw C-channel area (between polygons) - fill original then overlay inset
        # Create mask for C-channel
        cchannel_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(cchannel_mask, [polygon_points], 255)
        cv2.fillPoly(cchannel_mask, [inset_points], 0)

        # Fill C-channel area with distinct tan/beige color (BGR format)
        cchannel_color = (180, 200, 240)  # Warmer tan/beige (more orange)
        image[cchannel_mask == 255] = cchannel_color

        # Draw boundaries
        cv2.polylines(image, [polygon_points], True, (100, 100, 100), 2)  # Outer boundary
        cv2.polylines(image, [inset_points], True, (150, 150, 150), 1)    # Inner boundary (lighter)

        # Add C-channel width labels on each side of the floor plan
        if 'cchannel_widths' in statistics:
            widths = statistics['cchannel_widths']

            # Get polygon bounds for label positioning
            poly_min_x = min(pt[0] for pt in polygon_points)
            poly_max_x = max(pt[0] for pt in polygon_points)
            poly_min_y = min(pt[1] for pt in polygon_points)
            poly_max_y = max(pt[1] for pt in polygon_points)
            poly_center_x = (poly_min_x + poly_max_x) // 2
            poly_center_y = (poly_min_y + poly_max_y) // 2

            # North (top) - above the floor plan
            if 'N' in widths:
                label = f"{widths['N']:.1f}\""
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.putText(image, label,
                           (int(poly_center_x - text_size[0]//2), int(poly_min_y - 15)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            # South (bottom) - below the floor plan
            if 'S' in widths:
                label = f"{widths['S']:.1f}\""
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.putText(image, label,
                           (int(poly_center_x - text_size[0]//2), int(poly_max_y + 30)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            # East (right) - to the right of floor plan
            if 'E' in widths:
                label = f"{widths['E']:.1f}\""
                cv2.putText(image, label,
                           (int(poly_max_x + 15), int(poly_center_y + 5)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            # West (left) - to the left of floor plan
            if 'W' in widths:
                label = f"{widths['W']:.1f}\""
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.putText(image, label,
                           (int(poly_min_x - text_size[0] - 15), int(poly_center_y + 5)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    else:
        # Standard visualization mode
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

    # Check if per-cassette C-channel mode
    per_cassette_mode = statistics.get('per_cassette_cchannel', False)
    cchannel_widths_per_cassette = statistics.get('cchannel_widths_per_cassette', [])
    cchannel_geometries = statistics.get('cchannel_geometries', [])

    # HYBRID APPROACH: Draw C-channel geometries first (if available)
    # This uses the actual non-overlapping geometries from the optimizer
    if cchannel_geometries:
        cchannel_color = (180, 200, 240)  # Tan/beige
        for geom in cchannel_geometries:
            # Each geometry is a box with bounds (minx, miny, maxx, maxy)
            minx = geom['minx']
            miny = geom['miny']
            maxx = geom['maxx']
            maxy = geom['maxy']

            # Convert to pixel coordinates
            px1, py1 = to_pixel_coords(minx, maxy)  # Top-left (note: Y is flipped)
            px2, py2 = to_pixel_coords(maxx, miny)  # Bottom-right

            x1, y1 = int(px1), int(py1)
            x2, y2 = int(px2), int(py2)

            # Draw C-channel geometry
            cv2.rectangle(image, (x1, y1), (x2, y2), cchannel_color, -1)
            cv2.rectangle(image, (x1, y1), (x2, y2), (100, 100, 100), 1)

    # Draw cassettes with consistent colors per size
    for idx, cassette in enumerate(cassettes):

        # Draw cassette on top
        px1, py1 = to_pixel_coords(cassette['x'], cassette['y'] + cassette['height'])
        px2, py2 = to_pixel_coords(cassette['x'] + cassette['width'], cassette['y'])

        # Ensure coordinates are integers
        x1, y1 = int(px1), int(py1)
        x2, y2 = int(px2), int(py2)

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

    # Position legend and statistics BELOW floor plan
    legend_start_y = floor_plan_height + 20

    # Add statistics (left side, below floor plan)
    stats_x = 20
    stats_y = int(legend_start_y + 20)

    # Determine statistics height based on whether C-channel info is present
    has_cchannel = 'cchannel_area' in statistics
    stats_lines = 5 if has_cchannel else 3
    stats_height = int(stats_lines * 20 + 50)

    # Statistics background
    cv2.rectangle(image, (int(stats_x - 5), int(stats_y - 25)),
                 (int(stats_x + 280), int(stats_y + stats_height)),
                 (250, 250, 250), -1)
    cv2.rectangle(image, (int(stats_x - 5), int(stats_y - 25)),
                 (int(stats_x + 280), int(stats_y + stats_height)),
                 (100, 100, 100), 1)

    # Statistics text
    cv2.putText(image, "STATISTICS", (int(stats_x + 5), int(stats_y - 5)),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    y_offset = int(stats_y + 20)
    cv2.putText(image, f"Total Area: {statistics['total_area']:.0f} sq ft", (int(stats_x + 5), int(y_offset)),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    y_offset += 20
    cv2.putText(image, f"Coverage: {statistics['coverage']:.1f}%", (int(stats_x + 5), int(y_offset)),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    y_offset += 20
    cv2.putText(image, f"Cassettes: {statistics['cassettes']} units", (int(stats_x + 5), int(y_offset)),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    if has_cchannel:
        y_offset += 20
        cv2.putText(image, f"Cassette Area: {statistics.get('covered', 0) - statistics['cchannel_area']:.0f} sq ft",
                   (int(stats_x + 5), int(y_offset)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        y_offset += 20
        cv2.putText(image, f"C-Channel Area: {statistics['cchannel_area']:.1f} sq ft",
                   (int(stats_x + 5), int(y_offset)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Add C-channel widths if present
    if has_cchannel:
        cchannel_x = 320
        cchannel_y = int(legend_start_y + 20)

        # Check if per-cassette mode
        if per_cassette_mode and cchannel_widths_per_cassette:
            # Show per-cassette C-channel statistics (min, max, avg)
            cv2.rectangle(image, (int(cchannel_x - 5), int(cchannel_y - 25)),
                         (int(cchannel_x + 220), int(cchannel_y + 100)),
                         (250, 250, 250), -1)
            cv2.rectangle(image, (int(cchannel_x - 5), int(cchannel_y - 25)),
                         (int(cchannel_x + 220), int(cchannel_y + 100)),
                         (100, 100, 100), 1)

            cv2.putText(image, "C-CHANNEL (PER-CASSETTE)", (int(cchannel_x + 5), int(cchannel_y - 5)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # Calculate and display min, max, avg
            min_c = min(cchannel_widths_per_cassette)
            max_c = max(cchannel_widths_per_cassette)
            avg_c = sum(cchannel_widths_per_cassette) / len(cchannel_widths_per_cassette)

            y_offset = int(cchannel_y + 20)
            cv2.putText(image, f"Min: {min_c:.2f}\"", (int(cchannel_x + 5), int(y_offset)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            y_offset += 25
            cv2.putText(image, f"Max: {max_c:.2f}\"", (int(cchannel_x + 5), int(y_offset)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            y_offset += 25
            cv2.putText(image, f"Avg: {avg_c:.2f}\"", (int(cchannel_x + 5), int(y_offset)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        elif 'cchannel_widths' in statistics:
            # Show perimeter C-channel widths (original mode)
            cv2.rectangle(image, (int(cchannel_x - 5), int(cchannel_y - 25)),
                         (int(cchannel_x + 180), int(cchannel_y + 130)),
                         (250, 250, 250), -1)
            cv2.rectangle(image, (int(cchannel_x - 5), int(cchannel_y - 25)),
                         (int(cchannel_x + 180), int(cchannel_y + 130)),
                         (100, 100, 100), 1)

            cv2.putText(image, "C-CHANNEL WIDTHS", (int(cchannel_x + 5), int(cchannel_y - 5)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            widths = statistics['cchannel_widths']
            y_offset = int(cchannel_y + 20)
            for direction in ['N', 'S', 'E', 'W']:
                width = widths.get(direction, 0)
                cv2.putText(image, f"{direction}: {width:.1f}\"", (int(cchannel_x + 5), int(y_offset)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                y_offset += 25

    # Add legend (right side, below floor plan, after C-channel widths if present)
    if has_cchannel:
        legend_x = int(cchannel_x + 200)  # Position to right of C-channel widths
    else:
        legend_x = int(stats_x + 300)  # Position to right of statistics
    legend_y = int(legend_start_y + 20)

    # Add C-channel legend entry if present
    legend_entries = len(sorted_sizes) + (2 if has_cchannel else 0)
    legend_height = int(legend_entries * 30 + 40)

    # Legend background
    legend_width = 220
    cv2.rectangle(image, (int(legend_x - 5), int(legend_y - 25)),
                 (int(legend_x + legend_width), int(legend_y + legend_height)),
                 (250, 250, 250), -1)
    cv2.rectangle(image, (int(legend_x - 5), int(legend_y - 25)),
                 (int(legend_x + legend_width), int(legend_y + legend_height)),
                 (100, 100, 100), 1)

    # Legend title
    cv2.putText(image, "LEGEND", (int(legend_x + 5), int(legend_y - 5)),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Legend entries
    entry_idx = 0

    # C-channel entries first (if present)
    if has_cchannel:
        y_pos = int(legend_y + 20 + entry_idx * 30)
        cv2.rectangle(image, (int(legend_x + 5), int(y_pos - 15)),
                     (int(legend_x + 25), int(y_pos + 5)), (180, 200, 240), -1)
        cv2.rectangle(image, (int(legend_x + 5), int(y_pos - 15)),
                     (int(legend_x + 25), int(y_pos + 5)), (50, 50, 50), 1)
        cv2.putText(image, "C-Channel", (int(legend_x + 35), int(y_pos)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        entry_idx += 1

        y_pos = int(legend_y + 20 + entry_idx * 30)
        cv2.rectangle(image, (int(legend_x + 5), int(y_pos - 15)),
                     (int(legend_x + 25), int(y_pos + 5)), (230, 230, 230), -1)
        cv2.rectangle(image, (int(legend_x + 5), int(y_pos - 15)),
                     (int(legend_x + 25), int(y_pos + 5)), (50, 50, 50), 1)
        cv2.putText(image, "Empty Space", (int(legend_x + 35), int(y_pos)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        entry_idx += 1

    # Cassette size entries
    for i, size in enumerate(sorted_sizes):
        y_pos = int(legend_y + 20 + entry_idx * 30)

        # Color box
        color = size_colors[size]
        cv2.rectangle(image, (int(legend_x + 5), int(y_pos - 15)),
                     (int(legend_x + 25), int(y_pos + 5)), color, -1)
        cv2.rectangle(image, (int(legend_x + 5), int(y_pos - 15)),
                     (int(legend_x + 25), int(y_pos + 5)), (50, 50, 50), 1)

        # Size and count text
        text = f"{size}: {size_counts[size]} units"
        cv2.putText(image, text, (int(legend_x + 35), int(y_pos)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        entry_idx += 1

    # Save image
    cv2.imwrite(output_path, image)
    logger.info(f"Visualization saved to {output_path}")
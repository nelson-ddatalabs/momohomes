"""
Dual-output visualizer for 100% coverage optimizer
Creates both PNG and SVG visualizations
"""

import numpy as np
import cv2
import svgwrite
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import logging
import math

logger = logging.getLogger(__name__)


def create_simple_visualization(cassettes: List[Dict], polygon: List[Tuple[float, float]],
                               statistics: Dict, output_path: str, floor_plan_name: str = None,
                               inset_polygon: List[Tuple[float, float]] = None):
    """Create both PNG and SVG visualizations of cassette layout

    Args:
        cassettes: List of cassette dictionaries
        polygon: Original outer polygon
        statistics: Statistics dictionary
        output_path: Path to save visualization (with or without extension)
        floor_plan_name: Name of floor plan
        inset_polygon: Optional inset polygon for special panel visualization
    """

    # Determine output paths
    output_path_obj = Path(output_path)
    if output_path_obj.suffix in ['.png', '.svg']:
        # Has extension - use stem for both
        base_path = output_path_obj.parent / output_path_obj.stem
    else:
        # No extension - use as is
        base_path = output_path_obj

    png_path = str(base_path) + '.png'
    svg_path = str(base_path) + '.svg'

    # Prepare all shared data
    shared_data = _prepare_shared_data(cassettes, polygon, statistics,
                                      floor_plan_name, inset_polygon)

    # Render PNG
    logger.info(f"Rendering PNG: {png_path}")
    _render_png(shared_data, png_path)

    # Render SVG
    logger.info(f"Rendering SVG: {svg_path}")
    _render_svg(shared_data, svg_path)

    logger.info(f"Visualizations saved to {png_path} and {svg_path}")


def _prepare_shared_data(cassettes, polygon, statistics, floor_plan_name, inset_polygon):
    """Prepare all shared calculations for both renderers"""

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

    # Helper function to convert coordinates
    def to_pixel_coords(x, y):
        px = int((x - min_x) * scale + margin)
        py = floor_plan_height - int((y - min_y) * scale + margin)  # Flip Y axis
        return px, py

    # Convert polygon points
    polygon_points = [to_pixel_coords(x, y) for x, y in polygon]
    inset_points = [to_pixel_coords(x, y) for x, y in inset_polygon] if inset_polygon else None

    # Count cassette sizes and assign colors
    size_counts = {}
    for cassette in cassettes:
        size = cassette.get('size', '')
        if size:
            size_counts[size] = size_counts.get(size, 0) + 1

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

    size_colors = {}
    sorted_sizes = sorted(size_counts.keys())
    for i, size in enumerate(sorted_sizes):
        size_colors[size] = color_palette[i % len(color_palette)]

    # Process special panels
    has_special_panel = 'cchannel_area' in statistics
    per_cassette_mode = statistics.get('per_cassette_cchannel', False)
    special_panel_widths_per_cassette = statistics.get('cchannel_widths_per_cassette', [])
    special_panel_geometries = statistics.get('cchannel_geometries', [])

    # Calculate special panel labels and identify small ones
    special_panel_labels = []
    small_special_panels = []

    for geom in special_panel_geometries:
        minx, miny = geom['minx'], geom['miny']
        maxx, maxy = geom['maxx'], geom['maxy']

        # Calculate dimensions in feet
        width_dim = maxx - minx
        height_dim = maxy - miny
        special_panel_width = min(width_dim, height_dim)  # Gap-filling dimension (1.5"-18")
        special_panel_length = max(width_dim, height_dim)  # Length dimension (up to 8')

        # Determine orientation
        is_horizontal = width_dim > height_dim

        # Convert to pixel coordinates
        px1, py1 = to_pixel_coords(minx, maxy)  # Top-left
        px2, py2 = to_pixel_coords(maxx, miny)  # Bottom-right

        pixel_width = abs(px2 - px1)
        pixel_height = abs(py2 - py1)
        min_pixel_dim = min(pixel_width, pixel_height)

        # Create label
        label = f"{special_panel_width:.1f}' x {special_panel_length:.1f}'"

        # Check if label fits
        can_show_label = min_pixel_dim >= 50

        special_panel_data = {
            'bounds': (minx, miny, maxx, maxy),
            'pixel_bounds': (px1, py1, px2, py2),
            'label': label,
            'can_show_label': can_show_label,
            'is_horizontal': is_horizontal,
            'center': ((px1 + px2) // 2, (py1 + py2) // 2)
        }

        if can_show_label:
            special_panel_labels.append(special_panel_data)
        else:
            small_special_panels.append(label)

    # Count special panels by size (for legend)
    special_panel_size_counts = {}
    for geom in special_panel_geometries:
        width_dim = geom['maxx'] - geom['minx']
        height_dim = geom['maxy'] - geom['miny']
        special_panel_width = min(width_dim, height_dim)
        special_panel_length = max(width_dim, height_dim)
        size_label = f"{special_panel_width:.1f}' × {special_panel_length:.1f}'"
        special_panel_size_counts[size_label] = special_panel_size_counts.get(size_label, 0) + 1

    # Prepare cassette data
    cassette_data = []
    for cassette in cassettes:
        px1, py1 = to_pixel_coords(cassette['x'], cassette['y'] + cassette['height'])
        px2, py2 = to_pixel_coords(cassette['x'] + cassette['width'], cassette['y'])

        size = cassette.get('size', '')
        color = size_colors.get(size, (200, 200, 200))

        # Calculate dynamic font size based on cassette dimensions
        pixel_width = abs(px2 - px1)
        pixel_height = abs(py2 - py1)
        min_dim = min(pixel_width, pixel_height)

        if min_dim < 40:
            font_scale = 0.3
        elif min_dim < 80:
            font_scale = 0.5
        elif min_dim < 120:
            font_scale = 0.7
        else:
            font_scale = 0.9

        cassette_data.append({
            'pixel_bounds': (px1, py1, px2, py2),
            'size': size,
            'color': color,
            'font_scale': font_scale,
            'center': ((px1 + px2) // 2, (py1 + py2) // 2)
        })

    # Calculate horizontal three-column layout
    layout = _calculate_horizontal_layout(
        floor_plan_width,
        floor_plan_height,
        has_special_panel,
        len(sorted_sizes),
        small_special_panels,
        len(special_panel_size_counts),
        inset_polygon is not None
    )

    # Prepare title
    title = f"{floor_plan_name.upper()} FLOOR JOIST CASSETTE PLAN" if floor_plan_name else "FLOOR JOIST CASSETTE PLAN"

    return {
        'min_x': min_x,
        'min_y': min_y,
        'max_x': max_x,
        'max_y': max_y,
        'margin': margin,
        'scale': scale,
        'floor_plan_width': floor_plan_width,
        'floor_plan_height': floor_plan_height,
        'width': layout['total_width'],
        'height': layout['total_height'],
        'to_pixel_coords': to_pixel_coords,
        'polygon_points': polygon_points,
        'inset_points': inset_points,
        'cassette_data': cassette_data,
        'special_panel_labels': special_panel_labels,
        'small_special_panels': small_special_panels,
        'special_panel_geometries': special_panel_geometries,
        'special_panel_size_counts': special_panel_size_counts,
        'size_counts': size_counts,
        'size_colors': size_colors,
        'sorted_sizes': sorted_sizes,
        'statistics': statistics,
        'has_special_panel': has_special_panel,
        'per_cassette_mode': per_cassette_mode,
        'special_panel_widths_per_cassette': special_panel_widths_per_cassette,
        'title': title,
        'layout': layout
    }


def _calculate_horizontal_layout(floor_plan_width, floor_plan_height, has_special_panel, num_cassette_sizes, small_special_panels, num_special_panel_sizes, has_inset_polygon):
    """Calculate positions for horizontal three-column layout"""

    # Start position for bottom section (below floor plan)
    bottom_start_y = floor_plan_height + 20

    # Gap between columns
    column_gap = 20

    # Calculate total width (ensure minimum for three columns)
    total_content_width = floor_plan_width
    min_column_width = 250
    min_total_width = min_column_width * 3 + column_gap * 2 + 40  # 3 columns + 2 gaps + margins

    if total_content_width < min_total_width:
        total_content_width = min_total_width

    # Calculate column width (equal thirds with gaps)
    available_width = total_content_width - column_gap * 2 - 40  # Subtract gaps and margins
    column_width = available_width // 3

    # Calculate box heights based on content
    # Legend box height
    legend_entries = num_cassette_sizes + num_special_panel_sizes + (1 if has_inset_polygon else 0)
    legend_height = max(legend_entries * 30 + 40, 150)  # Minimum 150px

    # Special panel box height
    if has_special_panel:
        special_panel_lines = 3  # Base lines
        if small_special_panels:
            special_panel_lines += 1 + math.ceil(len(small_special_panels) / 2)
        special_panel_height = max(special_panel_lines * 25 + 40, 150)
    else:
        special_panel_height = 150

    # Statistics box height
    stats_lines = 5 if has_special_panel else 3
    stats_height = max(stats_lines * 25 + 40, 150)

    # Use maximum height for all boxes to make them equal
    box_height = max(legend_height, special_panel_height, stats_height, 180)

    # Calculate positions for three columns
    margin_x = 20

    # Column 1: Legend
    legend_box = {
        'x': margin_x,
        'y': bottom_start_y,
        'width': column_width,
        'height': box_height
    }

    # Column 2: Special Panel
    special_panel_box = {
        'x': margin_x + column_width + column_gap,
        'y': bottom_start_y,
        'width': column_width,
        'height': box_height
    } if has_special_panel else None

    # Column 3: Statistics
    stats_box = {
        'x': margin_x + (column_width + column_gap) * 2,
        'y': bottom_start_y,
        'width': column_width,
        'height': box_height
    }

    # Calculate total dimensions
    total_height = bottom_start_y + box_height + 20  # Add bottom margin
    total_width = total_content_width

    return {
        'legend_start_y': bottom_start_y,
        'stats_box': stats_box,
        'special_panel_box': special_panel_box,
        'legend_box': legend_box,
        'total_width': total_width,
        'total_height': total_height,
        'column_width': column_width,
        'column_gap': column_gap,
        'box_height': box_height
    }


def _render_png(data, output_path):
    """Render PNG using OpenCV + PIL for rotated text"""

    width, height = data['width'], data['height']

    # Create white background using PIL (for better text rotation support)
    pil_image = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(pil_image)

    # Try to load a font (fallback to default if not available)
    try:
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
        font_medium = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        font_large = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
    except:
        font_small = ImageFont.load_default()
        font_medium = ImageFont.load_default()
        font_large = ImageFont.load_default()
        font_title = ImageFont.load_default()

    # Draw polygon
    if data['polygon_points']:
        draw.polygon(data['polygon_points'], fill=(240, 240, 240), outline=(100, 100, 100))

    # Draw inset polygon if present
    if data['inset_points']:
        draw.polygon(data['inset_points'], fill=(230, 230, 230), outline=(150, 150, 150))

    # Draw special panel geometries
    special_panel_color = (240, 200, 180)  # Tan/beige (RGB format for PIL)
    for geom in data['special_panel_geometries']:
        minx, miny, maxx, maxy = geom['minx'], geom['miny'], geom['maxx'], geom['maxy']
        px1, py1 = data['to_pixel_coords'](minx, maxy)
        px2, py2 = data['to_pixel_coords'](maxx, miny)

        draw.rectangle([px1, py1, px2, py2], fill=special_panel_color, outline=(100, 100, 100))

    # Draw special panel labels (with rotation)
    for sp_data in data['special_panel_labels']:
        label = sp_data['label']
        cx, cy = sp_data['center']
        is_horizontal = sp_data['is_horizontal']

        # Create text image
        if is_horizontal:
            # Horizontal text
            bbox = draw.textbbox((0, 0), label, font=font_medium)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            draw.text((cx - text_width//2, cy - text_height//2), label, fill=(0, 0, 0), font=font_medium)
        else:
            # Vertical text - need to rotate
            bbox = draw.textbbox((0, 0), label, font=font_medium)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # Create temporary image for rotated text
            temp_img = Image.new('RGBA', (text_width + 10, text_height + 10), (255, 255, 255, 0))
            temp_draw = ImageDraw.Draw(temp_img)
            temp_draw.text((5, 5), label, fill=(0, 0, 0, 255), font=font_medium)

            # Rotate 90 degrees
            rotated = temp_img.rotate(90, expand=True)

            # Paste onto main image
            paste_x = cx - rotated.width // 2
            paste_y = cy - rotated.height // 2
            pil_image.paste(rotated, (paste_x, paste_y), rotated)

    # Draw cassettes
    for cass_data in data['cassette_data']:
        px1, py1, px2, py2 = cass_data['pixel_bounds']
        color = cass_data['color']
        size = cass_data['size']
        cx, cy = cass_data['center']

        draw.rectangle([px1, py1, px2, py2], fill=color, outline=(50, 50, 50))

        # Add size label
        if size:
            # Scale font based on cassette size
            if cass_data['font_scale'] < 0.4:
                font = font_small
            elif cass_data['font_scale'] < 0.6:
                font = font_medium
            else:
                font = font_large

            bbox = draw.textbbox((0, 0), size, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            draw.text((cx - text_width//2, cy - text_height//2), size, fill=(0, 0, 0), font=font)

    # Draw title
    bbox = draw.textbbox((0, 0), data['title'], font=font_title)
    draw.text((20, 20), data['title'], fill=(0, 0, 0), font=font_title)

    # Draw statistics, C-channel info, and legend boxes
    _draw_legend_boxes_pil(draw, data, font_small, font_medium, font_large)

    # Save PNG
    pil_image.save(output_path)
    logger.info(f"PNG saved: {output_path}")


def _draw_legend_boxes_pil(draw, data, font_small, font_medium, font_large):
    """Draw statistics, special panel info, and legend boxes on PIL image"""

    layout = data['layout']

    # Statistics box
    stats_box = layout['stats_box']
    draw.rectangle(
        [stats_box['x'] - 5, stats_box['y'] - 5,
         stats_box['x'] + stats_box['width'], stats_box['y'] + stats_box['height']],
        fill=(250, 250, 250), outline=(100, 100, 100)
    )

    draw.text((stats_box['x'] + 10, stats_box['y'] + 10), "STATISTICS", fill=(0, 0, 0), font=font_large)

    y_offset = stats_box['y'] + 40
    stats = data['statistics']

    # Handle both 'coverage' and 'coverage_percent' keys
    coverage = stats.get('coverage', stats.get('coverage_percent', 0))
    cassette_count = stats.get('cassettes', stats.get('cassette_count', 0))

    draw.text((stats_box['x'] + 10, y_offset), f"Total Area: {stats['total_area']:.0f} sq ft", fill=(0, 0, 0), font=font_medium)
    y_offset += 25
    draw.text((stats_box['x'] + 10, y_offset), f"Coverage: {coverage:.1f}%", fill=(0, 0, 0), font=font_medium)
    y_offset += 25
    draw.text((stats_box['x'] + 10, y_offset), f"Cassettes: {cassette_count} units", fill=(0, 0, 0), font=font_medium)

    if data['has_special_panel']:
        y_offset += 25
        cassette_area = stats.get('covered', 0) - stats['cchannel_area']
        draw.text((stats_box['x'] + 10, y_offset), f"Cassette Area: {cassette_area:.0f} sq ft", fill=(0, 0, 0), font=font_medium)
        y_offset += 25
        draw.text((stats_box['x'] + 10, y_offset), f"Special Panel Area: {stats['cchannel_area']:.1f} sq ft", fill=(0, 0, 0), font=font_medium)

    # Special panel info box
    if data['has_special_panel'] and layout['special_panel_box']:
        sp_box = layout['special_panel_box']
        draw.rectangle(
            [sp_box['x'] - 5, sp_box['y'] - 5,
             sp_box['x'] + sp_box['width'], sp_box['y'] + sp_box['height']],
            fill=(250, 250, 250), outline=(100, 100, 100)
        )

        draw.text((sp_box['x'] + 10, sp_box['y'] + 10), "SPECIAL PANEL INFO", fill=(0, 0, 0), font=font_large)

        y_offset = sp_box['y'] + 40

        if data['per_cassette_mode'] and data['special_panel_widths_per_cassette']:
            widths = data['special_panel_widths_per_cassette']
            min_sp = min(widths)
            max_sp = max(widths)
            avg_sp = sum(widths) / len(widths)

            draw.text((sp_box['x'] + 10, y_offset), f"Min: {min_sp:.2f}\"", fill=(0, 0, 0), font=font_medium)
            y_offset += 25
            draw.text((sp_box['x'] + 10, y_offset), f"Max: {max_sp:.2f}\"", fill=(0, 0, 0), font=font_medium)
            y_offset += 25
            draw.text((sp_box['x'] + 10, y_offset), f"Avg: {avg_sp:.2f}\"", fill=(0, 0, 0), font=font_medium)

        # Add small special panels if any
        if data['small_special_panels']:
            y_offset += 25
            draw.text((sp_box['x'] + 10, y_offset), "Small Special Panels:", fill=(0, 0, 0), font=font_medium)
            y_offset += 20

            # Wrap labels
            labels_text = ", ".join(data['small_special_panels'])
            draw.text((sp_box['x'] + 10, y_offset), labels_text, fill=(0, 0, 0), font=font_small)

    # Legend box
    legend_box = layout['legend_box']
    draw.rectangle(
        [legend_box['x'] - 5, legend_box['y'] - 5,
         legend_box['x'] + legend_box['width'], legend_box['y'] + legend_box['height']],
        fill=(250, 250, 250), outline=(100, 100, 100)
    )

    draw.text((legend_box['x'] + 10, legend_box['y'] + 10), "LEGEND", fill=(0, 0, 0), font=font_large)

    y_offset = legend_box['y'] + 40
    entry_idx = 0

    # Special panel entries (treated like cassettes with size and count)
    special_panel_color = (240, 200, 180)  # Tan/beige
    for size, count in sorted(data['special_panel_size_counts'].items()):
        draw.rectangle([legend_box['x'] + 10, y_offset - 12, legend_box['x'] + 30, y_offset + 8],
                      fill=special_panel_color, outline=(50, 50, 50))
        unit_text = "unit" if count == 1 else "units"
        draw.text((legend_box['x'] + 40, y_offset - 5), f"{size}: {count} {unit_text}", fill=(0, 0, 0), font=font_medium)
        y_offset += 30

    # Empty space entry (only if inset polygon exists)
    if data['inset_points']:
        draw.rectangle([legend_box['x'] + 10, y_offset - 12, legend_box['x'] + 30, y_offset + 8],
                      fill=(230, 230, 230), outline=(50, 50, 50))
        draw.text((legend_box['x'] + 40, y_offset - 5), "Empty Space", fill=(0, 0, 0), font=font_medium)
        y_offset += 30

    # Cassette entries
    for size in data['sorted_sizes']:
        color = data['size_colors'][size]
        count = data['size_counts'][size]

        draw.rectangle([legend_box['x'] + 10, y_offset - 12, legend_box['x'] + 30, y_offset + 8],
                      fill=color, outline=(50, 50, 50))
        draw.text((legend_box['x'] + 40, y_offset - 5), f"{size}: {count} units", fill=(0, 0, 0), font=font_medium)
        y_offset += 30


def _render_svg(data, output_path):
    """Render SVG using svgwrite"""

    width, height = data['width'], data['height']

    dwg = svgwrite.Drawing(output_path, size=(width, height))

    # Draw polygon
    if data['polygon_points']:
        dwg.add(dwg.polygon(points=data['polygon_points'],
                           fill='rgb(240,240,240)',
                           stroke='rgb(100,100,100)',
                           stroke_width=2))

    # Draw inset polygon
    if data['inset_points']:
        dwg.add(dwg.polygon(points=data['inset_points'],
                           fill='rgb(230,230,230)',
                           stroke='rgb(150,150,150)',
                           stroke_width=1))

    # Draw special panel geometries
    for geom in data['special_panel_geometries']:
        minx, miny, maxx, maxy = geom['minx'], geom['miny'], geom['maxx'], geom['maxy']
        px1, py1 = data['to_pixel_coords'](minx, maxy)
        px2, py2 = data['to_pixel_coords'](maxx, miny)

        w = abs(px2 - px1)
        h = abs(py2 - py1)

        dwg.add(dwg.rect(insert=(min(px1, px2), min(py1, py2)),
                        size=(w, h),
                        fill='rgb(240,200,180)',
                        stroke='rgb(100,100,100)',
                        stroke_width=1))

    # Draw special panel labels
    for sp_data in data['special_panel_labels']:
        label = sp_data['label']
        cx, cy = sp_data['center']
        is_horizontal = sp_data['is_horizontal']

        if is_horizontal:
            dwg.add(dwg.text(label, insert=(cx, cy),
                           text_anchor='middle',
                           dominant_baseline='middle',
                           font_size=16,
                           font_family='Arial',
                           fill='black'))
        else:
            # Rotated text for vertical C-channels
            text_elem = dwg.text(label, insert=(cx, cy),
                               text_anchor='middle',
                               dominant_baseline='middle',
                               font_size=16,
                               font_family='Arial',
                               fill='black',
                               transform=f'rotate(90, {cx}, {cy})')
            dwg.add(text_elem)

    # Draw cassettes
    for cass_data in data['cassette_data']:
        px1, py1, px2, py2 = cass_data['pixel_bounds']
        color = cass_data['color']
        size = cass_data['size']
        cx, cy = cass_data['center']

        w = abs(px2 - px1)
        h = abs(py2 - py1)

        r, g, b = color
        dwg.add(dwg.rect(insert=(min(px1, px2), min(py1, py2)),
                        size=(w, h),
                        fill=f'rgb({r},{g},{b})',
                        stroke='rgb(50,50,50)',
                        stroke_width=1))

        # Add size label
        if size:
            font_size = 12 + int(cass_data['font_scale'] * 10)
            dwg.add(dwg.text(size, insert=(cx, cy),
                           text_anchor='middle',
                           dominant_baseline='middle',
                           font_size=font_size,
                           font_family='Arial',
                           fill='black'))

    # Draw title
    dwg.add(dwg.text(data['title'], insert=(20, 40),
                    font_size=28,
                    font_family='Arial',
                    font_weight='bold',
                    fill='black'))

    # Draw legend boxes
    _draw_legend_boxes_svg(dwg, data)

    dwg.save()
    logger.info(f"SVG saved: {output_path}")


def _draw_legend_boxes_svg(dwg, data):
    """Draw statistics, special panel info, and legend boxes on SVG"""

    layout = data['layout']

    # Statistics box
    stats_box = layout['stats_box']
    dwg.add(dwg.rect(insert=(stats_box['x'] - 5, stats_box['y'] - 5),
                    size=(stats_box['width'], stats_box['height']),
                    fill='rgb(250,250,250)',
                    stroke='rgb(100,100,100)',
                    stroke_width=1))

    dwg.add(dwg.text("STATISTICS", insert=(stats_box['x'] + 10, stats_box['y'] + 25),
                    font_size=20, font_family='Arial', font_weight='bold', fill='black'))

    y_offset = stats_box['y'] + 55
    stats = data['statistics']

    # Handle both 'coverage' and 'coverage_percent' keys
    coverage = stats.get('coverage', stats.get('coverage_percent', 0))
    cassette_count = stats.get('cassettes', stats.get('cassette_count', 0))

    dwg.add(dwg.text(f"Total Area: {stats['total_area']:.0f} sq ft",
                    insert=(stats_box['x'] + 10, y_offset),
                    font_size=16, font_family='Arial', fill='black'))
    y_offset += 25
    dwg.add(dwg.text(f"Coverage: {coverage:.1f}%",
                    insert=(stats_box['x'] + 10, y_offset),
                    font_size=16, font_family='Arial', fill='black'))
    y_offset += 25
    dwg.add(dwg.text(f"Cassettes: {cassette_count} units",
                    insert=(stats_box['x'] + 10, y_offset),
                    font_size=16, font_family='Arial', fill='black'))

    if data['has_special_panel']:
        y_offset += 25
        cassette_area = stats.get('covered', 0) - stats['cchannel_area']
        dwg.add(dwg.text(f"Cassette Area: {cassette_area:.0f} sq ft",
                        insert=(stats_box['x'] + 10, y_offset),
                        font_size=16, font_family='Arial', fill='black'))
        y_offset += 25
        dwg.add(dwg.text(f"Special Panel Area: {stats['cchannel_area']:.1f} sq ft",
                        insert=(stats_box['x'] + 10, y_offset),
                        font_size=16, font_family='Arial', fill='black'))

    # Special panel info box
    if data['has_special_panel'] and layout['special_panel_box']:
        sp_box = layout['special_panel_box']
        dwg.add(dwg.rect(insert=(sp_box['x'] - 5, sp_box['y'] - 5),
                        size=(sp_box['width'], sp_box['height']),
                        fill='rgb(250,250,250)',
                        stroke='rgb(100,100,100)',
                        stroke_width=1))

        dwg.add(dwg.text("SPECIAL PANEL INFO", insert=(sp_box['x'] + 10, sp_box['y'] + 25),
                        font_size=20, font_family='Arial', font_weight='bold', fill='black'))

        y_offset = sp_box['y'] + 55

        if data['per_cassette_mode'] and data['special_panel_widths_per_cassette']:
            widths = data['special_panel_widths_per_cassette']
            min_sp = min(widths)
            max_sp = max(widths)
            avg_sp = sum(widths) / len(widths)

            dwg.add(dwg.text(f"Min: {min_sp:.2f}\"", insert=(sp_box['x'] + 10, y_offset),
                           font_size=16, font_family='Arial', fill='black'))
            y_offset += 25
            dwg.add(dwg.text(f"Max: {max_sp:.2f}\"", insert=(sp_box['x'] + 10, y_offset),
                           font_size=16, font_family='Arial', fill='black'))
            y_offset += 25
            dwg.add(dwg.text(f"Avg: {avg_sp:.2f}\"", insert=(sp_box['x'] + 10, y_offset),
                           font_size=16, font_family='Arial', fill='black'))

        # Add small special panels
        if data['small_special_panels']:
            y_offset += 25
            dwg.add(dwg.text("Small Special Panels:", insert=(sp_box['x'] + 10, y_offset),
                           font_size=16, font_family='Arial', fill='black'))
            y_offset += 20

            labels_text = ", ".join(data['small_special_panels'])
            dwg.add(dwg.text(labels_text, insert=(sp_box['x'] + 10, y_offset),
                           font_size=12, font_family='Arial', fill='black'))

    # Legend box
    legend_box = layout['legend_box']
    dwg.add(dwg.rect(insert=(legend_box['x'] - 5, legend_box['y'] - 5),
                    size=(legend_box['width'], legend_box['height']),
                    fill='rgb(250,250,250)',
                    stroke='rgb(100,100,100)',
                    stroke_width=1))

    dwg.add(dwg.text("LEGEND", insert=(legend_box['x'] + 10, legend_box['y'] + 25),
                    font_size=20, font_family='Arial', font_weight='bold', fill='black'))

    y_offset = legend_box['y'] + 50

    # Special panel entries (treated like cassettes with size and count)
    for size, count in sorted(data['special_panel_size_counts'].items()):
        dwg.add(dwg.rect(insert=(legend_box['x'] + 10, y_offset - 12),
                        size=(20, 20),
                        fill='rgb(240,200,180)',
                        stroke='rgb(50,50,50)',
                        stroke_width=1))
        unit_text = "unit" if count == 1 else "units"
        dwg.add(dwg.text(f"{size}: {count} {unit_text}", insert=(legend_box['x'] + 40, y_offset + 3),
                        font_size=16, font_family='Arial', fill='black'))
        y_offset += 30

    # Empty space entry (only if inset polygon exists)
    if data['inset_points']:
        dwg.add(dwg.rect(insert=(legend_box['x'] + 10, y_offset - 12),
                        size=(20, 20),
                        fill='rgb(230,230,230)',
                        stroke='rgb(50,50,50)',
                        stroke_width=1))
        dwg.add(dwg.text("Empty Space", insert=(legend_box['x'] + 40, y_offset + 3),
                        font_size=16, font_family='Arial', fill='black'))
        y_offset += 30

    # Cassette entries
    for size in data['sorted_sizes']:
        color = data['size_colors'][size]
        count = data['size_counts'][size]
        r, g, b = color

        dwg.add(dwg.rect(insert=(legend_box['x'] + 10, y_offset - 12),
                        size=(20, 20),
                        fill=f'rgb({r},{g},{b})',
                        stroke='rgb(50,50,50)',
                        stroke_width=1))
        dwg.add(dwg.text(f"{size}: {count} units", insert=(legend_box['x'] + 40, y_offset + 3),
                        font_size=16, font_family='Arial', fill='black'))
        y_offset += 30

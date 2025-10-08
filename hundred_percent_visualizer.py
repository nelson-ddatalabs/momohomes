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
        inset_polygon: Optional inset polygon for C-channel visualization
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

    # Process C-channels
    has_cchannel = 'cchannel_area' in statistics
    per_cassette_mode = statistics.get('per_cassette_cchannel', False)
    cchannel_widths_per_cassette = statistics.get('cchannel_widths_per_cassette', [])
    cchannel_geometries = statistics.get('cchannel_geometries', [])

    # Calculate C-channel labels and identify small ones
    cchannel_labels = []
    small_cchannels = []

    for geom in cchannel_geometries:
        minx, miny = geom['minx'], geom['miny']
        maxx, maxy = geom['maxx'], geom['maxy']

        # Calculate dimensions in feet
        width_dim = maxx - minx
        height_dim = maxy - miny
        cchannel_width = min(width_dim, height_dim)  # Gap-filling dimension (1.5"-18")
        cchannel_length = max(width_dim, height_dim)  # Length dimension (up to 8')

        # Determine orientation
        is_horizontal = width_dim > height_dim

        # Convert to pixel coordinates
        px1, py1 = to_pixel_coords(minx, maxy)  # Top-left
        px2, py2 = to_pixel_coords(maxx, miny)  # Bottom-right

        pixel_width = abs(px2 - px1)
        pixel_height = abs(py2 - py1)
        min_pixel_dim = min(pixel_width, pixel_height)

        # Create label
        label = f"{cchannel_width:.1f}' x {cchannel_length:.1f}'"

        # Check if label fits
        can_show_label = min_pixel_dim >= 50

        cchannel_data = {
            'bounds': (minx, miny, maxx, maxy),
            'pixel_bounds': (px1, py1, px2, py2),
            'label': label,
            'can_show_label': can_show_label,
            'is_horizontal': is_horizontal,
            'center': ((px1 + px2) // 2, (py1 + py2) // 2)
        }

        if can_show_label:
            cchannel_labels.append(cchannel_data)
        else:
            small_cchannels.append(label)

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

    # Calculate vertical stacking layout
    layout = _calculate_vertical_layout(
        floor_plan_width,
        floor_plan_height,
        has_cchannel,
        len(sorted_sizes),
        small_cchannels
    )

    # Prepare title
    title = f"{floor_plan_name.upper()} CASSETTE PLAN" if floor_plan_name else "CASSETTE FLOOR PLAN"

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
        'cchannel_labels': cchannel_labels,
        'small_cchannels': small_cchannels,
        'cchannel_geometries': cchannel_geometries,
        'size_counts': size_counts,
        'size_colors': size_colors,
        'sorted_sizes': sorted_sizes,
        'statistics': statistics,
        'has_cchannel': has_cchannel,
        'per_cassette_mode': per_cassette_mode,
        'cchannel_widths_per_cassette': cchannel_widths_per_cassette,
        'title': title,
        'layout': layout
    }


def _calculate_vertical_layout(floor_plan_width, floor_plan_height, has_cchannel, num_cassette_sizes, small_cchannels):
    """Calculate positions for vertical stacking layout"""

    # Start position for legend area
    legend_start_y = floor_plan_height + 20

    # Calculate box dimensions
    box_spacing = 30
    current_y = legend_start_y

    # Statistics box
    stats_lines = 5 if has_cchannel else 3
    stats_height = stats_lines * 25 + 40
    stats_box = {
        'x': 20,
        'y': current_y,
        'width': 300,
        'height': stats_height
    }
    current_y += stats_height + box_spacing

    # C-channel info box (if applicable)
    cchannel_box = None
    if has_cchannel:
        # Base lines: min/max/avg = 3 lines
        cchannel_lines = 3
        # Add lines for small C-channels
        if small_cchannels:
            cchannel_lines += 1 + math.ceil(len(small_cchannels) / 2)  # Title + wrapped labels

        cchannel_height = cchannel_lines * 25 + 40
        cchannel_box = {
            'x': 20,
            'y': current_y,
            'width': 400,
            'height': cchannel_height
        }
        current_y += cchannel_height + box_spacing

    # Legend box
    legend_entries = num_cassette_sizes + (2 if has_cchannel else 0)
    legend_height = legend_entries * 30 + 40
    legend_box = {
        'x': 20,
        'y': current_y,
        'width': 250,
        'height': legend_height
    }
    current_y += legend_height

    # Calculate total dimensions
    total_height = current_y + 20  # Add bottom margin
    total_width = max(floor_plan_width, 450)  # Ensure minimum width

    return {
        'legend_start_y': legend_start_y,
        'stats_box': stats_box,
        'cchannel_box': cchannel_box,
        'legend_box': legend_box,
        'total_width': total_width,
        'total_height': total_height
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

    # Draw C-channel geometries
    cchannel_color = (240, 200, 180)  # Tan/beige (RGB format for PIL)
    for geom in data['cchannel_geometries']:
        minx, miny, maxx, maxy = geom['minx'], geom['miny'], geom['maxx'], geom['maxy']
        px1, py1 = data['to_pixel_coords'](minx, maxy)
        px2, py2 = data['to_pixel_coords'](maxx, miny)

        draw.rectangle([px1, py1, px2, py2], fill=cchannel_color, outline=(100, 100, 100))

    # Draw C-channel labels (with rotation)
    for cc_data in data['cchannel_labels']:
        label = cc_data['label']
        cx, cy = cc_data['center']
        is_horizontal = cc_data['is_horizontal']

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
    """Draw statistics, C-channel info, and legend boxes on PIL image"""

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

    if data['has_cchannel']:
        y_offset += 25
        cassette_area = stats.get('covered', 0) - stats['cchannel_area']
        draw.text((stats_box['x'] + 10, y_offset), f"Cassette Area: {cassette_area:.0f} sq ft", fill=(0, 0, 0), font=font_medium)
        y_offset += 25
        draw.text((stats_box['x'] + 10, y_offset), f"C-Channel Area: {stats['cchannel_area']:.1f} sq ft", fill=(0, 0, 0), font=font_medium)

    # C-channel info box
    if data['has_cchannel'] and layout['cchannel_box']:
        cc_box = layout['cchannel_box']
        draw.rectangle(
            [cc_box['x'] - 5, cc_box['y'] - 5,
             cc_box['x'] + cc_box['width'], cc_box['y'] + cc_box['height']],
            fill=(250, 250, 250), outline=(100, 100, 100)
        )

        draw.text((cc_box['x'] + 10, cc_box['y'] + 10), "C-CHANNEL INFO", fill=(0, 0, 0), font=font_large)

        y_offset = cc_box['y'] + 40

        if data['per_cassette_mode'] and data['cchannel_widths_per_cassette']:
            widths = data['cchannel_widths_per_cassette']
            min_c = min(widths)
            max_c = max(widths)
            avg_c = sum(widths) / len(widths)

            draw.text((cc_box['x'] + 10, y_offset), f"Min: {min_c:.2f}\"", fill=(0, 0, 0), font=font_medium)
            y_offset += 25
            draw.text((cc_box['x'] + 10, y_offset), f"Max: {max_c:.2f}\"", fill=(0, 0, 0), font=font_medium)
            y_offset += 25
            draw.text((cc_box['x'] + 10, y_offset), f"Avg: {avg_c:.2f}\"", fill=(0, 0, 0), font=font_medium)

        # Add small C-channels if any
        if data['small_cchannels']:
            y_offset += 25
            draw.text((cc_box['x'] + 10, y_offset), "Small C-Channels:", fill=(0, 0, 0), font=font_medium)
            y_offset += 20

            # Wrap labels
            labels_text = ", ".join(data['small_cchannels'])
            draw.text((cc_box['x'] + 10, y_offset), labels_text, fill=(0, 0, 0), font=font_small)

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

    # C-channel entries
    if data['has_cchannel']:
        # C-Channel color box
        draw.rectangle([legend_box['x'] + 10, y_offset - 12, legend_box['x'] + 30, y_offset + 8],
                      fill=(240, 200, 180), outline=(50, 50, 50))
        draw.text((legend_box['x'] + 40, y_offset - 5), "C-Channel", fill=(0, 0, 0), font=font_medium)
        y_offset += 30

        # Empty space color box
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

    # Draw C-channel geometries
    for geom in data['cchannel_geometries']:
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

    # Draw C-channel labels
    for cc_data in data['cchannel_labels']:
        label = cc_data['label']
        cx, cy = cc_data['center']
        is_horizontal = cc_data['is_horizontal']

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
    """Draw statistics, C-channel info, and legend boxes on SVG"""

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

    if data['has_cchannel']:
        y_offset += 25
        cassette_area = stats.get('covered', 0) - stats['cchannel_area']
        dwg.add(dwg.text(f"Cassette Area: {cassette_area:.0f} sq ft",
                        insert=(stats_box['x'] + 10, y_offset),
                        font_size=16, font_family='Arial', fill='black'))
        y_offset += 25
        dwg.add(dwg.text(f"C-Channel Area: {stats['cchannel_area']:.1f} sq ft",
                        insert=(stats_box['x'] + 10, y_offset),
                        font_size=16, font_family='Arial', fill='black'))

    # C-channel info box
    if data['has_cchannel'] and layout['cchannel_box']:
        cc_box = layout['cchannel_box']
        dwg.add(dwg.rect(insert=(cc_box['x'] - 5, cc_box['y'] - 5),
                        size=(cc_box['width'], cc_box['height']),
                        fill='rgb(250,250,250)',
                        stroke='rgb(100,100,100)',
                        stroke_width=1))

        dwg.add(dwg.text("C-CHANNEL INFO", insert=(cc_box['x'] + 10, cc_box['y'] + 25),
                        font_size=20, font_family='Arial', font_weight='bold', fill='black'))

        y_offset = cc_box['y'] + 55

        if data['per_cassette_mode'] and data['cchannel_widths_per_cassette']:
            widths = data['cchannel_widths_per_cassette']
            min_c = min(widths)
            max_c = max(widths)
            avg_c = sum(widths) / len(widths)

            dwg.add(dwg.text(f"Min: {min_c:.2f}\"", insert=(cc_box['x'] + 10, y_offset),
                           font_size=16, font_family='Arial', fill='black'))
            y_offset += 25
            dwg.add(dwg.text(f"Max: {max_c:.2f}\"", insert=(cc_box['x'] + 10, y_offset),
                           font_size=16, font_family='Arial', fill='black'))
            y_offset += 25
            dwg.add(dwg.text(f"Avg: {avg_c:.2f}\"", insert=(cc_box['x'] + 10, y_offset),
                           font_size=16, font_family='Arial', fill='black'))

        # Add small C-channels
        if data['small_cchannels']:
            y_offset += 25
            dwg.add(dwg.text("Small C-Channels:", insert=(cc_box['x'] + 10, y_offset),
                           font_size=16, font_family='Arial', fill='black'))
            y_offset += 20

            labels_text = ", ".join(data['small_cchannels'])
            dwg.add(dwg.text(labels_text, insert=(cc_box['x'] + 10, y_offset),
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

    # C-channel entries
    if data['has_cchannel']:
        dwg.add(dwg.rect(insert=(legend_box['x'] + 10, y_offset - 12),
                        size=(20, 20),
                        fill='rgb(240,200,180)',
                        stroke='rgb(50,50,50)',
                        stroke_width=1))
        dwg.add(dwg.text("C-Channel", insert=(legend_box['x'] + 40, y_offset + 3),
                        font_size=16, font_family='Arial', fill='black'))
        y_offset += 30

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

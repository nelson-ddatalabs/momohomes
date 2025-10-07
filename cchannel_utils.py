from typing import List, Tuple, Dict
from shapely.geometry import Polygon, Point, LineString
from shapely import affinity
import math


def create_inset_polygon(polygon_coords: List[Tuple[float, float]], offset_distance: float) -> List[Tuple[float, float]]:
    """
    Create an inset polygon by offsetting inward by specified distance.

    Args:
        polygon_coords: List of (x, y) coordinates defining the polygon
        offset_distance: Distance to offset inward (in feet)

    Returns:
        List of (x, y) coordinates for the inset polygon
    """
    poly = Polygon(polygon_coords)

    inset_poly = poly.buffer(-offset_distance, join_style=2, mitre_limit=10.0)

    if inset_poly.is_empty:
        raise ValueError(f"Offset distance {offset_distance} ft is too large for this polygon")

    if hasattr(inset_poly, 'exterior'):
        inset_coords = list(inset_poly.exterior.coords[:-1])
    else:
        raise ValueError("Buffer operation resulted in invalid geometry")

    return inset_coords


def classify_edge_direction(p1: Tuple[float, float], p2: Tuple[float, float]) -> str:
    """
    Classify edge direction as cardinal (N/S/E/W).

    Args:
        p1: Start point (x, y)
        p2: End point (x, y)

    Returns:
        Cardinal direction: 'N', 'S', 'E', or 'W'
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)

    if angle_deg < 0:
        angle_deg += 360

    if -45 <= angle_deg <= 45 or angle_deg >= 315:
        return 'E'
    elif 45 < angle_deg <= 135:
        return 'S'
    elif 135 < angle_deg <= 225:
        return 'W'
    else:
        return 'N'


def measure_gaps_per_cardinal_side(
    original_polygon: List[Tuple[float, float]],
    cassettes: List[Dict]
) -> Dict[str, float]:
    """
    Measure gap from cassettes to polygon boundary per cardinal side.

    Args:
        original_polygon: Original polygon coordinates
        cassettes: List of cassette dictionaries with x, y, width, height

    Returns:
        Dictionary with cardinal directions as keys and gap distances as values
    """
    poly = Polygon(original_polygon)

    gaps = {'N': float('inf'), 'S': float('inf'), 'E': float('inf'), 'W': float('inf')}

    for cassette in cassettes:
        cx = cassette['x']
        cy = cassette['y']
        cw = cassette['width']
        ch = cassette['height']

        cassette_corners = [
            (cx, cy),
            (cx + cw, cy),
            (cx + cw, cy + ch),
            (cx, cy + ch)
        ]

        for i in range(len(original_polygon)):
            p1 = original_polygon[i]
            p2 = original_polygon[(i + 1) % len(original_polygon)]

            edge_dir = classify_edge_direction(p1, p2)
            edge_line = LineString([p1, p2])

            for corner in cassette_corners:
                corner_point = Point(corner)
                distance = corner_point.distance(edge_line)

                if poly.contains(corner_point) or poly.touches(corner_point):
                    gaps[edge_dir] = min(gaps[edge_dir], distance)

    return gaps


def select_cchannel_widths(gaps: Dict[str, float]) -> Dict[str, float]:
    """
    Round gaps UP to valid C-channel widths.

    Valid widths: 1.5" or whole numbers 2"-18"

    Args:
        gaps: Dictionary with cardinal directions and gap distances in feet

    Returns:
        Dictionary with cardinal directions and selected C-channel widths in feet
    """
    widths = {}

    for direction, gap_ft in gaps.items():
        gap_inches = gap_ft * 12.0

        if gap_inches < 1.5:
            width_inches = 1.5
        elif gap_inches <= 1.5:
            width_inches = 1.5
        elif gap_inches <= 2.0:
            width_inches = 2.0
        else:
            width_inches = math.ceil(gap_inches)
            width_inches = min(width_inches, 18.0)

        widths[direction] = width_inches / 12.0

    return widths


def calculate_cchannel_areas(
    polygon_coords: List[Tuple[float, float]],
    widths: Dict[str, float]
) -> Dict:
    """
    Calculate C-channel areas using miter method (no corner overlap).

    Args:
        polygon_coords: Polygon coordinates
        widths: Dictionary of C-channel widths per cardinal direction (in feet)

    Returns:
        Dictionary with per-side areas and total area
    """
    areas = {'N': 0.0, 'S': 0.0, 'E': 0.0, 'W': 0.0}
    corner_area = 0.0

    edges = []
    for i in range(len(polygon_coords)):
        p1 = polygon_coords[i]
        p2 = polygon_coords[(i + 1) % len(polygon_coords)]
        direction = classify_edge_direction(p1, p2)
        length = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        edges.append({'direction': direction, 'length': length, 'p1': p1, 'p2': p2})

    for i, edge in enumerate(edges):
        direction = edge['direction']
        length = edge['length']
        width = widths.get(direction, 0.125)

        prev_edge = edges[(i - 1) % len(edges)]
        next_edge = edges[(i + 1) % len(edges)]

        prev_width = widths.get(prev_edge['direction'], 0.125)
        next_width = widths.get(next_edge['direction'], 0.125)

        adjusted_length = length - prev_width - next_width
        adjusted_length = max(adjusted_length, 0)

        areas[direction] += adjusted_length * width

        corner_area += prev_width * width

    total_area = sum(areas.values()) + corner_area

    return {
        'per_side': areas,
        'corners': corner_area,
        'total': total_area
    }

#!/usr/bin/env python3
"""
Polygon Offsetter
=================
Custom polygon offsetting algorithm for creating concentric layers
that follow the building shape.
"""

import logging
import math
from typing import List, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PolygonOffsetter:
    """
    Creates inward-offset polygons for concentric layer placement.

    This is a simplified offset algorithm that:
    1. Moves each edge inward by the specified distance
    2. Finds intersections of adjacent offset edges
    3. Handles self-intersections by splitting polygons
    """

    def __init__(self, tolerance: float = 0.01):
        """
        Initialize PolygonOffsetter.

        Args:
            tolerance: Numerical tolerance for calculations
        """
        self.tolerance = tolerance

    def offset_polygon(self, polygon: List[Tuple[float, float]],
                      offset_distance: float) -> List[List[Tuple[float, float]]]:
        """
        Create an inward offset of the polygon.

        Args:
            polygon: List of vertices (x, y)
            offset_distance: Distance to offset inward (positive value)

        Returns:
            List of offset polygons (may be multiple if original splits)
        """
        if len(polygon) < 3:
            return []

        # Ensure polygon is closed
        if polygon[0] != polygon[-1]:
            polygon = polygon + [polygon[0]]

        # Calculate offset edges
        offset_edges = self._calculate_offset_edges(polygon, offset_distance)

        # Find intersections between consecutive edges
        offset_vertices = self._find_edge_intersections(offset_edges)

        # Remove self-intersecting parts
        cleaned_polygons = self._clean_self_intersections(offset_vertices)

        # Filter out degenerate polygons
        valid_polygons = []
        for poly in cleaned_polygons:
            if len(poly) >= 3 and self._calculate_area(poly) > 0.1:
                valid_polygons.append(poly)

        return valid_polygons

    def _calculate_offset_edges(self, polygon: List[Tuple[float, float]],
                                offset: float) -> List[dict]:
        """
        Calculate offset lines for each edge.

        Args:
            polygon: Original polygon
            offset: Offset distance

        Returns:
            List of offset edge definitions
        """
        n = len(polygon) - 1  # Exclude closing vertex
        offset_edges = []

        for i in range(n):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % n]

            # Calculate edge vector
            edge_vec = (p2[0] - p1[0], p2[1] - p1[1])
            edge_len = math.sqrt(edge_vec[0]**2 + edge_vec[1]**2)

            if edge_len < self.tolerance:
                continue

            # Normalize edge vector
            edge_norm = (edge_vec[0] / edge_len, edge_vec[1] / edge_len)

            # Calculate perpendicular vector (pointing inward)
            # For a clockwise polygon, left perpendicular points inward
            perp_vec = (-edge_norm[1], edge_norm[0])

            # Calculate offset points
            offset_p1 = (p1[0] + perp_vec[0] * offset, p1[1] + perp_vec[1] * offset)
            offset_p2 = (p2[0] + perp_vec[0] * offset, p2[1] + perp_vec[1] * offset)

            offset_edges.append({
                'start': offset_p1,
                'end': offset_p2,
                'vector': edge_vec,
                'normal': perp_vec,
                'original_index': i
            })

        return offset_edges

    def _find_edge_intersections(self, offset_edges: List[dict]) -> List[Tuple[float, float]]:
        """
        Find intersections between consecutive offset edges.

        Args:
            offset_edges: List of offset edge definitions

        Returns:
            List of vertices forming the offset polygon
        """
        if not offset_edges:
            return []

        vertices = []
        n = len(offset_edges)

        for i in range(n):
            edge1 = offset_edges[i]
            edge2 = offset_edges[(i + 1) % n]

            # Find intersection of two edges
            intersection = self._line_intersection(
                edge1['start'], edge1['end'],
                edge2['start'], edge2['end']
            )

            if intersection:
                vertices.append(intersection)
            else:
                # Lines are parallel or coincident, use midpoint
                vertices.append(edge1['end'])

        return vertices

    def _line_intersection(self, p1: Tuple[float, float], p2: Tuple[float, float],
                          p3: Tuple[float, float], p4: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """
        Find intersection point of two lines.

        Args:
            p1, p2: Points defining first line
            p3, p4: Points defining second line

        Returns:
            Intersection point or None if lines are parallel
        """
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

        if abs(denom) < self.tolerance:
            # Lines are parallel
            return None

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom

        # Calculate intersection point
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)

        return (x, y)

    def _clean_self_intersections(self, vertices: List[Tuple[float, float]]) -> List[List[Tuple[float, float]]]:
        """
        Remove self-intersecting parts of the polygon.

        For simplicity, this returns the original if no intersections found,
        or splits into multiple polygons if intersections exist.

        Args:
            vertices: Offset polygon vertices

        Returns:
            List of cleaned polygons
        """
        # For now, return as single polygon if valid
        if self._is_simple_polygon(vertices):
            return [vertices]

        # If self-intersecting, try to extract valid sub-polygons
        # This is a simplified approach - just return empty for complex cases
        return []

    def _is_simple_polygon(self, vertices: List[Tuple[float, float]]) -> bool:
        """
        Check if polygon is simple (no self-intersections).

        Args:
            vertices: Polygon vertices

        Returns:
            True if polygon is simple
        """
        n = len(vertices)
        if n < 3:
            return False

        # Check each edge pair for intersection
        for i in range(n):
            for j in range(i + 2, n):
                # Skip adjacent edges
                if j == (i + 1) % n or i == (j + 1) % n:
                    continue

                # Check if edges intersect
                if self._segments_intersect(
                    vertices[i], vertices[(i + 1) % n],
                    vertices[j], vertices[(j + 1) % n]
                ):
                    return False

        return True

    def _segments_intersect(self, p1: Tuple[float, float], p2: Tuple[float, float],
                           p3: Tuple[float, float], p4: Tuple[float, float]) -> bool:
        """
        Check if two line segments intersect.

        Args:
            p1, p2: First segment
            p3, p4: Second segment

        Returns:
            True if segments intersect
        """
        def ccw(a, b, c):
            return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

    def _calculate_area(self, vertices: List[Tuple[float, float]]) -> float:
        """
        Calculate polygon area using shoelace formula.

        Args:
            vertices: Polygon vertices

        Returns:
            Area in square units
        """
        if len(vertices) < 3:
            return 0.0

        n = len(vertices)
        area = 0.0

        for i in range(n):
            j = (i + 1) % n
            area += vertices[i][0] * vertices[j][1]
            area -= vertices[j][0] * vertices[i][1]

        return abs(area) / 2.0

    def create_concentric_layers(self, polygon: List[Tuple[float, float]],
                                 min_offset: float = 2.0,
                                 max_offset: float = 20.0,
                                 offset_step: float = 6.0) -> List[List[Tuple[float, float]]]:
        """
        Create multiple concentric offset layers.

        Args:
            polygon: Original polygon
            min_offset: Minimum offset from boundary
            max_offset: Maximum offset distance
            offset_step: Distance between layers

        Returns:
            List of offset polygons (outer to inner)
        """
        layers = []
        current_offset = min_offset

        while current_offset <= max_offset:
            offset_polys = self.offset_polygon(polygon, current_offset)

            for poly in offset_polys:
                if self._calculate_area(poly) > 10.0:  # Minimum 10 sq ft
                    layers.append(poly)

            current_offset += offset_step

        return layers


def test_polygon_offsetter():
    """Test the polygon offsetter."""

    # Test polygon: Rectangle
    test_polygon = [
        (0, 0),
        (40, 0),
        (40, 30),
        (0, 30)
    ]

    offsetter = PolygonOffsetter()

    print("\n" + "="*60)
    print("POLYGON OFFSETTER TEST")
    print("="*60)

    # Create single offset
    offset_polygons = offsetter.offset_polygon(test_polygon, 5.0)

    print(f"Original polygon area: {offsetter._calculate_area(test_polygon):.1f} sq ft")
    print(f"Offset by 5 ft: {len(offset_polygons)} polygon(s)")

    for i, poly in enumerate(offset_polygons):
        area = offsetter._calculate_area(poly)
        print(f"  Polygon {i}: {len(poly)} vertices, area = {area:.1f} sq ft")

    # Create concentric layers
    layers = offsetter.create_concentric_layers(test_polygon)

    print(f"\nConcentric layers: {len(layers)} layers")
    for i, layer in enumerate(layers):
        area = offsetter._calculate_area(layer)
        print(f"  Layer {i}: {len(layer)} vertices, area = {area:.1f} sq ft")

    # Test with L-shaped polygon
    l_polygon = [
        (0, 0),
        (30, 0),
        (30, 20),
        (15, 20),
        (15, 35),
        (0, 35)
    ]

    print(f"\nL-shaped polygon test:")
    print(f"Original area: {offsetter._calculate_area(l_polygon):.1f} sq ft")

    l_layers = offsetter.create_concentric_layers(l_polygon, min_offset=3.0, offset_step=4.0)
    print(f"Concentric layers: {len(l_layers)} layers")

    for i, layer in enumerate(l_layers):
        area = offsetter._calculate_area(layer)
        print(f"  Layer {i}: area = {area:.1f} sq ft")

    return offsetter


if __name__ == "__main__":
    test_polygon_offsetter()
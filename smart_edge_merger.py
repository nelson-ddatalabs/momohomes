#!/usr/bin/env python3
"""
Smart Edge Merger
=================
Intelligently merges zero-length edges with adjacent edges to simplify polygons
while maintaining closure.
"""

import logging
from typing import List, Dict, Tuple
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SmartEdgeMerger:
    """
    Merges zero-length or very small edges with adjacent edges.

    When user enters 0 for an edge measurement, it means they don't
    see it as a meaningful edge, so we merge it with adjacent edges.
    """

    def __init__(self, min_edge_length: float = 0.5):
        """
        Initialize the edge merger.

        Args:
            min_edge_length: Minimum edge length in feet. Edges smaller than this
                           will be merged with adjacent edges.
        """
        self.min_edge_length = min_edge_length
        self.merged_edges = []
        self.merged_measurements = {}

    def merge_edges(self, edges: List, measurements: Dict[int, float]) -> Tuple[List, Dict[int, float]]:
        """
        Merge zero-length and small edges with adjacent edges.

        Args:
            edges: List of edge objects with cardinal directions
            measurements: Dictionary mapping edge index to measurement in feet

        Returns:
            Tuple of (merged_edges, merged_measurements)
        """
        if not edges or not measurements:
            return edges, measurements

        logger.info(f"Starting edge merging with {len(edges)} edges")

        # Identify edges to remove (0 ft) and edges to merge (small but non-zero)
        edges_to_remove = []  # Edges with 0 measurement
        edges_to_merge = []   # Edges with small non-zero measurement

        for i, edge in enumerate(edges):
            if i in measurements:
                if measurements[i] == 0:
                    edges_to_remove.append(i)
                    logger.info(f"Edge {i} ({edge.cardinal_direction}, 0 ft) marked for removal")
                elif measurements[i] <= self.min_edge_length:
                    edges_to_merge.append(i)
                    logger.info(f"Edge {i} ({edge.cardinal_direction}, {measurements[i]} ft) marked for merging")

        if not edges_to_remove and not edges_to_merge:
            logger.info("No edges need removal or merging")
            return edges, measurements

        # Create merged edge list
        merged_edges = []
        merged_measurements = {}
        skip_indices = set()

        new_idx = 0
        for i, edge in enumerate(edges):
            if i in skip_indices:
                continue

            if i in edges_to_remove:
                # Edge has 0 measurement - completely remove it
                logger.info(f"Removing edge {i} (0 ft measurement)")
                skip_indices.add(i)
            elif i in edges_to_merge:
                # Edge has small non-zero measurement - try to merge
                logger.info(f"Processing small edge {i} ({measurements[i]} ft)")

                prev_idx = (i - 1) % len(edges)
                next_idx = (i + 1) % len(edges)

                prev_edge = edges[prev_idx] if prev_idx not in edges_to_remove and prev_idx not in edges_to_merge else None
                next_edge = edges[next_idx] if next_idx not in edges_to_remove and next_idx not in edges_to_merge else None

                if prev_edge and next_edge and self._can_merge_adjacent(prev_edge, next_edge):
                    logger.info(f"Merging adjacent edges {prev_idx} and {next_idx}")
                    skip_indices.add(i)
                else:
                    # Keep small edge with its actual measurement
                    logger.info(f"Keeping small edge {i} with measurement {measurements[i]} ft")
                    merged_edges.append(edge)
                    merged_measurements[new_idx] = measurements[i]
                    new_idx += 1
            elif i not in skip_indices:
                # Normal edge - keep as-is
                merged_edges.append(edge)
                merged_measurements[new_idx] = measurements.get(i, 0.0)
                new_idx += 1

        logger.info(f"Edge merging complete: {len(edges)} -> {len(merged_edges)} edges")

        # Verify closure after merging
        if self._verify_closure(merged_edges, merged_measurements):
            logger.info("✓ Polygon closure maintained after merging")
        else:
            logger.warning("⚠ Polygon closure may be affected by merging")

        self.merged_edges = merged_edges
        self.merged_measurements = merged_measurements

        return merged_edges, merged_measurements

    def _can_merge_adjacent(self, edge1, edge2) -> bool:
        """
        Check if two edges can be merged (are they perpendicular or opposite).

        Args:
            edge1: First edge
            edge2: Second edge

        Returns:
            True if edges are perpendicular (form a corner)
        """
        # Edges in same direction can't be merged (would create duplicate)
        if edge1.cardinal_direction == edge2.cardinal_direction:
            return False

        # Opposite directions can't be merged (would cancel out)
        opposites = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}
        if edge1.cardinal_direction == opposites.get(edge2.cardinal_direction):
            return False

        # Perpendicular edges can be merged (form a corner)
        return True

    def _verify_closure(self, edges: List, measurements: Dict[int, float]) -> bool:
        """
        Verify that the polygon still closes after merging.

        Args:
            edges: List of edges
            measurements: Edge measurements

        Returns:
            True if polygon closes
        """
        # Sum measurements by direction
        sums = {'N': 0.0, 'S': 0.0, 'E': 0.0, 'W': 0.0}

        for i, edge in enumerate(edges):
            if i in measurements:
                sums[edge.cardinal_direction] += measurements[i]

        # Check if opposite directions balance
        ns_diff = abs(sums['N'] - sums['S'])
        ew_diff = abs(sums['E'] - sums['W'])

        tolerance = 0.01  # Allow small rounding errors

        return ns_diff < tolerance and ew_diff < tolerance

    def simplify_polygon(self, polygon: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Simplify polygon by removing duplicate vertices and merging collinear edges.

        Args:
            polygon: List of (x, y) vertices

        Returns:
            Simplified polygon
        """
        if len(polygon) < 3:
            return polygon

        simplified = []

        # Remove consecutive duplicate vertices
        for i, vertex in enumerate(polygon):
            if i == 0 or vertex != polygon[i-1]:
                simplified.append(vertex)

        # Remove last vertex if it duplicates the first
        if len(simplified) > 1 and simplified[-1] == simplified[0]:
            simplified = simplified[:-1]

        logger.info(f"Polygon simplified: {len(polygon)} -> {len(simplified)} vertices")

        # Further simplification: remove collinear points
        final_polygon = self._remove_collinear_points(simplified)

        return final_polygon

    def _remove_collinear_points(self, polygon: List[Tuple[float, float]],
                                 tolerance: float = 0.01) -> List[Tuple[float, float]]:
        """
        Remove vertices that are collinear with their neighbors.

        Args:
            polygon: List of vertices
            tolerance: Angle tolerance in radians

        Returns:
            Polygon with collinear points removed
        """
        if len(polygon) < 3:
            return polygon

        result = []
        n = len(polygon)

        for i in range(n):
            prev_point = polygon[(i - 1) % n]
            curr_point = polygon[i]
            next_point = polygon[(i + 1) % n]

            # Check if current point is collinear with neighbors
            if not self._is_collinear(prev_point, curr_point, next_point, tolerance):
                result.append(curr_point)
            else:
                logger.debug(f"Removing collinear vertex at index {i}: {curr_point}")

        if len(result) != len(polygon):
            logger.info(f"Removed {len(polygon) - len(result)} collinear vertices")

        return result if len(result) >= 3 else polygon

    def _is_collinear(self, p1: Tuple[float, float], p2: Tuple[float, float],
                      p3: Tuple[float, float], tolerance: float = 0.01) -> bool:
        """
        Check if three points are collinear.

        Uses cross product method: if cross product is near zero, points are collinear.

        Args:
            p1, p2, p3: Three points
            tolerance: Tolerance for collinearity check

        Returns:
            True if points are collinear
        """
        # Vector from p1 to p2
        v1 = (p2[0] - p1[0], p2[1] - p1[1])
        # Vector from p2 to p3
        v2 = (p3[0] - p2[0], p3[1] - p2[1])

        # Cross product (in 2D, this gives the z-component)
        cross = v1[0] * v2[1] - v1[1] * v2[0]

        # Check if cross product is near zero
        return abs(cross) < tolerance

    def get_merge_summary(self) -> str:
        """
        Get a summary of the edge merging results.

        Returns:
            String summary of merging operations
        """
        summary = []
        summary.append("=" * 50)
        summary.append("EDGE MERGING SUMMARY")
        summary.append("=" * 50)

        if self.merged_edges:
            summary.append(f"Original edges: {len(self.merged_measurements) + len(self.merged_edges)}")
            summary.append(f"Merged edges: {len(self.merged_edges)}")
            summary.append(f"Edges removed: {len(self.merged_measurements) + len(self.merged_edges) - len(self.merged_edges)}")

            # Show direction totals
            direction_totals = {'N': 0, 'S': 0, 'E': 0, 'W': 0}
            for i, edge in enumerate(self.merged_edges):
                if i in self.merged_measurements:
                    direction_totals[edge.cardinal_direction] += self.merged_measurements[i]

            summary.append("\nDirection totals after merging:")
            summary.append(f"  North: {direction_totals['N']:.1f} ft")
            summary.append(f"  South: {direction_totals['S']:.1f} ft")
            summary.append(f"  East: {direction_totals['E']:.1f} ft")
            summary.append(f"  West: {direction_totals['W']:.1f} ft")

            summary.append("\nClosure check:")
            ns_diff = abs(direction_totals['N'] - direction_totals['S'])
            ew_diff = abs(direction_totals['E'] - direction_totals['W'])
            summary.append(f"  N-S difference: {ns_diff:.3f} ft")
            summary.append(f"  E-W difference: {ew_diff:.3f} ft")

            if ns_diff < 0.01 and ew_diff < 0.01:
                summary.append("  ✓ Perfect closure maintained!")
            else:
                summary.append("  ⚠ Closure may need adjustment")
        else:
            summary.append("No merging performed yet")

        return "\n".join(summary)


def test_edge_merger():
    """Test the smart edge merger."""

    # Create mock edge class for testing
    class MockEdge:
        def __init__(self, direction):
            self.cardinal_direction = direction

    # Test case: Building with zero-length edges
    edges = [
        MockEdge('S'),  # Edge 0: 0 ft (zero-length)
        MockEdge('E'),  # Edge 1: 23.5 ft
        MockEdge('S'),  # Edge 2: 6.5 ft
        MockEdge('E'),  # Edge 3: 18.5 ft
        MockEdge('S'),  # Edge 4: 8 ft
        MockEdge('E'),  # Edge 5: 0 ft (zero-length)
        MockEdge('S'),  # Edge 6: 8 ft
        MockEdge('W'),  # Edge 7: 0 ft (zero-length)
        MockEdge('S'),  # Edge 8: 8 ft
        MockEdge('W'),  # Edge 9: 12.5 ft
        MockEdge('S'),  # Edge 10: 6.5 ft
        MockEdge('W'),  # Edge 11: 21 ft
        MockEdge('N'),  # Edge 12: 15.5 ft
        MockEdge('W'),  # Edge 13: 8.5 ft
        MockEdge('N'),  # Edge 14: 21.5 ft
    ]

    measurements = {
        0: 0.0,
        1: 23.5,
        2: 6.5,
        3: 18.5,
        4: 8.0,
        5: 0.0,
        6: 8.0,
        7: 0.0,
        8: 8.0,
        9: 12.5,
        10: 6.5,
        11: 21.0,
        12: 15.5,
        13: 8.5,
        14: 21.5
    }

    # Test merging
    merger = SmartEdgeMerger(min_edge_length=0.5)
    merged_edges, merged_measurements = merger.merge_edges(edges, measurements)

    print(merger.get_merge_summary())

    # Test polygon simplification
    polygon = [
        (0.0, 37.0),
        (0.0, 37.0),  # Duplicate
        (23.5, 37.0),
        (23.5, 30.5),
        (42.0, 30.5),
        (42.0, 22.5),
        (42.0, 22.5),  # Duplicate
        (42.0, 14.5),
        (42.0, 14.5),  # Duplicate
        (42.0, 6.5),
        (29.5, 6.5),
        (29.5, 0.0),
        (8.5, 0.0),
        (8.5, 15.5),
        (0.0, 15.5),
    ]

    simplified = merger.simplify_polygon(polygon)
    print(f"\nPolygon simplification: {len(polygon)} -> {len(simplified)} vertices")

    return merger


if __name__ == "__main__":
    test_edge_merger()
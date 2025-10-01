#!/usr/bin/env python3
"""
Perimeter Tracer Stage
======================
Traces the polygon perimeter and places cassettes along edges
with micro-adjustments for optimal fit.
"""

import logging
import math
from typing import List, Tuple, Optional

from optimization_pipeline import PipelineStage, PipelineContext, Cassette

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerimeterTracer(PipelineStage):
    """
    Places cassettes along the polygon perimeter.

    Strategy:
    1. Trace each edge of the polygon clockwise
    2. For each edge, determine the best cassette orientation
    3. Place cassettes with micro-adjustments (±0.125 ft)
    4. Prioritize larger cassettes to minimize count
    5. Fill remaining gaps with smaller cassettes
    """

    # Available cassette sizes (width, height) in feet, sorted by area (largest first)
    CASSETTE_SIZES = [
        (6, 8), (5, 8), (4, 8),  # Large cassettes
        (6, 6), (5, 6), (4, 6),  # Medium cassettes
        (3, 8), (3, 6),          # Narrow cassettes
        (2, 8), (2, 6), (2, 4),  # Small cassettes
    ]

    def __init__(self, micro_adjustment: float = 0.125, max_gap: float = 0.5):
        """
        Initialize PerimeterTracer.

        Args:
            micro_adjustment: Step size for micro-adjustments in feet (default 0.125)
            max_gap: Maximum allowed gap between cassettes in feet (default 0.5)
        """
        self.micro_adjustment = micro_adjustment
        self.max_gap = max_gap

    @property
    def name(self) -> str:
        return "PerimeterTracer"

    def process(self, context: PipelineContext) -> PipelineContext:
        """Place cassettes along polygon perimeter."""
        polygon = context.polygon
        if len(polygon) < 3:
            logger.warning("Polygon has less than 3 vertices")
            return context

        initial_count = len(context.cassettes)

        # Trace each edge
        n = len(polygon)
        for i in range(n):
            next_idx = (i + 1) % n
            p1 = polygon[i]
            p2 = polygon[next_idx]

            # Place cassettes along this edge
            edge_cassettes = self._trace_edge(p1, p2, context.polygon, context.cassettes)

            for cassette in edge_cassettes:
                context.cassettes.append(cassette)

            if edge_cassettes:
                logger.info(f"  Edge {i}->{next_idx}: placed {len(edge_cassettes)} cassettes")

        # Update metadata
        placed_count = len(context.cassettes) - initial_count
        context.metadata['perimeter_tracer'] = {
            'cassettes_placed': placed_count,
            'coverage_after': context.get_coverage()
        }

        logger.info(f"  Perimeter tracing complete: {placed_count} cassettes placed")

        return context

    def _trace_edge(self, p1: Tuple[float, float], p2: Tuple[float, float],
                   polygon: List[Tuple[float, float]],
                   existing_cassettes: List[Cassette]) -> List[Cassette]:
        """
        Place cassettes along a single edge.

        Args:
            p1: Start point of edge
            p2: End point of edge
            polygon: Building polygon
            existing_cassettes: Already placed cassettes

        Returns:
            List of cassettes placed along this edge
        """
        edge_length = self._distance(p1, p2)
        if edge_length < 2.0:  # Skip very short edges
            return []

        # Determine edge direction
        edge_vector = (p2[0] - p1[0], p2[1] - p1[1])
        edge_angle = math.atan2(edge_vector[1], edge_vector[0])

        # Determine if edge is horizontal or vertical
        is_horizontal = abs(edge_vector[1]) < abs(edge_vector[0])

        placed_cassettes = []

        # Try to place cassettes along the edge
        current_position = 0.0

        while current_position < edge_length - 1.0:  # Leave at least 1 ft
            # Calculate current point along edge
            t = current_position / edge_length
            current_x = p1[0] + t * edge_vector[0]
            current_y = p1[1] + t * edge_vector[1]

            # Try to place a cassette at this position
            cassette, used_length = self._place_edge_cassette(
                (current_x, current_y),
                edge_vector,
                edge_length - current_position,
                is_horizontal,
                polygon,
                existing_cassettes + placed_cassettes
            )

            if cassette:
                placed_cassettes.append(cassette)
                current_position += used_length
            else:
                # Move forward by minimum amount if can't place
                current_position += 1.0

        return placed_cassettes

    def _place_edge_cassette(self, start_point: Tuple[float, float],
                            edge_vector: Tuple[float, float],
                            remaining_length: float,
                            is_horizontal: bool,
                            polygon: List[Tuple[float, float]],
                            existing_cassettes: List[Cassette]) -> Tuple[Optional[Cassette], float]:
        """
        Try to place a cassette along an edge.

        Args:
            start_point: Starting position on edge
            edge_vector: Direction vector of edge
            remaining_length: Remaining length on edge
            is_horizontal: Whether edge is primarily horizontal
            polygon: Building polygon
            existing_cassettes: Already placed cassettes

        Returns:
            Tuple of (cassette, length_used) or (None, 0)
        """
        # Normalize edge vector
        edge_length = math.sqrt(edge_vector[0]**2 + edge_vector[1]**2)
        edge_normal = (edge_vector[0]/edge_length, edge_vector[1]/edge_length)

        # Calculate perpendicular vector (pointing inward)
        perp_vector = (-edge_normal[1], edge_normal[0])

        # Try cassette sizes from largest to smallest
        for width, height in self.CASSETTE_SIZES:
            # Determine orientation based on edge direction
            if is_horizontal:
                # Edge is horizontal - cassette width along edge
                cassette_along = width
                cassette_perp = height
            else:
                # Edge is vertical - cassette height along edge
                cassette_along = height
                cassette_perp = width

            # Check if cassette fits along remaining edge
            if cassette_along > remaining_length + self.max_gap:
                continue

            # Try different positions perpendicular to edge (with micro-adjustments)
            for perp_offset in self._generate_micro_adjustments(0, cassette_perp):
                # Calculate cassette position
                x = start_point[0] + perp_vector[0] * perp_offset
                y = start_point[1] + perp_vector[1] * perp_offset

                # Adjust for cassette orientation
                if is_horizontal:
                    cassette = Cassette(x, y, width, height)
                else:
                    cassette = Cassette(x, y, height, width)

                # Check validity
                if self._is_valid_placement(cassette, polygon, existing_cassettes):
                    return cassette, cassette_along

        return None, 0

    def _generate_micro_adjustments(self, base: float, max_offset: float) -> List[float]:
        """
        Generate micro-adjustment positions.

        Args:
            base: Base position
            max_offset: Maximum offset allowed

        Returns:
            List of positions to try
        """
        positions = [base]

        # Add micro-adjustments
        step = self.micro_adjustment
        current = step
        while current <= max_offset:
            positions.append(base - current)  # Try inward first
            positions.append(base + current)  # Then outward
            current += step

        return positions

    def _is_valid_placement(self, cassette: Cassette,
                           polygon: List[Tuple[float, float]],
                           existing_cassettes: List[Cassette]) -> bool:
        """
        Check if cassette placement is valid.

        Args:
            cassette: Cassette to place
            polygon: Building polygon
            existing_cassettes: Already placed cassettes

        Returns:
            True if placement is valid
        """
        # Check 1: No overlap with existing cassettes
        for existing in existing_cassettes:
            if cassette.overlaps(existing):
                return False

        # Check 2: Cassette is within polygon boundaries
        # Use a more thorough check with multiple sample points
        sample_points = self._get_cassette_sample_points(cassette)

        for point in sample_points:
            if not self._point_in_polygon(point, polygon):
                return False

        return True

    def _get_cassette_sample_points(self, cassette: Cassette) -> List[Tuple[float, float]]:
        """
        Get sample points for cassette boundary checking.

        Args:
            cassette: Cassette to sample

        Returns:
            List of sample points
        """
        points = []

        # Corners
        points.extend(cassette.get_corners())

        # Mid-points of edges
        points.append((cassette.x + cassette.width/2, cassette.y))
        points.append((cassette.x + cassette.width, cassette.y + cassette.height/2))
        points.append((cassette.x + cassette.width/2, cassette.y + cassette.height))
        points.append((cassette.x, cassette.y + cassette.height/2))

        # Center
        points.append((cassette.x + cassette.width/2, cassette.y + cassette.height/2))

        return points

    def _distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Calculate distance between two points."""
        return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

    def _point_in_polygon(self, point: Tuple[float, float],
                         polygon: List[Tuple[float, float]]) -> bool:
        """
        Check if a point is inside the polygon using ray casting.

        Args:
            point: (x, y) coordinates
            polygon: List of vertices

        Returns:
            True if point is inside polygon
        """
        x, y = point
        n = len(polygon)
        inside = False

        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside


def test_perimeter_tracer():
    """Test the PerimeterTracer stage."""
    from optimization_pipeline import PipelineContext
    from corner_placer_stage import CornerPlacer

    # Test polygon: Rectangle with notch
    test_polygon = [
        (0, 0),
        (40, 0),
        (40, 25),
        (25, 25),
        (25, 20),
        (15, 20),
        (15, 25),
        (0, 25)
    ]

    # Create context
    context = PipelineContext(test_polygon)

    # First place corners
    corner_placer = CornerPlacer()
    context = corner_placer.process(context)
    print(f"After corner placement: {len(context.cassettes)} cassettes, {context.get_coverage():.1f}% coverage")

    # Then trace perimeter
    tracer = PerimeterTracer()
    result_context = tracer.process(context)

    print("\n" + "="*60)
    print("PERIMETER TRACER TEST RESULTS")
    print("="*60)
    print(f"Total cassettes: {len(result_context.cassettes)}")
    print(f"Coverage: {result_context.get_coverage():.1f}%")

    # Show size distribution
    size_dist = {}
    for cassette in result_context.cassettes:
        size = cassette.size
        size_dist[size] = size_dist.get(size, 0) + 1

    print("\nCassette size distribution:")
    for size, count in sorted(size_dist.items()):
        print(f"  {size}: {count} cassettes")

    return result_context


if __name__ == "__main__":
    test_perimeter_tracer()
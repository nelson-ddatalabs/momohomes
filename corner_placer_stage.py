#!/usr/bin/env python3
"""
Corner Placer Stage
===================
Places medium cassettes at polygon corners/vertices to establish
a foundation for perimeter tracing.
"""

import logging
import math
from typing import List, Tuple, Optional

from optimization_pipeline import PipelineStage, PipelineContext, Cassette

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CornerPlacer(PipelineStage):
    """
    Places cassettes at polygon corners.

    Strategy:
    1. Identify all corners (vertices)
    2. Calculate angle at each corner
    3. Place medium cassettes (4x6, 5x6) at suitable corners
    4. For acute angles: use smaller cassettes
    5. For obtuse angles: use larger cassettes
    6. Ensure no overlaps and stay within boundaries
    """

    # Available cassette sizes (width, height) in feet
    CASSETTE_SIZES = [
        (4, 8), (4, 6), (3, 8),  # Medium cassettes
        (5, 8), (5, 6),          # Larger medium
        (2, 8), (2, 6), (2, 4),  # Small cassettes for tight corners
    ]

    def __init__(self, min_angle: float = 30.0):
        """
        Initialize CornerPlacer.

        Args:
            min_angle: Minimum angle in degrees to consider placing a cassette
        """
        self.min_angle = min_angle
        self.placed_cassettes = []

    @property
    def name(self) -> str:
        return "CornerPlacer"

    def process(self, context: PipelineContext) -> PipelineContext:
        """Place cassettes at polygon corners."""
        polygon = context.polygon
        if len(polygon) < 3:
            logger.warning("Polygon has less than 3 vertices")
            return context

        # Identify and analyze corners
        corners = self._analyze_corners(polygon)

        # Sort corners by angle (acute angles first, as they're harder to fill)
        corners.sort(key=lambda c: c['angle'])

        # Place cassettes at corners
        placed_count = 0
        for corner in corners:
            if corner['angle'] < self.min_angle:
                logger.debug(f"Skipping corner at {corner['vertex']} with angle {corner['angle']:.1f}°")
                continue

            cassette = self._place_corner_cassette(
                corner,
                context.polygon,
                context.cassettes
            )

            if cassette:
                context.cassettes.append(cassette)
                placed_count += 1
                logger.info(f"  Placed {cassette.size} cassette at corner {corner['index']} "
                          f"({corner['vertex'][0]:.1f}, {corner['vertex'][1]:.1f}), angle={corner['angle']:.1f}°")

        # Update metadata
        context.metadata['corner_placer'] = {
            'corners_analyzed': len(corners),
            'cassettes_placed': placed_count,
            'coverage_after': context.get_coverage()
        }

        logger.info(f"  Corner placement complete: {placed_count} cassettes placed at corners")

        return context

    def _analyze_corners(self, polygon: List[Tuple[float, float]]) -> List[dict]:
        """
        Analyze all corners in the polygon.

        Returns:
            List of corner dictionaries with vertex, angle, and metadata
        """
        corners = []
        n = len(polygon)

        for i in range(n):
            prev_idx = (i - 1) % n
            next_idx = (i + 1) % n

            prev_vertex = polygon[prev_idx]
            curr_vertex = polygon[i]
            next_vertex = polygon[next_idx]

            # Calculate angle at this corner
            angle = self._calculate_angle(prev_vertex, curr_vertex, next_vertex)

            # Calculate edge lengths
            edge_before = self._distance(prev_vertex, curr_vertex)
            edge_after = self._distance(curr_vertex, next_vertex)

            corners.append({
                'index': i,
                'vertex': curr_vertex,
                'angle': angle,
                'edge_before': edge_before,
                'edge_after': edge_after,
                'prev_vertex': prev_vertex,
                'next_vertex': next_vertex,
                'type': self._classify_angle(angle)
            })

        return corners

    def _calculate_angle(self, p1: Tuple[float, float],
                        p2: Tuple[float, float],
                        p3: Tuple[float, float]) -> float:
        """
        Calculate interior angle at p2.

        Args:
            p1: Previous vertex
            p2: Current vertex (corner)
            p3: Next vertex

        Returns:
            Interior angle in degrees
        """
        # Vector from p2 to p1
        v1 = (p1[0] - p2[0], p1[1] - p2[1])
        # Vector from p2 to p3
        v2 = (p3[0] - p2[0], p3[1] - p2[1])

        # Calculate magnitudes
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2)

        if mag1 == 0 or mag2 == 0:
            return 0.0

        # Normalize vectors
        v1_norm = (v1[0]/mag1, v1[1]/mag1)
        v2_norm = (v2[0]/mag2, v2[1]/mag2)

        # Calculate dot product
        dot_product = v1_norm[0] * v2_norm[0] + v1_norm[1] * v2_norm[1]

        # Clamp to [-1, 1] to avoid numerical errors
        dot_product = max(-1.0, min(1.0, dot_product))

        # Calculate angle in radians then convert to degrees
        angle_rad = math.acos(dot_product)
        angle_deg = math.degrees(angle_rad)

        # Check if angle is reflex (> 180°) using cross product
        cross = v1[0] * v2[1] - v1[1] * v2[0]

        if cross < 0:
            # Interior angle is actually the reflex angle
            angle_deg = 360 - angle_deg

        return angle_deg

    def _classify_angle(self, angle: float) -> str:
        """Classify angle type."""
        if angle < 90:
            return "acute"
        elif angle == 90:
            return "right"
        elif angle < 180:
            return "obtuse"
        else:
            return "reflex"

    def _distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Calculate distance between two points."""
        return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

    def _place_corner_cassette(self, corner: dict,
                               polygon: List[Tuple[float, float]],
                               existing_cassettes: List[Cassette]) -> Optional[Cassette]:
        """
        Place a cassette at the given corner.

        Args:
            corner: Corner dictionary with vertex, angle, etc.
            polygon: Building polygon
            existing_cassettes: Already placed cassettes

        Returns:
            Placed cassette or None if couldn't place
        """
        vertex = corner['vertex']
        angle = corner['angle']

        # Select cassette sizes based on angle
        if angle < 60:
            # Very acute angle - use smallest cassettes
            sizes_to_try = [(2, 4), (2, 6), (3, 6)]
        elif angle < 90:
            # Acute angle - use small to medium cassettes
            sizes_to_try = [(3, 6), (4, 6), (3, 8)]
        elif angle < 120:
            # Right to slightly obtuse - use medium cassettes
            sizes_to_try = [(4, 6), (5, 6), (4, 8)]
        else:
            # Obtuse or reflex - can use larger cassettes
            sizes_to_try = [(5, 6), (5, 8), (4, 8)]

        # Try each size in both orientations
        for width, height in sizes_to_try:
            for rotated in [False, True]:
                w, h = (height, width) if rotated else (width, height)

                # Try different positions relative to the corner
                positions = self._generate_corner_positions(vertex, w, h, corner)

                for x, y in positions:
                    cassette = Cassette(x, y, w, h)

                    # Check if cassette is valid
                    if self._is_valid_placement(cassette, polygon, existing_cassettes):
                        return cassette

        logger.debug(f"Could not place cassette at corner {corner['index']}")
        return None

    def _generate_corner_positions(self, vertex: Tuple[float, float],
                                  width: float, height: float,
                                  corner: dict) -> List[Tuple[float, float]]:
        """
        Generate potential positions for a cassette at a corner.

        Args:
            vertex: Corner vertex
            width: Cassette width
            height: Cassette height
            corner: Corner information

        Returns:
            List of (x, y) positions to try
        """
        positions = []

        # Position 1: Cassette corner at vertex
        positions.append((vertex[0], vertex[1]))

        # Position 2: Cassette centered on vertex
        positions.append((vertex[0] - width/2, vertex[1] - height/2))

        # Position 3: Cassette edge at vertex
        positions.append((vertex[0] - width, vertex[1]))
        positions.append((vertex[0], vertex[1] - height))

        # Position 4: Cassette slightly inside from vertex
        inset = 0.5  # 0.5 ft inset
        positions.append((vertex[0] + inset, vertex[1] + inset))
        positions.append((vertex[0] - width - inset, vertex[1] + inset))
        positions.append((vertex[0] + inset, vertex[1] - height - inset))
        positions.append((vertex[0] - width - inset, vertex[1] - height - inset))

        # For acute angles, try positions along the angle bisector
        if corner['angle'] < 90:
            # Calculate angle bisector direction
            prev = corner['prev_vertex']
            next = corner['next_vertex']

            # Vectors from vertex
            v1 = (prev[0] - vertex[0], prev[1] - vertex[1])
            v2 = (next[0] - vertex[0], next[1] - vertex[1])

            # Normalize
            mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
            mag2 = math.sqrt(v2[0]**2 + v2[1]**2)

            if mag1 > 0 and mag2 > 0:
                v1_norm = (v1[0]/mag1, v1[1]/mag1)
                v2_norm = (v2[0]/mag2, v2[1]/mag2)

                # Bisector direction (average of normalized vectors)
                bisector = ((v1_norm[0] + v2_norm[0])/2, (v1_norm[1] + v2_norm[1])/2)

                # Try positions along bisector
                for dist in [1.0, 2.0]:
                    x = vertex[0] + bisector[0] * dist - width/2
                    y = vertex[1] + bisector[1] * dist - height/2
                    positions.append((x, y))

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
        # Check all four corners
        corners = cassette.get_corners()
        for corner in corners:
            if not self._point_in_polygon(corner, polygon):
                return False

        # Check 3: Cassette center is within polygon (additional safety)
        center = (cassette.x + cassette.width/2, cassette.y + cassette.height/2)
        if not self._point_in_polygon(center, polygon):
            return False

        return True

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


def test_corner_placer():
    """Test the CornerPlacer stage."""
    from optimization_pipeline import PipelineContext

    # Test polygon: L-shaped building
    test_polygon = [
        (0, 0),
        (30, 0),
        (30, 20),
        (15, 20),
        (15, 35),
        (0, 35)
    ]

    # Create context
    context = PipelineContext(test_polygon)

    # Create and run CornerPlacer
    placer = CornerPlacer()
    result_context = placer.process(context)

    print("\n" + "="*60)
    print("CORNER PLACER TEST RESULTS")
    print("="*60)
    print(f"Polygon vertices: {len(test_polygon)}")
    print(f"Cassettes placed: {len(result_context.cassettes)}")
    print(f"Coverage: {result_context.get_coverage():.1f}%")

    print("\nCorner Analysis:")
    corners = placer._analyze_corners(test_polygon)
    for i, corner in enumerate(corners):
        print(f"  Corner {i}: ({corner['vertex'][0]:.1f}, {corner['vertex'][1]:.1f})")
        print(f"    Angle: {corner['angle']:.1f}° ({corner['type']})")
        print(f"    Edge before: {corner['edge_before']:.1f} ft")
        print(f"    Edge after: {corner['edge_after']:.1f} ft")

    print("\nPlaced Cassettes:")
    for i, cassette in enumerate(result_context.cassettes):
        print(f"  {i}: {cassette.size} at ({cassette.x:.1f}, {cassette.y:.1f})")

    return result_context


if __name__ == "__main__":
    test_corner_placer()
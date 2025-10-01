#!/usr/bin/env python3
"""
Cardinal Polygon Reconstructor
===============================
Reconstructs polygon using exact cardinal directions (N/S/E/W).
Guarantees perfect polygon closure for properly measured floor plans.
"""

import math
from typing import List, Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CardinalPolygonReconstructor:
    """Reconstructs polygon using cardinal directions for perfect closure."""

    def __init__(self):
        """Initialize reconstructor."""
        self.vertices = []
        self.area = 0.0
        self.perimeter = 0.0
        self.is_closed = False
        self.closure_error = 0.0

    def build_from_cardinal_measurements(self,
                                        edges: List,
                                        measurements: Dict[int, float]) -> List[Tuple[float, float]]:
        """
        Build polygon using cardinal directions and measurements.

        Args:
            edges: List of CardinalEdge objects
            measurements: Dictionary mapping edge index to measurement in feet

        Returns:
            List of (x, y) vertices in feet
        """
        self.vertices = []
        x, y = 0.0, 0.0
        self.vertices.append((x, y))
        self.perimeter = 0.0

        # Cardinal direction vectors (exact movements)
        direction_vectors = {
            'E': (1.0, 0.0),   # East: move right
            'W': (-1.0, 0.0),  # West: move left
            'S': (0.0, -1.0),  # South: move down
            'N': (0.0, 1.0)    # North: move up
        }

        for i, edge in enumerate(edges):
            if i not in measurements:
                logger.error(f"Missing measurement for edge {i} ({edge.cardinal_direction})")
                continue

            # Get measurement
            length = measurements[i]
            self.perimeter += length

            # Get exact cardinal direction vector
            direction = edge.cardinal_direction
            dx_unit, dy_unit = direction_vectors[direction]

            # Move exactly in cardinal direction
            dx = dx_unit * length
            dy = dy_unit * length

            # Add vertex
            x += dx
            y += dy
            self.vertices.append((x, y))

            logger.debug(f"Edge {i}: {direction} {length:.1f}ft -> ({x:.1f}, {y:.1f})")

        # Validate closure BEFORE removing duplicate
        self._validate_closure()

        # Remove duplicate last vertex if polygon closed perfectly
        # This is for downstream processing - we keep N vertices for N edges
        if len(self.vertices) > 1 and self.is_closed:
            first = self.vertices[0]
            last = self.vertices[-1]
            if abs(last[0] - first[0]) < 0.01 and abs(last[1] - first[1]) < 0.01:
                self.vertices = self.vertices[:-1]
                logger.debug("Removed duplicate closing vertex for downstream processing")

        # Calculate area
        self.area = self._calculate_area()

        return self.vertices

    def _validate_closure(self):
        """Validate that polygon closes perfectly."""
        if len(self.vertices) < 3:
            self.is_closed = False
            self.closure_error = float('inf')
            return

        # Check distance between first and last vertex
        first = self.vertices[0]
        last = self.vertices[-1]
        self.closure_error = math.sqrt((last[0] - first[0])**2 + (last[1] - first[1])**2)

        # With cardinal directions, closure should be nearly perfect
        self.is_closed = self.closure_error < 0.1  # Very tight tolerance

        if self.is_closed:
            logger.info(f"✓ Polygon closes PERFECTLY (error: {self.closure_error:.4f} feet)")
        else:
            logger.warning(f"✗ Polygon does not close (error: {self.closure_error:.2f} feet)")
            logger.warning(f"  Start: ({first[0]:.2f}, {first[1]:.2f})")
            logger.warning(f"  End: ({last[0]:.2f}, {last[1]:.2f})")

    def _calculate_area(self) -> float:
        """
        Calculate polygon area using shoelace formula.

        Returns:
            Area in square feet
        """
        if len(self.vertices) < 3:
            return 0.0

        # Shoelace formula
        area = 0.0
        n = len(self.vertices)

        for i in range(n):
            j = (i + 1) % n
            area += self.vertices[i][0] * self.vertices[j][1]
            area -= self.vertices[j][0] * self.vertices[i][1]

        area = abs(area) / 2.0
        logger.info(f"Calculated area: {area:.1f} square feet")

        return area

    def get_bounds(self) -> Tuple[float, float, float, float]:
        """
        Get bounding box of polygon.

        Returns:
            Tuple of (min_x, min_y, max_x, max_y)
        """
        if not self.vertices:
            return (0, 0, 0, 0)

        xs = [v[0] for v in self.vertices]
        ys = [v[1] for v in self.vertices]

        return (min(xs), min(ys), max(xs), max(ys))

    def normalize_to_positive(self) -> List[Tuple[float, float]]:
        """
        Normalize polygon so all vertices are positive.

        Returns:
            Normalized vertices
        """
        if not self.vertices:
            return []

        min_x, min_y, _, _ = self.get_bounds()

        # Translate to positive coordinates
        normalized = [(x - min_x, y - min_y) for x, y in self.vertices]
        self.vertices = normalized

        logger.info(f"Normalized polygon to positive coordinates (offset: {-min_x:.1f}, {-min_y:.1f})")

        return normalized

    def point_in_polygon(self, x: float, y: float) -> bool:
        """
        Check if point is inside polygon using ray casting.

        Args:
            x: X coordinate in feet
            y: Y coordinate in feet

        Returns:
            True if point is inside polygon
        """
        if len(self.vertices) < 3:
            return False

        inside = False
        n = len(self.vertices)

        p1x, p1y = self.vertices[0]
        for i in range(1, n + 1):
            p2x, p2y = self.vertices[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def get_statistics(self) -> Dict:
        """
        Get polygon statistics.

        Returns:
            Dictionary with polygon metrics
        """
        min_x, min_y, max_x, max_y = self.get_bounds()

        return {
            'area': self.area,
            'perimeter': self.perimeter,
            'num_vertices': len(self.vertices),
            'is_closed': self.is_closed,
            'closure_error': self.closure_error,
            'width': max_x - min_x,
            'height': max_y - min_y,
            'bounds': (min_x, min_y, max_x, max_y)
        }

    def debug_trace(self) -> str:
        """
        Create debug trace of polygon construction.

        Returns:
            String with step-by-step trace
        """
        trace = "Polygon Construction Trace:\n"
        trace += "Starting at (0.0, 0.0)\n"

        for i, vertex in enumerate(self.vertices[1:], 1):
            prev = self.vertices[i-1]
            dx = vertex[0] - prev[0]
            dy = vertex[1] - prev[1]

            direction = "?"
            if abs(dx) > abs(dy):
                direction = "E" if dx > 0 else "W"
                distance = abs(dx)
            else:
                direction = "N" if dy > 0 else "S"
                distance = abs(dy)

            trace += f"Step {i}: {direction} {distance:.1f}ft -> ({vertex[0]:.1f}, {vertex[1]:.1f})\n"

        trace += f"Final position: ({self.vertices[-1][0]:.1f}, {self.vertices[-1][1]:.1f})\n"
        trace += f"Closure error: {self.closure_error:.4f} feet\n"

        return trace
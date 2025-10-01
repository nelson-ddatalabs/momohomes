#!/usr/bin/env python3
"""
Polygon Reconstructor
=====================
Builds real-world polygon from edge measurements and validates closure.
Calculates area using shoelace formula.
"""

import math
from typing import List, Tuple, Dict
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PolygonReconstructor:
    """Reconstructs floor plan polygon from measurements."""

    def __init__(self):
        """Initialize reconstructor."""
        self.vertices = []
        self.area = 0.0
        self.perimeter = 0.0
        self.is_closed = False
        self.closure_error = 0.0

    def build_from_measurements(self, edges: List, measurements: Dict[int, float]) -> List[Tuple[float, float]]:
        """
        Build polygon from edges and measurements.

        Args:
            edges: List of Edge objects from detection
            measurements: Dictionary mapping edge index to measurement in feet

        Returns:
            List of (x, y) vertices in feet
        """
        self.vertices = []
        x, y = 0.0, 0.0
        self.vertices.append((x, y))

        for i, edge in enumerate(edges):
            if i not in measurements:
                logger.error(f"Missing measurement for edge {i}")
                continue

            # Get measurement
            length = measurements[i]
            self.perimeter += length

            # Calculate direction from edge pixels
            dx = edge.end[0] - edge.start[0]
            dy = edge.end[1] - edge.start[1]
            pixel_length = math.sqrt(dx * dx + dy * dy)

            # Normalize and scale
            if pixel_length > 0:
                dx = (dx / pixel_length) * length
                dy = (dy / pixel_length) * length
            else:
                logger.warning(f"Edge {i} has zero length")
                continue

            # Add vertex
            x += dx
            y += dy
            self.vertices.append((x, y))

        # Remove duplicate last vertex if it exists
        if len(self.vertices) > 1 and self._distance(self.vertices[-1], self.vertices[0]) < 0.01:
            self.vertices = self.vertices[:-1]

        # Validate closure
        self._validate_closure()

        # Calculate area
        self.area = self._calculate_area()

        return self.vertices

    def _validate_closure(self):
        """Validate that polygon closes properly."""
        if len(self.vertices) < 3:
            self.is_closed = False
            self.closure_error = float('inf')
            return

        # Check distance between first and last vertex
        first = self.vertices[0]
        last = self.vertices[-1]
        self.closure_error = self._distance(first, last)

        # Consider closed if error is less than 1 foot
        self.is_closed = self.closure_error < 1.0

        if self.is_closed:
            logger.info(f"Polygon closes successfully (error: {self.closure_error:.3f} feet)")
        else:
            logger.warning(f"Polygon does not close (error: {self.closure_error:.1f} feet)")

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

    def _distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Calculate distance between two points."""
        return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

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

    def to_dict(self) -> Dict:
        """Convert to dictionary for saving."""
        return {
            'vertices': self.vertices,
            'area': self.area,
            'perimeter': self.perimeter,
            'is_closed': self.is_closed,
            'closure_error': self.closure_error,
            'statistics': self.get_statistics()
        }
#!/usr/bin/env python3
"""
Edge Detection System
=====================
Finds and simplifies building contours from binary images.
Implements Douglas-Peucker algorithm for edge simplification.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict
import logging
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Edge:
    """Represents a single edge in the building perimeter."""

    def __init__(self, start: Tuple[int, int], end: Tuple[int, int]):
        """
        Initialize edge with start and end points.

        Args:
            start: (x, y) starting point
            end: (x, y) ending point
        """
        self.start = start
        self.end = end
        self.length = self._calculate_length()
        self.orientation = self._determine_orientation()

    def _calculate_length(self) -> float:
        """Calculate edge length in pixels."""
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        return math.sqrt(dx * dx + dy * dy)

    def _determine_orientation(self) -> str:
        """Determine if edge is horizontal, vertical, or diagonal."""
        dx = abs(self.end[0] - self.start[0])
        dy = abs(self.end[1] - self.start[1])

        angle_threshold = 10  # degrees

        if dy < dx * math.tan(math.radians(angle_threshold)):
            return "horizontal"
        elif dx < dy * math.tan(math.radians(angle_threshold)):
            return "vertical"
        else:
            return "diagonal"

    def to_dict(self) -> Dict:
        """Convert edge to dictionary."""
        return {
            'start': self.start,
            'end': self.end,
            'length': self.length,
            'orientation': self.orientation
        }


class EdgeDetectionSystem:
    """Detects and simplifies edges from binary floor plan images."""

    def __init__(self, simplification_epsilon: float = 5.0):
        """
        Initialize edge detection system.

        Args:
            simplification_epsilon: Douglas-Peucker epsilon for simplification
        """
        self.simplification_epsilon = simplification_epsilon
        self.edges = []
        self.simplified_contour = None

    def detect_edges(self, binary_image: np.ndarray) -> List[Edge]:
        """
        Detect edges from binary image.

        Args:
            binary_image: Binary image (white=indoor, black=outdoor)

        Returns:
            List of Edge objects ordered clockwise
        """
        # Find contours
        contours, _ = cv2.findContours(
            binary_image,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            logger.warning("No contours found in binary image")
            return []

        # Get largest contour (main building)
        main_contour = max(contours, key=cv2.contourArea)
        logger.info(f"Main contour has {len(main_contour)} points")

        # Simplify contour using Douglas-Peucker
        epsilon = self.simplification_epsilon
        self.simplified_contour = cv2.approxPolyDP(main_contour, epsilon, True)
        logger.info(f"Simplified to {len(self.simplified_contour)} points")

        # Extract edges from simplified contour
        self.edges = self._extract_edges(self.simplified_contour)

        # Ensure clockwise ordering
        self.edges = self._ensure_clockwise(self.edges)

        return self.edges

    def _extract_edges(self, contour: np.ndarray) -> List[Edge]:
        """
        Extract edges from contour points.

        Args:
            contour: OpenCV contour array

        Returns:
            List of Edge objects
        """
        edges = []
        num_points = len(contour)

        for i in range(num_points):
            start = tuple(contour[i][0])
            end = tuple(contour[(i + 1) % num_points][0])
            edge = Edge(start, end)
            edges.append(edge)

        return edges

    def _ensure_clockwise(self, edges: List[Edge]) -> List[Edge]:
        """
        Ensure edges are ordered clockwise.

        Args:
            edges: List of edges

        Returns:
            Edges in clockwise order
        """
        # Calculate signed area to determine orientation
        signed_area = 0
        for edge in edges:
            signed_area += (edge.end[0] - edge.start[0]) * (edge.end[1] + edge.start[1])

        # If counter-clockwise (positive area), reverse
        if signed_area > 0:
            edges = edges[::-1]
            # Also reverse each edge's start and end
            for edge in edges:
                edge.start, edge.end = edge.end, edge.start
            logger.info("Reversed edges to ensure clockwise order")

        return edges

    def get_edge_list(self) -> List[Dict]:
        """
        Get list of edges with details.

        Returns:
            List of edge dictionaries
        """
        edge_list = []
        for i, edge in enumerate(self.edges):
            edge_dict = edge.to_dict()
            edge_dict['index'] = i
            edge_list.append(edge_dict)
        return edge_list

    def simplify_further(self, target_edges: int = 12):
        """
        Further simplify contour to target number of edges.

        Args:
            target_edges: Target number of edges

        Returns:
            Simplified edges
        """
        if not self.simplified_contour:
            logger.warning("No contour to simplify")
            return self.edges

        # Binary search for optimal epsilon
        min_epsilon = 1.0
        max_epsilon = 100.0
        best_contour = self.simplified_contour

        for _ in range(10):  # Max iterations
            epsilon = (min_epsilon + max_epsilon) / 2
            simplified = cv2.approxPolyDP(self.simplified_contour, epsilon, True)

            if len(simplified) == target_edges:
                best_contour = simplified
                break
            elif len(simplified) > target_edges:
                min_epsilon = epsilon
            else:
                max_epsilon = epsilon
                best_contour = simplified

        self.simplified_contour = best_contour
        self.edges = self._extract_edges(best_contour)
        self.edges = self._ensure_clockwise(self.edges)

        logger.info(f"Further simplified to {len(self.edges)} edges")
        return self.edges

    def visualize_edges(self, image: np.ndarray, output_path: str = None) -> np.ndarray:
        """
        Visualize detected edges on image.

        Args:
            image: Original image or binary image
            output_path: Optional path to save visualization

        Returns:
            Image with edges drawn
        """
        # Convert grayscale to BGR if needed
        if len(image.shape) == 2:
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis = image.copy()

        # Draw edges with different colors by orientation
        colors = {
            'horizontal': (255, 0, 0),  # Blue
            'vertical': (0, 255, 0),    # Green
            'diagonal': (0, 0, 255)      # Red
        }

        for i, edge in enumerate(self.edges):
            color = colors[edge.orientation]
            cv2.line(vis, edge.start, edge.end, color, 2)

            # Add edge number at midpoint
            mid_x = (edge.start[0] + edge.end[0]) // 2
            mid_y = (edge.start[1] + edge.end[1]) // 2
            cv2.putText(vis, str(i), (mid_x, mid_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if output_path:
            cv2.imwrite(output_path, vis)
            logger.info(f"Edge visualization saved to: {output_path}")

        return vis
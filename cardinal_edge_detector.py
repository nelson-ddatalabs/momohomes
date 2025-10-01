#!/usr/bin/env python3
"""
Cardinal Edge Detector
======================
Classifies edges as cardinal directions (N/S/E/W) for perfect polygon closure.
Buildings have right angles - we should respect that.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict
import logging
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CardinalEdge:
    """Edge with cardinal direction classification."""

    def __init__(self, start: Tuple[int, int], end: Tuple[int, int]):
        """
        Initialize edge and determine cardinal direction.

        Args:
            start: (x, y) starting point in pixels
            end: (x, y) ending point in pixels
        """
        self.start = start
        self.end = end
        self.pixel_length = self._calculate_pixel_length()
        self.cardinal_direction = self._determine_cardinal_direction()
        self.measurement = None  # To be filled by user

    def _calculate_pixel_length(self) -> float:
        """Calculate edge length in pixels."""
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        return math.sqrt(dx * dx + dy * dy)

    def _determine_cardinal_direction(self) -> str:
        """
        Determine cardinal direction (N/S/E/W) of edge.

        Returns:
            'E' for East (right), 'W' for West (left),
            'S' for South (down), 'N' for North (up)
        """
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]

        # Calculate angle from horizontal
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)

        # Normalize to 0-360
        if angle_deg < 0:
            angle_deg += 360

        # Classify into cardinal directions with 45-degree tolerance
        # East: -45 to 45 degrees
        if -45 <= angle_deg <= 45 or angle_deg >= 315:
            return 'E'
        # South: 45 to 135 degrees (down in image coordinates)
        elif 45 < angle_deg <= 135:
            return 'S'
        # West: 135 to 225 degrees
        elif 135 < angle_deg <= 225:
            return 'W'
        # North: 225 to 315 degrees (up in image coordinates)
        else:
            return 'N'

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'start': self.start,
            'end': self.end,
            'pixel_length': self.pixel_length,
            'cardinal_direction': self.cardinal_direction,
            'measurement': self.measurement
        }


class CardinalEdgeDetector:
    """Detects edges and classifies them as cardinal directions."""

    def __init__(self, simplification_epsilon: float = 5.0):
        """
        Initialize detector.

        Args:
            simplification_epsilon: Douglas-Peucker epsilon for simplification
        """
        self.simplification_epsilon = simplification_epsilon
        self.edges = []
        self.simplified_contour = None

    def detect_cardinal_edges(self, binary_image: np.ndarray) -> List[CardinalEdge]:
        """
        Detect edges and classify as cardinal directions.

        Args:
            binary_image: Binary image (white=indoor, black=outdoor)

        Returns:
            List of CardinalEdge objects ordered clockwise
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

        # Simplify contour
        epsilon = self.simplification_epsilon
        self.simplified_contour = cv2.approxPolyDP(main_contour, epsilon, True)
        logger.info(f"Simplified to {len(self.simplified_contour)} points")

        # Extract cardinal edges
        self.edges = self._extract_cardinal_edges(self.simplified_contour)

        # Ensure clockwise ordering
        self.edges = self._ensure_clockwise(self.edges)

        # Merge consecutive edges with same direction
        self.edges = self._merge_same_direction_edges(self.edges)

        logger.info(f"Final edge count: {len(self.edges)} cardinal edges")

        return self.edges

    def _extract_cardinal_edges(self, contour: np.ndarray) -> List[CardinalEdge]:
        """
        Extract edges from contour as cardinal edges.

        Args:
            contour: OpenCV contour array

        Returns:
            List of CardinalEdge objects
        """
        edges = []
        num_points = len(contour)

        for i in range(num_points):
            start = tuple(contour[i][0])
            end = tuple(contour[(i + 1) % num_points][0])
            edge = CardinalEdge(start, end)
            edges.append(edge)

        return edges

    def _ensure_clockwise(self, edges: List[CardinalEdge]) -> List[CardinalEdge]:
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
                # Recalculate cardinal direction after reversal
                edge.cardinal_direction = edge._determine_cardinal_direction()
            logger.info("Reversed edges to ensure clockwise order")

        return edges

    def _merge_same_direction_edges(self, edges: List[CardinalEdge]) -> List[CardinalEdge]:
        """
        Merge consecutive edges that have the same cardinal direction.

        Args:
            edges: List of cardinal edges

        Returns:
            Merged edge list
        """
        if len(edges) <= 1:
            return edges

        merged = []
        current_edge = edges[0]

        for i in range(1, len(edges)):
            next_edge = edges[i]

            # Check if same direction and connected
            if (current_edge.cardinal_direction == next_edge.cardinal_direction and
                current_edge.end == next_edge.start):
                # Merge edges
                current_edge = CardinalEdge(current_edge.start, next_edge.end)
                logger.debug(f"Merged two {current_edge.cardinal_direction} edges")
            else:
                # Different direction, save current and move to next
                merged.append(current_edge)
                current_edge = next_edge

        # Add last edge
        merged.append(current_edge)

        # Check if first and last can be merged
        if (len(merged) > 1 and
            merged[0].cardinal_direction == merged[-1].cardinal_direction and
            merged[-1].end == merged[0].start):
            # Merge first and last
            new_last = CardinalEdge(merged[-1].start, merged[0].end)
            merged = [new_last] + merged[1:-1]
            logger.debug(f"Merged first and last {new_last.cardinal_direction} edges")

        return merged

    def visualize_cardinal_edges(self, image: np.ndarray,
                                output_path: str = None,
                                measurements: Dict[int, float] = None) -> np.ndarray:
        """
        Visualize edges with cardinal directions and measurements.

        Args:
            image: Original or binary image
            output_path: Optional path to save visualization
            measurements: Optional dictionary of edge measurements in feet

        Returns:
            Image with cardinal edges and measurements drawn
        """
        # Convert grayscale to BGR if needed
        if len(image.shape) == 2:
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis = image.copy()

        # Colors for cardinal directions
        colors = {
            'E': (255, 0, 0),    # Blue for East
            'W': (255, 0, 255),  # Magenta for West
            'N': (0, 255, 0),    # Green for North
            'S': (0, 0, 255)     # Red for South
        }

        for i, edge in enumerate(self.edges):
            color = colors[edge.cardinal_direction]
            cv2.line(vis, edge.start, edge.end, color, 3)

            # Add edge info at midpoint
            mid_x = (edge.start[0] + edge.end[0]) // 2
            mid_y = (edge.start[1] + edge.end[1]) // 2

            # Create label with measurement if available
            if measurements and i in measurements:
                label = f"{i}:{edge.cardinal_direction} {measurements[i]:.1f}ft"
            else:
                label = f"{i}:{edge.cardinal_direction}"

            # Draw background for visibility
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            (text_width, text_height), _ = cv2.getTextSize(
                label, font, font_scale, thickness
            )
            cv2.rectangle(vis,
                         (mid_x - text_width // 2 - 5, mid_y - text_height // 2 - 5),
                         (mid_x + text_width // 2 + 5, mid_y + text_height // 2 + 5),
                         (255, 255, 255), -1)

            cv2.putText(vis, label, (mid_x - text_width // 2, mid_y + text_height // 2),
                       font, font_scale, (0, 0, 0), thickness)

        # Add legend
        legend_y = 30
        cv2.putText(vis, "Cardinal Directions: E=Blue, W=Magenta, N=Green, S=Red",
                   (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Add measurement totals if provided
        if measurements:
            # Calculate direction sums
            sums = {'E': 0, 'W': 0, 'N': 0, 'S': 0}
            for i, edge in enumerate(self.edges):
                if i in measurements:
                    sums[edge.cardinal_direction] += measurements[i]

            # Display sums
            info_y = legend_y + 25
            balance_text = f"E:{sums['E']:.1f}ft W:{sums['W']:.1f}ft N:{sums['N']:.1f}ft S:{sums['S']:.1f}ft"
            cv2.putText(vis, balance_text,
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # Check balance
            e_w_balanced = abs(sums['E'] - sums['W']) < 0.1
            n_s_balanced = abs(sums['N'] - sums['S']) < 0.1
            if e_w_balanced and n_s_balanced:
                cv2.putText(vis, "CLOSED", (10, info_y + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(vis, f"ERROR: E-W:{sums['E']-sums['W']:.1f}ft N-S:{sums['N']-sums['S']:.1f}ft",
                           (10, info_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if output_path:
            cv2.imwrite(output_path, vis)
            logger.info(f"Cardinal edge visualization saved to: {output_path}")

        return vis
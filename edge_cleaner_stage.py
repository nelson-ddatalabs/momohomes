#!/usr/bin/env python3
"""
Edge Cleaner Stage
==================
Cleans polygon edges by removing protrusions, merging collinear edges,
and validating minimum edge lengths.
"""

import logging
import math
from typing import List, Tuple

from optimization_pipeline import PipelineStage, PipelineContext

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EdgeCleaner(PipelineStage):
    """
    Cleans polygon edges to prepare for cassette placement.

    Operations:
    1. Remove zero-area protrusions
    2. Merge collinear edges
    3. Validate minimum edge lengths (2 ft for smallest cassette)
    """

    def __init__(self, min_edge_length: float = 2.0, collinear_tolerance: float = 0.01):
        """
        Initialize EdgeCleaner.

        Args:
            min_edge_length: Minimum edge length in feet (default 2.0 for smallest cassette)
            collinear_tolerance: Tolerance for collinearity check in radians
        """
        self.min_edge_length = min_edge_length
        self.collinear_tolerance = collinear_tolerance

    @property
    def name(self) -> str:
        return "EdgeCleaner"

    def process(self, context: PipelineContext) -> PipelineContext:
        """Clean polygon edges."""
        original_vertices = len(context.polygon)

        # Step 1: Remove narrow protrusions (< 2 ft wide)
        context.polygon = self._remove_narrow_protrusions(context.polygon)

        # Step 2: Remove zero-area protrusions
        context.polygon = self._remove_zero_area_protrusions(context.polygon)

        # Step 3: Merge collinear edges
        context.polygon = self._merge_collinear_edges(context.polygon)

        # Step 4: Validate and fix minimum edge lengths
        context.polygon = self._validate_minimum_edges(context.polygon)

        # Update metadata
        final_vertices = len(context.polygon)
        context.metadata['edge_cleaner'] = {
            'original_vertices': original_vertices,
            'final_vertices': final_vertices,
            'vertices_removed': original_vertices - final_vertices
        }

        logger.info(f"  Edge cleaning: {original_vertices} -> {final_vertices} vertices")

        return context

    def _remove_zero_area_protrusions(self, polygon: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Remove protrusions that have zero or near-zero area.

        A protrusion is identified when:
        1. Three consecutive vertices form a spike
        2. The spike goes out and returns on the same line
        """
        if len(polygon) < 4:
            return polygon

        cleaned = []
        i = 0
        removed_count = 0

        while i < len(polygon):
            if i < len(polygon) - 2:
                p1 = polygon[i]
                p2 = polygon[i + 1]
                p3 = polygon[i + 2]

                # Check if p1, p2, p3 form a zero-area protrusion
                if self._is_zero_area_protrusion(p1, p2, p3):
                    # Skip p2 (the protrusion point)
                    cleaned.append(p1)
                    i += 2  # Skip p2, continue from p3
                    removed_count += 1
                    logger.debug(f"Removed zero-area protrusion at vertex {i+1}")
                else:
                    cleaned.append(p1)
                    i += 1
            else:
                cleaned.append(polygon[i])
                i += 1

        if removed_count > 0:
            logger.info(f"  Removed {removed_count} zero-area protrusions")

        return cleaned

    def _is_zero_area_protrusion(self, p1: Tuple[float, float],
                                 p2: Tuple[float, float],
                                 p3: Tuple[float, float]) -> bool:
        """
        Check if three points form a zero-area protrusion.

        This happens when:
        1. The distance from p2 perpendicular to line p1-p3 is very small
        2. p2 is between p1 and p3 along the line
        """
        # Calculate area of triangle using cross product
        area = abs((p2[0] - p1[0]) * (p3[1] - p1[1]) -
                  (p3[0] - p1[0]) * (p2[1] - p1[1])) / 2.0

        # Check if area is essentially zero
        if area < 0.01:  # Less than 0.01 sq ft
            # Check if p2 is between p1 and p3
            dist_13 = math.sqrt((p3[0] - p1[0])**2 + (p3[1] - p1[1])**2)
            dist_12 = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            dist_23 = math.sqrt((p3[0] - p2[0])**2 + (p3[1] - p2[1])**2)

            # If p2 is between p1 and p3, then dist_12 + dist_23 ≈ dist_13
            if abs((dist_12 + dist_23) - dist_13) < 0.1:
                return True

        return False

    def _merge_collinear_edges(self, polygon: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Merge consecutive edges that are collinear.

        If three consecutive vertices are collinear, remove the middle one.
        """
        if len(polygon) < 3:
            return polygon

        merged = []
        merged_count = 0

        for i in range(len(polygon)):
            prev_idx = (i - 1) % len(polygon)
            next_idx = (i + 1) % len(polygon)

            if self._are_collinear(polygon[prev_idx], polygon[i], polygon[next_idx]):
                # Skip this vertex as it's collinear with its neighbors
                merged_count += 1
                logger.debug(f"Merged collinear vertex at index {i}")
            else:
                merged.append(polygon[i])

        if merged_count > 0:
            logger.info(f"  Merged {merged_count} collinear vertices")

        return merged if len(merged) >= 3 else polygon

    def _are_collinear(self, p1: Tuple[float, float],
                       p2: Tuple[float, float],
                       p3: Tuple[float, float]) -> bool:
        """
        Check if three points are collinear using cross product.

        Points are collinear if the cross product is near zero.
        """
        # Vector from p1 to p2
        v1 = (p2[0] - p1[0], p2[1] - p1[1])
        # Vector from p2 to p3
        v2 = (p3[0] - p2[0], p3[1] - p2[1])

        # Cross product (in 2D, this gives the z-component)
        cross = v1[0] * v2[1] - v1[1] * v2[0]

        # Check if cross product is near zero
        return abs(cross) < self.collinear_tolerance

    def _validate_minimum_edges(self, polygon: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Validate that all edges meet minimum length requirement.

        If an edge is too short (< 2 ft), either:
        1. Merge it with adjacent edge if possible
        2. Extend it to minimum length
        3. Remove the vertex if it creates a negligible change
        """
        if len(polygon) < 3:
            return polygon

        validated = []
        modified_count = 0

        for i in range(len(polygon)):
            next_idx = (i + 1) % len(polygon)
            edge_length = self._edge_length(polygon[i], polygon[next_idx])

            if edge_length < self.min_edge_length:
                # Edge is too short
                if edge_length < 0.5:  # Very short, just remove
                    modified_count += 1
                    logger.debug(f"Removed very short edge ({edge_length:.2f} ft) at index {i}")
                    # Skip this vertex
                    continue
                else:
                    # Keep but warn
                    logger.warning(f"Edge {i} is {edge_length:.2f} ft (< {self.min_edge_length} ft minimum)")

            validated.append(polygon[i])

        if modified_count > 0:
            logger.info(f"  Removed {modified_count} edges shorter than 0.5 ft")

        return validated if len(validated) >= 3 else polygon

    def _edge_length(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Calculate edge length between two points."""
        return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

    def _remove_narrow_protrusions(self, polygon: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Remove protrusions narrower than minimum cassette width (2 ft).

        A narrow protrusion is identified as a sequence of edges that:
        1. Goes out perpendicular to the main edge
        2. Has width < 2 ft
        3. Returns back parallel to the original direction
        """
        if len(polygon) < 6:
            return polygon

        cleaned = []
        i = 0
        removed_count = 0

        while i < len(polygon):
            if i < len(polygon) - 3:
                # Check for rectangular protrusion pattern
                # Looking for pattern: straight -> perpendicular out -> parallel -> perpendicular back
                p1 = polygon[i]
                p2 = polygon[(i + 1) % len(polygon)]
                p3 = polygon[(i + 2) % len(polygon)]
                p4 = polygon[(i + 3) % len(polygon)]

                # Check if this forms a narrow protrusion
                edge1_length = self._edge_length(p1, p2)
                edge2_length = self._edge_length(p2, p3)
                edge3_length = self._edge_length(p3, p4)

                # Check if edge1 is perpendicular to edge2
                v1 = (p2[0] - p1[0], p2[1] - p1[1])
                v2 = (p3[0] - p2[0], p3[1] - p2[1])
                dot_product = v1[0] * v2[0] + v1[1] * v2[1]

                # If perpendicular (dot product ≈ 0) and edge1 is narrow
                if abs(dot_product) < 0.1 and edge1_length < self.min_edge_length:
                    # This is a narrow protrusion, skip it
                    cleaned.append(p1)
                    i += 3  # Skip the protrusion vertices
                    removed_count += 2
                    logger.info(f"  Removed narrow protrusion ({edge1_length:.1f} ft wide)")
                else:
                    cleaned.append(p1)
                    i += 1
            else:
                cleaned.append(polygon[i])
                i += 1

        if removed_count > 0:
            logger.info(f"  Total removed from narrow protrusions: {removed_count} vertices")

        return cleaned if len(cleaned) >= 3 else polygon


def test_edge_cleaner():
    """Test the EdgeCleaner stage."""
    from optimization_pipeline import PipelineContext

    # Test polygon with issues:
    # - Zero-area protrusion at (42, 22.5) -> (42.5, 22.5) -> (42.5, 14.5) -> (42, 14.5)
    # - Collinear points
    # - Short edges
    test_polygon = [
        (0, 0),
        (23.5, 0),
        (23.5, 0.1),  # Almost collinear
        (23.5, 6.5),
        (42, 6.5),
        (42, 14.5),
        (42.5, 14.5),  # Start of protrusion
        (42.5, 22.5),  # Protrusion point
        (42, 22.5),    # End of protrusion
        (42, 30.5),
        (23.5, 30.5),
        (23.5, 37),
        (0, 37),
        (0, 15.5),
        (0.1, 15.5),   # Very short edge
        (0, 15.5)      # Duplicate
    ]

    # Create context
    context = PipelineContext(test_polygon)

    # Create and run EdgeCleaner
    cleaner = EdgeCleaner()
    cleaned_context = cleaner.process(context)

    print("\n" + "="*60)
    print("EDGE CLEANER TEST RESULTS")
    print("="*60)
    print(f"Original vertices: {len(test_polygon)}")
    print(f"Cleaned vertices: {len(cleaned_context.polygon)}")
    print(f"Vertices removed: {len(test_polygon) - len(cleaned_context.polygon)}")

    print("\nCleaned polygon:")
    for i, vertex in enumerate(cleaned_context.polygon):
        print(f"  {i}: ({vertex[0]:.1f}, {vertex[1]:.1f})")

    # Check for short edges
    print("\nEdge lengths:")
    for i in range(len(cleaned_context.polygon)):
        next_idx = (i + 1) % len(cleaned_context.polygon)
        p1 = cleaned_context.polygon[i]
        p2 = cleaned_context.polygon[next_idx]
        length = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        status = "OK" if length >= 2.0 else "SHORT"
        print(f"  Edge {i}->{next_idx}: {length:.2f} ft [{status}]")

    return cleaned_context


if __name__ == "__main__":
    test_edge_cleaner()
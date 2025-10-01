#!/usr/bin/env python3
"""
Concentric Filler Stage
=======================
Fills the interior of the building with cassettes in concentric layers,
following the building shape.
"""

import logging
import math
from typing import List, Tuple, Optional

from optimization_pipeline import PipelineStage, PipelineContext, Cassette
from polygon_offsetter import PolygonOffsetter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConcentricFiller(PipelineStage):
    """
    Fills building interior with cassettes in concentric layers.

    Strategy:
    1. Create inward-offset polygons at regular intervals
    2. For each layer, place cassettes following the perimeter
    3. Use variable spacing based on remaining area
    4. Prioritize larger cassettes in outer layers
    5. Use smaller cassettes in inner layers
    """

    # Cassette sizes by layer preference
    OUTER_LAYER_SIZES = [
        (6, 8), (5, 8), (6, 6), (5, 6), (4, 8), (4, 6)
    ]

    INNER_LAYER_SIZES = [
        (4, 6), (3, 6), (4, 4), (3, 4), (2, 8), (2, 6), (2, 4)
    ]

    def __init__(self, initial_offset: float = 8.0, layer_spacing: float = 6.0):
        """
        Initialize ConcentricFiller.

        Args:
            initial_offset: Distance from perimeter for first layer (accounts for perimeter cassettes)
            layer_spacing: Distance between concentric layers
        """
        self.initial_offset = initial_offset
        self.layer_spacing = layer_spacing
        self.offsetter = PolygonOffsetter()

    @property
    def name(self) -> str:
        return "ConcentricFiller"

    def process(self, context: PipelineContext) -> PipelineContext:
        """Fill interior with concentric cassette layers."""
        polygon = context.polygon
        if len(polygon) < 3:
            logger.warning("Polygon has less than 3 vertices")
            return context

        initial_count = len(context.cassettes)

        # Calculate maximum offset based on polygon size
        polygon_area = context.metadata['total_area']
        max_offset = self._estimate_max_offset(polygon, polygon_area)

        # Create concentric layers
        layers = self.offsetter.create_concentric_layers(
            polygon,
            min_offset=self.initial_offset,
            max_offset=max_offset,
            offset_step=self.layer_spacing
        )

        logger.info(f"  Created {len(layers)} concentric layers")

        # Fill each layer - IMPORTANT: pass original polygon for boundary checking
        for layer_idx, layer_polygon in enumerate(layers):
            layer_cassettes = self._fill_layer(
                layer_polygon,
                layer_idx,
                len(layers),
                context.cassettes,
                polygon  # Pass original polygon for boundary checking
            )

            for cassette in layer_cassettes:
                context.cassettes.append(cassette)

            if layer_cassettes:
                logger.info(f"  Layer {layer_idx}: placed {len(layer_cassettes)} cassettes")

        # Update metadata
        placed_count = len(context.cassettes) - initial_count
        context.metadata['concentric_filler'] = {
            'layers_created': len(layers),
            'cassettes_placed': placed_count,
            'coverage_after': context.get_coverage()
        }

        logger.info(f"  Concentric filling complete: {placed_count} cassettes placed")

        return context

    def _estimate_max_offset(self, polygon: List[Tuple[float, float]],
                            area: float) -> float:
        """
        Estimate maximum offset distance based on polygon dimensions.

        Args:
            polygon: Building polygon
            area: Polygon area

        Returns:
            Maximum offset distance
        """
        # Find bounding box
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]

        width = max(xs) - min(xs)
        height = max(ys) - min(ys)

        # Maximum offset is roughly half of the smaller dimension
        max_offset = min(width, height) / 2

        # Adjust based on area (larger areas can have more layers)
        if area > 2000:
            max_offset = min(max_offset, 30.0)
        elif area > 1000:
            max_offset = min(max_offset, 20.0)
        else:
            max_offset = min(max_offset, 15.0)

        return max_offset

    def _fill_layer(self, layer_polygon: List[Tuple[float, float]],
                   layer_idx: int,
                   total_layers: int,
                   existing_cassettes: List[Cassette],
                   original_polygon: List[Tuple[float, float]]) -> List[Cassette]:
        """
        Fill a single concentric layer with cassettes.

        Args:
            layer_polygon: Offset polygon for this layer
            layer_idx: Index of this layer (0 = outermost)
            total_layers: Total number of layers
            existing_cassettes: Already placed cassettes
            original_polygon: Original building polygon for boundary checking

        Returns:
            List of cassettes placed in this layer
        """
        if len(layer_polygon) < 3:
            return []

        placed_cassettes = []

        # Select cassette sizes based on layer position
        layer_ratio = layer_idx / max(total_layers - 1, 1)
        if layer_ratio < 0.5:
            # Outer layers - use larger cassettes
            sizes_to_try = self.OUTER_LAYER_SIZES
        else:
            # Inner layers - use smaller cassettes
            sizes_to_try = self.INNER_LAYER_SIZES

        # Try grid-based placement within the layer
        placed_cassettes.extend(
            self._grid_fill_polygon(
                layer_polygon,
                sizes_to_try,
                existing_cassettes + placed_cassettes,
                original_polygon  # Pass original polygon for boundary checking
            )
        )

        return placed_cassettes

    def _grid_fill_polygon(self, polygon: List[Tuple[float, float]],
                          cassette_sizes: List[Tuple[float, float]],
                          existing_cassettes: List[Cassette],
                          original_polygon: List[Tuple[float, float]] = None) -> List[Cassette]:
        """
        Fill a polygon using a grid-based approach.

        Args:
            polygon: Polygon to fill (may be offset layer)
            cassette_sizes: Sizes to try (in order of preference)
            existing_cassettes: Already placed cassettes
            original_polygon: Original building polygon for boundary checking (if None, uses polygon)

        Returns:
            List of placed cassettes
        """
        # Use original polygon for boundary checking if provided
        boundary_polygon = original_polygon if original_polygon is not None else polygon

        # Find bounding box of the layer polygon
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        placed = []

        # Try each cassette size
        for width, height in cassette_sizes:
            # Grid placement with both orientations
            for rotated in [False, True]:
                w, h = (height, width) if rotated else (width, height)

                # Grid step (with small overlap to maximize coverage)
                step_x = w - 0.25
                step_y = h - 0.25

                y = min_y
                while y + h <= max_y + 0.5:
                    x = min_x
                    while x + w <= max_x + 0.5:
                        cassette = Cassette(x, y, w, h)

                        # Check against ORIGINAL polygon boundaries
                        if self._is_valid_placement(
                            cassette,
                            boundary_polygon,  # Use original polygon for boundary check
                            existing_cassettes + placed
                        ):
                            placed.append(cassette)

                        x += step_x
                    y += step_y

        return placed

    def _is_valid_placement(self, cassette: Cassette,
                           polygon: List[Tuple[float, float]],
                           existing_cassettes: List[Cassette]) -> bool:
        """
        Check if cassette placement is valid.

        Args:
            cassette: Cassette to place
            polygon: Polygon boundary
            existing_cassettes: Already placed cassettes

        Returns:
            True if placement is valid
        """
        # Check overlap with existing cassettes
        for existing in existing_cassettes:
            if cassette.overlaps(existing):
                return False

        # Check if ALL corners are within polygon (strict boundary check)
        corners = cassette.get_corners()
        for corner in corners:
            if not self._point_in_polygon(corner, polygon):
                return False

        # Also check center and edge midpoints for extra safety
        additional_points = [
            (cassette.x + cassette.width/2, cassette.y + cassette.height/2),  # Center
            (cassette.x + cassette.width/2, cassette.y),  # Bottom mid
            (cassette.x + cassette.width/2, cassette.y + cassette.height),  # Top mid
            (cassette.x, cassette.y + cassette.height/2),  # Left mid
            (cassette.x + cassette.width, cassette.y + cassette.height/2),  # Right mid
        ]

        for point in additional_points:
            if not self._point_in_polygon(point, polygon):
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


def test_concentric_filler():
    """Test the ConcentricFiller stage."""
    from optimization_pipeline import PipelineContext
    from corner_placer_stage import CornerPlacer
    from perimeter_tracer_stage import PerimeterTracer

    # Test polygon: Simple rectangle
    test_polygon = [
        (0, 0),
        (50, 0),
        (50, 40),
        (0, 40)
    ]

    # Create context
    context = PipelineContext(test_polygon)

    # Place corners and perimeter first
    corner_placer = CornerPlacer()
    context = corner_placer.process(context)
    print(f"After corners: {len(context.cassettes)} cassettes, {context.get_coverage():.1f}% coverage")

    perimeter_tracer = PerimeterTracer()
    context = perimeter_tracer.process(context)
    print(f"After perimeter: {len(context.cassettes)} cassettes, {context.get_coverage():.1f}% coverage")

    # Fill interior with concentric layers
    filler = ConcentricFiller()
    result_context = filler.process(context)

    print("\n" + "="*60)
    print("CONCENTRIC FILLER TEST RESULTS")
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

    # Check if meets requirement
    print(f"\nMeets 94% requirement: {result_context.get_coverage() >= 94.0}")

    return result_context


if __name__ == "__main__":
    test_concentric_filler()
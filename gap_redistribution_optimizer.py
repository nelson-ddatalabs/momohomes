#!/usr/bin/env python3
"""
Gap Redistribution C-Channel Optimizer
=======================================
Achieves 100% coverage by:
1. Resizing boundary cassettes to eliminate gaps (if possible)
2. If gaps remain, shift cassettes to move gaps from boundaries to center
3. Fill center gaps with C-channels

Constraints:
- Cassettes: 2-8 feet dimensions, 0.5' increments, ≤48 sq ft, ≤500 lbs (10.4 lbs/sq ft)
- C-channels: 1.5" to 18" width
- Coverage: Exactly 100.00%
- Structural: No gaps at boundaries, gaps only inside (filled with C-channels)
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import logging
from shapely.geometry import Polygon, box, Point
from shapely.ops import unary_union
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GapRedistributionOptimizer:
    """
    Gap Redistribution C-Channel Optimizer

    Strategy:
    1. Place standard cassettes edge-to-edge (greedy)
    2. Detect boundary gaps
    3. Resize boundary cassettes to eliminate gaps (if feasible)
    4. If gaps remain, shift cassettes to move gaps to center
    5. Place C-channels in center gaps
    6. Ensure exactly 100% coverage
    """

    # Cassette constraints
    MIN_CASSETTE_DIM = 2.0  # feet
    MAX_CASSETTE_DIM = 8.0  # feet
    MAX_CASSETTE_AREA = 48.0  # sq ft
    MAX_CASSETTE_WEIGHT = 500.0  # lbs
    WEIGHT_PER_SQ_FT = 10.4  # lbs/sq ft
    CASSETTE_INCREMENT = 0.5  # feet

    # C-channel constraints
    MIN_CCHANNEL = 1.5 / 12.0  # 1.5 inches in feet
    MAX_CCHANNEL = 18.0 / 12.0  # 18 inches in feet

    # Tolerance
    COVERAGE_TOLERANCE = 0.01  # 0.01% tolerance for 100% coverage
    GEOMETRIC_TOLERANCE = 0.001  # feet

    def __init__(self, polygon: List[Tuple[float, float]]):
        """
        Initialize optimizer.

        Args:
            polygon: List of (x, y) coordinates defining the boundary
        """
        self.polygon = polygon
        self.polygon_shapely = Polygon(polygon)
        self.polygon_area = self.polygon_shapely.area

        self.cassettes = []
        self.c_channels = []

        logger.info("=" * 80)
        logger.info("GAP REDISTRIBUTION C-CHANNEL OPTIMIZER")
        logger.info("=" * 80)
        logger.info(f"Polygon area: {self.polygon_area:.2f} sq ft")
        logger.info(f"Cassette constraints: {self.MIN_CASSETTE_DIM}'-{self.MAX_CASSETTE_DIM}', "
                   f"≤{self.MAX_CASSETTE_AREA} sq ft, ≤{self.MAX_CASSETTE_WEIGHT} lbs")
        logger.info(f"C-channel range: {self.MIN_CCHANNEL*12:.1f}\" to {self.MAX_CCHANNEL*12:.1f}\"")

    def _greedy_cassette_placement(self) -> List[Dict]:
        """
        Phase 1: Initial cassette placement using greedy optimizer.

        Returns:
            List of cassette dictionaries
        """
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 1: INITIAL CASSETTE PLACEMENT (GREEDY)")
        logger.info("=" * 80)

        from ultra_smart_optimizer import UltraSmartOptimizer
        optimizer = UltraSmartOptimizer(self.polygon)
        result = optimizer.optimize()

        cassettes = result['cassettes']
        coverage = result.get('coverage_percent', result.get('coverage', 0) * 100)

        logger.info(f"✓ Placed {len(cassettes)} cassettes")
        logger.info(f"✓ Initial coverage: {coverage:.2f}% (cassettes only)")

        return cassettes

    def _detect_gaps(self, cassettes: List[Dict]) -> Dict:
        """
        Phase 2: Detect gaps between cassettes and polygon boundary.

        Args:
            cassettes: List of cassette dictionaries

        Returns:
            Dictionary with gap analysis
        """
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 2: GAP DETECTION")
        logger.info("=" * 80)

        # Create cassette union
        cassette_geoms = [
            box(c['x'], c['y'], c['x'] + c['width'], c['y'] + c['height'])
            for c in cassettes
        ]
        cassette_union = unary_union(cassette_geoms)

        # Calculate gap
        gap_geom = self.polygon_shapely.difference(cassette_union)
        gap_area = gap_geom.area

        cassette_area = cassette_union.area
        coverage = (cassette_area / self.polygon_area) * 100

        logger.info(f"Cassette area: {cassette_area:.2f} sq ft ({coverage:.2f}%)")
        logger.info(f"Gap area: {gap_area:.2f} sq ft ({gap_area/self.polygon_area*100:.2f}%)")

        if gap_area < self.GEOMETRIC_TOLERANCE:
            logger.info("✓ No gaps detected - 100% coverage achieved!")
            return {
                'has_gaps': False,
                'gap_area': 0,
                'gap_geometry': None,
                'coverage': 100.0
            }

        # Analyze gap location (boundary vs interior)
        poly_bounds = self.polygon_shapely.bounds
        cass_bounds = cassette_union.bounds

        boundary_gaps = {
            'left': abs(cass_bounds[0] - poly_bounds[0]),
            'right': abs(poly_bounds[2] - cass_bounds[2]),
            'bottom': abs(cass_bounds[1] - poly_bounds[1]),
            'top': abs(poly_bounds[3] - cass_bounds[3])
        }

        logger.info("\nBoundary gap analysis:")
        total_boundary_gap = 0
        for edge, gap_dist in boundary_gaps.items():
            if gap_dist > self.GEOMETRIC_TOLERANCE:
                logger.info(f"  {edge.capitalize()}: {gap_dist:.3f}' ({gap_dist*12:.1f}\")")
                total_boundary_gap += gap_dist

        return {
            'has_gaps': True,
            'gap_area': gap_area,
            'gap_geometry': gap_geom,
            'coverage': coverage,
            'boundary_gaps': boundary_gaps,
            'cassette_union': cassette_union
        }

    def _is_valid_cassette_size(self, width: float, height: float) -> bool:
        """
        Check if cassette size is valid within constraints.

        Args:
            width: Cassette width in feet
            height: Cassette height in feet

        Returns:
            True if valid, False otherwise
        """
        # Check dimension limits
        if width < self.MIN_CASSETTE_DIM or width > self.MAX_CASSETTE_DIM:
            return False
        if height < self.MIN_CASSETTE_DIM or height > self.MAX_CASSETTE_DIM:
            return False

        # Check area limit
        area = width * height
        if area > self.MAX_CASSETTE_AREA:
            return False

        # Check weight limit
        weight = area * self.WEIGHT_PER_SQ_FT
        if weight > self.MAX_CASSETTE_WEIGHT:
            return False

        # Check increment alignment
        if abs(width % self.CASSETTE_INCREMENT) > 0.001:
            return False
        if abs(height % self.CASSETTE_INCREMENT) > 0.001:
            return False

        return True

    def _identify_boundary_cassettes(self, cassettes: List[Dict], boundary_gaps: Dict) -> Dict:
        """
        Identify which cassettes are at polygon boundaries.

        Args:
            cassettes: List of cassette dictionaries
            boundary_gaps: Dictionary of boundary gaps

        Returns:
            Dictionary mapping edges to cassettes at those edges
        """
        poly_bounds = self.polygon_shapely.bounds
        boundary_cassettes = {
            'left': [],
            'right': [],
            'bottom': [],
            'top': []
        }

        for i, c in enumerate(cassettes):
            c_left = c['x']
            c_right = c['x'] + c['width']
            c_bottom = c['y']
            c_top = c['y'] + c['height']

            # Check if cassette is at each boundary
            # For left/bottom: cassette edge must match polygon edge
            if abs(c_left - poly_bounds[0]) < self.GEOMETRIC_TOLERANCE:
                boundary_cassettes['left'].append(i)
            if abs(c_bottom - poly_bounds[1]) < self.GEOMETRIC_TOLERANCE:
                boundary_cassettes['bottom'].append(i)

            # For right/top: cassette edge must be NEAR polygon edge (within gap distance)
            # This identifies cassettes that are closest to the boundary where gaps exist
            if boundary_gaps['right'] > self.GEOMETRIC_TOLERANCE:
                # Find cassettes with maximum X coordinate
                if c_right >= poly_bounds[2] - boundary_gaps['right'] - self.GEOMETRIC_TOLERANCE:
                    boundary_cassettes['right'].append(i)

            if boundary_gaps['top'] > self.GEOMETRIC_TOLERANCE:
                # Find cassettes with maximum Y coordinate
                if c_top >= poly_bounds[3] - boundary_gaps['top'] - self.GEOMETRIC_TOLERANCE:
                    boundary_cassettes['top'].append(i)

        return boundary_cassettes

    def _try_resize_cassette(self, cassette: Dict, gap_amount: float, direction: str) -> Optional[Dict]:
        """
        Attempt to resize a cassette to fill a gap.

        Args:
            cassette: Cassette dictionary
            gap_amount: Gap size to fill (in feet)
            direction: 'width' or 'height' - which dimension to resize

        Returns:
            Updated cassette dict if resize is valid, None otherwise
        """
        new_cassette = cassette.copy()

        if direction == 'width':
            new_width = cassette['width'] + gap_amount
            new_height = cassette['height']

            # Round to nearest 0.5' increment
            new_width = round(new_width / self.CASSETTE_INCREMENT) * self.CASSETTE_INCREMENT

            if not self._is_valid_cassette_size(new_width, new_height):
                return None

            new_cassette['width'] = new_width
            new_cassette['area'] = new_width * new_height
            new_cassette['weight'] = new_cassette['area'] * self.WEIGHT_PER_SQ_FT

        elif direction == 'height':
            new_width = cassette['width']
            new_height = cassette['height'] + gap_amount

            # Round to nearest 0.5' increment
            new_height = round(new_height / self.CASSETTE_INCREMENT) * self.CASSETTE_INCREMENT

            if not self._is_valid_cassette_size(new_width, new_height):
                return None

            new_cassette['height'] = new_height
            new_cassette['area'] = new_width * new_height
            new_cassette['weight'] = new_cassette['area'] * self.WEIGHT_PER_SQ_FT

        return new_cassette

    def _resize_boundary_cassettes(self, cassettes: List[Dict], gap_info: Dict) -> Tuple[List[Dict], float]:
        """
        Phase 3: Attempt to resize boundary cassettes to eliminate gaps.

        Args:
            cassettes: List of cassette dictionaries
            gap_info: Gap information from _detect_gaps

        Returns:
            Tuple of (updated cassettes, remaining gap area)
        """
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 3: CASSETTE RESIZE OPTIMIZATION")
        logger.info("=" * 80)

        boundary_gaps = gap_info['boundary_gaps']
        cassettes = [c.copy() for c in cassettes]  # Work on copies

        # Identify boundary cassettes
        boundary_cassettes = self._identify_boundary_cassettes(cassettes, boundary_gaps)

        logger.info("\nBoundary cassette analysis:")
        for edge, indices in boundary_cassettes.items():
            if indices and boundary_gaps[edge] > self.GEOMETRIC_TOLERANCE:
                logger.info(f"  {edge.capitalize()}: {len(indices)} cassettes, gap = {boundary_gaps[edge]:.3f}' ({boundary_gaps[edge]*12:.1f}\")")

        # Attempt to resize boundary cassettes for each edge with gaps
        total_gap_filled = 0.0
        resize_attempts = []

        for edge, gap_amount in boundary_gaps.items():
            if gap_amount < self.GEOMETRIC_TOLERANCE:
                continue

            edge_cassettes = boundary_cassettes[edge]
            if not edge_cassettes:
                continue

            logger.info(f"\nAttempting to resize {edge} boundary cassettes (gap = {gap_amount:.3f}')...")

            # Determine resize direction based on edge
            if edge in ['left', 'right']:
                resize_direction = 'width'
            else:  # top or bottom
                resize_direction = 'height'

            # Try to distribute gap among boundary cassettes
            gap_per_cassette = gap_amount / len(edge_cassettes)

            logger.info(f"  Strategy: Distribute {gap_amount:.3f}' among {len(edge_cassettes)} cassettes")
            logger.info(f"  Gap per cassette: {gap_per_cassette:.3f}' ({gap_per_cassette*12:.1f}\")")

            # Check if we can resize all cassettes at this edge
            all_resizes_valid = True
            potential_resizes = []

            for idx in edge_cassettes:
                cassette = cassettes[idx]
                resized = self._try_resize_cassette(cassette, gap_per_cassette, resize_direction)

                if resized is None:
                    all_resizes_valid = False
                    logger.info(f"  ✗ Cassette #{idx} cannot resize (exceeds constraints)")
                    break
                else:
                    potential_resizes.append((idx, resized))
                    logger.info(f"  ✓ Cassette #{idx} can resize: {cassette['width']:.1f}'×{cassette['height']:.1f}' → {resized['width']:.1f}'×{resized['height']:.1f}'")

            # Apply resizes if all are valid
            if all_resizes_valid and potential_resizes:
                for idx, resized in potential_resizes:
                    cassettes[idx] = resized

                total_gap_filled += gap_amount
                logger.info(f"  ✓ SUCCESS: Resized {len(potential_resizes)} cassettes, filled {gap_amount:.3f}' gap")
                resize_attempts.append({
                    'edge': edge,
                    'gap_filled': gap_amount,
                    'cassettes_resized': len(potential_resizes)
                })
            else:
                logger.info(f"  ✗ FAILED: Cannot resize {edge} cassettes (constraints violated)")

        # Recalculate coverage after resizing
        cassette_geoms = [
            box(c['x'], c['y'], c['x'] + c['width'], c['y'] + c['height'])
            for c in cassettes
        ]
        cassette_union = unary_union(cassette_geoms)
        gap_geom = self.polygon_shapely.difference(cassette_union)
        remaining_gap = gap_geom.area

        logger.info(f"\n" + "-" * 80)
        logger.info(f"RESIZE OPTIMIZATION RESULTS:")
        logger.info(f"  Initial gap: {gap_info['gap_area']:.2f} sq ft")
        logger.info(f"  Gap filled by resizing: {total_gap_filled:.2f} linear feet")
        logger.info(f"  Remaining gap: {remaining_gap:.2f} sq ft")
        logger.info(f"  Coverage: {(cassette_union.area / self.polygon_area) * 100:.2f}%")

        if remaining_gap < self.GEOMETRIC_TOLERANCE:
            logger.info(f"  ✓ 100% COVERAGE ACHIEVED BY RESIZING!")
            return cassettes, 0.0

        return cassettes, remaining_gap

    def _determine_split_orientation(self, gap_geom) -> str:
        """
        Determine whether to split vertically or horizontally.

        Args:
            gap_geom: Gap geometry

        Returns:
            'vertical' or 'horizontal'
        """
        bounds = gap_geom.bounds
        gap_width = bounds[2] - bounds[0]
        gap_height = bounds[3] - bounds[1]

        # Choose orientation based on longest dimension
        if gap_width > gap_height:
            return 'horizontal'  # Gap is wider, so split horizontally
        else:
            return 'vertical'  # Gap is taller, so split vertically

    def _find_optimal_split_location(self, cassettes: List[Dict], orientation: str) -> float:
        """
        Find optimal location to create center gap.

        Args:
            cassettes: List of cassettes
            orientation: 'vertical' or 'horizontal'

        Returns:
            Split coordinate (X for vertical, Y for horizontal)
        """
        if orientation == 'vertical':
            # Find median X coordinate of cassettes
            x_coords = [c['x'] + c['width'] / 2 for c in cassettes]
            return np.median(x_coords)
        else:
            # Find median Y coordinate of cassettes
            y_coords = [c['y'] + c['height'] / 2 for c in cassettes]
            return np.median(y_coords)

    def _shift_cassettes_to_create_gap(self, cassettes: List[Dict], gap_area: float,
                                       orientation: str, split_location: float) -> Tuple[List[Dict], Dict]:
        """
        Phase 4: Shift cassettes to move boundary gaps to center.

        Args:
            cassettes: List of cassettes
            gap_area: Total gap area to redistribute
            orientation: 'vertical' or 'horizontal'
            split_location: Where to create the gap

        Returns:
            Tuple of (shifted cassettes, gap_info dict)
        """
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 4: CASSETTE SHIFTING (CREATE CENTER GAP)")
        logger.info("=" * 80)

        logger.info(f"Split orientation: {orientation}")
        logger.info(f"Split location: {split_location:.2f}'")
        logger.info(f"Gap area to redistribute: {gap_area:.2f} sq ft")

        # Calculate gap width needed
        # For vertical split: gap_width × polygon_height = gap_area
        # For horizontal split: polygon_width × gap_height = gap_area

        cassettes = [c.copy() for c in cassettes]

        if orientation == 'vertical':
            # Calculate gap width and height needed
            # We need to determine the actual height that the gap spans
            # This should be based on the cassettes that will be separated

            # Find all cassettes that cross the split location
            cassettes_crossing_split = [c for c in cassettes
                                       if c['x'] < split_location < c['x'] + c['width'] or
                                       abs(c['x'] - split_location) < self.GEOMETRIC_TOLERANCE or
                                       abs(c['x'] + c['width'] - split_location) < self.GEOMETRIC_TOLERANCE]

            # Calculate the total vertical span covered by cassettes at this location
            if cassettes_crossing_split:
                min_y = min(c['y'] for c in cassettes_crossing_split)
                max_y = max(c['y'] + c['height'] for c in cassettes_crossing_split)
                gap_height = max_y - min_y
            else:
                # No cassettes crossing - use cassettes on either side
                cassettes_left_of_split = [c for c in cassettes if c['x'] + c['width'] <= split_location]
                cassettes_right_of_split = [c for c in cassettes if c['x'] >= split_location]

                if cassettes_left_of_split and cassettes_right_of_split:
                    # Use the maximum vertical extent
                    left_max_x = max(c['x'] + c['width'] for c in cassettes_left_of_split)
                    # Get cassettes at the rightmost edge of left group
                    cassettes_at_edge = [c for c in cassettes_left_of_split
                                        if abs(c['x'] + c['width'] - left_max_x) < self.GEOMETRIC_TOLERANCE]
                    min_y = min(c['y'] for c in cassettes_at_edge)
                    max_y = max(c['y'] + c['height'] for c in cassettes_at_edge)
                    gap_height = max_y - min_y
                else:
                    # Fallback to polygon height
                    poly_bounds = self.polygon_shapely.bounds
                    gap_height = poly_bounds[3] - poly_bounds[1]

            gap_width = gap_area / gap_height if gap_height > 0 else 0
            gap_width_inches = gap_width * 12

            logger.info(f"Vertical gap dimensions: {gap_width:.3f}' × {gap_height:.1f}' = {gap_area:.2f} sq ft")
            logger.info(f"Gap width: {gap_width_inches:.1f}\"")

            # Check if gap width is within C-channel limits
            if gap_width_inches < 1.5:
                logger.warning(f"⚠️  Gap width ({gap_width_inches:.1f}\") < C-channel min (1.5\")")
                logger.warning("Cannot use C-channel for this gap (too small)")
                return cassettes, {'success': False, 'reason': 'gap_too_small'}

            if gap_width_inches > 18:
                logger.info(f"Gap width ({gap_width_inches:.1f}\") > C-channel max (18\")")
                logger.info(f"Will use multiple C-channels side-by-side")
                num_cchannels = math.ceil(gap_width_inches / 18)
                logger.info(f"Number of C-channels needed: {num_cchannels}")

            # Shift cassettes
            # Cassettes left of split stay in place
            # Cassettes right of split move right by gap_width
            cassettes_left = []
            cassettes_right = []

            for c in cassettes:
                c_center_x = c['x'] + c['width'] / 2
                if c_center_x < split_location:
                    cassettes_left.append(c)
                else:
                    cassettes_right.append(c)

            logger.info(f"Cassettes left of split: {len(cassettes_left)}")
            logger.info(f"Cassettes right of split: {len(cassettes_right)}")

            # Shift right cassettes
            for c in cassettes_right:
                c['x'] += gap_width

            # Combine back
            shifted_cassettes = cassettes_left + cassettes_right

            # Verify cassettes stay within polygon bounds
            poly_bounds = self.polygon_shapely.bounds
            for c in shifted_cassettes:
                c_right = c['x'] + c['width']
                if c_right > poly_bounds[2] + self.GEOMETRIC_TOLERANCE:
                    logger.warning(f"⚠️  Cassette at ({c['x']:.2f}, {c['y']:.2f}) extends beyond polygon boundary")
                    logger.warning("Shift amount may be too large")
                    return cassettes, {'success': False, 'reason': 'exceeds_bounds'}

            logger.info(f"✓ Shifted {len(cassettes_right)} cassettes right by {gap_width:.3f}'")

            return shifted_cassettes, {
                'success': True,
                'orientation': 'vertical',
                'split_location': split_location,
                'gap_width': gap_width,
                'gap_width_inches': gap_width_inches,
                'gap_height': gap_height,
                'gap_area': gap_area
            }

        else:  # horizontal
            # Calculate gap height needed
            cassettes_at_split = [c for c in cassettes
                                 if c['y'] <= split_location <= c['y'] + c['height']]

            if cassettes_at_split:
                avg_width = np.mean([c['width'] for c in cassettes_at_split])
            else:
                # Use polygon width as fallback
                poly_bounds = self.polygon_shapely.bounds
                avg_width = poly_bounds[2] - poly_bounds[0]

            gap_height = gap_area / avg_width if avg_width > 0 else 0
            gap_height_inches = gap_height * 12

            logger.info(f"Horizontal gap dimensions: {avg_width:.1f}' × {gap_height:.3f}' = {gap_area:.2f} sq ft")
            logger.info(f"Gap height: {gap_height_inches:.1f}\"")

            # Check if gap height is within C-channel limits
            if gap_height_inches < 1.5:
                logger.warning(f"⚠️  Gap height ({gap_height_inches:.1f}\") < C-channel min (1.5\")")
                logger.warning("Cannot use C-channel for this gap (too small)")
                return cassettes, {'success': False, 'reason': 'gap_too_small'}

            if gap_height_inches > 18:
                logger.info(f"Gap height ({gap_height_inches:.1f}\") > C-channel max (18\")")
                logger.info(f"Will use multiple C-channels stacked")
                num_cchannels = math.ceil(gap_height_inches / 18)
                logger.info(f"Number of C-channels needed: {num_cchannels}")

            # Shift cassettes
            # Cassettes below split stay in place
            # Cassettes above split move up by gap_height
            cassettes_below = []
            cassettes_above = []

            for c in cassettes:
                c_center_y = c['y'] + c['height'] / 2
                if c_center_y < split_location:
                    cassettes_below.append(c)
                else:
                    cassettes_above.append(c)

            logger.info(f"Cassettes below split: {len(cassettes_below)}")
            logger.info(f"Cassettes above split: {len(cassettes_above)}")

            # Shift above cassettes
            for c in cassettes_above:
                c['y'] += gap_height

            # Combine back
            shifted_cassettes = cassettes_below + cassettes_above

            # Verify cassettes stay within polygon bounds
            poly_bounds = self.polygon_shapely.bounds
            for c in shifted_cassettes:
                c_top = c['y'] + c['height']
                if c_top > poly_bounds[3] + self.GEOMETRIC_TOLERANCE:
                    logger.warning(f"⚠️  Cassette at ({c['x']:.2f}, {c['y']:.2f}) extends beyond polygon boundary")
                    logger.warning("Shift amount may be too large")
                    return cassettes, {'success': False, 'reason': 'exceeds_bounds'}

            logger.info(f"✓ Shifted {len(cassettes_above)} cassettes up by {gap_height:.3f}'")

            return shifted_cassettes, {
                'success': True,
                'orientation': 'horizontal',
                'split_location': split_location,
                'gap_width': avg_width,
                'gap_height': gap_height,
                'gap_height_inches': gap_height_inches,
                'gap_area': gap_area
            }

    def _place_cchannels(self, gap_info: Dict) -> List[Dict]:
        """
        Phase 5: Place C-channels to fill center gap.

        Args:
            gap_info: Gap information from shifting phase

        Returns:
            List of C-channel dictionaries
        """
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 5: C-CHANNEL PLACEMENT")
        logger.info("=" * 80)

        if not gap_info.get('success', False):
            logger.error("Cannot place C-channels - gap creation failed")
            return []

        orientation = gap_info['orientation']
        split_location = gap_info['split_location']
        gap_area = gap_info['gap_area']

        # Calculate ACTUAL gap geometry after shifting by comparing cassettes with polygon
        cassette_geoms = [
            box(c['x'], c['y'], c['x'] + c['width'], c['y'] + c['height'])
            for c in self.cassettes
        ]
        cassette_union = unary_union(cassette_geoms)
        actual_gap_geom = self.polygon_shapely.difference(cassette_union)
        actual_gap_area = actual_gap_geom.area

        logger.info(f"Actual gap geometry after shifting: {actual_gap_area:.2f} sq ft")

        c_channels = []

        if orientation == 'vertical':
            gap_width = gap_info['gap_width']
            gap_height = gap_info['gap_height']
            gap_width_inches = gap_info['gap_width_inches']

            logger.info(f"Filling vertical gap: {gap_width:.3f}' × {gap_height:.1f}' = {gap_area:.2f} sq ft")

            # Use the ACTUAL gap geometry to determine C-channel placement
            # Get the gap bounds
            if not actual_gap_geom.is_empty:
                gap_bounds = actual_gap_geom.bounds
                gap_x = gap_bounds[0]
                gap_y_start = gap_bounds[1]
                gap_x_end = gap_bounds[2]
                gap_y_end = gap_bounds[3]
                actual_gap_width = gap_x_end - gap_x
                actual_gap_height = gap_y_end - gap_y_start
                actual_gap_width_inches = actual_gap_width * 12

                logger.info(f"Actual gap bounds: x=[{gap_x:.1f}, {gap_x_end:.1f}], y=[{gap_y_start:.1f}, {gap_y_end:.1f}]")
                logger.info(f"Actual gap dimensions: {actual_gap_width:.3f}' × {actual_gap_height:.1f}' = {actual_gap_area:.2f} sq ft")
            else:
                # No gap - use planned dimensions
                gap_x = split_location
                gap_y_start = 0
                actual_gap_width = gap_width
                actual_gap_height = gap_height
                actual_gap_width_inches = gap_width_inches

            if actual_gap_width_inches <= 18:
                # Single C-channel fills the gap
                c_channel = {
                    'x': gap_x,
                    'y': gap_y_start,
                    'width': actual_gap_width,
                    'height': actual_gap_height,
                    'width_inches': actual_gap_width_inches,
                    'area': actual_gap_width * actual_gap_height,
                    'orientation': 'vertical'
                }
                c_channels.append(c_channel)
                logger.info(f"✓ Placed single C-channel: {actual_gap_width_inches:.1f}\" × {actual_gap_height:.1f}' at ({gap_x:.1f}, {gap_y_start:.1f})")

            else:
                # Multiple C-channels side-by-side
                num_cchannels = math.ceil(gap_width_inches / 18)
                cchannel_width_inches = gap_width_inches / num_cchannels
                cchannel_width_feet = cchannel_width_inches / 12

                logger.info(f"Placing {num_cchannels} C-channels side-by-side")
                logger.info(f"Each C-channel: {cchannel_width_inches:.1f}\" × {gap_height:.1f}'")

                for i in range(num_cchannels):
                    c_channel = {
                        'x': split_location + (i * cchannel_width_feet),
                        'y': 0,
                        'width': cchannel_width_feet,
                        'height': gap_height,
                        'width_inches': cchannel_width_inches,
                        'area': cchannel_width_feet * gap_height,
                        'orientation': 'vertical'
                    }
                    c_channels.append(c_channel)

                logger.info(f"✓ Placed {num_cchannels} C-channels")

        else:  # horizontal
            gap_width = gap_info['gap_width']
            gap_height = gap_info['gap_height']
            gap_height_inches = gap_info['gap_height_inches']

            logger.info(f"Filling horizontal gap: {gap_width:.1f}' × {gap_height:.3f}' = {gap_area:.2f} sq ft")

            if gap_height_inches <= 18:
                # Single C-channel
                c_channel = {
                    'x': 0,
                    'y': split_location,
                    'width': gap_width,
                    'height': gap_height,
                    'height_inches': gap_height_inches,
                    'area': gap_area,
                    'orientation': 'horizontal'
                }
                c_channels.append(c_channel)
                logger.info(f"✓ Placed single C-channel: {gap_width:.1f}' × {gap_height_inches:.1f}\"")

            else:
                # Multiple C-channels stacked
                num_cchannels = math.ceil(gap_height_inches / 18)
                cchannel_height_inches = gap_height_inches / num_cchannels
                cchannel_height_feet = cchannel_height_inches / 12

                logger.info(f"Placing {num_cchannels} C-channels stacked")
                logger.info(f"Each C-channel: {gap_width:.1f}' × {cchannel_height_inches:.1f}\"")

                for i in range(num_cchannels):
                    c_channel = {
                        'x': 0,
                        'y': split_location + (i * cchannel_height_feet),
                        'width': gap_width,
                        'height': cchannel_height_feet,
                        'height_inches': cchannel_height_inches,
                        'area': gap_width * cchannel_height_feet,
                        'orientation': 'horizontal'
                    }
                    c_channels.append(c_channel)

                logger.info(f"✓ Placed {num_cchannels} C-channels")

        # Calculate total C-channel area
        total_cchannel_area = sum(c['area'] for c in c_channels)
        logger.info(f"\nTotal C-channel area: {total_cchannel_area:.2f} sq ft")

        return c_channels

    def _validate_coverage(self, cassettes: List[Dict], c_channels: List[Dict]) -> Dict:
        """
        Phase 6: Validate 100% coverage and constraints.

        Args:
            cassettes: List of cassettes
            c_channels: List of C-channels

        Returns:
            Validation results dictionary
        """
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 6: VALIDATION")
        logger.info("=" * 80)

        # Calculate total coverage
        cassette_geoms = [
            box(c['x'], c['y'], c['x'] + c['width'], c['y'] + c['height'])
            for c in cassettes
        ]
        cassette_union = unary_union(cassette_geoms) if cassette_geoms else Polygon()

        cchannel_geoms = [
            box(c['x'], c['y'], c['x'] + c['width'], c['y'] + c['height'])
            for c in c_channels
        ]
        cchannel_union = unary_union(cchannel_geoms) if cchannel_geoms else Polygon()

        # Combined coverage
        total_union = unary_union([cassette_union, cchannel_union])
        covered_area = total_union.area
        coverage_percent = (covered_area / self.polygon_area) * 100

        cassette_area = cassette_union.area
        cchannel_area = cchannel_union.area

        logger.info(f"Coverage validation:")
        logger.info(f"  Polygon area: {self.polygon_area:.2f} sq ft")
        logger.info(f"  Cassette area: {cassette_area:.2f} sq ft")
        logger.info(f"  C-channel area: {cchannel_area:.2f} sq ft")
        logger.info(f"  Total covered: {covered_area:.2f} sq ft")
        logger.info(f"  Coverage: {coverage_percent:.4f}%")

        # Check if coverage is exactly 100%
        coverage_diff = abs(100.0 - coverage_percent)
        if coverage_diff <= self.COVERAGE_TOLERANCE:
            logger.info(f"  ✓ 100% COVERAGE ACHIEVED! (within {self.COVERAGE_TOLERANCE}% tolerance)")
        else:
            logger.warning(f"  ⚠️  Coverage deviation: {coverage_diff:.4f}%")

        # Check for overlaps
        overlap = cassette_union.intersection(cchannel_union)
        overlap_area = overlap.area if not overlap.is_empty else 0

        if overlap_area > self.GEOMETRIC_TOLERANCE:
            logger.warning(f"  ⚠️  Overlap detected: {overlap_area:.4f} sq ft")
        else:
            logger.info(f"  ✓ No overlaps between cassettes and C-channels")

        # Check cassette constraints
        invalid_cassettes = []
        for i, c in enumerate(cassettes):
            if not self._is_valid_cassette_size(c['width'], c['height']):
                invalid_cassettes.append(i)

        if invalid_cassettes:
            logger.warning(f"  ⚠️  {len(invalid_cassettes)} cassettes violate constraints")
        else:
            logger.info(f"  ✓ All {len(cassettes)} cassettes within constraints")

        # Check C-channel constraints
        invalid_cchannels = []
        for i, c in enumerate(c_channels):
            if c.get('orientation') == 'vertical':
                width_inches = c.get('width_inches', c['width'] * 12)
                if width_inches < 1.5 or width_inches > 18:
                    invalid_cchannels.append(i)
            else:
                height_inches = c.get('height_inches', c['height'] * 12)
                if height_inches < 1.5 or height_inches > 18:
                    invalid_cchannels.append(i)

        if invalid_cchannels:
            logger.warning(f"  ⚠️  {len(invalid_cchannels)} C-channels outside 1.5\"-18\" range")
        else:
            logger.info(f"  ✓ All {len(c_channels)} C-channels within 1.5\"-18\" range")

        # Check boundary gaps
        gap_geom = self.polygon_shapely.difference(total_union)
        remaining_gap = gap_geom.area

        if remaining_gap > self.GEOMETRIC_TOLERANCE:
            logger.warning(f"  ⚠️  Remaining gap: {remaining_gap:.4f} sq ft")
        else:
            logger.info(f"  ✓ No boundary gaps remaining")

        return {
            'valid': (coverage_diff <= self.COVERAGE_TOLERANCE and
                     overlap_area <= self.GEOMETRIC_TOLERANCE and
                     not invalid_cassettes and
                     not invalid_cchannels and
                     remaining_gap <= self.GEOMETRIC_TOLERANCE),
            'coverage_percent': coverage_percent,
            'coverage_diff': coverage_diff,
            'overlap_area': overlap_area,
            'remaining_gap': remaining_gap,
            'cassette_area': cassette_area,
            'cchannel_area': cchannel_area
        }

    def optimize(self) -> Dict:
        """
        Main optimization method.

        Returns:
            Dictionary with optimization results
        """
        # Phase 1: Initial cassette placement
        self.cassettes = self._greedy_cassette_placement()

        # Phase 2: Detect gaps
        gap_info = self._detect_gaps(self.cassettes)

        if not gap_info['has_gaps']:
            # Already at 100% coverage
            return self._prepare_results(coverage=100.0, cchannel_area=0)

        # Phase 3: Resize boundary cassettes
        self.cassettes, remaining_gap = self._resize_boundary_cassettes(self.cassettes, gap_info)

        if remaining_gap < self.GEOMETRIC_TOLERANCE:
            # 100% coverage achieved by resizing alone
            return self._prepare_results(coverage=100.0, cchannel_area=0)

        # Phase 4: Shift cassettes to move gaps to center
        logger.info(f"\nResizing could not achieve 100% coverage (remaining gap: {remaining_gap:.2f} sq ft)")
        logger.info("Proceeding to cassette shifting + C-channel placement...")

        # Re-detect gaps after resizing
        cassette_geoms = [
            box(c['x'], c['y'], c['x'] + c['width'], c['y'] + c['height'])
            for c in self.cassettes
        ]
        cassette_union = unary_union(cassette_geoms)
        gap_geom = self.polygon_shapely.difference(cassette_union)

        # Determine split orientation
        orientation = self._determine_split_orientation(gap_geom)

        # Find optimal split location
        split_location = self._find_optimal_split_location(self.cassettes, orientation)

        # Shift cassettes to create center gap
        self.cassettes, shift_info = self._shift_cassettes_to_create_gap(
            self.cassettes, remaining_gap, orientation, split_location
        )

        if not shift_info.get('success', False):
            logger.error(f"⚠️  Cassette shifting failed: {shift_info.get('reason', 'unknown')}")
            logger.error("Cannot achieve 100% coverage with current strategy")

            # Return partial results
            cassette_geoms = [
                box(c['x'], c['y'], c['x'] + c['width'], c['y'] + c['height'])
                for c in self.cassettes
            ]
            cassette_union = unary_union(cassette_geoms)
            coverage = (cassette_union.area / self.polygon_area) * 100

            return self._prepare_results(coverage=coverage, cchannel_area=0)

        # Phase 5: Place C-channels
        self.c_channels = self._place_cchannels(shift_info)

        # Phase 6: Validate 100% coverage
        validation = self._validate_coverage(self.cassettes, self.c_channels)

        # Prepare final results
        logger.info("\n" + "=" * 80)
        logger.info("OPTIMIZATION COMPLETE")
        logger.info("=" * 80)

        if validation['valid']:
            logger.info("✓ ALL VALIDATION CHECKS PASSED")
            logger.info(f"✓ Coverage: {validation['coverage_percent']:.4f}%")
            logger.info(f"✓ Cassettes: {len(self.cassettes)}")
            logger.info(f"✓ C-channels: {len(self.c_channels)}")
        else:
            logger.warning("⚠️  Some validation checks failed")
            logger.warning(f"Coverage: {validation['coverage_percent']:.4f}%")

        return self._prepare_results(
            coverage=validation['coverage_percent'],
            cchannel_area=validation['cchannel_area']
        )

    def _prepare_results(self, coverage: float, cchannel_area: float) -> Dict:
        """
        Prepare final results dictionary.

        Args:
            coverage: Coverage percentage
            cchannel_area: C-channel area in sq ft

        Returns:
            Results dictionary
        """
        cassette_area = sum(c['area'] for c in self.cassettes)

        # Extract C-channel widths/heights in inches
        c_channels_inches = []
        for c in self.c_channels:
            if c.get('orientation') == 'vertical':
                c_channels_inches.append(c.get('width_inches', c['width'] * 12))
            else:
                c_channels_inches.append(c.get('height_inches', c['height'] * 12))

        # Convert C-channels to geometry format for visualizer
        cchannel_geometries = []
        for c in self.c_channels:
            cchannel_geometries.append({
                'minx': c['x'],
                'miny': c['y'],
                'maxx': c['x'] + c['width'],
                'maxy': c['y'] + c['height']
            })

        return {
            'cassettes': self.cassettes,
            'c_channels': self.c_channels,
            'c_channels_inches': c_channels_inches,
            'cchannel_geometries': cchannel_geometries,  # Add for visualizer
            'coverage_percent': coverage,
            'polygon': self.polygon,
            'statistics': {
                'total_area': self.polygon_area,
                'cassette_area': cassette_area,
                'cchannel_area': cchannel_area,
                'cassette_count': len(self.cassettes),
                'coverage_percent': coverage,
            }
        }


if __name__ == "__main__":
    # Test with Umbra XL polygon
    umbra_polygon = [
        (0.0, 28.0),
        (55.5, 28.0),
        (55.5, 12.0),
        (16.0, 12.0),
        (16.0, 0.0),
        (0.0, 0.0)
    ]

    print("\nTesting Gap Redistribution Optimizer")
    print("=" * 80)

    optimizer = GapRedistributionOptimizer(umbra_polygon)
    result = optimizer.optimize()

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Coverage: {result['coverage_percent']:.2f}%")
    print(f"Cassettes: {result['statistics']['cassette_count']}")
    print(f"C-channel area: {result['statistics']['cchannel_area']:.2f} sq ft")

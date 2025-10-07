"""
C-Channel Aware Cassette Optimizer
Optimizes cassette placement with C-channel perimeter
"""

from typing import List, Tuple, Dict
from cchannel_utils import (
    create_inset_polygon,
    measure_gaps_per_cardinal_side,
    calculate_cchannel_areas
)
import logging
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CChannelOptimizer:
    """Optimizes cassettes with C-channel perimeter"""

    MIN_CCHANNEL_WIDTH = 1.5 / 12.0  # 1.5 inches in feet
    MAX_CCHANNEL_WIDTH = 18.0 / 12.0  # 18 inches in feet

    def __init__(self, polygon: List[Tuple[float, float]]):
        """
        Initialize optimizer with original polygon.

        Args:
            polygon: List of (x, y) coordinates
        """
        self.original_polygon = polygon
        self.inset_polygon = None
        self.cassettes = None
        self.cchannel_widths = None
        self.cchannel_areas = None
        self.statistics = None

    def _iterative_gap_filling(self, cassettes: List[Dict], polygon: List[Tuple[float, float]]) -> List[Dict]:
        """
        Iteratively fill gaps to achieve 100% coverage.
        Try smallest cassettes in gaps, shift cassettes if needed.
        """
        from shapely.geometry import Polygon, Point
        import numpy as np

        poly = Polygon(polygon)
        min_x = min(p[0] for p in polygon)
        max_x = max(p[0] for p in polygon)
        min_y = min(p[1] for p in polygon)
        max_y = max(p[1] for p in polygon)

        # Minimum cassette size
        min_size = 3.5

        # Cassette sizes to try (smallest to largest)
        filler_sizes = [
            (3.5, 3.5), (3.5, 4.0), (4.0, 4.5), (4.5, 5.0),
            (5.0, 5.5), (5.5, 6.0), (6.0, 6.0)
        ]

        max_iterations = 20
        iteration = 0
        resolution = 0.5

        while iteration < max_iterations:
            # Find uncovered points
            uncovered = []
            y = min_y
            while y <= max_y:
                x = min_x
                while x <= max_x:
                    point = Point(x, y)
                    if poly.contains(point) or poly.touches(point):
                        # Check if covered by any cassette
                        covered = False
                        for c in cassettes:
                            if (c['x'] <= x <= c['x'] + c['width'] and
                                c['y'] <= y <= c['y'] + c['height']):
                                covered = True
                                break
                        if not covered:
                            uncovered.append((x, y))
                    x += resolution
                y += resolution

            if not uncovered:
                logger.info("  Achieved 100% coverage!")
                break

            if iteration == 0:
                logger.info(f"  Found {len(uncovered)} uncovered points, attempting to fill...")

            # Try to place a filler cassette
            placed = False
            for ux, uy in uncovered[:50]:  # Try more uncovered points
                # Try different positions around this uncovered point with finer granularity
                for dx in np.arange(0, -3.5, -0.5):
                    for dy in np.arange(0, -3.5, -0.5):
                        test_x = round((ux + dx) * 2) / 2
                        test_y = round((uy + dy) * 2) / 2

                        # Try each filler size
                        for width, height in filler_sizes:
                            for w, h in [(width, height), (height, width)]:
                                # Check if cassette fits in polygon
                                corners = [
                                    (test_x, test_y),
                                    (test_x + w, test_y),
                                    (test_x + w, test_y + h),
                                    (test_x, test_y + h)
                                ]

                                all_in = all(
                                    poly.contains(Point(cx, cy)) or poly.touches(Point(cx, cy))
                                    for cx, cy in corners
                                )

                                if not all_in:
                                    continue

                                # Check for overlap with existing cassettes
                                overlap = False
                                for c in cassettes:
                                    if not (test_x >= c['x'] + c['width'] or
                                           test_x + w <= c['x'] or
                                           test_y >= c['y'] + c['height'] or
                                           test_y + h <= c['y']):
                                        overlap = True
                                        break

                                if not overlap:
                                    # Place this cassette
                                    new_cassette = {
                                        'x': test_x,
                                        'y': test_y,
                                        'width': w,
                                        'height': h,
                                        'size': f"{w}x{h}",
                                        'area': w * h,
                                        'weight': w * h * 10.4
                                    }
                                    cassettes.append(new_cassette)
                                    placed = True
                                    logger.info(f"  Added {w}x{h} cassette at ({test_x:.1f}, {test_y:.1f})")
                                    break
                            if placed:
                                break
                        if placed:
                            break
                    if placed:
                        break
                if placed:
                    break

            if not placed:
                logger.info(f"  Gap filling iteration {iteration+1}: No more cassettes can be placed")
                logger.info(f"  Remaining uncovered points: {len(uncovered)}")

                # Calculate remaining gap area
                gap_area = len(uncovered) * (resolution ** 2)
                gap_percent = (gap_area / poly.area) * 100
                logger.info(f"  Remaining gap area: ~{gap_area:.1f} sq ft ({gap_percent:.1f}%)")

                # If gaps are very small (<1% of area), consider them acceptable
                if gap_percent < 1.0:
                    logger.info(f"  Gaps are very small (<1%), considering as acceptable tolerance")

                break

            iteration += 1

        return cassettes

    def _calculate_inset_coverage(self, cassettes: List[Dict], polygon: List[Tuple[float, float]]) -> float:
        """Calculate coverage percentage of inset polygon"""
        from shapely.geometry import Polygon, Point

        poly = Polygon(polygon)
        total_area = poly.area

        cassette_area = sum(c['area'] for c in cassettes)

        # Use grid sampling for accurate coverage
        min_x = min(p[0] for p in polygon)
        max_x = max(p[0] for p in polygon)
        min_y = min(p[1] for p in polygon)
        max_y = max(p[1] for p in polygon)

        covered_points = 0
        total_points = 0
        resolution = 0.25

        y = min_y
        while y <= max_y:
            x = min_x
            while x <= max_x:
                point = Point(x, y)
                if poly.contains(point) or poly.touches(point):
                    total_points += 1
                    # Check if covered
                    covered = False
                    for c in cassettes:
                        if (c['x'] <= x <= c['x'] + c['width'] and
                            c['y'] <= y <= c['y'] + c['height']):
                            covered = True
                            break
                    if covered:
                        covered_points += 1
                x += resolution
            y += resolution

        if total_points == 0:
            return 0.0

        return (covered_points / total_points) * 100

    def optimize(self) -> Dict:
        """
        Run optimization with C-channel.

        Returns:
            Dictionary with cassettes, C-channel info, and statistics
        """
        logger.info("Step 1: Creating inset polygon with minimum C-channel (1.5\")")
        try:
            self.inset_polygon = create_inset_polygon(
                self.original_polygon,
                self.MIN_CCHANNEL_WIDTH
            )
        except ValueError as e:
            logger.error(f"Failed to create inset polygon: {e}")
            raise

        logger.info("Step 2: Optimizing cassettes on inset polygon using ultra-smart optimizer")
        from ultra_smart_optimizer import UltraSmartOptimizer

        optimizer = UltraSmartOptimizer(self.inset_polygon)
        result = optimizer.optimize()

        self.cassettes = result['cassettes']
        coverage_percent = result.get('coverage_percent', result.get('coverage', 0) * 100)
        logger.info(f"Placed {len(self.cassettes)} cassettes with {coverage_percent:.1f}% coverage of inset area")

        if coverage_percent < 100.0:
            logger.info("Step 2.5: Iterative adjustment to achieve 100% coverage of inset area")
            self.cassettes = self._iterative_gap_filling(self.cassettes, self.inset_polygon)
            coverage_percent = self._calculate_inset_coverage(self.cassettes, self.inset_polygon)
            logger.info(f"After gap filling: {len(self.cassettes)} cassettes with {coverage_percent:.1f}% coverage")

        logger.info("Step 3: Measuring gaps from cassettes to INSET boundary")
        gaps = measure_gaps_per_cardinal_side(self.inset_polygon, self.cassettes)
        logger.info(f"Gaps: N={gaps['N']*12:.1f}\", S={gaps['S']*12:.1f}\", "
                   f"E={gaps['E']*12:.1f}\", W={gaps['W']*12:.1f}\"")

        logger.info("Step 4: Calculating C-channel widths (initial 1.5\" + gaps)")
        self.cchannel_widths = {}
        for direction, gap_ft in gaps.items():
            total_gap_ft = self.MIN_CCHANNEL_WIDTH + gap_ft
            gap_inches = total_gap_ft * 12.0

            if gap_inches < 1.5:
                width_inches = 1.5
            elif gap_inches <= 1.5:
                width_inches = 1.5
            elif gap_inches <= 2.0:
                width_inches = 2.0
            else:
                width_inches = math.ceil(gap_inches)
                width_inches = min(width_inches, 18.0)

            self.cchannel_widths[direction] = width_inches / 12.0
        logger.info(f"C-channel widths: N={self.cchannel_widths['N']*12:.1f}\", "
                   f"S={self.cchannel_widths['S']*12:.1f}\", "
                   f"E={self.cchannel_widths['E']*12:.1f}\", "
                   f"W={self.cchannel_widths['W']*12:.1f}\"")

        logger.info("Step 5: Validating C-channel widths")
        self._validate_cchannel_widths()

        logger.info("Step 6: Calculating C-channel areas")
        self.cchannel_areas = calculate_cchannel_areas(
            self.original_polygon,
            self.cchannel_widths
        )

        logger.info("Step 7: Calculating final statistics")
        self.statistics = self._calculate_statistics()

        return {
            'cassettes': self.cassettes,
            'cchannel_widths': self.cchannel_widths,
            'cchannel_areas': self.cchannel_areas,
            'statistics': self.statistics,
            'original_polygon': self.original_polygon,
            'inset_polygon': self.inset_polygon
        }

    def _validate_cchannel_widths(self):
        """Validate all C-channel widths are within range"""
        for direction, width_ft in self.cchannel_widths.items():
            width_inches = width_ft * 12.0

            if width_inches < 1.5:
                raise ValueError(
                    f"C-channel {direction} is {width_inches:.1f}\" (< 1.5\" minimum)"
                )

            if width_inches > 18.0:
                raise ValueError(
                    f"C-channel {direction} is {width_inches:.1f}\" (> 18\" maximum)"
                )

            logger.info(f"C-channel {direction}: {width_inches:.1f}\" - VALID")

    def _calculate_statistics(self) -> Dict:
        """Calculate comprehensive statistics"""
        from shapely.geometry import Polygon

        original_poly = Polygon(self.original_polygon)
        total_area = original_poly.area

        cassette_area = sum(c['area'] for c in self.cassettes)
        cchannel_area = self.cchannel_areas['total']

        cassette_counts = {}
        for cassette in self.cassettes:
            size = cassette['size']
            cassette_counts[size] = cassette_counts.get(size, 0) + 1

        total_weight = sum(c['weight'] for c in self.cassettes)

        coverage_percent = ((cassette_area + cchannel_area) / total_area) * 100

        return {
            'total_area': total_area,
            'cassette_area': cassette_area,
            'cassette_percent': (cassette_area / total_area) * 100,
            'cchannel_area': cchannel_area,
            'cchannel_percent': (cchannel_area / total_area) * 100,
            'coverage_percent': coverage_percent,
            'cassette_count': len(self.cassettes),
            'cassette_counts': cassette_counts,
            'total_weight': total_weight,
            'cchannel_widths_inches': {
                k: v * 12.0 for k, v in self.cchannel_widths.items()
            },
            'cchannel_per_side': self.cchannel_areas['per_side'],
            'cchannel_corners': self.cchannel_areas['corners']
        }

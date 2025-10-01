#!/usr/bin/env python3
"""
Coverage Analyzer Stage
========================
Analyzes why target coverage cannot be achieved and suggests improvements.
"""

import logging
import math
from typing import List, Tuple, Dict, Any

from optimization_pipeline import PipelineStage, PipelineContext, Cassette

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CoverageAnalyzer(PipelineStage):
    """
    Analyzes coverage limitations and suggests improvements.

    Features:
    1. Identifies structural limitations
    2. Calculates theoretical maximum coverage
    3. Suggests polygon modifications
    4. Provides detailed gap analysis
    5. Recommends cassette size optimizations
    """

    def __init__(self, target_coverage: float = 94.0):
        """
        Initialize CoverageAnalyzer.

        Args:
            target_coverage: Target coverage percentage
        """
        self.target_coverage = target_coverage

    @property
    def name(self) -> str:
        return "CoverageAnalyzer"

    def process(self, context: PipelineContext) -> PipelineContext:
        """Analyze coverage limitations and provide recommendations."""
        polygon = context.polygon
        cassettes = context.cassettes
        current_coverage = context.get_coverage()

        analysis = {
            'current_coverage': current_coverage,
            'target_coverage': self.target_coverage,
            'meets_target': current_coverage >= self.target_coverage,
            'shortfall': max(0, self.target_coverage - current_coverage)
        }

        # Analyze polygon characteristics
        polygon_analysis = self._analyze_polygon(polygon)
        analysis['polygon_analysis'] = polygon_analysis

        # Identify problem areas
        problem_areas = self._identify_problem_areas(polygon, cassettes)
        analysis['problem_areas'] = problem_areas

        # Calculate theoretical maximum
        theoretical_max = self._calculate_theoretical_maximum(polygon)
        analysis['theoretical_maximum'] = theoretical_max

        # Generate recommendations
        recommendations = self._generate_recommendations(
            polygon, cassettes, current_coverage, theoretical_max, problem_areas
        )
        analysis['recommendations'] = recommendations

        # Store analysis in metadata
        context.metadata['coverage_analysis'] = analysis

        # Log analysis results
        self._log_analysis(analysis)

        return context

    def _analyze_polygon(self, polygon: List[Tuple[float, float]]) -> Dict[str, Any]:
        """
        Analyze polygon characteristics that affect coverage.

        Args:
            polygon: Building polygon

        Returns:
            Polygon analysis dictionary
        """
        n = len(polygon)

        # Calculate area and perimeter
        area = self._calculate_area(polygon)
        perimeter = self._calculate_perimeter(polygon)

        # Find narrow sections
        narrow_sections = self._find_narrow_sections(polygon)

        # Find small protrusions
        protrusions = self._find_protrusions(polygon)

        # Calculate complexity score
        complexity = perimeter / (2 * math.sqrt(math.pi * area))  # Circularity measure

        # Check for rectilinearity
        is_rectilinear = self._is_rectilinear(polygon)

        return {
            'vertices': n,
            'area': area,
            'perimeter': perimeter,
            'complexity': complexity,
            'is_rectilinear': is_rectilinear,
            'narrow_sections': narrow_sections,
            'protrusions': protrusions,
            'perimeter_to_area_ratio': perimeter / area if area > 0 else 0
        }

    def _identify_problem_areas(self, polygon: List[Tuple[float, float]],
                               cassettes: List[Cassette]) -> List[Dict[str, Any]]:
        """
        Identify specific areas preventing full coverage.

        Args:
            polygon: Building polygon
            cassettes: Placed cassettes

        Returns:
            List of problem area descriptions
        """
        problems = []

        # Find gaps
        gaps = self._find_gaps(polygon, cassettes)
        for gap in gaps:
            if gap['area'] < 8.0:  # Smaller than smallest cassette
                problems.append({
                    'type': 'unfillable_gap',
                    'area': gap['area'],
                    'location': gap['center'],
                    'reason': f"Gap of {gap['area']:.1f} sq ft is smaller than minimum cassette (8 sq ft)"
                })

        # Find edge irregularities
        edge_issues = self._find_edge_issues(polygon)
        for issue in edge_issues:
            problems.append({
                'type': 'edge_irregularity',
                'location': issue['location'],
                'reason': issue['description']
            })

        return problems

    def _calculate_theoretical_maximum(self, polygon: List[Tuple[float, float]]) -> float:
        """
        Calculate theoretical maximum coverage for the polygon.

        Args:
            polygon: Building polygon

        Returns:
            Theoretical maximum coverage percentage
        """
        area = self._calculate_area(polygon)
        perimeter = self._calculate_perimeter(polygon)
        n_vertices = len(polygon)

        # Edge loss: minimum 0.5 ft buffer from edges
        edge_buffer = 0.5
        edge_loss_area = perimeter * edge_buffer

        # Corner loss: additional area lost at corners
        # Assume average 2 sq ft lost per corner
        corner_loss_area = n_vertices * 2.0

        # Minimum cassette constraint
        # Areas smaller than 8 sq ft (2x4) cannot be filled
        min_cassette_area = 8.0

        # Calculate usable area
        usable_area = area - edge_loss_area - corner_loss_area

        # Account for geometric inefficiency
        # Complex shapes have lower packing efficiency
        complexity = perimeter / (2 * math.sqrt(math.pi * area))
        efficiency_factor = 1.0 / (1.0 + 0.1 * (complexity - 1.0))

        # Calculate theoretical maximum
        theoretical_area = usable_area * efficiency_factor
        theoretical_coverage = (theoretical_area / area * 100) if area > 0 else 0

        # Cap at realistic maximum
        return min(theoretical_coverage, 96.0)

    def _generate_recommendations(self, polygon: List[Tuple[float, float]],
                                 cassettes: List[Cassette],
                                 current_coverage: float,
                                 theoretical_max: float,
                                 problem_areas: List[Dict]) -> List[str]:
        """
        Generate actionable recommendations.

        Args:
            polygon: Building polygon
            cassettes: Placed cassettes
            current_coverage: Current coverage percentage
            theoretical_max: Theoretical maximum coverage
            problem_areas: Identified problem areas

        Returns:
            List of recommendations
        """
        recommendations = []

        # Check if target is achievable
        if theoretical_max < self.target_coverage:
            recommendations.append(
                f"⚠️ CRITICAL: Theoretical maximum coverage ({theoretical_max:.1f}%) "
                f"is below target ({self.target_coverage}%). Polygon modification required."
            )

        # Polygon modifications
        polygon_analysis = self._analyze_polygon(polygon)

        if polygon_analysis['protrusions']:
            recommendations.append(
                f"📐 Remove {len(polygon_analysis['protrusions'])} small protrusions "
                f"to simplify polygon and improve coverage"
            )

        if polygon_analysis['narrow_sections']:
            recommendations.append(
                f"🔧 Widen {len(polygon_analysis['narrow_sections'])} narrow sections "
                f"(< 2 ft) to accommodate smallest cassettes"
            )

        # Cassette optimization
        size_distribution = self._get_size_distribution(cassettes)
        small_cassette_ratio = sum(count for size, count in size_distribution.items()
                                  if int(size.split('x')[0]) * int(size.split('x')[1]) <= 12) / len(cassettes)

        if small_cassette_ratio > 0.3:
            recommendations.append(
                f"🔄 High ratio of small cassettes ({small_cassette_ratio*100:.0f}%). "
                f"Consider algorithmic improvements to place larger cassettes"
            )

        # Gap analysis
        unfillable_gaps = [p for p in problem_areas if p['type'] == 'unfillable_gap']
        if unfillable_gaps:
            total_gap_area = sum(p['area'] for p in unfillable_gaps)
            recommendations.append(
                f"🕳️ {len(unfillable_gaps)} unfillable gaps totaling {total_gap_area:.1f} sq ft. "
                f"These gaps are too small for standard cassettes"
            )

        # Performance optimization
        efficiency = current_coverage / theoretical_max * 100 if theoretical_max > 0 else 0
        if efficiency < 90:
            recommendations.append(
                f"⚙️ Algorithm efficiency is {efficiency:.0f}%. "
                f"Consider more iterations or different placement strategies"
            )

        # Specific modifications
        if polygon_analysis['complexity'] > 1.5:
            recommendations.append(
                f"📏 Polygon complexity score: {polygon_analysis['complexity']:.2f}. "
                f"Simplifying to a more rectangular shape would improve coverage"
            )

        return recommendations

    def _find_narrow_sections(self, polygon: List[Tuple[float, float]]) -> List[Dict[str, Any]]:
        """Find sections of polygon narrower than minimum cassette width."""
        narrow_sections = []
        min_width = 2.0  # Minimum cassette width

        # Simplified check - would need more sophisticated algorithm for production
        for i, vertex in enumerate(polygon):
            # Check distance to non-adjacent edges
            for j in range(len(polygon)):
                if abs(i - j) <= 1 or abs(i - j) >= len(polygon) - 1:
                    continue

                dist = self._point_to_line_distance(
                    vertex,
                    polygon[j],
                    polygon[(j + 1) % len(polygon)]
                )

                if dist < min_width and dist > 0:
                    narrow_sections.append({
                        'vertex_index': i,
                        'width': dist,
                        'location': vertex
                    })

        return narrow_sections

    def _find_protrusions(self, polygon: List[Tuple[float, float]]) -> List[Dict[str, Any]]:
        """Find small protrusions that reduce coverage efficiency."""
        protrusions = []

        # Look for vertices that create small areas
        n = len(polygon)
        for i in range(n):
            # Check if removing this vertex would improve coverage
            # This is a simplified check
            prev_idx = (i - 1) % n
            next_idx = (i + 1) % n

            # Calculate area of triangle formed by these three vertices
            triangle_area = abs(
                (polygon[next_idx][0] - polygon[prev_idx][0]) *
                (polygon[i][1] - polygon[prev_idx][1]) -
                (polygon[i][0] - polygon[prev_idx][0]) *
                (polygon[next_idx][1] - polygon[prev_idx][1])
            ) / 2.0

            if triangle_area < 4.0:  # Small protrusion
                protrusions.append({
                    'vertex_index': i,
                    'area': triangle_area,
                    'location': polygon[i]
                })

        return protrusions

    def _find_gaps(self, polygon: List[Tuple[float, float]],
                  cassettes: List[Cassette]) -> List[Dict[str, Any]]:
        """Find gaps in coverage."""
        # Simplified gap detection
        gaps = []

        # Get bounding box
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # Grid-based gap detection
        grid_size = 2.0
        y = min_y
        while y < max_y:
            x = min_x
            while x < max_x:
                center = (x + grid_size/2, y + grid_size/2)

                if self._point_in_polygon(center, polygon):
                    covered = False
                    for cassette in cassettes:
                        if (cassette.x <= center[0] <= cassette.x + cassette.width and
                            cassette.y <= center[1] <= cassette.y + cassette.height):
                            covered = True
                            break

                    if not covered:
                        gaps.append({
                            'center': center,
                            'area': grid_size * grid_size
                        })

                x += grid_size
            y += grid_size

        return gaps

    def _find_edge_issues(self, polygon: List[Tuple[float, float]]) -> List[Dict[str, Any]]:
        """Find edge-related issues."""
        issues = []

        n = len(polygon)
        for i in range(n):
            j = (i + 1) % n
            edge_length = self._distance(polygon[i], polygon[j])

            if edge_length < 2.0:
                issues.append({
                    'location': polygon[i],
                    'description': f"Edge {i}->{j} is {edge_length:.1f} ft, less than minimum cassette width"
                })

        return issues

    def _get_size_distribution(self, cassettes: List[Cassette]) -> Dict[str, int]:
        """Get cassette size distribution."""
        distribution = {}
        for cassette in cassettes:
            size = cassette.size
            distribution[size] = distribution.get(size, 0) + 1
        return distribution

    def _log_analysis(self, analysis: Dict[str, Any]):
        """Log analysis results."""
        logger.info("=" * 60)
        logger.info("COVERAGE ANALYSIS REPORT")
        logger.info("=" * 60)

        logger.info(f"Current Coverage: {analysis['current_coverage']:.1f}%")
        logger.info(f"Target Coverage: {analysis['target_coverage']:.1f}%")
        logger.info(f"Theoretical Maximum: {analysis['theoretical_maximum']:.1f}%")

        if analysis['meets_target']:
            logger.info("✅ Target coverage achieved!")
        else:
            logger.info(f"❌ {analysis['shortfall']:.1f}% short of target")

        polygon_info = analysis['polygon_analysis']
        logger.info(f"\nPolygon Characteristics:")
        logger.info(f"  • Area: {polygon_info['area']:.1f} sq ft")
        logger.info(f"  • Perimeter: {polygon_info['perimeter']:.1f} ft")
        logger.info(f"  • Complexity: {polygon_info['complexity']:.2f}")
        logger.info(f"  • Vertices: {polygon_info['vertices']}")

        if analysis['problem_areas']:
            logger.info(f"\nProblem Areas: {len(analysis['problem_areas'])} identified")
            for problem in analysis['problem_areas'][:3]:  # Show top 3
                logger.info(f"  • {problem['reason']}")

        if analysis['recommendations']:
            logger.info(f"\nRecommendations:")
            for rec in analysis['recommendations']:
                logger.info(f"  {rec}")

    def _calculate_area(self, polygon: List[Tuple[float, float]]) -> float:
        """Calculate polygon area."""
        n = len(polygon)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += polygon[i][0] * polygon[j][1]
            area -= polygon[j][0] * polygon[i][1]
        return abs(area) / 2.0

    def _calculate_perimeter(self, polygon: List[Tuple[float, float]]) -> float:
        """Calculate polygon perimeter."""
        perimeter = 0.0
        n = len(polygon)
        for i in range(n):
            j = (i + 1) % n
            perimeter += self._distance(polygon[i], polygon[j])
        return perimeter

    def _distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Calculate distance between two points."""
        return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

    def _point_to_line_distance(self, point: Tuple[float, float],
                               line_start: Tuple[float, float],
                               line_end: Tuple[float, float]) -> float:
        """Calculate distance from point to line segment."""
        # Vector from line_start to line_end
        line_vec = (line_end[0] - line_start[0], line_end[1] - line_start[1])
        line_len = math.sqrt(line_vec[0]**2 + line_vec[1]**2)

        if line_len == 0:
            return self._distance(point, line_start)

        # Normalize line vector
        line_norm = (line_vec[0] / line_len, line_vec[1] / line_len)

        # Vector from line_start to point
        point_vec = (point[0] - line_start[0], point[1] - line_start[1])

        # Project point onto line
        t = max(0, min(line_len, point_vec[0] * line_norm[0] + point_vec[1] * line_norm[1]))

        # Find closest point on line segment
        closest = (line_start[0] + t * line_norm[0], line_start[1] + t * line_norm[1])

        return self._distance(point, closest)

    def _is_rectilinear(self, polygon: List[Tuple[float, float]]) -> bool:
        """Check if polygon is rectilinear (all edges horizontal/vertical)."""
        n = len(polygon)
        for i in range(n):
            j = (i + 1) % n
            dx = abs(polygon[j][0] - polygon[i][0])
            dy = abs(polygon[j][1] - polygon[i][1])

            # Check if edge is not horizontal or vertical
            if dx > 0.1 and dy > 0.1:
                return False

        return True

    def _point_in_polygon(self, point: Tuple[float, float],
                         polygon: List[Tuple[float, float]]) -> bool:
        """Check if point is inside polygon."""
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


def test_coverage_analyzer():
    """Test the CoverageAnalyzer on Bungalow."""
    from test_bungalow_complete import create_full_pipeline, get_bungalow_polygon

    # Get Bungalow polygon
    polygon = get_bungalow_polygon()

    # Create pipeline with analyzer
    pipeline = create_full_pipeline()
    pipeline.add_stage(CoverageAnalyzer())

    # Run optimization
    results = pipeline.optimize(polygon)

    # Get analysis
    if 'coverage_analysis' in results.get('stage_results', {}).get('CoverageAnalyzer', {}).get('metadata', {}):
        analysis = results['stage_results']['CoverageAnalyzer']['metadata']['coverage_analysis']

        print("\n" + "="*60)
        print("COVERAGE ANALYSIS SUMMARY")
        print("="*60)
        print(f"Can achieve 94%: {'YES' if analysis['theoretical_maximum'] >= 94 else 'NO'}")
        print(f"Theoretical max: {analysis['theoretical_maximum']:.1f}%")
        print(f"Current: {analysis['current_coverage']:.1f}%")
        print(f"Efficiency: {(analysis['current_coverage']/analysis['theoretical_maximum']*100):.0f}%")

    return results


if __name__ == "__main__":
    test_coverage_analyzer()
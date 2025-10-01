"""
Coverage Analyzer Module
========================
Analyzes cassette coverage and identifies gaps for custom work.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

from cassette_models import CassetteLayout, Cassette, Point, FloorBoundary
from floor_geometry import Rectangle, GeometryUtils
from config_cassette import CassetteConfig

logger = logging.getLogger(__name__)


class CoverageAnalyzer:
    """Analyzes coverage and identifies gaps in cassette layouts."""
    
    def __init__(self):
        """Initialize coverage analyzer."""
        self.config = CassetteConfig()
        self.min_custom_area = self.config.OPTIMIZATION['min_custom_area']
        
    def analyze(self, layout: CassetteLayout) -> Dict:
        """
        Perform complete coverage analysis.
        
        Args:
            layout: Cassette layout to analyze
            
        Returns:
            Dictionary with analysis results
        """
        # Calculate basic metrics
        coverage_percentage = layout.coverage_percentage
        covered_area = layout.covered_area
        uncovered_area = layout.uncovered_area
        
        # Find gaps
        gaps = self.identify_gaps(layout)
        
        # Classify gaps
        gap_classification = self.classify_gaps(gaps)
        
        # Calculate custom work requirements
        custom_work = self.calculate_custom_work(gaps)
        
        # Generate heatmap data
        heatmap = self.generate_coverage_heatmap(layout)
        
        # Check coverage targets
        targets_met = self.check_coverage_targets(coverage_percentage)
        
        results = {
            'coverage_percentage': coverage_percentage,
            'covered_area': covered_area,
            'uncovered_area': uncovered_area,
            'gaps': gaps,
            'gap_classification': gap_classification,
            'custom_work': custom_work,
            'heatmap': heatmap,
            'targets_met': targets_met,
            'summary': self._generate_summary(
                coverage_percentage, len(gaps), custom_work
            )
        }
        
        logger.info(f"Coverage analysis: {coverage_percentage:.1f}% coverage, "
                   f"{len(gaps)} gaps, {custom_work['total_area']:.1f} sq ft custom work")
        
        return results
    
    def identify_gaps(self, layout: CassetteLayout) -> List[Rectangle]:
        """
        Identify uncovered gaps in the layout.
        
        Args:
            layout: Cassette layout
            
        Returns:
            List of rectangles representing gaps
        """
        boundary = layout.floor_boundary
        cassettes = layout.cassettes
        
        # Create grid for gap detection
        grid_size = 1.0  # 1 ft resolution
        
        gaps = []
        gap_points = []
        
        # Scan through floor area
        y = 0
        while y < boundary.height:
            x = 0
            while x < boundary.width:
                point = Point(x + grid_size/2, y + grid_size/2)
                
                # Check if point is in floor boundary
                if boundary.contains_point(point):
                    # Check if point is covered by any cassette
                    covered = False
                    for cassette in cassettes:
                        if cassette.contains_point(point):
                            covered = True
                            break
                    
                    if not covered:
                        gap_points.append(Point(x, y))
                
                x += grid_size
            y += grid_size
        
        # Group adjacent gap points into rectangles
        if gap_points:
            gaps = self._group_points_to_rectangles(gap_points, grid_size)
        
        # Filter out very small gaps
        significant_gaps = [
            gap for gap in gaps 
            if gap.area >= self.min_custom_area
        ]
        
        logger.debug(f"Found {len(significant_gaps)} significant gaps")
        
        return significant_gaps
    
    def _group_points_to_rectangles(self, points: List[Point], 
                                   grid_size: float) -> List[Rectangle]:
        """Group adjacent points into rectangles."""
        if not points:
            return []
        
        rectangles = []
        used = set()
        
        for point in points:
            if point in used:
                continue
            
            # Start a new rectangle from this point
            min_x = point.x
            min_y = point.y
            max_x = point.x + grid_size
            max_y = point.y + grid_size
            
            # Expand rectangle to include adjacent points
            expanded = True
            while expanded:
                expanded = False
                
                for other in points:
                    if other in used:
                        continue
                    
                    # Check if point is adjacent to current rectangle
                    if (min_x - grid_size <= other.x <= max_x and
                        min_y - grid_size <= other.y <= max_y):
                        
                        min_x = min(min_x, other.x)
                        min_y = min(min_y, other.y)
                        max_x = max(max_x, other.x + grid_size)
                        max_y = max(max_y, other.y + grid_size)
                        
                        used.add(other)
                        expanded = True
            
            used.add(point)
            
            # Create rectangle
            rect = Rectangle(min_x, min_y, max_x - min_x, max_y - min_y)
            rectangles.append(rect)
        
        # Merge overlapping rectangles
        merged = GeometryUtils._merge_rectangles(rectangles)
        
        return merged
    
    def classify_gaps(self, gaps: List[Rectangle]) -> Dict:
        """
        Classify gaps by size and location.
        
        Args:
            gaps: List of gap rectangles
            
        Returns:
            Classification dictionary
        """
        classification = {
            'edge_gaps': [],
            'corner_gaps': [],
            'interior_gaps': [],
            'large_gaps': [],
            'small_gaps': []
        }
        
        for gap in gaps:
            # Classify by size
            if gap.area >= 16:  # >= 4x4
                classification['large_gaps'].append(gap)
            else:
                classification['small_gaps'].append(gap)
            
            # Classify by location (simplified for MVP)
            # In production, would check actual position relative to boundary
            if gap.width < 3 or gap.height < 3:
                classification['edge_gaps'].append(gap)
            else:
                classification['interior_gaps'].append(gap)
        
        return classification
    
    def calculate_custom_work(self, gaps: List[Rectangle]) -> Dict:
        """
        Calculate custom work requirements.
        
        Args:
            gaps: List of gap rectangles
            
        Returns:
            Custom work details
        """
        total_area = sum(gap.area for gap in gaps)
        
        # Estimate joist lengths needed
        joist_lengths = []
        for gap in gaps:
            # Assume joists run along shorter dimension
            if gap.width < gap.height:
                joist_length = gap.width
                joist_count = int(gap.height / 1.33)  # 16" spacing
            else:
                joist_length = gap.height
                joist_count = int(gap.width / 1.33)  # 16" spacing
            
            for _ in range(joist_count):
                joist_lengths.append(joist_length)
        
        # Calculate total linear feet
        total_linear_feet = sum(joist_lengths)
        
        # Estimate weight (approximate)
        weight_per_linear_ft = 2.5  # lbs per linear foot
        total_weight = total_linear_feet * weight_per_linear_ft
        
        return {
            'total_area': total_area,
            'gap_count': len(gaps),
            'total_linear_feet': total_linear_feet,
            'estimated_weight': total_weight,
            'joist_lengths': sorted(joist_lengths, reverse=True),
            'percentage_of_floor': (total_area / (total_area + sum(c.area for c in gaps)) * 100) 
                                  if gaps else 0
        }
    
    def generate_coverage_heatmap(self, layout: CassetteLayout) -> np.ndarray:
        """
        Generate coverage heatmap data.
        
        Args:
            layout: Cassette layout
            
        Returns:
            2D array representing coverage (1=covered, 0=uncovered)
        """
        # Create grid
        resolution = 2  # 2 ft grid
        width_cells = int(layout.floor_boundary.width / resolution) + 1
        height_cells = int(layout.floor_boundary.height / resolution) + 1
        
        heatmap = np.zeros((height_cells, width_cells))
        
        # Mark covered cells
        for j in range(height_cells):
            for i in range(width_cells):
                x = i * resolution + resolution / 2
                y = j * resolution + resolution / 2
                point = Point(x, y)
                
                # Check if in boundary
                if layout.floor_boundary.contains_point(point):
                    # Check if covered by cassette
                    for cassette in layout.cassettes:
                        if cassette.contains_point(point):
                            heatmap[j, i] = 1
                            break
        
        return heatmap
    
    def check_coverage_targets(self, coverage_percentage: float) -> Dict:
        """
        Check if coverage meets various targets.
        
        Args:
            coverage_percentage: Current coverage percentage
            
        Returns:
            Dictionary with target status
        """
        targets = {
            'minimum_met': coverage_percentage >= self.config.OPTIMIZATION['min_coverage'] * 100,
            'target_met': coverage_percentage >= self.config.OPTIMIZATION['target_coverage'] * 100,
            'ideal_met': coverage_percentage >= self.config.OPTIMIZATION['ideal_coverage'] * 100,
            'gap_to_minimum': max(0, self.config.OPTIMIZATION['min_coverage'] * 100 - coverage_percentage),
            'gap_to_target': max(0, self.config.OPTIMIZATION['target_coverage'] * 100 - coverage_percentage),
            'gap_to_ideal': max(0, self.config.OPTIMIZATION['ideal_coverage'] * 100 - coverage_percentage)
        }
        
        return targets
    
    def _generate_summary(self, coverage: float, gap_count: int, 
                         custom_work: Dict) -> str:
        """Generate text summary of coverage analysis."""
        lines = [
            f"Coverage: {coverage:.1f}%",
            f"Gaps: {gap_count}",
            f"Custom Area: {custom_work['total_area']:.1f} sq ft",
            f"Custom Percentage: {custom_work.get('percentage_of_floor', 0):.1f}%"
        ]
        
        if coverage >= self.config.OPTIMIZATION['min_coverage'] * 100:
            lines.append("Status: PASS - Minimum coverage achieved")
        else:
            gap = self.config.OPTIMIZATION['min_coverage'] * 100 - coverage
            lines.append(f"Status: FAIL - {gap:.1f}% below minimum")
        
        return "\n".join(lines)
    
    def recommend_improvements(self, layout: CassetteLayout, 
                              gaps: List[Rectangle]) -> List[str]:
        """
        Recommend improvements to achieve better coverage.
        
        Args:
            layout: Current layout
            gaps: Identified gaps
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Check for large gaps that could fit cassettes
        for gap in gaps:
            if gap.area >= 16:  # Could fit 4x4 cassette
                if gap.width >= 4 and gap.height >= 4:
                    recommendations.append(
                        f"Gap at ({gap.x:.1f}, {gap.y:.1f}) could fit a 4x4 cassette"
                    )
                elif gap.width >= 6 and gap.height >= 2:
                    recommendations.append(
                        f"Gap at ({gap.x:.1f}, {gap.y:.1f}) could fit a 6x2 cassette"
                    )
                elif gap.width >= 2 and gap.height >= 6:
                    recommendations.append(
                        f"Gap at ({gap.x:.1f}, {gap.y:.1f}) could fit a 2x6 cassette"
                    )
        
        # Check cassette density
        cassette_density = len(layout.cassettes) / layout.floor_boundary.area
        if cassette_density < 0.02:  # Less than 1 cassette per 50 sq ft
            recommendations.append(
                "Consider using smaller cassettes for better coverage"
            )
        
        # Check edge utilization
        edge_cassette_count = sum(
            1 for c in layout.cassettes 
            if c.size.area <= 12  # Edge cassettes
        )
        if edge_cassette_count < len(layout.cassettes) * 0.1:
            recommendations.append(
                "Consider using more edge cassettes to fill gaps"
            )
        
        return recommendations
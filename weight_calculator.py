"""
Weight Distribution Calculator and Validator
============================================
Calculates and validates weight distribution for cassette layouts.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

from cassette_models import CassetteLayout, Cassette
from config_cassette import CassetteConfig

logger = logging.getLogger(__name__)


class WeightCalculator:
    """Calculates and validates weight distribution."""
    
    def __init__(self):
        """Initialize weight calculator."""
        self.config = CassetteConfig()
        self.max_cassette_weight = self.config.MAX_CASSETTE_WEIGHT
        self.weight_per_sqft = self.config.WEIGHT_PER_SQFT
        
    def calculate_weight_distribution(self, layout: CassetteLayout) -> Dict:
        """
        Calculate comprehensive weight distribution analysis.
        
        Args:
            layout: Cassette layout
            
        Returns:
            Dictionary with weight distribution data
        """
        # Basic metrics
        total_weight = layout.total_weight
        cassette_count = layout.cassette_count
        avg_weight = total_weight / cassette_count if cassette_count > 0 else 0
        
        # Weight by size
        weight_by_size = self._calculate_weight_by_size(layout)
        
        # Weight distribution grid
        weight_grid = self._create_weight_grid(layout)
        
        # Center of gravity
        center_of_gravity = self._calculate_center_of_gravity(layout)
        
        # Weight zones
        weight_zones = self._analyze_weight_zones(layout, weight_grid)
        
        # Transport requirements
        transport = self._calculate_transport_requirements(layout)
        
        # Validation results
        validation = self.validate_weights(layout)
        
        return {
            'total_weight': total_weight,
            'average_weight': avg_weight,
            'min_weight': min(c.weight for c in layout.cassettes) if layout.cassettes else 0,
            'max_weight': max(c.weight for c in layout.cassettes) if layout.cassettes else 0,
            'weight_by_size': weight_by_size,
            'weight_grid': weight_grid,
            'center_of_gravity': center_of_gravity,
            'weight_zones': weight_zones,
            'transport': transport,
            'validation': validation
        }
    
    def _calculate_weight_by_size(self, layout: CassetteLayout) -> Dict:
        """Calculate weight breakdown by cassette size."""
        weight_by_size = {}
        
        for cassette in layout.cassettes:
            size_name = cassette.size.name
            if size_name not in weight_by_size:
                weight_by_size[size_name] = {
                    'count': 0,
                    'unit_weight': cassette.weight,
                    'total_weight': 0
                }
            weight_by_size[size_name]['count'] += 1
            weight_by_size[size_name]['total_weight'] += cassette.weight
        
        return weight_by_size
    
    def _create_weight_grid(self, layout: CassetteLayout, 
                           grid_size: float = 4.0) -> np.ndarray:
        """
        Create weight density grid for visualization.
        
        Args:
            layout: Cassette layout
            grid_size: Grid cell size in feet
            
        Returns:
            2D array of weight per grid cell
        """
        boundary = layout.floor_boundary
        
        # Create grid dimensions
        width_cells = int(np.ceil(boundary.width / grid_size))
        height_cells = int(np.ceil(boundary.height / grid_size))
        
        weight_grid = np.zeros((height_cells, width_cells))
        
        # Calculate weight per cell
        for cassette in layout.cassettes:
            # Find cells covered by cassette
            start_x = int(cassette.x / grid_size)
            end_x = int(np.ceil((cassette.x + cassette.width) / grid_size))
            start_y = int(cassette.y / grid_size)
            end_y = int(np.ceil((cassette.y + cassette.height) / grid_size))
            
            # Calculate cells covered
            cells_covered = (end_x - start_x) * (end_y - start_y)
            if cells_covered > 0:
                weight_per_cell = cassette.weight / cells_covered
                
                # Add weight to each cell
                for j in range(start_y, min(end_y, height_cells)):
                    for i in range(start_x, min(end_x, width_cells)):
                        weight_grid[j, i] += weight_per_cell
        
        return weight_grid
    
    def _calculate_center_of_gravity(self, layout: CassetteLayout) -> Tuple[float, float]:
        """
        Calculate center of gravity for the layout.
        
        Args:
            layout: Cassette layout
            
        Returns:
            Tuple of (x, y) coordinates for center of gravity
        """
        if not layout.cassettes:
            return (0, 0)
        
        total_weight = 0
        weighted_x = 0
        weighted_y = 0
        
        for cassette in layout.cassettes:
            # Use cassette center
            center_x = cassette.x + cassette.width / 2
            center_y = cassette.y + cassette.height / 2
            
            weighted_x += center_x * cassette.weight
            weighted_y += center_y * cassette.weight
            total_weight += cassette.weight
        
        if total_weight > 0:
            cog_x = weighted_x / total_weight
            cog_y = weighted_y / total_weight
        else:
            cog_x = cog_y = 0
        
        return (cog_x, cog_y)
    
    def _analyze_weight_zones(self, layout: CassetteLayout, 
                             weight_grid: np.ndarray) -> Dict:
        """
        Analyze weight distribution by zones.
        
        Args:
            layout: Cassette layout
            weight_grid: Weight distribution grid
            
        Returns:
            Dictionary with zone analysis
        """
        height, width = weight_grid.shape
        
        # Divide into quadrants
        mid_x = width // 2
        mid_y = height // 2
        
        zones = {
            'bottom_left': np.sum(weight_grid[:mid_y, :mid_x]),
            'bottom_right': np.sum(weight_grid[:mid_y, mid_x:]),
            'top_left': np.sum(weight_grid[mid_y:, :mid_x]),
            'top_right': np.sum(weight_grid[mid_y:, mid_x:]),
        }
        
        # Calculate balance metrics
        horizontal_balance = abs(
            (zones['bottom_left'] + zones['top_left']) -
            (zones['bottom_right'] + zones['top_right'])
        )
        
        vertical_balance = abs(
            (zones['bottom_left'] + zones['bottom_right']) -
            (zones['top_left'] + zones['top_right'])
        )
        
        total_weight = np.sum(weight_grid)
        
        return {
            'zones': zones,
            'horizontal_balance': horizontal_balance,
            'vertical_balance': vertical_balance,
            'balance_ratio': max(horizontal_balance, vertical_balance) / total_weight 
                           if total_weight > 0 else 0,
            'is_balanced': max(horizontal_balance, vertical_balance) < total_weight * 0.2
        }
    
    def _calculate_transport_requirements(self, layout: CassetteLayout) -> Dict:
        """
        Calculate transportation requirements.
        
        Args:
            layout: Cassette layout
            
        Returns:
            Dictionary with transport requirements
        """
        # Categorize by handling requirements
        crane_required = []
        forklift_required = []
        manual_possible = []
        
        for cassette in layout.cassettes:
            if cassette.weight > 400:
                crane_required.append(cassette.cassette_id)
            elif cassette.weight > 200:
                forklift_required.append(cassette.cassette_id)
            else:
                manual_possible.append(cassette.cassette_id)
        
        # Estimate truck loads (assuming 40,000 lbs per truck)
        truck_capacity = 40000
        truck_loads = int(np.ceil(layout.total_weight / truck_capacity))
        
        # Crew requirements
        heavy_count = len(crane_required) + len(forklift_required)
        crew_size = 3 if heavy_count > 10 else 2
        
        return {
            'crane_required': len(crane_required),
            'forklift_required': len(forklift_required),
            'manual_possible': len(manual_possible),
            'truck_loads': truck_loads,
            'recommended_crew_size': crew_size,
            'equipment_needed': self._get_equipment_list(
                len(crane_required), 
                len(forklift_required)
            )
        }
    
    def _get_equipment_list(self, crane_count: int, forklift_count: int) -> List[str]:
        """Get list of required equipment."""
        equipment = []
        
        if crane_count > 0:
            equipment.append(f"Mobile crane (for {crane_count} cassettes)")
        
        if forklift_count > 0:
            equipment.append(f"Forklift (for {forklift_count} cassettes)")
        
        equipment.append("Lifting straps and rigging")
        equipment.append("Safety equipment (hard hats, gloves)")
        
        return equipment
    
    def validate_weights(self, layout: CassetteLayout) -> Dict:
        """
        Validate weight constraints.
        
        Args:
            layout: Cassette layout
            
        Returns:
            Validation results
        """
        violations = []
        warnings = []
        
        # Check individual cassette weights
        for cassette in layout.cassettes:
            if cassette.weight > self.max_cassette_weight:
                violations.append(
                    f"Cassette {cassette.cassette_id} exceeds max weight: "
                    f"{cassette.weight} > {self.max_cassette_weight} lbs"
                )
            elif cassette.weight > self.max_cassette_weight * 0.9:
                warnings.append(
                    f"Cassette {cassette.cassette_id} near weight limit: "
                    f"{cassette.weight} lbs"
                )
        
        # Check total weight
        max_total = self.config.VALIDATION.get('max_total_weight', 50000)
        if layout.total_weight > max_total:
            violations.append(
                f"Total weight exceeds limit: {layout.total_weight} > {max_total} lbs"
            )
        
        # Check weight distribution
        weight_grid = self._create_weight_grid(layout, grid_size=2.0)
        max_concentration = np.max(weight_grid)
        
        if max_concentration > 1000:  # 1000 lbs per 4 sq ft
            warnings.append(
                f"High weight concentration detected: {max_concentration:.0f} lbs/4sqft"
            )
        
        return {
            'valid': len(violations) == 0,
            'violations': violations,
            'warnings': warnings,
            'max_concentration': max_concentration,
            'summary': 'PASS' if len(violations) == 0 else 'FAIL'
        }
    
    def generate_weight_report(self, layout: CassetteLayout) -> str:
        """
        Generate text report of weight analysis.
        
        Args:
            layout: Cassette layout
            
        Returns:
            Text report
        """
        analysis = self.calculate_weight_distribution(layout)
        
        lines = [
            "WEIGHT DISTRIBUTION ANALYSIS",
            "=" * 40,
            f"Total Weight: {analysis['total_weight']:,.0f} lbs",
            f"Average per Cassette: {analysis['average_weight']:.0f} lbs",
            f"Range: {analysis['min_weight']:.0f} - {analysis['max_weight']:.0f} lbs",
            "",
            "Weight by Size:",
        ]
        
        for size, data in analysis['weight_by_size'].items():
            lines.append(
                f"  {size}: {data['count']} units, "
                f"{data['total_weight']:,.0f} lbs total"
            )
        
        lines.extend([
            "",
            "Center of Gravity:",
            f"  X: {analysis['center_of_gravity'][0]:.1f} ft",
            f"  Y: {analysis['center_of_gravity'][1]:.1f} ft",
            "",
            "Transport Requirements:",
            f"  Truck Loads: {analysis['transport']['truck_loads']}",
            f"  Crew Size: {analysis['transport']['recommended_crew_size']}",
            f"  Crane Required: {analysis['transport']['crane_required']} cassettes",
            f"  Forklift Required: {analysis['transport']['forklift_required']} cassettes",
            "",
            f"Validation: {analysis['validation']['summary']}",
        ])
        
        if analysis['validation']['violations']:
            lines.append("Violations:")
            for v in analysis['validation']['violations']:
                lines.append(f"  - {v}")
        
        if analysis['validation']['warnings']:
            lines.append("Warnings:")
            for w in analysis['validation']['warnings']:
                lines.append(f"  - {w}")
        
        return "\n".join(lines)
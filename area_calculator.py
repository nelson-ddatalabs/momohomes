"""
Area Calculator Module
======================
Calculates areas using Shoelace formula and classifies space types.
"""

import math
import numpy as np
from typing import List, Tuple, Dict, Optional
from enum import Enum
import cv2
import logging

logger = logging.getLogger(__name__)


class SpaceType(Enum):
    """Types of spaces in floor plan."""
    INDOOR_LIVING = "indoor_living"  # Cassette-eligible
    GARAGE = "garage"                 # Not cassette-eligible
    PATIO = "patio"                   # Not cassette-eligible
    DECK = "deck"                     # Not cassette-eligible
    UTILITY = "utility"               # May be cassette-eligible
    OUTDOOR = "outdoor"               # Not cassette-eligible
    UNKNOWN = "unknown"


class AreaCalculator:
    """Calculates areas and classifies spaces."""
    
    def __init__(self):
        """Initialize area calculator."""
        self.space_classifications = {}
        
    def calculate_polygon_area(self, vertices: List[Tuple[float, float]]) -> float:
        """
        Calculate polygon area using Shoelace formula.
        
        Args:
            vertices: List of (x, y) coordinates
            
        Returns:
            Area in square feet
        """
        if len(vertices) < 3:
            return 0.0
        
        # Ensure polygon is closed
        if vertices[0] != vertices[-1]:
            vertices = vertices + [vertices[0]]
        
        # Apply Shoelace formula
        n = len(vertices)
        area = 0.0
        
        for i in range(n - 1):
            area += vertices[i][0] * vertices[i + 1][1]
            area -= vertices[i + 1][0] * vertices[i][1]
        
        area = abs(area) / 2.0
        
        logger.debug(f"Calculated area: {area:.2f} sq ft from {n-1} vertices")
        
        return area
    
    def calculate_areas_with_classification(self, polygon, image: np.ndarray) -> Dict:
        """
        Calculate all areas with space classification.
        
        Args:
            polygon: Polygon object
            image: Floor plan image for space detection
            
        Returns:
            Dictionary with classified areas
        """
        # Get total area
        vertices = [(v.x, v.y) for v in polygon.vertices]
        total_area = self.calculate_polygon_area(vertices)
        
        # Classify spaces
        space_polygons = self._identify_space_polygons(polygon, image)
        
        # Calculate areas by type
        areas_by_type = {}
        for space_type in SpaceType:
            areas_by_type[space_type.value] = 0.0
        
        for space_poly in space_polygons:
            space_area = self.calculate_polygon_area(space_poly['vertices'])
            space_type = space_poly['type']
            areas_by_type[space_type.value] += space_area
        
        # Calculate cassette-eligible area
        cassette_eligible = (
            areas_by_type[SpaceType.INDOOR_LIVING.value] +
            areas_by_type[SpaceType.UTILITY.value]
        )
        
        result = {
            'total_area': total_area,
            'cassette_eligible_area': cassette_eligible,
            'areas_by_type': areas_by_type,
            'space_polygons': space_polygons,
            'coverage_ratio': cassette_eligible / total_area if total_area > 0 else 0
        }
        
        logger.info(f"Area breakdown: Total={total_area:.0f} sq ft, "
                   f"Cassette-eligible={cassette_eligible:.0f} sq ft "
                   f"({result['coverage_ratio']*100:.1f}%)")
        
        return result
    
    def _identify_space_polygons(self, polygon, image: np.ndarray) -> List[Dict]:
        """
        Identify different space polygons within the main polygon.
        
        Args:
            polygon: Main polygon
            image: Floor plan image
            
        Returns:
            List of space polygon dictionaries
        """
        space_polygons = []
        
        # Detect garage
        garage_poly = self._detect_garage_polygon(polygon, image)
        if garage_poly:
            space_polygons.append({
                'type': SpaceType.GARAGE,
                'vertices': garage_poly,
                'confidence': 0.8
            })
        
        # Detect patio/deck
        outdoor_polys = self._detect_outdoor_spaces(polygon, image)
        for poly in outdoor_polys:
            space_polygons.append(poly)
        
        # Remaining is indoor living space
        indoor_poly = self._calculate_indoor_polygon(polygon, space_polygons)
        if indoor_poly:
            space_polygons.append({
                'type': SpaceType.INDOOR_LIVING,
                'vertices': indoor_poly,
                'confidence': 0.9
            })
        
        return space_polygons
    
    def _detect_garage_polygon(self, polygon, image: np.ndarray) -> Optional[List[Tuple[float, float]]]:
        """
        Detect garage area in floor plan.
        
        Args:
            polygon: Main polygon
            image: Floor plan image
            
        Returns:
            Garage polygon vertices or None
        """
        # Look for garage indicators
        # 1. Text containing "garage"
        # 2. Different color region
        # 3. Typical garage dimensions (20x20 to 30x30 feet)
        
        # For now, use edge labels to identify garage
        garage_edges = []
        for edge in polygon.edges:
            if edge.room_label and 'garage' in edge.room_label.lower():
                garage_edges.append(edge)
        
        if not garage_edges:
            return None
        
        # Estimate garage polygon from edges
        # Simplified: assume rectangular garage
        if len(garage_edges) >= 2:
            # Find bounds of garage edges
            all_points = []
            for edge in garage_edges:
                all_points.append((edge.start_vertex.x, edge.start_vertex.y))
                all_points.append((edge.end_vertex.x, edge.end_vertex.y))
            
            if all_points:
                xs = [p[0] for p in all_points]
                ys = [p[1] for p in all_points]
                
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
                
                # Create rectangular garage polygon
                garage_poly = [
                    (min_x, min_y),
                    (max_x, min_y),
                    (max_x, max_y),
                    (min_x, max_y)
                ]
                
                # Validate size (typical garage 400-900 sq ft)
                area = self.calculate_polygon_area(garage_poly)
                if 300 <= area <= 1200:
                    logger.info(f"Detected garage: {area:.0f} sq ft")
                    return garage_poly
        
        return None
    
    def _detect_outdoor_spaces(self, polygon, image: np.ndarray) -> List[Dict]:
        """
        Detect outdoor spaces (patio, deck).
        
        Args:
            polygon: Main polygon
            image: Floor plan image
            
        Returns:
            List of outdoor space polygons
        """
        outdoor_spaces = []
        
        # Look for patio/deck indicators
        for edge in polygon.edges:
            if edge.room_label:
                label_lower = edge.room_label.lower()
                
                space_type = None
                if 'patio' in label_lower:
                    space_type = SpaceType.PATIO
                elif 'deck' in label_lower:
                    space_type = SpaceType.DECK
                elif 'porch' in label_lower:
                    space_type = SpaceType.DECK
                
                if space_type:
                    # Create simplified polygon for outdoor space
                    # This is a placeholder - would need more sophisticated detection
                    outdoor_spaces.append({
                        'type': space_type,
                        'vertices': [],  # Would calculate actual vertices
                        'confidence': 0.6
                    })
        
        return outdoor_spaces
    
    def _calculate_indoor_polygon(self, main_polygon, excluded_spaces: List[Dict]) -> Optional[List[Tuple[float, float]]]:
        """
        Calculate indoor living space polygon by excluding other spaces.
        
        Args:
            main_polygon: Main polygon
            excluded_spaces: List of non-indoor space polygons
            
        Returns:
            Indoor polygon vertices or None
        """
        # For now, if no exclusions, entire polygon is indoor
        if not excluded_spaces:
            return [(v.x, v.y) for v in main_polygon.vertices]
        
        # Would implement polygon subtraction here
        # For now, return main polygon minus excluded area estimate
        main_vertices = [(v.x, v.y) for v in main_polygon.vertices]
        
        # This is simplified - actual implementation would do proper polygon operations
        return main_vertices
    
    def classify_space_by_color(self, image: np.ndarray, polygon_mask: np.ndarray) -> Dict[str, float]:
        """
        Classify spaces based on color in floor plan.
        
        Args:
            image: Floor plan image
            polygon_mask: Binary mask of polygon area
            
        Returns:
            Dictionary of space type to area ratio
        """
        # Convert to HSV for color analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for different spaces
        color_ranges = {
            'green': {  # Indoor living
                'lower': np.array([35, 30, 30]),
                'upper': np.array([85, 255, 255]),
                'space_type': SpaceType.INDOOR_LIVING
            },
            'gray': {  # Garage
                'lower': np.array([0, 0, 50]),
                'upper': np.array([180, 30, 150]),
                'space_type': SpaceType.GARAGE
            }
        }
        
        space_ratios = {}
        total_pixels = np.sum(polygon_mask > 0)
        
        for color_name, color_info in color_ranges.items():
            # Create color mask
            color_mask = cv2.inRange(hsv, color_info['lower'], color_info['upper'])
            
            # Combine with polygon mask
            combined_mask = cv2.bitwise_and(color_mask, polygon_mask)
            
            # Calculate ratio
            color_pixels = np.sum(combined_mask > 0)
            ratio = color_pixels / total_pixels if total_pixels > 0 else 0
            
            space_type = color_info['space_type']
            space_ratios[space_type.value] = ratio
        
        return space_ratios
    
    def validate_area_calculation(self, calculated_area: float, 
                                 expected_area: Optional[float] = None) -> Dict:
        """
        Validate calculated area against expected values.
        
        Args:
            calculated_area: Calculated area in sq ft
            expected_area: Expected area if known
            
        Returns:
            Validation results
        """
        validation = {
            'valid': True,
            'warnings': [],
            'calculated': calculated_area
        }
        
        # Check reasonable bounds
        if calculated_area < 500:
            validation['warnings'].append(f"Area unusually small: {calculated_area:.0f} sq ft")
        elif calculated_area > 10000:
            validation['warnings'].append(f"Area unusually large: {calculated_area:.0f} sq ft")
        
        # Compare with expected if provided
        if expected_area:
            error = abs(calculated_area - expected_area)
            error_pct = (error / expected_area) * 100 if expected_area > 0 else 0
            
            validation['expected'] = expected_area
            validation['error'] = error
            validation['error_percent'] = error_pct
            
            if error_pct > 5:
                validation['warnings'].append(f"Area error: {error_pct:.1f}% "
                                            f"(calculated: {calculated_area:.0f}, "
                                            f"expected: {expected_area:.0f})")
                if error_pct > 10:
                    validation['valid'] = False
        
        return validation
    
    def get_centroid(self, vertices: List[Tuple[float, float]]) -> Tuple[float, float]:
        """
        Calculate polygon centroid.
        
        Args:
            vertices: List of (x, y) coordinates
            
        Returns:
            (cx, cy) centroid coordinates
        """
        if not vertices:
            return (0, 0)
        
        # Calculate area first
        area = self.calculate_polygon_area(vertices)
        
        if area == 0:
            # Fall back to simple average
            cx = sum(v[0] for v in vertices) / len(vertices)
            cy = sum(v[1] for v in vertices) / len(vertices)
            return (cx, cy)
        
        # Calculate centroid using area formula
        cx = 0
        cy = 0
        
        n = len(vertices)
        for i in range(n - 1):
            factor = vertices[i][0] * vertices[i + 1][1] - vertices[i + 1][0] * vertices[i][1]
            cx += (vertices[i][0] + vertices[i + 1][0]) * factor
            cy += (vertices[i][1] + vertices[i + 1][1]) * factor
        
        cx = cx / (6 * area)
        cy = cy / (6 * area)
        
        return (cx, cy)
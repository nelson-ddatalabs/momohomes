"""
Improved Floor Plan Extraction System
=====================================
Combines room detection, dimension line detection, and color-based extraction
for reliable floor plan processing.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import logging

from room_detector import RoomDetector
from dimension_line_detector import DimensionLineDetector
from cassette_models import FloorBoundary, Point

logger = logging.getLogger(__name__)


class ImprovedFloorExtractor:
    """Improved floor plan extraction using multiple detection methods."""
    
    def __init__(self):
        """Initialize the improved extractor."""
        self.room_detector = RoomDetector()
        self.dimension_detector = DimensionLineDetector()
        
    def extract(self, image_path: str) -> Dict:
        """
        Extract floor plan information using comprehensive detection.
        
        Args:
            image_path: Path to floor plan image
            
        Returns:
            Dictionary with extraction results
        """
        logger.info(f"Extracting floor plan from: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Step 1: Detect rooms using OCR
        logger.info("Step 1: Detecting rooms with OCR")
        room_results = self.room_detector.detect_rooms(image)
        
        # Step 2: Get indoor space from color
        logger.info("Step 2: Extracting indoor space from green areas")
        indoor_validation = self.room_detector.validate_indoor_spaces(image, room_results)
        
        # Step 3: Extract dimensions from all sides
        logger.info("Step 3: Extracting dimensions from all sides")
        dimensions = self.dimension_detector.detect_dimension_lines(image)
        width_feet, height_feet = self.dimension_detector.extract_floor_dimensions(image)
        
        # Step 4: Create floor boundary
        logger.info("Step 4: Creating floor boundary")
        boundary = self._create_floor_boundary(
            indoor_validation, 
            width_feet, 
            height_feet,
            image.shape
        )
        
        # Step 5: Calculate metrics
        area_sqft = width_feet * height_feet
        indoor_area = self._calculate_indoor_area(indoor_validation, image.shape, area_sqft)
        
        # Compile results
        results = {
            'success': True,
            'image_path': image_path,
            'dimensions': {
                'width_feet': width_feet,
                'height_feet': height_feet,
                'area_sqft': area_sqft,
                'details': dimensions
            },
            'rooms': {
                'indoor': room_results['indoor_rooms'],
                'outdoor': room_results['outdoor_spaces'],
                'validated': indoor_validation['validated_rooms']
            },
            'indoor_space': {
                'mask': indoor_validation['indoor_mask'],
                'boundary': boundary,
                'area_sqft': indoor_area,
                'coverage': indoor_area / area_sqft if area_sqft > 0 else 0
            },
            'extraction_quality': self._assess_extraction_quality(
                room_results, indoor_validation, dimensions
            )
        }
        
        logger.info(f"Extraction complete: {width_feet:.1f}' x {height_feet:.1f}' = {area_sqft:.0f} sqft")
        logger.info(f"Indoor area: {indoor_area:.0f} sqft ({indoor_area/area_sqft*100:.1f}%)")
        logger.info(f"Detected {len(room_results['indoor_rooms'])} indoor rooms")
        
        return results
    
    def _create_floor_boundary(self, indoor_validation: Dict, 
                              width_feet: float, height_feet: float,
                              image_shape: Tuple) -> FloorBoundary:
        """Create floor boundary from detection results."""
        img_height, img_width = image_shape[:2]
        
        # Calculate scale
        scale_x = width_feet / img_width
        scale_y = height_feet / img_height
        
        # Use the indoor contour if available
        if indoor_validation['main_contour'] is not None:
            contour = indoor_validation['main_contour']
            
            # Convert contour points to feet
            points = []
            for point in contour:
                x_px, y_px = point[0]
                x_ft = x_px * scale_x
                y_ft = y_px * scale_y
                points.append(Point(x_ft, y_ft))
            
            # Create boundary
            boundary = FloorBoundary(points=points)
        else:
            # Fallback to rectangular boundary
            logger.warning("No indoor contour detected, using rectangular boundary")
            boundary = FloorBoundary(
                points=[
                    Point(0, 0),
                    Point(width_feet, 0),
                    Point(width_feet, height_feet),
                    Point(0, height_feet)
                ]
            )
        
        return boundary
    
    def _calculate_indoor_area(self, indoor_validation: Dict, 
                              image_shape: Tuple, total_area: float) -> float:
        """Calculate indoor area in square feet."""
        if indoor_validation['area_pixels'] > 0:
            # Calculate percentage of image that is indoor
            total_pixels = image_shape[0] * image_shape[1]
            indoor_percentage = indoor_validation['area_pixels'] / total_pixels
            
            # Apply to total area
            indoor_area = total_area * indoor_percentage
        else:
            # Assume most of the area is indoor (conservative estimate)
            indoor_area = total_area * 0.85
            logger.warning("Could not calculate exact indoor area, using 85% estimate")
        
        return indoor_area
    
    def _assess_extraction_quality(self, room_results: Dict, 
                                  indoor_validation: Dict,
                                  dimensions: Dict) -> Dict:
        """Assess the quality of the extraction."""
        quality_score = 100
        issues = []
        
        # Check room detection
        if len(room_results['indoor_rooms']) == 0:
            quality_score -= 30
            issues.append("No indoor rooms detected")
        elif len(room_results['indoor_rooms']) < 3:
            quality_score -= 15
            issues.append(f"Only {len(room_results['indoor_rooms'])} rooms detected")
        
        # Check indoor space detection
        if indoor_validation['main_contour'] is None:
            quality_score -= 25
            issues.append("Could not detect main indoor contour")
        
        # Check dimension detection
        dim_count = sum(len(dimensions[side]) for side in ['top', 'bottom', 'left', 'right'])
        if dim_count == 0:
            quality_score -= 30
            issues.append("No dimensions detected")
        elif dim_count < 4:
            quality_score -= 15
            issues.append(f"Only {dim_count} dimensions detected")
        
        # Check if dimensions are complete
        if dimensions['total_width'] is None:
            quality_score -= 10
            issues.append("Could not calculate total width")
        if dimensions['total_height'] is None:
            quality_score -= 10
            issues.append("Could not calculate total height")
        
        # Determine overall quality
        if quality_score >= 80:
            quality = "high"
        elif quality_score >= 60:
            quality = "medium"
        else:
            quality = "low"
        
        return {
            'score': quality_score,
            'quality': quality,
            'issues': issues,
            'room_count': len(room_results['indoor_rooms']),
            'dimension_count': dim_count,
            'has_contour': indoor_validation['main_contour'] is not None
        }
    
    def validate_extraction(self, results: Dict) -> Tuple[bool, List[str]]:
        """
        Validate extraction results for cassette optimization.
        
        Args:
            results: Extraction results
            
        Returns:
            Tuple of (is_valid, warnings)
        """
        warnings = []
        
        # Check extraction quality
        quality = results['extraction_quality']
        if quality['score'] < 60:
            warnings.append(f"Low extraction quality: {quality['score']}/100")
            for issue in quality['issues']:
                warnings.append(f"  - {issue}")
        
        # Check dimensions
        dims = results['dimensions']
        if dims['width_feet'] < 10 or dims['width_feet'] > 200:
            warnings.append(f"Unusual width: {dims['width_feet']:.1f} feet")
        if dims['height_feet'] < 10 or dims['height_feet'] > 200:
            warnings.append(f"Unusual height: {dims['height_feet']:.1f} feet")
        
        # Check indoor space
        if results['indoor_space']['coverage'] < 0.5:
            warnings.append(f"Low indoor coverage: {results['indoor_space']['coverage']*100:.1f}%")
        
        # Determine if valid
        is_valid = quality['score'] >= 40 and dims['width_feet'] > 0 and dims['height_feet'] > 0
        
        return is_valid, warnings
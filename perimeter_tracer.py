"""
Perimeter Tracer System
========================
Main system for sequential perimeter tracing of floor plans.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import logging
import json
import math

# Import all modules
from corner_detector import CornerDetector
from dimension_sequence_extractor import DimensionSequenceExtractor
from direction_tracker import DirectionTracker
from polygon_builder import PolygonBuilder
from closure_validator import ClosureValidator
from error_corrector import ErrorCorrector, CorrectionMethod
from area_calculator import AreaCalculator

logger = logging.getLogger(__name__)


class PerimeterTracer:
    """Main system for floor plan perimeter tracing."""
    
    def __init__(self):
        """Initialize perimeter tracer with all components."""
        self.corner_detector = CornerDetector()
        self.dimension_extractor = DimensionSequenceExtractor()
        self.direction_tracker = DirectionTracker()
        self.polygon_builder = PolygonBuilder()
        self.closure_validator = ClosureValidator()
        self.error_corrector = ErrorCorrector(CorrectionMethod.BOWDITCH)
        self.area_calculator = AreaCalculator()
        
        self.image = None
        self.results = {}
        
    def trace_perimeter(self, image_path: str) -> Dict:
        """
        Complete perimeter tracing pipeline.
        
        Args:
            image_path: Path to floor plan image
            
        Returns:
            Dictionary with complete tracing results
        """
        logger.info(f"Starting perimeter tracing for: {image_path}")
        
        try:
            # Phase 1: Initialization
            phase1 = self._phase1_initialization(image_path)
            if not phase1['success']:
                return {'success': False, 'error': 'Initialization failed', 'phase': 1}
            
            # Phase 2: Starting Point
            phase2 = self._phase2_starting_point()
            if not phase2['success']:
                return {'success': False, 'error': 'Starting point detection failed', 'phase': 2}
            
            # Phase 3: Sequential Extraction
            phase3 = self._phase3_sequential_extraction(phase2['starting_corner'])
            if not phase3['success']:
                return {'success': False, 'error': 'Dimension extraction failed', 'phase': 3}
            
            # Phase 4: Polygon Construction
            phase4 = self._phase4_polygon_construction(phase3['dimensions'])
            if not phase4['success']:
                return {'success': False, 'error': 'Polygon construction failed', 'phase': 4}
            
            # Phase 5: Validation & Correction
            phase5 = self._phase5_validation_correction(phase4['polygon'])
            if not phase5['success']:
                logger.warning("Validation failed, attempting correction")
            
            # Phase 6: Output Generation
            phase6 = self._phase6_output_generation(phase5['polygon'])
            
            # Compile results
            results = {
                'success': True,
                'image_path': image_path,
                'polygon': phase5['polygon'],
                'areas': phase6['areas'],
                'validation': phase5['validation'],
                'metadata': {
                    'starting_corner': phase2['starting_corner'],
                    'dimensions_extracted': len(phase3['dimensions']),
                    'vertices': len(phase5['polygon'].vertices),
                    'closure_error': phase5['validation'].get('geometric_closure', {}).get('error_feet', 0),
                    'confidence': phase6.get('confidence', 0)
                }
            }
            
            self.results = results
            logger.info("Perimeter tracing complete")
            
            return results
            
        except Exception as e:
            logger.error(f"Perimeter tracing failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _phase1_initialization(self, image_path: str) -> Dict:
        """
        Phase 1: Load image and perform initial detection.
        
        Args:
            image_path: Path to floor plan image
            
        Returns:
            Phase 1 results
        """
        logger.info("Phase 1: Initialization")
        
        # Load image
        self.image = cv2.imread(image_path)
        if self.image is None:
            return {'success': False, 'error': 'Failed to load image'}
        
        # Detect color regions
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        
        # Green mask for indoor areas
        lower_green = np.array([35, 30, 30])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Identify legend/scale
        legend_info = self._detect_legend()
        
        # Locate dimension lines
        dimension_lines = self._detect_dimension_lines()
        
        return {
            'success': True,
            'image_shape': self.image.shape,
            'green_area_pixels': np.sum(green_mask > 0),
            'legend': legend_info,
            'dimension_lines_count': len(dimension_lines)
        }
    
    def _phase2_starting_point(self) -> Dict:
        """
        Phase 2: Find optimal starting corner and set initial bearing.
        
        Returns:
            Phase 2 results
        """
        logger.info("Phase 2: Starting point determination")
        
        # Find optimal corner
        starting_corner = self.corner_detector.detect_starting_corner(self.image)
        
        # Determine initial bearing based on corner type
        corner_bearings = {
            'NW': 0,    # Start going East
            'NE': 180,  # Start going West
            'SE': 180,  # Start going West
            'SW': 0     # Start going East
        }
        
        corner_str = str(starting_corner['type']).split('.')[-1] if hasattr(starting_corner['type'], 'value') else 'NW'
        initial_bearing = corner_bearings.get(corner_str[:2], 0)
        
        # Initialize direction tracker
        self.direction_tracker.initialize(self.image, initial_bearing)
        
        # Set coordinate origin
        self.polygon_builder = PolygonBuilder(origin=(0, 0))
        
        return {
            'success': True,
            'starting_corner': starting_corner,
            'initial_bearing': initial_bearing
        }
    
    def _phase3_sequential_extraction(self, starting_corner: Dict) -> Dict:
        """
        Phase 3: Extract dimensions sequentially around perimeter.
        
        Args:
            starting_corner: Starting corner information
            
        Returns:
            Phase 3 results
        """
        logger.info("Phase 3: Sequential dimension extraction")
        
        # Extract perimeter dimensions
        perimeter_dims = self.dimension_extractor.extract_perimeter_dimensions(
            self.image, starting_corner
        )
        
        # Flatten dimensions into sequential list
        dimensions = []
        for edge_dir, edge_dims in perimeter_dims.items():
            for dim in edge_dims:
                dimensions.append({
                    'length': dim.total_feet,
                    'text': dim.text,
                    'edge': edge_dir,
                    'confidence': dim.confidence,
                    'room_label': dim.room_label
                })
        
        # Handle complex features
        dimensions = self._handle_complex_features(dimensions)
        
        return {
            'success': len(dimensions) > 0,
            'dimensions': dimensions,
            'perimeter_dims': perimeter_dims
        }
    
    def _phase4_polygon_construction(self, dimensions: List[Dict]) -> Dict:
        """
        Phase 4: Build polygon from dimensions.
        
        Args:
            dimensions: Sequential dimension list
            
        Returns:
            Phase 4 results
        """
        logger.info("Phase 4: Polygon construction")
        
        # Build polygon
        polygon = self.polygon_builder.build_polygon(dimensions, self.direction_tracker)
        
        # Add closing edge if needed
        self.polygon_builder.add_closing_edge()
        
        return {
            'success': len(polygon.vertices) >= 3,
            'polygon': polygon
        }
    
    def _phase5_validation_correction(self, polygon) -> Dict:
        """
        Phase 5: Validate and correct polygon.
        
        Args:
            polygon: Constructed polygon
            
        Returns:
            Phase 5 results
        """
        logger.info("Phase 5: Validation and correction")
        
        # Validate polygon
        validation = self.closure_validator.validate_polygon(polygon)
        
        # If not valid, apply corrections
        if not validation['valid']:
            closure_error = self.closure_validator.get_closure_error_vector(polygon)
            correction = self.error_corrector.correct_polygon(polygon, closure_error)
            
            # Update polygon with corrected vertices
            polygon.vertices = correction['corrected_vertices']
            polygon.closure_error = math.sqrt(
                correction['final_error'][0]**2 + 
                correction['final_error'][1]**2
            )
            
            # Re-validate
            validation = self.closure_validator.validate_polygon(polygon)
        
        # Validate against green mask
        green_validation = self._validate_against_green_mask(polygon)
        validation['green_mask_match'] = green_validation
        
        return {
            'success': validation['valid'] or validation['geometric_closure']['error_feet'] < 5,
            'polygon': polygon,
            'validation': validation
        }
    
    def _phase6_output_generation(self, polygon) -> Dict:
        """
        Phase 6: Generate final outputs.
        
        Args:
            polygon: Final polygon
            
        Returns:
            Phase 6 results
        """
        logger.info("Phase 6: Output generation")
        
        # Calculate areas
        areas = self.area_calculator.calculate_areas_with_classification(polygon, self.image)
        
        # Generate confidence metrics
        confidence = self._calculate_confidence(polygon, areas)
        
        # Create output specification
        output = {
            'polygon_vertices': [(v.x, v.y) for v in polygon.vertices],
            'edges': [
                {
                    'length': e.length,
                    'bearing': e.bearing,
                    'text': e.dimension_text,
                    'room': e.room_label
                }
                for e in polygon.edges
            ],
            'areas': areas,
            'confidence': confidence,
            'perimeter': polygon.perimeter,
            'is_closed': polygon.is_closed
        }
        
        return output
    
    def _detect_legend(self) -> Dict:
        """Detect legend box in image."""
        # Simplified legend detection
        height, width = self.image.shape[:2]
        bottom_region = self.image[int(height * 0.8):, :]
        
        # Look for white/light regions
        gray = cv2.cvtColor(bottom_region, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            
            return {
                'found': True,
                'position': (x, y + int(height * 0.8)),
                'size': (w, h)
            }
        
        return {'found': False}
    
    def _detect_dimension_lines(self) -> List:
        """Detect dimension lines in image."""
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, 
                                minLineLength=50, maxLineGap=10)
        
        return lines if lines is not None else []
    
    def _handle_complex_features(self, dimensions: List[Dict]) -> List[Dict]:
        """Handle complex features like indentations and extensions."""
        # This would handle L-shapes, garages, etc.
        # For now, return as-is
        return dimensions
    
    def _validate_against_green_mask(self, polygon) -> float:
        """Validate polygon against green area mask."""
        # Create polygon mask
        height, width = self.image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Scale polygon vertices to pixels (approximate)
        scale = 10  # pixels per foot (to be refined)
        points = np.array([(int(v.x * scale), int(v.y * scale)) for v in polygon.vertices])
        
        if len(points) > 2:
            cv2.fillPoly(mask, [points], 255)
        
        # Compare with green mask
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 30, 30])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Calculate overlap
        intersection = cv2.bitwise_and(mask, green_mask)
        
        mask_area = np.sum(mask > 0)
        intersection_area = np.sum(intersection > 0)
        
        if mask_area > 0:
            overlap_ratio = intersection_area / mask_area
            return overlap_ratio
        
        return 0.0
    
    def _calculate_confidence(self, polygon, areas: Dict) -> float:
        """Calculate overall confidence score."""
        confidence_factors = []
        
        # Closure confidence
        if polygon.closure_error < 1:
            confidence_factors.append(1.0)
        elif polygon.closure_error < 5:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.5)
        
        # Area confidence
        total_area = areas['total_area']
        if 500 <= total_area <= 10000:
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.6)
        
        # Coverage confidence
        coverage = areas['coverage_ratio']
        if coverage > 0.7:
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.7)
        
        # Average confidence
        return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5
    
    def visualize_result(self, output_path: str):
        """Create visualization of traced perimeter."""
        if self.image is None or not self.results:
            return
        
        vis_image = self.image.copy()
        
        # Draw polygon
        if 'polygon' in self.results:
            polygon = self.results['polygon']
            
            # Scale vertices to pixels
            scale = 10  # pixels per foot
            points = np.array([(int(v.x * scale), int(v.y * scale)) 
                              for v in polygon.vertices])
            
            # Draw polygon
            cv2.polylines(vis_image, [points], True, (0, 255, 0), 2)
            
            # Draw vertices
            for point in points:
                cv2.circle(vis_image, tuple(point), 5, (255, 0, 0), -1)
        
        # Save visualization
        cv2.imwrite(output_path, vis_image)
        logger.info(f"Visualization saved to: {output_path}")
    
    def save_results(self, output_path: str):
        """Save results to JSON file."""
        if not self.results:
            return
        
        # Convert to serializable format
        output = {
            'success': self.results.get('success', False),
            'image_path': self.results.get('image_path', ''),
            'areas': self.results.get('areas', {}),
            'metadata': self.results.get('metadata', {})
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Results saved to: {output_path}")
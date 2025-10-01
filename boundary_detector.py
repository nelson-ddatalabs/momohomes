"""
Boundary Detector Module
========================
Detects and extracts floor plan boundaries representing indoor living space.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging

from cassette_models import Point, FloorBoundary
from floor_geometry import GeometryUtils
from config_cassette import CassetteConfig

logger = logging.getLogger(__name__)


class BoundaryDetector:
    """Detects floor boundaries in floor plan images."""
    
    def __init__(self):
        """Initialize boundary detector."""
        self.config = CassetteConfig.IMAGE_PROCESSING
        self.color_ranges = self.config['boundary_color_ranges']
        
    def detect_boundary(self, image_path: str, 
                       scale: Optional[float] = None) -> FloorBoundary:
        """
        Detect the floor boundary from an image.
        
        Args:
            image_path: Path to floor plan image
            scale: Pixels per foot (if known)
            
        Returns:
            FloorBoundary object with boundary points
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        # Try color-based detection first (green area)
        boundary = self._detect_by_color(image, 'green')
        
        # If no green area found, try gray area
        if boundary is None or len(boundary.points) < 3:
            logger.info("No green boundary found, trying gray detection")
            boundary = self._detect_by_color(image, 'gray')
        
        # If still no boundary, use image edges
        if boundary is None or len(boundary.points) < 3:
            logger.warning("No color boundary found, using image edges")
            boundary = self._create_boundary_from_image(image)
        
        # Convert to feet if scale is provided
        if scale and scale > 0:
            boundary = self._scale_boundary(boundary, scale)
        
        logger.info(f"Detected boundary with {len(boundary.points)} points, "
                   f"area: {boundary.area:.1f} sq units")
        
        return boundary
    
    def _detect_by_color(self, image: np.ndarray, color: str) -> Optional[FloorBoundary]:
        """Detect boundary using color range."""
        if color not in self.color_ranges:
            return None
        
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create mask for color range
        color_range = self.color_ranges[color]
        lower = np.array(color_range['lower_hsv'])
        upper = np.array(color_range['upper_hsv'])
        mask = cv2.inRange(hsv, lower, upper)
        
        # Clean up mask
        mask = self._clean_mask(mask)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Check if contour is significant
        if cv2.contourArea(largest_contour) < self.config['min_contour_area']:
            return None
        
        # Simplify contour
        epsilon = self.config['epsilon_factor'] * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # Convert to points
        points = [Point(float(p[0][0]), float(p[0][1])) for p in approx]
        
        # Further simplify if too many points
        if len(points) > 20:
            points = GeometryUtils.simplify_polygon(points, tolerance=2.0)
        
        return FloorBoundary(points)
    
    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """Clean up binary mask using morphological operations."""
        # Remove small noise
        kernel_small = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
        
        # Close small gaps
        kernel_medium = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_medium)
        
        # Fill holes
        mask_filled = mask.copy()
        h, w = mask.shape
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(mask_filled, flood_mask, (0, 0), 255)
        mask_filled = cv2.bitwise_not(mask_filled)
        mask = cv2.bitwise_or(mask, mask_filled)
        
        return mask
    
    def _create_boundary_from_image(self, image: np.ndarray) -> FloorBoundary:
        """Create boundary from image edges as fallback."""
        h, w = image.shape[:2]
        
        # Create rectangle with small margin
        margin = 10
        points = [
            Point(margin, margin),
            Point(w - margin, margin),
            Point(w - margin, h - margin),
            Point(margin, h - margin)
        ]
        
        return FloorBoundary(points)
    
    def _scale_boundary(self, boundary: FloorBoundary, scale: float) -> FloorBoundary:
        """Convert boundary from pixels to feet."""
        scaled_points = [
            Point(p.x / scale, p.y / scale) for p in boundary.points
        ]
        return FloorBoundary(scaled_points)
    
    def detect_luna_boundary(self, image_path: str) -> FloorBoundary:
        """
        Special handling for Luna floor plan.
        Uses known dimensions from luna_dimension_extractor.
        """
        # Luna floor plan known boundary (approximate)
        # Total dimensions: ~78' x ~34'
        points = [
            Point(0, 0),
            Point(78, 0),
            Point(78, 34),
            Point(0, 34)
        ]
        
        logger.info("Using known Luna floor plan boundary")
        return FloorBoundary(points)
    
    def refine_boundary(self, boundary: FloorBoundary, 
                       dimensions: dict) -> FloorBoundary:
        """
        Refine boundary using extracted dimensions.
        
        Args:
            boundary: Initial boundary
            dimensions: Extracted dimensions from edges
            
        Returns:
            Refined boundary
        """
        # Calculate total dimensions
        width = 0
        height = 0
        
        # Use top or bottom dimensions for width
        if 'top' in dimensions and dimensions['top']:
            width = sum(d.value_feet for d in dimensions['top'])
        elif 'bottom' in dimensions and dimensions['bottom']:
            width = sum(d.value_feet for d in dimensions['bottom'])
        
        # Use left or right dimensions for height
        if 'left' in dimensions and dimensions['left']:
            height = sum(d.value_feet for d in dimensions['left'])
        elif 'right' in dimensions and dimensions['right']:
            height = sum(d.value_feet for d in dimensions['right'])
        
        if width > 0 and height > 0:
            # Create refined rectangular boundary
            points = [
                Point(0, 0),
                Point(width, 0),
                Point(width, height),
                Point(0, height)
            ]
            
            logger.info(f"Refined boundary to {width}' x {height}'")
            return FloorBoundary(points)
        
        return boundary
    
    def validate_boundary(self, boundary: FloorBoundary) -> bool:
        """
        Validate that boundary is reasonable for a floor plan.
        
        Args:
            boundary: Boundary to validate
            
        Returns:
            True if valid
        """
        if not boundary.points or len(boundary.points) < 3:
            logger.warning("Boundary has insufficient points")
            return False
        
        # Check area
        area = boundary.area
        min_area = CassetteConfig.VALIDATION['min_floor_area']
        max_area = CassetteConfig.VALIDATION['max_floor_area']
        
        if area < min_area:
            logger.warning(f"Boundary area too small: {area:.1f} < {min_area}")
            return False
        
        if area > max_area:
            logger.warning(f"Boundary area too large: {area:.1f} > {max_area}")
            return False
        
        # Check aspect ratio
        if boundary.width > 0 and boundary.height > 0:
            aspect_ratio = max(boundary.width, boundary.height) / min(boundary.width, boundary.height)
            if aspect_ratio > 5:
                logger.warning(f"Boundary aspect ratio unusual: {aspect_ratio:.1f}")
                return False
        
        return True
    
    def visualize_boundary(self, image_path: str, boundary: FloorBoundary, 
                          output_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize detected boundary on the original image.
        
        Args:
            image_path: Original image path
            boundary: Detected boundary
            output_path: Optional path to save visualization
            
        Returns:
            Image with boundary overlay
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Draw boundary
        if boundary.points:
            points = np.array([(int(p.x), int(p.y)) for p in boundary.points])
            
            # Draw filled polygon with transparency
            overlay = image.copy()
            cv2.fillPoly(overlay, [points], (0, 255, 0))
            image = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
            
            # Draw boundary outline
            cv2.polylines(image, [points], True, (0, 255, 0), 2)
            
            # Draw corner points
            for point in points:
                cv2.circle(image, tuple(point), 5, (255, 0, 0), -1)
        
        # Add text
        text = f"Boundary: {boundary.width:.1f} x {boundary.height:.1f}, Area: {boundary.area:.1f}"
        cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 255), 2)
        
        # Save if requested
        if output_path:
            cv2.imwrite(output_path, image)
            logger.info(f"Saved boundary visualization to {output_path}")
        
        return image
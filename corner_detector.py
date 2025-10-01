"""
Corner Detector Module
======================
Identifies optimal starting corner for perimeter tracing.
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Optional, List
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class CornerType(Enum):
    """Corner position types."""
    NORTHWEST = "NW"
    NORTHEAST = "NE"
    SOUTHWEST = "SW"
    SOUTHEAST = "SE"


class CornerDetector:
    """Detects optimal starting corner for floor plan perimeter tracing."""
    
    def __init__(self):
        """Initialize corner detector."""
        self.image = None
        self.height = None
        self.width = None
        
    def detect_starting_corner(self, image: np.ndarray) -> Dict:
        """
        Detect the optimal starting corner for perimeter tracing.
        
        Args:
            image: Floor plan image (BGR)
            
        Returns:
            Dictionary with corner information
        """
        self.image = image
        self.height, self.width = image.shape[:2]
        
        # Method 1: Detect dimension line intersections
        line_corner = self._detect_dimension_line_corner()
        
        # Method 2: Use legend box position
        legend_corner = self._detect_legend_corner()
        
        # Method 3: Use green area boundaries
        green_corner = self._detect_green_area_corner()
        
        # Select best corner based on confidence
        best_corner = self._select_best_corner(line_corner, legend_corner, green_corner)
        
        logger.info(f"Selected starting corner: {best_corner['type']} at ({best_corner['x']}, {best_corner['y']}) "
                   f"with confidence {best_corner['confidence']:.2f}")
        
        return best_corner
    
    def _detect_dimension_line_corner(self) -> Dict:
        """
        Detect corner from dimension line intersections.
        
        Returns:
            Corner information dictionary
        """
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, 
                                minLineLength=100, maxLineGap=10)
        
        if lines is None:
            return {'x': 0, 'y': 0, 'type': CornerType.NORTHWEST, 'confidence': 0.0}
        
        # Find corner regions (within 15% of image edges)
        margin = 0.15
        corner_regions = {
            CornerType.NORTHWEST: (0, int(self.width * margin), 0, int(self.height * margin)),
            CornerType.NORTHEAST: (int(self.width * (1-margin)), self.width, 0, int(self.height * margin)),
            CornerType.SOUTHWEST: (0, int(self.width * margin), int(self.height * (1-margin)), self.height),
            CornerType.SOUTHEAST: (int(self.width * (1-margin)), self.width, int(self.height * (1-margin)), self.height)
        }
        
        # Count line intersections in each corner
        corner_scores = {}
        for corner_type, (x1, x2, y1, y2) in corner_regions.items():
            intersections = 0
            for line in lines:
                lx1, ly1, lx2, ly2 = line[0]
                # Check if line passes through corner region
                if ((x1 <= lx1 <= x2 and y1 <= ly1 <= y2) or 
                    (x1 <= lx2 <= x2 and y1 <= ly2 <= y2)):
                    intersections += 1
            corner_scores[corner_type] = intersections
        
        # Select corner with most intersections
        best_corner = max(corner_scores, key=corner_scores.get)
        corner_region = corner_regions[best_corner]
        
        # Find exact corner point
        corner_x = (corner_region[0] + corner_region[1]) // 2
        corner_y = (corner_region[2] + corner_region[3]) // 2
        
        # Refine to actual line intersection
        for line in lines:
            lx1, ly1, lx2, ly2 = line[0]
            if (corner_region[0] <= lx1 <= corner_region[1] and 
                corner_region[2] <= ly1 <= corner_region[3]):
                corner_x, corner_y = lx1, ly1
                break
        
        confidence = min(corner_scores[best_corner] / 10.0, 1.0)  # Normalize confidence
        
        return {
            'x': corner_x,
            'y': corner_y,
            'type': best_corner,
            'confidence': confidence,
            'method': 'dimension_lines'
        }
    
    def _detect_legend_corner(self) -> Dict:
        """
        Detect corner based on legend box position.
        
        Returns:
            Corner information dictionary
        """
        # Look for legend box (usually white/light gray rectangle at bottom)
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Focus on bottom 20% of image
        bottom_region = gray[int(self.height * 0.8):, :]
        
        # Threshold to find bright regions (legend is usually white)
        _, binary = cv2.threshold(bottom_region, 200, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {'x': 0, 'y': self.height, 'type': CornerType.SOUTHWEST, 'confidence': 0.0}
        
        # Find largest rectangular contour (likely legend box)
        legend_contour = None
        max_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                # Check if roughly rectangular
                x, y, w, h = cv2.boundingRect(contour)
                if w > self.width * 0.3 and h > 20:  # Legend is wide and has some height
                    max_area = area
                    legend_contour = contour
        
        if legend_contour is not None:
            x, y, w, h = cv2.boundingRect(legend_contour)
            y += int(self.height * 0.8)  # Adjust for region offset
            
            # Legend typically indicates bottom edge, so start from top corners
            if x < self.width / 2:
                # Legend on left, start from top-left
                return {
                    'x': 0,
                    'y': 0,
                    'type': CornerType.NORTHWEST,
                    'confidence': 0.8,
                    'method': 'legend_box'
                }
            else:
                # Legend on right, start from top-right
                return {
                    'x': self.width - 1,
                    'y': 0,
                    'type': CornerType.NORTHEAST,
                    'confidence': 0.8,
                    'method': 'legend_box'
                }
        
        # Default if no legend found
        return {
            'x': 0,
            'y': 0,
            'type': CornerType.NORTHWEST,
            'confidence': 0.3,
            'method': 'legend_box'
        }
    
    def _detect_green_area_corner(self) -> Dict:
        """
        Detect corner from green area boundaries.
        
        Returns:
            Corner information dictionary
        """
        # Convert to HSV for color detection
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        
        # Define green range
        lower_green = np.array([35, 30, 30])
        upper_green = np.array([85, 255, 255])
        
        # Create mask for green areas
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {'x': 0, 'y': 0, 'type': CornerType.NORTHWEST, 'confidence': 0.0}
        
        # Find largest contour (main floor area)
        main_contour = max(contours, key=cv2.contourArea)
        
        # Find extreme points
        leftmost = tuple(main_contour[main_contour[:, :, 0].argmin()][0])
        rightmost = tuple(main_contour[main_contour[:, :, 0].argmax()][0])
        topmost = tuple(main_contour[main_contour[:, :, 1].argmin()][0])
        bottommost = tuple(main_contour[main_contour[:, :, 1].argmax()][0])
        
        # Determine corners based on extreme points
        corners = []
        
        # Top-left corner approximation - use simple approach
        # Just use the extreme points to define corner
        corners.append((CornerType.NORTHWEST, int(leftmost[0]), int(topmost[1])))
        
        # Find the best corner (closest to actual corner)
        if corners:
            if len(corners[0]) == 3:
                corner_type, x, y = corners[0]
            else:
                corner_type = corners[0][0]
                x, y = leftmost[0], topmost[1]  # Default position
            return {
                'x': x,
                'y': y,
                'type': corner_type,
                'confidence': 0.7,
                'method': 'green_area'
            }
        
        # Default to top-left
        return {
            'x': leftmost[0],
            'y': topmost[1],
            'type': CornerType.NORTHWEST,
            'confidence': 0.5,
            'method': 'green_area'
        }
    
    def _select_best_corner(self, line_corner: Dict, legend_corner: Dict, 
                           green_corner: Dict) -> Dict:
        """
        Select the best corner based on confidence scores.
        
        Args:
            line_corner: Corner from dimension lines
            legend_corner: Corner from legend box
            green_corner: Corner from green area
            
        Returns:
            Best corner dictionary
        """
        corners = [line_corner, legend_corner, green_corner]
        
        # Weight different methods
        weights = {
            'dimension_lines': 1.2,  # Most reliable if found
            'legend_box': 1.0,
            'green_area': 0.8
        }
        
        # Apply weights
        for corner in corners:
            method = corner.get('method', '')
            corner['weighted_confidence'] = corner['confidence'] * weights.get(method, 1.0)
        
        # Select corner with highest weighted confidence
        best_corner = max(corners, key=lambda c: c['weighted_confidence'])
        
        # If confidence is too low, default to top-left
        if best_corner['confidence'] < 0.3:
            logger.warning("Low confidence in corner detection, using default top-left")
            best_corner = {
                'x': 0,
                'y': 0,
                'type': CornerType.NORTHWEST,
                'confidence': 0.5,
                'method': 'default'
            }
        
        return best_corner
    
    def get_corner_coordinates(self, corner_type: CornerType) -> Tuple[int, int]:
        """
        Get pixel coordinates for a specific corner type.
        
        Args:
            corner_type: Type of corner
            
        Returns:
            Tuple of (x, y) coordinates
        """
        if self.image is None:
            return (0, 0)
        
        margin = 50  # Pixels from edge
        
        corners = {
            CornerType.NORTHWEST: (margin, margin),
            CornerType.NORTHEAST: (self.width - margin, margin),
            CornerType.SOUTHWEST: (margin, self.height - margin),
            CornerType.SOUTHEAST: (self.width - margin, self.height - margin)
        }
        
        return corners.get(corner_type, (0, 0))
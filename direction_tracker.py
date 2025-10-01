"""
Direction Tracker Module
========================
Manages bearing changes and corner detection for perimeter tracing.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict
from enum import Enum
from dataclasses import dataclass
import math
import logging

logger = logging.getLogger(__name__)


class TurnType(Enum):
    """Types of turns in perimeter traversal."""
    STRAIGHT = "straight"
    RIGHT_90 = "right_90"
    LEFT_90 = "left_90"
    RIGHT_ANGLE = "right_angle"  # Non-90 degree right turn
    LEFT_ANGLE = "left_angle"    # Non-90 degree left turn
    INDENTATION = "indentation"   # Going into building
    EXTENSION = "extension"       # Coming out of building


@dataclass
class CornerInfo:
    """Information about a detected corner."""
    position: Tuple[int, int]  # Pixel coordinates
    turn_type: TurnType
    angle_degrees: float
    confidence: float
    is_interior: bool  # Interior vs exterior corner


class DirectionTracker:
    """Tracks bearing changes and detects corners during perimeter tracing."""
    
    def __init__(self):
        """Initialize direction tracker."""
        self.current_bearing = 0  # Degrees (0=East, 90=South, 180=West, 270=North)
        self.turn_history = []
        self.corners = []
        self.image = None
        
    def initialize(self, image: np.ndarray, starting_bearing: float = 0):
        """
        Initialize tracker with image and starting bearing.
        
        Args:
            image: Floor plan image
            starting_bearing: Initial bearing in degrees
        """
        self.image = image
        self.current_bearing = starting_bearing
        self.turn_history = []
        self.corners = []
        logger.info(f"Direction tracker initialized with bearing {starting_bearing}°")
    
    def detect_corner(self, current_position: Tuple[float, float], 
                     next_dimension: float) -> CornerInfo:
        """
        Detect if there's a corner at the current position.
        
        Args:
            current_position: Current (x, y) position in feet
            next_dimension: Length of next segment in feet
            
        Returns:
            CornerInfo about the detected corner
        """
        # Convert position to pixels for image analysis
        pixel_pos = self._feet_to_pixels(current_position)
        
        # Method 1: Detect perpendicular lines at position
        line_corner = self._detect_corner_from_lines(pixel_pos)
        
        # Method 2: Check green area boundary
        boundary_corner = self._detect_corner_from_boundary(pixel_pos)
        
        # Method 3: Analyze dimension pattern
        pattern_corner = self._detect_corner_from_pattern(next_dimension)
        
        # Combine detection methods
        corner = self._combine_corner_detections(line_corner, boundary_corner, pattern_corner)
        
        if corner.turn_type != TurnType.STRAIGHT:
            self.corners.append(corner)
            self._update_bearing(corner)
            logger.info(f"Corner detected at {current_position}: {corner.turn_type.value}, "
                       f"new bearing: {self.current_bearing}°")
        
        return corner
    
    def _detect_corner_from_lines(self, pixel_pos: Tuple[int, int]) -> CornerInfo:
        """
        Detect corner from perpendicular lines in image.
        
        Args:
            pixel_pos: Position in pixels
            
        Returns:
            CornerInfo from line detection
        """
        x, y = pixel_pos
        
        # Define region around position
        region_size = 100
        x1 = max(0, x - region_size // 2)
        y1 = max(0, y - region_size // 2)
        x2 = min(self.image.shape[1], x + region_size // 2)
        y2 = min(self.image.shape[0], y + region_size // 2)
        
        roi = self.image[y1:y2, x1:x2]
        
        if roi.size == 0:
            return CornerInfo(pixel_pos, TurnType.STRAIGHT, 0, 0.0, False)
        
        # Detect edges
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, 
                                minLineLength=20, maxLineGap=10)
        
        if lines is None:
            return CornerInfo(pixel_pos, TurnType.STRAIGHT, 0, 0.0, False)
        
        # Classify lines by angle
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = math.atan2(y2 - y1, x2 - x1) * 180 / math.pi
            
            if abs(angle) < 15 or abs(angle) > 165:
                horizontal_lines.append(line)
            elif 75 < abs(angle) < 105:
                vertical_lines.append(line)
        
        # Check for perpendicular intersection
        if horizontal_lines and vertical_lines:
            # Find intersection point
            h_line = horizontal_lines[0][0]
            v_line = vertical_lines[0][0]
            
            # Simple intersection check
            if self._lines_intersect(h_line, v_line):
                # Determine turn direction based on current bearing
                turn_type = self._determine_turn_type()
                return CornerInfo(pixel_pos, turn_type, 90, 0.8, False)
        
        return CornerInfo(pixel_pos, TurnType.STRAIGHT, 0, 0.2, False)
    
    def _detect_corner_from_boundary(self, pixel_pos: Tuple[int, int]) -> CornerInfo:
        """
        Detect corner from green area boundary.
        
        Args:
            pixel_pos: Position in pixels
            
        Returns:
            CornerInfo from boundary detection
        """
        # Convert to HSV for green detection
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        
        # Define green range
        lower_green = np.array([35, 30, 30])
        upper_green = np.array([85, 255, 255])
        
        # Create mask
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Check gradient around position
        x, y = pixel_pos
        window = 20
        
        if (x - window >= 0 and x + window < mask.shape[1] and 
            y - window >= 0 and y + window < mask.shape[0]):
            
            # Sample points around position
            north = mask[y - window, x]
            south = mask[y + window, x]
            east = mask[y, x + window]
            west = mask[y, x - window]
            
            # Check for corner patterns
            if self.current_bearing == 0:  # Going East
                if north > 0 and east == 0:  # Wall ahead, can go north
                    return CornerInfo(pixel_pos, TurnType.LEFT_90, 90, 0.7, False)
                elif south > 0 and east == 0:  # Wall ahead, can go south
                    return CornerInfo(pixel_pos, TurnType.RIGHT_90, 90, 0.7, False)
            
            elif self.current_bearing == 90:  # Going South
                if east > 0 and south == 0:  # Wall ahead, can go east
                    return CornerInfo(pixel_pos, TurnType.LEFT_90, 90, 0.7, False)
                elif west > 0 and south == 0:  # Wall ahead, can go west
                    return CornerInfo(pixel_pos, TurnType.RIGHT_90, 90, 0.7, False)
        
        return CornerInfo(pixel_pos, TurnType.STRAIGHT, 0, 0.3, False)
    
    def _detect_corner_from_pattern(self, next_dimension: float) -> CornerInfo:
        """
        Detect corner from dimension pattern.
        
        Args:
            next_dimension: Next dimension value
            
        Returns:
            CornerInfo from pattern detection
        """
        # If dimension suddenly changes significantly, might indicate corner
        if self.turn_history:
            recent_dims = [t.get('dimension', 0) for t in self.turn_history[-3:]]
            if recent_dims:
                avg_recent = sum(recent_dims) / len(recent_dims)
                if abs(next_dimension - avg_recent) > avg_recent * 0.5:
                    # Significant change might indicate corner
                    return CornerInfo((0, 0), TurnType.RIGHT_90, 90, 0.4, False)
        
        return CornerInfo((0, 0), TurnType.STRAIGHT, 0, 0.1, False)
    
    def _combine_corner_detections(self, line_corner: CornerInfo, 
                                  boundary_corner: CornerInfo,
                                  pattern_corner: CornerInfo) -> CornerInfo:
        """
        Combine multiple corner detection methods.
        
        Args:
            line_corner: Corner from line detection
            boundary_corner: Corner from boundary detection
            pattern_corner: Corner from pattern detection
            
        Returns:
            Combined corner information
        """
        # Weight different methods
        corners = [
            (line_corner, 1.0),
            (boundary_corner, 0.8),
            (pattern_corner, 0.5)
        ]
        
        # Calculate weighted confidence
        total_confidence = 0
        for corner, weight in corners:
            if corner.turn_type != TurnType.STRAIGHT:
                total_confidence += corner.confidence * weight
        
        # If high confidence in corner, use highest confidence detection
        if total_confidence > 0.6:
            best_corner = max(corners, key=lambda c: c[0].confidence * c[1])[0]
            return best_corner
        
        # Default to straight
        return CornerInfo(line_corner.position, TurnType.STRAIGHT, 0, total_confidence, False)
    
    def _determine_turn_type(self) -> TurnType:
        """
        Determine turn type based on current bearing.
        
        Returns:
            Type of turn to make
        """
        # For rectangular buildings, we typically make 90-degree turns
        # This is a simplified version - could be enhanced with more logic
        
        # Count existing turns to determine if we should turn right or left
        right_turns = sum(1 for c in self.corners if c.turn_type == TurnType.RIGHT_90)
        left_turns = sum(1 for c in self.corners if c.turn_type == TurnType.LEFT_90)
        
        # For clockwise traversal, prefer right turns
        if right_turns <= 3:  # Haven't completed a rectangle yet
            return TurnType.RIGHT_90
        else:
            return TurnType.LEFT_90
    
    def _update_bearing(self, corner: CornerInfo):
        """
        Update current bearing based on turn.
        
        Args:
            corner: Corner information including turn type
        """
        if corner.turn_type == TurnType.RIGHT_90:
            self.current_bearing = (self.current_bearing + 90) % 360
        elif corner.turn_type == TurnType.LEFT_90:
            self.current_bearing = (self.current_bearing - 90) % 360
        elif corner.turn_type == TurnType.RIGHT_ANGLE:
            self.current_bearing = (self.current_bearing + corner.angle_degrees) % 360
        elif corner.turn_type == TurnType.LEFT_ANGLE:
            self.current_bearing = (self.current_bearing - corner.angle_degrees) % 360
        
        # Record in history
        self.turn_history.append({
            'position': corner.position,
            'turn_type': corner.turn_type,
            'new_bearing': self.current_bearing,
            'angle': corner.angle_degrees
        })
    
    def _lines_intersect(self, line1: Tuple[int, int, int, int], 
                        line2: Tuple[int, int, int, int]) -> bool:
        """
        Check if two lines intersect.
        
        Args:
            line1: First line (x1, y1, x2, y2)
            line2: Second line (x1, y1, x2, y2)
            
        Returns:
            True if lines intersect
        """
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        # Calculate intersection using cross product
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        
        if abs(denom) < 0.001:
            return False  # Lines are parallel
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        
        return 0 <= t <= 1 and 0 <= u <= 1
    
    def _feet_to_pixels(self, position_feet: Tuple[float, float]) -> Tuple[int, int]:
        """
        Convert position from feet to pixels.
        
        Args:
            position_feet: Position in feet
            
        Returns:
            Position in pixels
        """
        # Assuming approximate scale (to be refined with actual scale)
        pixels_per_foot = 10  # This should come from actual scale calculation
        
        x_pixels = int(position_feet[0] * pixels_per_foot)
        y_pixels = int(position_feet[1] * pixels_per_foot)
        
        return (x_pixels, y_pixels)
    
    def get_bearing_vector(self) -> Tuple[float, float]:
        """
        Get unit vector for current bearing.
        
        Returns:
            (dx, dy) unit vector
        """
        # Convert bearing to radians
        bearing_rad = math.radians(self.current_bearing)
        
        # Calculate unit vector
        dx = math.cos(bearing_rad)
        dy = math.sin(bearing_rad)
        
        return (dx, dy)
    
    def validate_turn_sequence(self) -> bool:
        """
        Validate that turn sequence forms a valid polygon.
        
        Returns:
            True if turn sequence is valid
        """
        # For a closed polygon, sum of exterior angles should be 360°
        total_turn = sum(t['angle'] for t in self.turn_history)
        
        # Check if we've made a complete circuit
        expected_turns = 360  # For exterior angles
        
        if abs(total_turn - expected_turns) < 10:  # Allow some tolerance
            logger.info(f"Valid turn sequence: total turn = {total_turn}°")
            return True
        else:
            logger.warning(f"Invalid turn sequence: total turn = {total_turn}°, expected ~{expected_turns}°")
            return False
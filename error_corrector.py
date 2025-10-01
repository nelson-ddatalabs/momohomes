"""
Error Corrector Module
======================
Applies surveying corrections to achieve polygon closure.
"""

import math
import numpy as np
from typing import List, Dict, Tuple, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class CorrectionMethod(Enum):
    """Available correction methods."""
    BOWDITCH = "bowditch"  # Compass Rule
    TRANSIT = "transit"     # Transit Rule
    LEAST_SQUARES = "least_squares"
    PROPORTIONAL = "proportional"


class ErrorCorrector:
    """Applies corrections to polygon dimensions for closure."""
    
    def __init__(self, method: CorrectionMethod = CorrectionMethod.BOWDITCH):
        """
        Initialize error corrector.
        
        Args:
            method: Correction method to use
        """
        self.method = method
        self.corrections_applied = []
        
    def correct_polygon(self, polygon, closure_error: Tuple[float, float]) -> Dict:
        """
        Apply corrections to achieve polygon closure.
        
        Args:
            polygon: Polygon object to correct
            closure_error: (dx, dy) closure error in feet
            
        Returns:
            Dictionary with corrected polygon and correction details
        """
        logger.info(f"Applying {self.method.value} correction for closure error: "
                   f"({closure_error[0]:.3f}, {closure_error[1]:.3f}) feet")
        
        if self.method == CorrectionMethod.BOWDITCH:
            result = self._apply_bowditch_rule(polygon, closure_error)
        elif self.method == CorrectionMethod.TRANSIT:
            result = self._apply_transit_rule(polygon, closure_error)
        elif self.method == CorrectionMethod.LEAST_SQUARES:
            result = self._apply_least_squares(polygon, closure_error)
        else:
            result = self._apply_proportional(polygon, closure_error)
        
        # Verify correction
        new_error = self._calculate_closure_error(result['corrected_vertices'])
        result['final_error'] = new_error
        result['error_reduction'] = math.sqrt(closure_error[0]**2 + closure_error[1]**2) - \
                                   math.sqrt(new_error[0]**2 + new_error[1]**2)
        
        logger.info(f"Correction complete. Final error: ({new_error[0]:.3f}, {new_error[1]:.3f}) feet")
        
        return result
    
    def _apply_bowditch_rule(self, polygon, closure_error: Tuple[float, float]) -> Dict:
        """
        Apply Bowditch Rule (Compass Rule) correction.
        
        Distributes error proportionally to the length of each traverse leg.
        
        Args:
            polygon: Polygon object
            closure_error: Closure error vector
            
        Returns:
            Correction results
        """
        vertices = polygon.vertices
        edges = polygon.edges
        
        # Calculate total perimeter
        total_perimeter = sum(edge.length for edge in edges)
        
        if total_perimeter == 0:
            return {'corrected_vertices': vertices, 'corrections': []}
        
        # Calculate cumulative distances
        cumulative_distances = [0]
        for edge in edges:
            cumulative_distances.append(cumulative_distances[-1] + edge.length)
        
        # Apply corrections to each vertex
        corrected_vertices = []
        corrections = []
        
        for i, vertex in enumerate(vertices):
            if i == 0:
                # First vertex stays fixed
                corrected_vertices.append(vertex)
                corrections.append((0, 0))
            else:
                # Calculate correction based on cumulative distance
                distance_ratio = cumulative_distances[i] / total_perimeter
                
                dx_correction = -closure_error[0] * distance_ratio
                dy_correction = -closure_error[1] * distance_ratio
                
                # Apply correction
                corrected_x = vertex.x + dx_correction
                corrected_y = vertex.y + dy_correction
                
                # Create corrected vertex
                corrected_vertex = type(vertex)(
                    x=corrected_x,
                    y=corrected_y,
                    index=vertex.index,
                    is_corner=vertex.is_corner
                )
                
                corrected_vertices.append(corrected_vertex)
                corrections.append((dx_correction, dy_correction))
                
                logger.debug(f"Vertex {i}: correction = ({dx_correction:.3f}, {dy_correction:.3f})")
        
        return {
            'corrected_vertices': corrected_vertices,
            'corrections': corrections,
            'method': 'Bowditch Rule',
            'total_perimeter': total_perimeter
        }
    
    def _apply_transit_rule(self, polygon, closure_error: Tuple[float, float]) -> Dict:
        """
        Apply Transit Rule correction.
        
        Distributes error based on the latitude and departure of each leg.
        
        Args:
            polygon: Polygon object
            closure_error: Closure error vector
            
        Returns:
            Correction results
        """
        vertices = polygon.vertices
        edges = polygon.edges
        
        # Calculate total latitude and departure
        total_latitude = sum(abs(edge.end_vertex.y - edge.start_vertex.y) for edge in edges)
        total_departure = sum(abs(edge.end_vertex.x - edge.start_vertex.x) for edge in edges)
        
        if total_latitude == 0 or total_departure == 0:
            # Fall back to Bowditch if can't apply Transit
            return self._apply_bowditch_rule(polygon, closure_error)
        
        # Apply corrections
        corrected_vertices = []
        corrections = []
        cumulative_lat = 0
        cumulative_dep = 0
        
        for i, vertex in enumerate(vertices):
            if i == 0:
                corrected_vertices.append(vertex)
                corrections.append((0, 0))
            else:
                # Calculate cumulative latitude and departure
                edge = edges[i-1]
                cumulative_lat += abs(edge.end_vertex.y - edge.start_vertex.y)
                cumulative_dep += abs(edge.end_vertex.x - edge.start_vertex.x)
                
                # Calculate corrections
                dx_correction = -closure_error[0] * (cumulative_dep / total_departure)
                dy_correction = -closure_error[1] * (cumulative_lat / total_latitude)
                
                # Apply correction
                corrected_x = vertex.x + dx_correction
                corrected_y = vertex.y + dy_correction
                
                corrected_vertex = type(vertex)(
                    x=corrected_x,
                    y=corrected_y,
                    index=vertex.index,
                    is_corner=vertex.is_corner
                )
                
                corrected_vertices.append(corrected_vertex)
                corrections.append((dx_correction, dy_correction))
        
        return {
            'corrected_vertices': corrected_vertices,
            'corrections': corrections,
            'method': 'Transit Rule',
            'total_latitude': total_latitude,
            'total_departure': total_departure
        }
    
    def _apply_least_squares(self, polygon, closure_error: Tuple[float, float]) -> Dict:
        """
        Apply Least Squares adjustment.
        
        Minimizes the sum of squared adjustments.
        
        Args:
            polygon: Polygon object
            closure_error: Closure error vector
            
        Returns:
            Correction results
        """
        vertices = polygon.vertices
        n = len(vertices)
        
        if n < 3:
            return {'corrected_vertices': vertices, 'corrections': []}
        
        # Build observation matrix
        A = np.zeros((2, n-1))
        
        for i in range(n-1):
            A[0, i] = 1  # X corrections
            A[1, i] = 1  # Y corrections
        
        # Error vector
        b = np.array([-closure_error[0], -closure_error[1]])
        
        # Solve using least squares
        # x = (A^T A)^-1 A^T b
        try:
            ATA = A @ A.T
            if np.linalg.det(ATA) != 0:
                corrections_vec = np.linalg.inv(ATA) @ A @ b
            else:
                # Fall back to equal distribution
                corrections_vec = b / (n - 1)
        except:
            # Fall back to Bowditch
            return self._apply_bowditch_rule(polygon, closure_error)
        
        # Apply corrections
        corrected_vertices = []
        corrections = []
        
        for i, vertex in enumerate(vertices):
            if i == 0:
                corrected_vertices.append(vertex)
                corrections.append((0, 0))
            else:
                dx_correction = corrections_vec[0] * i / (n - 1)
                dy_correction = corrections_vec[1] * i / (n - 1)
                
                corrected_x = vertex.x + dx_correction
                corrected_y = vertex.y + dy_correction
                
                corrected_vertex = type(vertex)(
                    x=corrected_x,
                    y=corrected_y,
                    index=vertex.index,
                    is_corner=vertex.is_corner
                )
                
                corrected_vertices.append(corrected_vertex)
                corrections.append((dx_correction, dy_correction))
        
        return {
            'corrected_vertices': corrected_vertices,
            'corrections': corrections,
            'method': 'Least Squares'
        }
    
    def _apply_proportional(self, polygon, closure_error: Tuple[float, float]) -> Dict:
        """
        Apply simple proportional correction.
        
        Args:
            polygon: Polygon object
            closure_error: Closure error vector
            
        Returns:
            Correction results
        """
        vertices = polygon.vertices
        n = len(vertices)
        
        if n < 2:
            return {'corrected_vertices': vertices, 'corrections': []}
        
        # Distribute error equally among all vertices except the first
        dx_per_vertex = -closure_error[0] / (n - 1)
        dy_per_vertex = -closure_error[1] / (n - 1)
        
        corrected_vertices = []
        corrections = []
        
        for i, vertex in enumerate(vertices):
            if i == 0:
                corrected_vertices.append(vertex)
                corrections.append((0, 0))
            else:
                dx_correction = dx_per_vertex * i
                dy_correction = dy_per_vertex * i
                
                corrected_x = vertex.x + dx_correction
                corrected_y = vertex.y + dy_correction
                
                corrected_vertex = type(vertex)(
                    x=corrected_x,
                    y=corrected_y,
                    index=vertex.index,
                    is_corner=vertex.is_corner
                )
                
                corrected_vertices.append(corrected_vertex)
                corrections.append((dx_correction, dy_correction))
        
        return {
            'corrected_vertices': corrected_vertices,
            'corrections': corrections,
            'method': 'Proportional'
        }
    
    def correct_ocr_errors(self, dimensions: List[Dict]) -> List[Dict]:
        """
        Detect and correct likely OCR errors in dimensions.
        
        Args:
            dimensions: List of dimension dictionaries
            
        Returns:
            Corrected dimensions
        """
        corrected = []
        
        for i, dim in enumerate(dimensions):
            corrected_dim = dim.copy()
            
            # Check for common OCR errors
            if 'text' in dim:
                text = dim['text']
                
                # Common substitutions
                replacements = {
                    'O': '0',  # Letter O to zero
                    'l': '1',  # Lowercase L to one
                    'S': '5',  # S to 5
                    'B': '8',  # B to 8
                    'G': '6',  # G to 6
                }
                
                for old, new in replacements.items():
                    if old in text and not text[0].isalpha():
                        # Only replace in numeric context
                        text = text.replace(old, new)
                        logger.debug(f"OCR correction: {dim['text']} -> {text}")
                
                corrected_dim['text'] = text
            
            # Check for unrealistic dimensions
            if 'length' in dim:
                length = dim['length']
                
                # Check against neighboring dimensions
                if i > 0 and i < len(dimensions) - 1:
                    prev_length = dimensions[i-1].get('length', length)
                    next_length = dimensions[i+1].get('length', length)
                    avg_neighbor = (prev_length + next_length) / 2
                    
                    # If this dimension is way off from neighbors
                    if length > avg_neighbor * 3 or length < avg_neighbor / 3:
                        logger.warning(f"Suspicious dimension: {length} ft (neighbors avg: {avg_neighbor:.1f} ft)")
                        corrected_dim['confidence'] = dim.get('confidence', 1.0) * 0.5
            
            corrected.append(corrected_dim)
        
        return corrected
    
    def _calculate_closure_error(self, vertices: List) -> Tuple[float, float]:
        """
        Calculate closure error for a set of vertices.
        
        Args:
            vertices: List of vertices
            
        Returns:
            (dx, dy) closure error
        """
        if len(vertices) < 2:
            return (0, 0)
        
        first = vertices[0]
        last = vertices[-1]
        
        dx = last.x - first.x
        dy = last.y - first.y
        
        return (dx, dy)
    
    def interpolate_missing_dimension(self, before: float, after: float, 
                                     confidence: float = 0.5) -> float:
        """
        Interpolate a missing dimension.
        
        Args:
            before: Dimension before the missing one
            after: Dimension after the missing one
            confidence: Confidence in interpolation
            
        Returns:
            Interpolated dimension value
        """
        # Simple average for now
        interpolated = (before + after) / 2
        
        logger.info(f"Interpolated missing dimension: {interpolated:.1f} ft "
                   f"(from {before:.1f} and {after:.1f})")
        
        return interpolated
"""
Closure Validator Module
========================
Validates polygon closure and dimensional consistency.
"""

import math
from typing import List, Tuple, Dict, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ClosureValidator:
    """Validates polygon closure and consistency."""
    
    def __init__(self, tolerance_feet: float = 1.0, angle_tolerance_degrees: float = 5.0):
        """
        Initialize closure validator.
        
        Args:
            tolerance_feet: Maximum acceptable closure error in feet
            angle_tolerance_degrees: Maximum acceptable angle error in degrees
        """
        self.tolerance_feet = tolerance_feet
        self.angle_tolerance_degrees = angle_tolerance_degrees
        self.validation_results = {}
        
    def validate_polygon(self, polygon) -> Dict:
        """
        Perform comprehensive polygon validation.
        
        Args:
            polygon: Polygon object from PolygonBuilder
            
        Returns:
            Dictionary with validation results
        """
        logger.info("Starting polygon validation")
        
        results = {
            'valid': True,
            'geometric_closure': self._check_geometric_closure(polygon),
            'angular_closure': self._check_angular_closure(polygon),
            'dimensional_consistency': self._check_dimensional_consistency(polygon),
            'self_intersection': self._check_self_intersection(polygon),
            'area_validation': self._check_area_bounds(polygon),
            'errors': [],
            'warnings': []
        }
        
        # Compile overall validity
        if not results['geometric_closure']['valid']:
            results['valid'] = False
            results['errors'].append(f"Geometric closure failed: {results['geometric_closure']['error_feet']:.2f} ft")
        
        if not results['angular_closure']['valid']:
            results['valid'] = False
            results['errors'].append(f"Angular closure failed: {results['angular_closure']['error_degrees']:.1f}°")
        
        if not results['dimensional_consistency']['valid']:
            results['warnings'].append("Dimensional inconsistencies detected")
        
        if results['self_intersection']['has_intersection']:
            results['valid'] = False
            results['errors'].append("Polygon has self-intersections")
        
        if not results['area_validation']['valid']:
            results['warnings'].append(f"Area outside expected bounds: {polygon.area:.0f} sq ft")
        
        self.validation_results = results
        
        logger.info(f"Validation complete: {'PASS' if results['valid'] else 'FAIL'}")
        
        return results
    
    def _check_geometric_closure(self, polygon) -> Dict:
        """
        Check if polygon closes geometrically.
        
        Args:
            polygon: Polygon object
            
        Returns:
            Closure check results
        """
        if len(polygon.vertices) < 3:
            return {
                'valid': False,
                'error_feet': float('inf'),
                'message': 'Insufficient vertices'
            }
        
        # Calculate closure error
        first = polygon.vertices[0]
        last = polygon.vertices[-1]
        
        dx = last.x - first.x
        dy = last.y - first.y
        error = math.sqrt(dx**2 + dy**2)
        
        valid = error <= self.tolerance_feet
        
        result = {
            'valid': valid,
            'error_feet': error,
            'error_vector': (dx, dy),
            'tolerance': self.tolerance_feet
        }
        
        if valid:
            logger.info(f"Geometric closure PASS: error = {error:.3f} ft")
        else:
            logger.warning(f"Geometric closure FAIL: error = {error:.3f} ft > {self.tolerance_feet} ft")
        
        return result
    
    def _check_angular_closure(self, polygon) -> Dict:
        """
        Check if interior angles sum correctly.
        
        Args:
            polygon: Polygon object
            
        Returns:
            Angular closure check results
        """
        n = len(polygon.vertices)
        if n < 3:
            return {
                'valid': False,
                'error_degrees': float('inf'),
                'message': 'Insufficient vertices'
            }
        
        # Calculate interior angles
        interior_angles = []
        
        for i in range(n):
            # Get three consecutive vertices
            p1 = polygon.vertices[i]
            p2 = polygon.vertices[(i + 1) % n]
            p3 = polygon.vertices[(i + 2) % n]
            
            # Calculate vectors
            v1 = (p1.x - p2.x, p1.y - p2.y)
            v2 = (p3.x - p2.x, p3.y - p2.y)
            
            # Calculate angle
            angle = self._calculate_angle(v1, v2)
            interior_angles.append(angle)
        
        # Sum should be (n-2) * 180 for interior angles
        expected_sum = (n - 2) * 180
        actual_sum = sum(interior_angles)
        error = abs(actual_sum - expected_sum)
        
        valid = error <= self.angle_tolerance_degrees
        
        result = {
            'valid': valid,
            'expected_sum': expected_sum,
            'actual_sum': actual_sum,
            'error_degrees': error,
            'interior_angles': interior_angles,
            'num_vertices': n
        }
        
        if valid:
            logger.info(f"Angular closure PASS: sum = {actual_sum:.1f}°, expected = {expected_sum}°")
        else:
            logger.warning(f"Angular closure FAIL: sum = {actual_sum:.1f}°, expected = {expected_sum}°, error = {error:.1f}°")
        
        return result
    
    def _check_dimensional_consistency(self, polygon) -> Dict:
        """
        Check dimensional consistency (e.g., opposite sides of rectangles).
        
        Args:
            polygon: Polygon object
            
        Returns:
            Consistency check results
        """
        inconsistencies = []
        
        # For rectangles, check opposite sides
        if len(polygon.edges) == 4:
            # Check if opposite edges have similar lengths
            edge_lengths = [e.length for e in polygon.edges]
            
            # Compare opposite edges
            diff1 = abs(edge_lengths[0] - edge_lengths[2])
            diff2 = abs(edge_lengths[1] - edge_lengths[3])
            
            tolerance = 2.0  # 2 feet tolerance for opposite sides
            
            if diff1 > tolerance:
                inconsistencies.append({
                    'type': 'opposite_sides',
                    'edges': [0, 2],
                    'lengths': [edge_lengths[0], edge_lengths[2]],
                    'difference': diff1
                })
            
            if diff2 > tolerance:
                inconsistencies.append({
                    'type': 'opposite_sides',
                    'edges': [1, 3],
                    'lengths': [edge_lengths[1], edge_lengths[3]],
                    'difference': diff2
                })
        
        # Check for parallel walls alignment
        parallel_pairs = self._find_parallel_edges(polygon)
        for pair in parallel_pairs:
            edge1, edge2 = pair
            # Check if parallel edges are properly aligned
            alignment_error = self._check_parallel_alignment(edge1, edge2)
            if alignment_error > 1.0:
                inconsistencies.append({
                    'type': 'parallel_alignment',
                    'edges': [edge1, edge2],
                    'error': alignment_error
                })
        
        valid = len(inconsistencies) == 0
        
        result = {
            'valid': valid,
            'inconsistencies': inconsistencies,
            'num_checks': len(parallel_pairs) + (2 if len(polygon.edges) == 4 else 0)
        }
        
        if valid:
            logger.info("Dimensional consistency PASS")
        else:
            logger.warning(f"Dimensional consistency issues: {len(inconsistencies)} found")
        
        return result
    
    def _check_self_intersection(self, polygon) -> Dict:
        """
        Check if polygon has self-intersections.
        
        Args:
            polygon: Polygon object
            
        Returns:
            Self-intersection check results
        """
        intersections = []
        
        edges = polygon.edges
        n = len(edges)
        
        # Check each pair of non-adjacent edges
        for i in range(n):
            for j in range(i + 2, n):
                # Skip adjacent edges
                if j == (i + 1) % n or i == (j + 1) % n:
                    continue
                
                edge1 = edges[i]
                edge2 = edges[j]
                
                # Check for intersection
                intersection = self._line_intersection(
                    (edge1.start_vertex.x, edge1.start_vertex.y),
                    (edge1.end_vertex.x, edge1.end_vertex.y),
                    (edge2.start_vertex.x, edge2.start_vertex.y),
                    (edge2.end_vertex.x, edge2.end_vertex.y)
                )
                
                if intersection:
                    intersections.append({
                        'edges': [i, j],
                        'point': intersection
                    })
        
        has_intersection = len(intersections) > 0
        
        result = {
            'has_intersection': has_intersection,
            'intersections': intersections,
            'num_intersections': len(intersections)
        }
        
        if has_intersection:
            logger.warning(f"Self-intersection detected: {len(intersections)} intersections")
        else:
            logger.info("No self-intersections detected")
        
        return result
    
    def _check_area_bounds(self, polygon) -> Dict:
        """
        Check if area is within reasonable bounds.
        
        Args:
            polygon: Polygon object
            
        Returns:
            Area validation results
        """
        area = polygon.area
        
        # Reasonable bounds for residential floor plans
        min_area = 500   # 500 sq ft minimum
        max_area = 10000 # 10,000 sq ft maximum
        
        valid = min_area <= area <= max_area
        
        result = {
            'valid': valid,
            'area': area,
            'min_expected': min_area,
            'max_expected': max_area
        }
        
        if valid:
            logger.info(f"Area validation PASS: {area:.0f} sq ft")
        else:
            logger.warning(f"Area validation FAIL: {area:.0f} sq ft not in [{min_area}, {max_area}]")
        
        return result
    
    def _calculate_angle(self, v1: Tuple[float, float], 
                        v2: Tuple[float, float]) -> float:
        """
        Calculate angle between two vectors in degrees.
        
        Args:
            v1: First vector
            v2: Second vector
            
        Returns:
            Angle in degrees
        """
        # Calculate dot product and magnitudes
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
        
        if mag1 == 0 or mag2 == 0:
            return 0
        
        # Calculate angle
        cos_angle = dot / (mag1 * mag2)
        cos_angle = max(-1, min(1, cos_angle))  # Clamp to [-1, 1]
        
        angle_rad = math.acos(cos_angle)
        angle_deg = math.degrees(angle_rad)
        
        return angle_deg
    
    def _find_parallel_edges(self, polygon) -> List[Tuple[int, int]]:
        """
        Find pairs of parallel edges.
        
        Args:
            polygon: Polygon object
            
        Returns:
            List of parallel edge pairs (indices)
        """
        parallel_pairs = []
        edges = polygon.edges
        n = len(edges)
        
        angle_tolerance = 5.0  # Degrees
        
        for i in range(n):
            for j in range(i + 1, n):
                bearing1 = edges[i].bearing
                bearing2 = edges[j].bearing
                
                # Check if parallel (same or opposite direction)
                diff = abs(bearing1 - bearing2)
                if diff > 180:
                    diff = 360 - diff
                
                if diff < angle_tolerance or abs(diff - 180) < angle_tolerance:
                    parallel_pairs.append((i, j))
        
        return parallel_pairs
    
    def _check_parallel_alignment(self, edge1_idx: int, edge2_idx: int) -> float:
        """
        Check alignment of parallel edges.
        
        Args:
            edge1_idx: First edge index
            edge2_idx: Second edge index
            
        Returns:
            Alignment error in feet
        """
        # Simplified alignment check
        # Would need actual edge objects to properly implement
        return 0.0
    
    def _line_intersection(self, p1: Tuple[float, float], p2: Tuple[float, float],
                          p3: Tuple[float, float], p4: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """
        Find intersection point of two line segments.
        
        Args:
            p1, p2: First line segment endpoints
            p3, p4: Second line segment endpoints
            
        Returns:
            Intersection point or None
        """
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        
        if abs(denom) < 1e-10:
            return None  # Lines are parallel
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        
        if 0 <= t <= 1 and 0 <= u <= 1:
            # Intersection exists
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            return (x, y)
        
        return None
    
    def get_closure_error_vector(self, polygon) -> Tuple[float, float]:
        """
        Get the closure error vector.
        
        Args:
            polygon: Polygon object
            
        Returns:
            (dx, dy) closure error in feet
        """
        if len(polygon.vertices) < 2:
            return (0, 0)
        
        first = polygon.vertices[0]
        last = polygon.vertices[-1]
        
        dx = last.x - first.x
        dy = last.y - first.y
        
        return (dx, dy)
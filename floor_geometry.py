"""
Floor Geometry Module
=====================
Handles polygon operations, area calculations, and geometric transformations
for floor plan processing.
"""

import numpy as np
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass
from cassette_models import Point, FloorBoundary


@dataclass
class Rectangle:
    """Represents an axis-aligned rectangle."""
    x: float
    y: float
    width: float
    height: float
    
    @property
    def area(self) -> float:
        """Calculate rectangle area."""
        return self.width * self.height
    
    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Get bounding box (x1, y1, x2, y2)."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)
    
    @property
    def center(self) -> Point:
        """Get center point."""
        return Point(self.x + self.width / 2, self.y + self.height / 2)
    
    @property
    def corners(self) -> List[Point]:
        """Get all four corners."""
        return [
            Point(self.x, self.y),
            Point(self.x + self.width, self.y),
            Point(self.x + self.width, self.y + self.height),
            Point(self.x, self.y + self.height)
        ]
    
    def contains_point(self, point: Point) -> bool:
        """Check if point is inside rectangle."""
        return (self.x <= point.x <= self.x + self.width and 
                self.y <= point.y <= self.y + self.height)
    
    def intersects(self, other: 'Rectangle') -> bool:
        """Check if this rectangle intersects with another."""
        x1, y1, x2, y2 = self.bounds
        ox1, oy1, ox2, oy2 = other.bounds
        
        return not (x2 <= ox1 or ox2 <= x1 or y2 <= oy1 or oy2 <= y1)
    
    def intersection_area(self, other: 'Rectangle') -> float:
        """Calculate intersection area with another rectangle."""
        if not self.intersects(other):
            return 0.0
        
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x + self.width, other.x + other.width)
        y2 = min(self.y + self.height, other.y + other.height)
        
        return max(0, x2 - x1) * max(0, y2 - y1)


class GeometryUtils:
    """Utility functions for geometric operations."""
    
    @staticmethod
    def calculate_polygon_area(points: List[Point]) -> float:
        """
        Calculate area of a polygon using the shoelace formula.
        Points should be in order (clockwise or counter-clockwise).
        """
        if len(points) < 3:
            return 0.0
        
        n = len(points)
        area = 0.0
        
        for i in range(n):
            j = (i + 1) % n
            area += points[i].x * points[j].y
            area -= points[j].x * points[i].y
        
        return abs(area) / 2.0
    
    @staticmethod
    def get_bounding_box(points: List[Point]) -> Rectangle:
        """Get axis-aligned bounding box for a set of points."""
        if not points:
            return Rectangle(0, 0, 0, 0)
        
        xs = [p.x for p in points]
        ys = [p.y for p in points]
        
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        return Rectangle(min_x, min_y, max_x - min_x, max_y - min_y)
    
    @staticmethod
    def point_in_polygon(point: Point, polygon: List[Point]) -> bool:
        """
        Check if a point is inside a polygon using ray casting algorithm.
        """
        n = len(polygon)
        inside = False
        
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i].x, polygon[i].y
            xj, yj = polygon[j].x, polygon[j].y
            
            if ((yi > point.y) != (yj > point.y)) and \
               (point.x < (xj - xi) * (point.y - yi) / (yj - yi) + xi):
                inside = not inside
            
            j = i
        
        return inside
    
    @staticmethod
    def simplify_polygon(points: List[Point], tolerance: float = 0.5) -> List[Point]:
        """
        Simplify a polygon by removing redundant points.
        Uses Douglas-Peucker algorithm.
        """
        if len(points) <= 3:
            return points
        
        # Find the point with maximum distance from line between first and last
        max_dist = 0
        max_idx = 0
        
        for i in range(1, len(points) - 1):
            dist = GeometryUtils._perpendicular_distance(
                points[i], points[0], points[-1]
            )
            if dist > max_dist:
                max_dist = dist
                max_idx = i
        
        # If max distance is greater than tolerance, recursively simplify
        if max_dist > tolerance:
            left = GeometryUtils.simplify_polygon(points[:max_idx + 1], tolerance)
            right = GeometryUtils.simplify_polygon(points[max_idx:], tolerance)
            return left[:-1] + right
        else:
            return [points[0], points[-1]]
    
    @staticmethod
    def _perpendicular_distance(point: Point, line_start: Point, line_end: Point) -> float:
        """Calculate perpendicular distance from point to line."""
        if line_start == line_end:
            return point.distance_to(line_start)
        
        num = abs(
            (line_end.y - line_start.y) * point.x -
            (line_end.x - line_start.x) * point.y +
            line_end.x * line_start.y -
            line_end.y * line_start.x
        )
        den = np.sqrt(
            (line_end.y - line_start.y) ** 2 +
            (line_end.x - line_start.x) ** 2
        )
        
        return num / den if den > 0 else 0
    
    @staticmethod
    def decompose_to_rectangles(polygon: List[Point], 
                               min_area: float = 16.0) -> List[Rectangle]:
        """
        Decompose a polygon into rectangles for easier cassette placement.
        Uses a simple grid-based approach.
        """
        bbox = GeometryUtils.get_bounding_box(polygon)
        rectangles = []
        
        # Grid size based on smallest cassette (4x4)
        grid_size = 4.0
        
        # Scan through the bounding box in grid cells
        y = bbox.y
        while y < bbox.y + bbox.height:
            x = bbox.x
            while x < bbox.x + bbox.width:
                # Check if this grid cell is mostly inside the polygon
                cell_center = Point(x + grid_size/2, y + grid_size/2)
                
                if GeometryUtils.point_in_polygon(cell_center, polygon):
                    # Try to expand this rectangle as much as possible
                    width = grid_size
                    height = grid_size
                    
                    # Expand horizontally
                    while x + width < bbox.x + bbox.width:
                        test_point = Point(x + width + grid_size/2, y + grid_size/2)
                        if GeometryUtils.point_in_polygon(test_point, polygon):
                            width += grid_size
                        else:
                            break
                    
                    # Expand vertically
                    while y + height < bbox.y + bbox.height:
                        test_point = Point(x + grid_size/2, y + height + grid_size/2)
                        if GeometryUtils.point_in_polygon(test_point, polygon):
                            height += grid_size
                        else:
                            break
                    
                    rect = Rectangle(x, y, width, height)
                    if rect.area >= min_area:
                        rectangles.append(rect)
                
                x += grid_size
            y += grid_size
        
        # Merge overlapping rectangles
        merged = GeometryUtils._merge_rectangles(rectangles)
        
        return merged
    
    @staticmethod
    def _merge_rectangles(rectangles: List[Rectangle]) -> List[Rectangle]:
        """Merge overlapping rectangles to reduce redundancy."""
        if len(rectangles) <= 1:
            return rectangles
        
        merged = []
        used = [False] * len(rectangles)
        
        for i, rect1 in enumerate(rectangles):
            if used[i]:
                continue
            
            # Try to merge with other rectangles
            current = rect1
            merged_any = True
            
            while merged_any:
                merged_any = False
                for j, rect2 in enumerate(rectangles):
                    if i == j or used[j]:
                        continue
                    
                    # Check if rectangles can be merged (adjacent or overlapping)
                    if current.intersects(rect2):
                        # Create merged rectangle
                        x1 = min(current.x, rect2.x)
                        y1 = min(current.y, rect2.y)
                        x2 = max(current.x + current.width, rect2.x + rect2.width)
                        y2 = max(current.y + current.height, rect2.y + rect2.height)
                        
                        current = Rectangle(x1, y1, x2 - x1, y2 - y1)
                        used[j] = True
                        merged_any = True
            
            merged.append(current)
            used[i] = True
        
        return merged
    
    @staticmethod
    def calculate_coverage(cassette_bounds: List[Tuple[float, float, float, float]], 
                          floor_boundary: FloorBoundary) -> float:
        """
        Calculate the coverage percentage of cassettes within floor boundary.
        """
        if not cassette_bounds or not floor_boundary.points:
            return 0.0
        
        # Simple approximation using bounding box
        # For production, use more sophisticated polygon intersection
        total_cassette_area = 0.0
        
        for bounds in cassette_bounds:
            x1, y1, x2, y2 = bounds
            # Check if cassette center is within boundary
            center = Point((x1 + x2) / 2, (y1 + y2) / 2)
            if floor_boundary.contains_point(center):
                total_cassette_area += (x2 - x1) * (y2 - y1)
        
        floor_area = floor_boundary.area
        if floor_area > 0:
            return (total_cassette_area / floor_area) * 100
        
        return 0.0
    
    @staticmethod
    def find_uncovered_regions(cassette_bounds: List[Tuple[float, float, float, float]],
                              floor_boundary: FloorBoundary,
                              min_area: float = 4.0) -> List[Rectangle]:
        """
        Find regions not covered by cassettes that need custom work.
        """
        uncovered = []
        bbox = floor_boundary.bounding_box
        
        if not bbox:
            return uncovered
        
        # Grid-based approach to find uncovered areas
        grid_size = 2.0  # 2x2 ft grid for finer resolution
        
        x1, y1, x2, y2 = bbox
        y = y1
        
        while y < y2:
            x = x1
            while x < x2:
                point = Point(x + grid_size/2, y + grid_size/2)
                
                # Check if point is in floor boundary
                if floor_boundary.contains_point(point):
                    # Check if point is covered by any cassette
                    covered = False
                    for bounds in cassette_bounds:
                        cx1, cy1, cx2, cy2 = bounds
                        if cx1 <= point.x <= cx2 and cy1 <= point.y <= cy2:
                            covered = True
                            break
                    
                    if not covered:
                        # Create a small rectangle for this uncovered area
                        rect = Rectangle(x, y, grid_size, grid_size)
                        uncovered.append(rect)
                
                x += grid_size
            y += grid_size
        
        # Merge adjacent uncovered rectangles
        merged = GeometryUtils._merge_rectangles(uncovered)
        
        # Filter out very small areas
        significant = [r for r in merged if r.area >= min_area]
        
        return significant
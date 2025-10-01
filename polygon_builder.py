"""
Polygon Builder Module
======================
Constructs polygon vertices from sequential dimensions and bearings.
"""

import math
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class Vertex:
    """Represents a polygon vertex."""
    x: float  # X coordinate in feet
    y: float  # Y coordinate in feet
    index: int  # Vertex index in sequence
    is_corner: bool = False
    metadata: Dict = field(default_factory=dict)


@dataclass
class Edge:
    """Represents a polygon edge."""
    start_vertex: Vertex
    end_vertex: Vertex
    length: float  # Length in feet
    bearing: float  # Bearing in degrees
    dimension_text: str = ""
    room_label: Optional[str] = None
    confidence: float = 1.0


@dataclass
class Polygon:
    """Represents the complete floor plan polygon."""
    vertices: List[Vertex]
    edges: List[Edge]
    is_closed: bool = False
    closure_error: float = 0.0
    area: float = 0.0
    perimeter: float = 0.0


class PolygonBuilder:
    """Builds polygon from sequential dimensions and bearings."""
    
    def __init__(self, origin: Tuple[float, float] = (0, 0)):
        """
        Initialize polygon builder.
        
        Args:
            origin: Starting point coordinates in feet
        """
        self.origin = origin
        self.current_position = list(origin)
        self.vertices = []
        self.edges = []
        self.vertex_index = 0
        
    def build_polygon(self, dimensions: List[Dict], 
                     direction_tracker) -> Polygon:
        """
        Build polygon from dimensions and direction information.
        
        Args:
            dimensions: List of dimension dictionaries with 'length', 'bearing', etc.
            direction_tracker: DirectionTracker instance for bearing management
            
        Returns:
            Complete Polygon object
        """
        logger.info(f"Building polygon from {len(dimensions)} dimensions")
        
        # Add origin as first vertex
        self._add_vertex(self.origin[0], self.origin[1], is_corner=True)
        
        # Process each dimension
        for i, dim in enumerate(dimensions):
            length = dim.get('length', 0)
            bearing = dim.get('bearing', direction_tracker.current_bearing)
            
            # Calculate next vertex position
            next_pos = self._calculate_next_position(length, bearing)
            
            # Check if this is a corner
            is_corner = dim.get('is_corner', False)
            
            # Add vertex and edge
            vertex = self._add_vertex(next_pos[0], next_pos[1], is_corner)
            edge = self._add_edge(self.vertices[-2], vertex, length, bearing, dim)
            
            # Update current position
            self.current_position = next_pos
            
            logger.debug(f"Added vertex {self.vertex_index} at ({next_pos[0]:.1f}, {next_pos[1]:.1f})")
        
        # Create polygon object
        polygon = self._finalize_polygon()
        
        return polygon
    
    def build_from_perimeter_dimensions(self, perimeter_dims: Dict) -> Polygon:
        """
        Build polygon from perimeter dimensions extracted by DimensionSequenceExtractor.
        
        Args:
            perimeter_dims: Dictionary of dimensions by edge direction
            
        Returns:
            Complete Polygon object
        """
        from dimension_sequence_extractor import EdgeDirection
        
        # Add origin vertex
        self._add_vertex(self.origin[0], self.origin[1], is_corner=True)
        
        current_bearing = 0  # Start facing East
        
        # Process dimensions in order
        edge_order = [EdgeDirection.NORTH, EdgeDirection.EAST, 
                     EdgeDirection.SOUTH, EdgeDirection.WEST]
        
        for edge_dir in edge_order:
            if edge_dir not in perimeter_dims:
                continue
            
            dimensions = perimeter_dims[edge_dir]
            
            # Set bearing based on edge direction
            bearings = {
                EdgeDirection.NORTH: 0,    # Going East along top
                EdgeDirection.EAST: 90,    # Going South along right
                EdgeDirection.SOUTH: 180,  # Going West along bottom
                EdgeDirection.WEST: 270    # Going North along left
            }
            current_bearing = bearings[edge_dir]
            
            # Process dimensions for this edge
            for dim in dimensions:
                length = dim.total_feet
                
                # Calculate next position
                next_pos = self._calculate_next_position(length, current_bearing)
                
                # Add vertex
                vertex = self._add_vertex(next_pos[0], next_pos[1], is_corner=False)
                
                # Add edge
                edge_data = {
                    'length': length,
                    'text': dim.text,
                    'room_label': dim.room_label,
                    'confidence': dim.confidence
                }
                self._add_edge(self.vertices[-2], vertex, length, current_bearing, edge_data)
                
                # Update position
                self.current_position = next_pos
            
            # Mark corner at end of edge
            if self.vertices:
                self.vertices[-1].is_corner = True
        
        # Create final polygon
        polygon = self._finalize_polygon()
        
        return polygon
    
    def _calculate_next_position(self, distance: float, bearing: float) -> List[float]:
        """
        Calculate next position from current position, distance, and bearing.
        
        Args:
            distance: Distance in feet
            bearing: Bearing in degrees (0=East, 90=South, 180=West, 270=North)
            
        Returns:
            [x, y] coordinates of next position
        """
        # Convert bearing to radians
        bearing_rad = math.radians(bearing)
        
        # Calculate displacement
        dx = distance * math.cos(bearing_rad)
        dy = distance * math.sin(bearing_rad)
        
        # Calculate new position
        new_x = self.current_position[0] + dx
        new_y = self.current_position[1] + dy
        
        return [new_x, new_y]
    
    def _add_vertex(self, x: float, y: float, is_corner: bool = False) -> Vertex:
        """
        Add a vertex to the polygon.
        
        Args:
            x: X coordinate in feet
            y: Y coordinate in feet
            is_corner: Whether this is a corner vertex
            
        Returns:
            Created Vertex object
        """
        vertex = Vertex(
            x=x,
            y=y,
            index=self.vertex_index,
            is_corner=is_corner
        )
        
        self.vertices.append(vertex)
        self.vertex_index += 1
        
        return vertex
    
    def _add_edge(self, start: Vertex, end: Vertex, length: float, 
                 bearing: float, dim_data: Dict) -> Edge:
        """
        Add an edge to the polygon.
        
        Args:
            start: Starting vertex
            end: Ending vertex
            length: Edge length in feet
            bearing: Edge bearing in degrees
            dim_data: Dimension data dictionary
            
        Returns:
            Created Edge object
        """
        edge = Edge(
            start_vertex=start,
            end_vertex=end,
            length=length,
            bearing=bearing,
            dimension_text=dim_data.get('text', ''),
            room_label=dim_data.get('room_label'),
            confidence=dim_data.get('confidence', 1.0)
        )
        
        self.edges.append(edge)
        
        return edge
    
    def _finalize_polygon(self) -> Polygon:
        """
        Finalize polygon by checking closure and calculating properties.
        
        Returns:
            Complete Polygon object
        """
        polygon = Polygon(
            vertices=self.vertices,
            edges=self.edges,
            is_closed=False,
            closure_error=0.0,
            area=0.0,
            perimeter=0.0
        )
        
        # Calculate closure error
        if len(self.vertices) > 2:
            first = self.vertices[0]
            last = self.vertices[-1]
            closure_error = math.sqrt((last.x - first.x)**2 + (last.y - first.y)**2)
            polygon.closure_error = closure_error
            
            # Check if polygon is closed (within tolerance)
            if closure_error < 1.0:  # 1 foot tolerance
                polygon.is_closed = True
                logger.info(f"Polygon closed with error: {closure_error:.2f} feet")
            else:
                logger.warning(f"Polygon not closed, error: {closure_error:.2f} feet")
        
        # Calculate area using Shoelace formula
        polygon.area = self._calculate_area()
        
        # Calculate perimeter
        polygon.perimeter = sum(edge.length for edge in self.edges)
        
        logger.info(f"Polygon complete: {len(self.vertices)} vertices, "
                   f"area={polygon.area:.1f} sq ft, perimeter={polygon.perimeter:.1f} ft")
        
        return polygon
    
    def _calculate_area(self) -> float:
        """
        Calculate polygon area using Shoelace formula.
        
        Returns:
            Area in square feet
        """
        if len(self.vertices) < 3:
            return 0.0
        
        # Extract x and y coordinates
        x = [v.x for v in self.vertices]
        y = [v.y for v in self.vertices]
        
        # Shoelace formula
        area = 0.0
        n = len(x)
        for i in range(n):
            j = (i + 1) % n
            area += x[i] * y[j]
            area -= x[j] * y[i]
        
        area = abs(area) / 2.0
        
        return area
    
    def get_vertex_array(self) -> np.ndarray:
        """
        Get vertices as numpy array.
        
        Returns:
            Nx2 array of vertex coordinates
        """
        if not self.vertices:
            return np.array([])
        
        coords = [[v.x, v.y] for v in self.vertices]
        return np.array(coords)
    
    def get_corner_vertices(self) -> List[Vertex]:
        """
        Get only corner vertices.
        
        Returns:
            List of corner vertices
        """
        return [v for v in self.vertices if v.is_corner]
    
    def add_closing_edge(self) -> Optional[Edge]:
        """
        Add edge to close the polygon if needed.
        
        Returns:
            Closing edge if added, None otherwise
        """
        if len(self.vertices) < 3:
            return None
        
        first = self.vertices[0]
        last = self.vertices[-1]
        
        # Calculate distance
        dx = first.x - last.x
        dy = first.y - last.y
        distance = math.sqrt(dx**2 + dy**2)
        
        if distance < 0.01:  # Already closed
            return None
        
        # Calculate bearing
        bearing = math.atan2(dy, dx) * 180 / math.pi
        
        # Add closing edge
        closing_edge = Edge(
            start_vertex=last,
            end_vertex=first,
            length=distance,
            bearing=bearing,
            dimension_text="closing",
            confidence=0.5
        )
        
        self.edges.append(closing_edge)
        logger.info(f"Added closing edge: {distance:.2f} feet at {bearing:.1f}°")
        
        return closing_edge
    
    def to_dict(self) -> Dict:
        """
        Convert polygon to dictionary representation.
        
        Returns:
            Dictionary with polygon data
        """
        return {
            'vertices': [
                {'x': v.x, 'y': v.y, 'index': v.index, 'is_corner': v.is_corner}
                for v in self.vertices
            ],
            'edges': [
                {
                    'start': e.start_vertex.index,
                    'end': e.end_vertex.index,
                    'length': e.length,
                    'bearing': e.bearing,
                    'text': e.dimension_text,
                    'room': e.room_label,
                    'confidence': e.confidence
                }
                for e in self.edges
            ],
            'properties': {
                'num_vertices': len(self.vertices),
                'num_edges': len(self.edges),
                'is_closed': getattr(self, 'is_closed', False),
                'area': self._calculate_area(),
                'perimeter': sum(e.length for e in self.edges)
            }
        }
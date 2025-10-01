"""
Dimension Sequence Extractor Module
====================================
Extracts dimensions in sequential order along floor plan perimeter.
"""

import cv2
import numpy as np
import pytesseract
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class EdgeDirection(Enum):
    """Direction of edge traversal."""
    NORTH = "N"  # Top edge, going right
    EAST = "E"   # Right edge, going down
    SOUTH = "S"  # Bottom edge, going left
    WEST = "W"   # Left edge, going up


@dataclass
class DimensionSegment:
    """Represents a dimension segment along an edge."""
    value_feet: float
    value_inches: float
    total_feet: float
    text: str
    room_label: Optional[str]
    start_pixel: int
    end_pixel: int
    confidence: float
    edge: EdgeDirection


class DimensionSequenceExtractor:
    """Extracts dimensions in sequential order for perimeter tracing."""
    
    def __init__(self):
        """Initialize dimension sequence extractor."""
        self.image = None
        self.height = None
        self.width = None
        self.segments = []
        
    def extract_perimeter_dimensions(self, image: np.ndarray, 
                                    starting_corner: Dict) -> Dict[EdgeDirection, List[DimensionSegment]]:
        """
        Extract all dimensions in sequential order around the perimeter.
        
        Args:
            image: Floor plan image
            starting_corner: Dictionary with corner information from CornerDetector
            
        Returns:
            Dictionary mapping edge directions to ordered dimension segments
        """
        self.image = image
        self.height, self.width = image.shape[:2]
        
        # Determine traversal order based on starting corner
        edge_order = self._determine_edge_order(starting_corner)
        
        # Extract dimensions for each edge in sequence
        perimeter_dimensions = {}
        for edge_dir in edge_order:
            dimensions = self._extract_edge_dimensions(edge_dir)
            perimeter_dimensions[edge_dir] = dimensions
            logger.info(f"Extracted {len(dimensions)} dimensions from {edge_dir.value} edge")
        
        return perimeter_dimensions
    
    def _determine_edge_order(self, starting_corner: Dict) -> List[EdgeDirection]:
        """
        Determine the order of edges to traverse based on starting corner.
        
        Args:
            starting_corner: Starting corner information
            
        Returns:
            Ordered list of edge directions for clockwise traversal
        """
        corner_type = starting_corner.get('type')
        
        # Define clockwise traversal from each corner
        traversal_orders = {
            'NW': [EdgeDirection.NORTH, EdgeDirection.EAST, EdgeDirection.SOUTH, EdgeDirection.WEST],
            'NE': [EdgeDirection.WEST, EdgeDirection.NORTH, EdgeDirection.EAST, EdgeDirection.SOUTH],
            'SE': [EdgeDirection.SOUTH, EdgeDirection.WEST, EdgeDirection.NORTH, EdgeDirection.EAST],
            'SW': [EdgeDirection.EAST, EdgeDirection.SOUTH, EdgeDirection.WEST, EdgeDirection.NORTH]
        }
        
        # Get corner type string
        corner_str = str(corner_type).split('.')[-1] if hasattr(corner_type, 'value') else 'NW'
        corner_str = corner_str.replace('NORTHWEST', 'NW').replace('NORTHEAST', 'NE')
        corner_str = corner_str.replace('SOUTHWEST', 'SW').replace('SOUTHEAST', 'SE')
        
        return traversal_orders.get(corner_str, traversal_orders['NW'])
    
    def _extract_edge_dimensions(self, edge_dir: EdgeDirection) -> List[DimensionSegment]:
        """
        Extract dimensions from a specific edge.
        
        Args:
            edge_dir: Direction of the edge
            
        Returns:
            Ordered list of dimension segments
        """
        # Define edge regions for dimension extraction
        edge_region = self._get_edge_region(edge_dir)
        
        # Extract image region
        x1, y1, x2, y2 = edge_region
        roi = self.image[y1:y2, x1:x2]
        
        # Detect dimension markers (perpendicular lines)
        markers = self._detect_dimension_markers(roi, edge_dir)
        
        # Create segments between markers
        segments = self._create_segments_from_markers(markers, edge_dir)
        
        # Extract dimension text for each segment
        dimensions = []
        for segment in segments:
            dimension = self._extract_segment_dimension(roi, segment, edge_dir)
            if dimension:
                dimensions.append(dimension)
        
        # Order dimensions based on edge direction
        dimensions = self._order_dimensions(dimensions, edge_dir)
        
        return dimensions
    
    def _get_edge_region(self, edge_dir: EdgeDirection) -> Tuple[int, int, int, int]:
        """
        Get the image region for an edge.
        
        Args:
            edge_dir: Edge direction
            
        Returns:
            Tuple of (x1, y1, x2, y2) for the region
        """
        margin_ratio = 0.20  # Look at 20% of image from edge
        
        regions = {
            EdgeDirection.NORTH: (0, 0, self.width, int(self.height * margin_ratio)),
            EdgeDirection.SOUTH: (0, int(self.height * (1 - margin_ratio)), self.width, self.height),
            EdgeDirection.EAST: (int(self.width * (1 - margin_ratio)), 0, self.width, self.height),
            EdgeDirection.WEST: (0, 0, int(self.width * margin_ratio), self.height)
        }
        
        return regions[edge_dir]
    
    def _detect_dimension_markers(self, roi: np.ndarray, edge_dir: EdgeDirection) -> List[int]:
        """
        Detect dimension marker lines (perpendicular to edge).
        
        Args:
            roi: Region of interest
            edge_dir: Edge direction
            
        Returns:
            List of marker positions in pixels
        """
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, 
                                minLineLength=20, maxLineGap=5)
        
        if lines is None:
            return []
        
        markers = []
        
        # For horizontal edges, look for vertical lines
        if edge_dir in [EdgeDirection.NORTH, EdgeDirection.SOUTH]:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Check if line is roughly vertical
                if abs(x2 - x1) < 10 and abs(y2 - y1) > 15:
                    markers.append((x1 + x2) // 2)
        
        # For vertical edges, look for horizontal lines
        else:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Check if line is roughly horizontal
                if abs(y2 - y1) < 10 and abs(x2 - x1) > 15:
                    markers.append((y1 + y2) // 2)
        
        # Remove duplicates and sort
        markers = sorted(list(set(markers)))
        
        # Filter out markers that are too close
        filtered_markers = []
        min_distance = 30  # Minimum pixels between markers
        for marker in markers:
            if not filtered_markers or marker - filtered_markers[-1] > min_distance:
                filtered_markers.append(marker)
        
        return filtered_markers
    
    def _create_segments_from_markers(self, markers: List[int], 
                                     edge_dir: EdgeDirection) -> List[Tuple[int, int]]:
        """
        Create segments between dimension markers.
        
        Args:
            markers: List of marker positions
            edge_dir: Edge direction
            
        Returns:
            List of (start, end) pixel positions for segments
        """
        if not markers:
            # Create one segment for entire edge
            if edge_dir in [EdgeDirection.NORTH, EdgeDirection.SOUTH]:
                return [(0, self.width)]
            else:
                return [(0, self.height)]
        
        segments = []
        
        # Add segment from edge start to first marker
        if markers[0] > 50:
            segments.append((0, markers[0]))
        
        # Add segments between markers
        for i in range(len(markers) - 1):
            segments.append((markers[i], markers[i + 1]))
        
        # Add segment from last marker to edge end
        edge_limit = self.width if edge_dir in [EdgeDirection.NORTH, EdgeDirection.SOUTH] else self.height
        if markers[-1] < edge_limit - 50:
            segments.append((markers[-1], edge_limit))
        
        return segments
    
    def _extract_segment_dimension(self, roi: np.ndarray, segment: Tuple[int, int], 
                                  edge_dir: EdgeDirection) -> Optional[DimensionSegment]:
        """
        Extract dimension text from a segment.
        
        Args:
            roi: Region of interest
            segment: Start and end positions of segment
            edge_dir: Edge direction
            
        Returns:
            DimensionSegment or None if no dimension found
        """
        start, end = segment
        
        # Create OCR window for segment
        if edge_dir in [EdgeDirection.NORTH, EdgeDirection.SOUTH]:
            # Horizontal edge - segment is along x-axis
            ocr_roi = roi[:, start:end]
        else:
            # Vertical edge - segment is along y-axis
            ocr_roi = roi[start:end, :]
            # Rotate for vertical text if needed
            if edge_dir == EdgeDirection.WEST:
                ocr_roi = cv2.rotate(ocr_roi, cv2.ROTATE_90_COUNTERCLOCKWISE)
            else:
                ocr_roi = cv2.rotate(ocr_roi, cv2.ROTATE_90_CLOCKWISE)
        
        if ocr_roi.size == 0:
            return None
        
        # Enhance for OCR
        gray = cv2.cvtColor(ocr_roi, cv2.COLOR_BGR2GRAY) if len(ocr_roi.shape) == 3 else ocr_roi
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Run OCR
        text = pytesseract.image_to_string(binary)
        
        # Parse dimension
        dimension = self._parse_dimension_text(text)
        if dimension:
            dimension['start_pixel'] = start
            dimension['end_pixel'] = end
            dimension['edge'] = edge_dir
            
            # Convert to DimensionSegment
            return DimensionSegment(
                value_feet=dimension['feet'],
                value_inches=dimension['inches'],
                total_feet=dimension['feet'] + dimension['inches'] / 12,
                text=dimension['text'],
                room_label=dimension.get('label'),
                start_pixel=start,
                end_pixel=end,
                confidence=dimension.get('confidence', 0.5),
                edge=edge_dir
            )
        
        return None
    
    def _parse_dimension_text(self, text: str) -> Optional[Dict]:
        """
        Parse dimension text to extract measurements.
        
        Args:
            text: OCR text
            
        Returns:
            Parsed dimension dictionary or None
        """
        # Pattern for feet and inches
        pattern = r"(\d+)'(?:-(\d+)(?:\"|'')?)?(?:\s+([A-Za-z\s]+))?"
        
        for line in text.split('\n'):
            # Skip area calculations
            if 'area' in line.lower() or 'square' in line.lower():
                continue
            
            match = re.search(pattern, line)
            if match:
                feet = int(match.group(1))
                inches = int(match.group(2)) if match.group(2) else 0
                label = match.group(3).strip() if match.group(3) else ""
                
                # Validate dimension
                if feet > 100:
                    logger.debug(f"Skipping unrealistic dimension: {feet}'")
                    continue
                
                return {
                    'feet': feet,
                    'inches': inches,
                    'text': match.group(0),
                    'label': label,
                    'confidence': 0.8 if feet < 50 else 0.6
                }
        
        return None
    
    def _order_dimensions(self, dimensions: List[DimensionSegment], 
                         edge_dir: EdgeDirection) -> List[DimensionSegment]:
        """
        Order dimensions based on edge direction.
        
        Args:
            dimensions: List of dimension segments
            edge_dir: Edge direction
            
        Returns:
            Ordered list of dimensions
        """
        if not dimensions:
            return dimensions
        
        # Sort by pixel position
        if edge_dir in [EdgeDirection.NORTH, EdgeDirection.EAST]:
            # Left to right or top to bottom
            dimensions.sort(key=lambda d: d.start_pixel)
        else:
            # Right to left or bottom to top
            dimensions.sort(key=lambda d: d.start_pixel, reverse=True)
        
        return dimensions
    
    def get_total_perimeter(self, perimeter_dimensions: Dict[EdgeDirection, List[DimensionSegment]]) -> float:
        """
        Calculate total perimeter from all dimensions.
        
        Args:
            perimeter_dimensions: Dictionary of dimensions by edge
            
        Returns:
            Total perimeter in feet
        """
        total = 0.0
        for edge_dir, dimensions in perimeter_dimensions.items():
            edge_total = sum(d.total_feet for d in dimensions)
            total += edge_total
            logger.debug(f"{edge_dir.value} edge: {edge_total:.1f} feet")
        
        return total
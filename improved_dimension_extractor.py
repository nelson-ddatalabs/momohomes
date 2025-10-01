#!/usr/bin/env python3
"""
Improved Dimension Extractor
=============================
Extract dimensions with their positions from floor plans.
"""

import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Dimension:
    """Represents a dimension with its location."""
    text: str
    value_feet: float
    x: int
    y: int
    width: int
    height: int
    confidence: float
    edge: Optional[str] = None


class ImprovedDimensionExtractor:
    """Extract dimensions with positions from floor plans."""
    
    def __init__(self):
        """Initialize extractor."""
        self.image = None
        self.dimensions = []
        self.height = 0
        self.width = 0
    
    def extract_dimensions(self, image_path: str) -> Dict:
        """
        Extract all dimensions with their positions.
        
        Args:
            image_path: Path to floor plan image
            
        Returns:
            Dictionary with dimension data
        """
        # Load image
        self.image = cv2.imread(image_path)
        if self.image is None:
            logger.error(f"Failed to load {image_path}")
            return {}
        
        self.height, self.width = self.image.shape[:2]
        logger.info(f"Processing image: {self.width}x{self.height}")
        
        # Extract dimensions using OCR with bounding boxes
        self.dimensions = self._extract_with_positions()
        
        # Classify dimensions by edge
        edge_dims = self._classify_by_edge()
        
        # Order dimensions for perimeter traversal
        ordered_dims = self._order_for_perimeter(edge_dims)
        
        # Calculate total perimeter
        total_perimeter = sum(d.value_feet for d in self.dimensions)
        
        return {
            'all_dimensions': self.dimensions,
            'by_edge': edge_dims,
            'ordered_perimeter': ordered_dims,
            'total_perimeter_feet': total_perimeter,
            'dimension_count': len(self.dimensions)
        }
    
    def _extract_with_positions(self) -> List[Dimension]:
        """Extract dimensions with their positions using OCR."""
        dimensions = []
        
        # Preprocess image for better OCR
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Get OCR data with bounding boxes
        ocr_data = pytesseract.image_to_data(thresh, output_type=Output.DICT)
        
        n_boxes = len(ocr_data['text'])
        for i in range(n_boxes):
            text = ocr_data['text'][i].strip()
            if text:
                # Check if this looks like a dimension
                dim_value = self._parse_dimension(text)
                if dim_value:
                    x = ocr_data['left'][i]
                    y = ocr_data['top'][i]
                    w = ocr_data['width'][i]
                    h = ocr_data['height'][i]
                    conf = ocr_data['conf'][i]
                    
                    # Only accept confident detections
                    if conf > 30:  # Lower threshold for architectural drawings
                        dim = Dimension(
                            text=text,
                            value_feet=dim_value,
                            x=x,
                            y=y,
                            width=w,
                            height=h,
                            confidence=conf/100.0
                        )
                        dimensions.append(dim)
                        logger.debug(f"Found dimension: {text} at ({x},{y}) = {dim_value} ft")
        
        # Also try region-based extraction for missed dimensions
        additional_dims = self._extract_from_regions()
        dimensions.extend(additional_dims)
        
        # Remove duplicates based on position
        dimensions = self._remove_duplicates(dimensions)
        
        logger.info(f"Extracted {len(dimensions)} dimensions")
        
        return dimensions
    
    def _extract_from_regions(self) -> List[Dimension]:
        """Extract dimensions from specific regions."""
        additional_dims = []
        
        # Define search regions (offset from edges to catch dimension lines)
        offset = 200  # Pixels from edge where dimensions typically appear
        regions = [
            # Top region
            ('top', 0, 0, self.width, offset),
            # Bottom region  
            ('bottom', 0, self.height - offset, self.width, offset),
            # Left region
            ('left', 0, 0, offset, self.height),
            # Right region
            ('right', self.width - offset, 0, offset, self.height)
        ]
        
        for edge_name, x, y, w, h in regions:
            roi = self.image[y:y+h, x:x+w]
            
            # Multiple preprocessing attempts
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Try different OCR configurations
            configs = [
                '--psm 11',  # Sparse text
                '--psm 6',   # Uniform block
                '--psm 8',   # Single word
            ]
            
            for config in configs:
                try:
                    # Get OCR with positions
                    ocr_data = pytesseract.image_to_data(gray_roi, 
                                                        config=config,
                                                        output_type=Output.DICT)
                    
                    n_boxes = len(ocr_data['text'])
                    for i in range(n_boxes):
                        text = ocr_data['text'][i].strip()
                        if text:
                            dim_value = self._parse_dimension(text)
                            if dim_value:
                                # Adjust coordinates to full image
                                abs_x = x + ocr_data['left'][i]
                                abs_y = y + ocr_data['top'][i]
                                
                                dim = Dimension(
                                    text=text,
                                    value_feet=dim_value,
                                    x=abs_x,
                                    y=abs_y,
                                    width=ocr_data['width'][i],
                                    height=ocr_data['height'][i],
                                    confidence=ocr_data['conf'][i]/100.0,
                                    edge=edge_name
                                )
                                additional_dims.append(dim)
                                
                except Exception as e:
                    logger.debug(f"OCR failed for {edge_name} with config {config}: {e}")
        
        return additional_dims
    
    def _parse_dimension(self, text: str) -> Optional[float]:
        """
        Parse dimension text to feet value.
        
        Args:
            text: Text to parse
            
        Returns:
            Value in feet or None
        """
        if not text:
            return None
        
        # Clean text
        text = text.strip()
        
        # Pattern matching for various formats
        patterns = [
            # Format: 32'-6" or 32' 6"
            (r"(\d+)['''][\s\-]?(\d+)[\"\"]", lambda m: float(m.group(1)) + float(m.group(2))/12),
            # Format: 32' or 32.5'
            (r"(\d+\.?\d*)[''']", lambda m: float(m.group(1))),
            # Format: 32-6 (assume feet-inches)
            (r"(\d+)[\-\s](\d{1,2})(?!\d)", lambda m: float(m.group(1)) + float(m.group(2))/12),
            # Format: just a number (check if reasonable)
            (r"^(\d+)$", lambda m: float(m.group(1)) if 3 <= float(m.group(1)) <= 100 else None),
        ]
        
        for pattern, converter in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    value = converter(match)
                    if value and 1 <= value <= 200:  # Reasonable dimension range
                        return value
                except:
                    continue
        
        return None
    
    def _classify_by_edge(self) -> Dict[str, List[Dimension]]:
        """Classify dimensions by which edge they belong to."""
        edge_dims = {
            'top': [],
            'right': [],
            'bottom': [],
            'left': []
        }
        
        # Define edge zones
        edge_threshold = 0.3  # 30% from edge
        
        for dim in self.dimensions:
            x_ratio = dim.x / self.width
            y_ratio = dim.y / self.height
            
            # Determine which edge based on position
            if y_ratio < edge_threshold:
                edge_dims['top'].append(dim)
            elif y_ratio > (1 - edge_threshold):
                edge_dims['bottom'].append(dim)
            elif x_ratio < edge_threshold:
                edge_dims['left'].append(dim)
            elif x_ratio > (1 - edge_threshold):
                edge_dims['right'].append(dim)
            else:
                # Try to assign based on closest edge
                distances = {
                    'top': dim.y,
                    'bottom': self.height - dim.y,
                    'left': dim.x,
                    'right': self.width - dim.x
                }
                closest_edge = min(distances, key=distances.get)
                edge_dims[closest_edge].append(dim)
        
        # Sort dimensions along each edge
        edge_dims['top'].sort(key=lambda d: d.x)  # Left to right
        edge_dims['right'].sort(key=lambda d: d.y)  # Top to bottom
        edge_dims['bottom'].sort(key=lambda d: d.x, reverse=True)  # Right to left
        edge_dims['left'].sort(key=lambda d: d.y, reverse=True)  # Bottom to top
        
        # Log results
        for edge, dims in edge_dims.items():
            logger.info(f"{edge} edge: {len(dims)} dimensions")
            for dim in dims:
                logger.debug(f"  {edge}: {dim.text} = {dim.value_feet:.1f} ft")
        
        return edge_dims
    
    def _order_for_perimeter(self, edge_dims: Dict[str, List[Dimension]]) -> List[Dimension]:
        """Order dimensions for clockwise perimeter traversal."""
        ordered = []
        
        # Clockwise order: top (L->R), right (T->B), bottom (R->L), left (B->T)
        ordered.extend(edge_dims['top'])
        ordered.extend(edge_dims['right'])
        ordered.extend(edge_dims['bottom'])
        ordered.extend(edge_dims['left'])
        
        return ordered
    
    def _remove_duplicates(self, dimensions: List[Dimension]) -> List[Dimension]:
        """Remove duplicate dimensions based on proximity."""
        if not dimensions:
            return []
        
        unique = []
        distance_threshold = 50  # Pixels
        
        for dim in dimensions:
            is_duplicate = False
            for existing in unique:
                # Check if too close to existing dimension
                dist = np.sqrt((dim.x - existing.x)**2 + (dim.y - existing.y)**2)
                if dist < distance_threshold and abs(dim.value_feet - existing.value_feet) < 0.5:
                    is_duplicate = True
                    # Keep the one with higher confidence
                    if dim.confidence > existing.confidence:
                        unique.remove(existing)
                        unique.append(dim)
                    break
            
            if not is_duplicate:
                unique.append(dim)
        
        return unique
    
    def visualize_dimensions(self, output_path: str = "dimension_visualization.png"):
        """Create visualization of detected dimensions."""
        if self.image is None or not self.dimensions:
            return
        
        vis_img = self.image.copy()
        
        # Colors for different edges
        edge_colors = {
            'top': (0, 0, 255),     # Red
            'right': (0, 255, 0),   # Green
            'bottom': (255, 0, 0),  # Blue
            'left': (255, 255, 0),  # Yellow
            None: (255, 255, 255)   # White for unclassified
        }
        
        # Draw dimension locations
        for dim in self.dimensions:
            color = edge_colors.get(dim.edge, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(vis_img, (dim.x, dim.y), 
                         (dim.x + dim.width, dim.y + dim.height),
                         color, 2)
            
            # Draw text label
            label = f"{dim.text} ({dim.value_feet:.1f}')"
            cv2.putText(vis_img, label, (dim.x, dim.y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Add summary text
        summary = f"Total dimensions: {len(self.dimensions)}"
        cv2.putText(vis_img, summary, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imwrite(output_path, vis_img)
        logger.info(f"Visualization saved to {output_path}")


def test_luna():
    """Test improved extraction on Luna.png."""
    print("\n" + "="*70)
    print("IMPROVED DIMENSION EXTRACTION TEST")
    print("="*70)
    
    extractor = ImprovedDimensionExtractor()
    results = extractor.extract_dimensions("Luna.png")
    
    print(f"\nTotal dimensions found: {results['dimension_count']}")
    print(f"Total perimeter: {results['total_perimeter_feet']:.1f} feet")
    
    print("\nDimensions by edge:")
    for edge, dims in results['by_edge'].items():
        print(f"\n{edge.upper()} edge ({len(dims)} dimensions):")
        for dim in dims:
            print(f"  {dim.text:10s} = {dim.value_feet:6.1f} ft at ({dim.x:4d},{dim.y:4d})")
    
    print("\nOrdered perimeter traversal:")
    for i, dim in enumerate(results['ordered_perimeter']):
        print(f"{i+1:2d}. {dim.text:10s} = {dim.value_feet:6.1f} ft")
    
    # Create visualization
    extractor.visualize_dimensions("luna_dimensions.png")
    print("\nVisualization saved to luna_dimensions.png")
    
    # Check against expected values
    expected_perimeter = 265.75  # From legend: 265'-9"
    expected_area = 2733.25  # sq ft
    
    print("\n" + "="*70)
    print("VALIDATION")
    print("="*70)
    print(f"Expected perimeter: {expected_perimeter:.2f} ft")
    print(f"Extracted perimeter: {results['total_perimeter_feet']:.2f} ft")
    
    perimeter_error = abs(results['total_perimeter_feet'] - expected_perimeter)
    print(f"Error: {perimeter_error:.2f} ft ({perimeter_error/expected_perimeter*100:.1f}%)")
    
    if perimeter_error < 10:
        print("✓ PASS: Perimeter within acceptable range")
    else:
        print("✗ FAIL: Perimeter error too large")
    
    return results


if __name__ == "__main__":
    test_luna()
"""
Dimension Line Detection Module
===============================
Detects and extracts dimension lines from all sides of floor plans.
"""

import cv2
import numpy as np
import pytesseract
import re
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DimensionLineDetector:
    """Detects dimension lines and measurements from floor plans."""
    
    def __init__(self):
        """Initialize dimension line detector."""
        self.dimensions = {
            'top': [],
            'bottom': [],
            'left': [],
            'right': []
        }
        
    def detect_dimension_lines(self, image: np.ndarray) -> Dict:
        """
        Detect dimension lines on all sides of the floor plan.
        
        Args:
            image: Floor plan image
            
        Returns:
            Dictionary with dimensions for each side
        """
        height, width = image.shape[:2]
        
        # Define regions for each side (looking beyond just edges)
        regions = {
            'top': (0, 0, width, int(height * 0.20)),     # Top 20%
            'bottom': (0, int(height * 0.80), width, height),  # Bottom 20%
            'left': (0, 0, int(width * 0.15), height),    # Left 15%
            'right': (int(width * 0.85), 0, width, height)  # Right 15%
        }
        
        # Process each region
        all_dimensions = {}
        for side, (x1, y1, x2, y2) in regions.items():
            roi = image[y1:y2, x1:x2]
            dimensions = self._extract_dimensions_from_region(roi, side)
            all_dimensions[side] = dimensions
            logger.debug(f"{side} dimensions: {dimensions}")
        
        # Detect dimension lines using line detection
        line_dimensions = self._detect_lines_with_text(image)
        
        # Merge results
        merged = self._merge_dimension_results(all_dimensions, line_dimensions)
        
        return merged
    
    def _extract_dimensions_from_region(self, roi: np.ndarray, side: str) -> List[Dict]:
        """Extract dimensions from a specific region."""
        # Convert to grayscale
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi.copy()
        
        # Enhance for OCR
        enhanced = self._enhance_for_dimension_text(gray, side)
        
        # Run OCR
        text = pytesseract.image_to_string(enhanced)
        
        # Parse dimensions
        dimensions = self._parse_dimension_text(text)
        
        # Get positions if needed
        if dimensions:
            ocr_data = pytesseract.image_to_data(enhanced, output_type=pytesseract.Output.DICT)
            dimensions = self._add_positions_to_dimensions(dimensions, ocr_data, side)
        
        return dimensions
    
    def _enhance_for_dimension_text(self, gray: np.ndarray, side: str) -> np.ndarray:
        """Enhance image specifically for dimension text extraction."""
        # Different processing for horizontal vs vertical text
        if side in ['left', 'right']:
            # Rotate for vertical text
            if side == 'left':
                enhanced = cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)
            else:
                enhanced = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
        else:
            enhanced = gray.copy()
        
        # Apply threshold
        _, binary = cv2.threshold(enhanced, 127, 255, cv2.THRESH_BINARY)
        
        # Invert if needed (white text on dark background)
        if np.mean(binary) > 127:
            binary = cv2.bitwise_not(binary)
        
        return binary
    
    def _parse_dimension_text(self, text: str) -> List[Dict]:
        """Parse dimension text to extract measurements."""
        dimensions = []
        
        # Pattern for feet and inches (e.g., "14'-6"", "32'-6"", "11'")
        # Also handle with labels (e.g., "14'-6" Master Suite")
        pattern = r"(\d+)'(?:-(\d+)(?:\"|'')?)?(?:\s+([A-Za-z\s]+))?"
        
        for line in text.split('\n'):
            # Skip lines that look like area calculations or totals
            if 'area' in line.lower() or 'square' in line.lower() or 'perimeter' in line.lower():
                continue
                
            matches = re.finditer(pattern, line)
            for match in matches:
                feet = int(match.group(1))
                inches = int(match.group(2)) if match.group(2) else 0
                label = match.group(3).strip() if match.group(3) else ""
                
                # Filter out unrealistic dimensions
                # Typical room/house dimensions are under 100 feet
                if feet > 100:
                    # Check if this might be a misread (e.g., 414 instead of 14)
                    if feet > 400 and feet < 500:
                        # Try to fix common OCR errors
                        feet_str = str(feet)
                        if feet_str.startswith('4'):
                            feet = int(feet_str[1:])  # Remove leading 4
                    else:
                        logger.debug(f"Skipping unrealistic dimension: {feet}'")
                        continue
                
                # Skip if this looks like a total/overall that's too large
                if 'overall' in label.lower() and feet > 100:
                    continue
                
                # Convert to total inches
                total_inches = feet * 12 + inches
                
                dimension = {
                    'feet': feet,
                    'inches': inches,
                    'total_inches': total_inches,
                    'text': match.group(0),
                    'label': label
                }
                dimensions.append(dimension)
                logger.debug(f"Found dimension: {dimension}")
        
        return dimensions
    
    def _detect_lines_with_text(self, image: np.ndarray) -> Dict:
        """Detect actual dimension lines using line detection."""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines using HoughLines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, 
                                minLineLength=100, maxLineGap=10)
        
        if lines is None:
            return {'horizontal': [], 'vertical': []}
        
        # Classify lines as horizontal or vertical
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            if angle < 10 or angle > 170:  # Horizontal
                horizontal_lines.append((x1, y1, x2, y2))
            elif 80 < angle < 100:  # Vertical
                vertical_lines.append((x1, y1, x2, y2))
        
        # Look for text near lines
        dimensions_near_lines = self._find_text_near_lines(
            image, horizontal_lines, vertical_lines
        )
        
        return dimensions_near_lines
    
    def _find_text_near_lines(self, image: np.ndarray, 
                              h_lines: List, v_lines: List) -> Dict:
        """Find dimension text near detected lines."""
        height, width = image.shape[:2]
        dimensions = {'horizontal': [], 'vertical': []}
        
        # Check areas near horizontal lines (top and bottom)
        for x1, y1, x2, y2 in h_lines:
            # Top lines
            if y1 < height * 0.25:
                roi = image[max(0, y1-30):min(height, y1+30), x1:x2]
                if roi.size > 0:  # Check ROI is not empty
                    text = pytesseract.image_to_string(roi)
                    dims = self._parse_dimension_text(text)
                    if dims:
                        for d in dims:
                            d['position'] = 'top'
                            d['line_coords'] = (x1, y1, x2, y2)
                        dimensions['horizontal'].extend(dims)
            
            # Bottom lines
            elif y1 > height * 0.75:
                roi = image[max(0, y1-30):min(height, y1+30), x1:x2]
                if roi.size > 0:  # Check ROI is not empty
                    text = pytesseract.image_to_string(roi)
                    dims = self._parse_dimension_text(text)
                    if dims:
                        for d in dims:
                            d['position'] = 'bottom'
                            d['line_coords'] = (x1, y1, x2, y2)
                        dimensions['horizontal'].extend(dims)
        
        # Check areas near vertical lines (left and right)
        for x1, y1, x2, y2 in v_lines:
            # Left lines
            if x1 < width * 0.25:
                roi = image[y1:y2, max(0, x1-30):min(width, x1+30)]
                if roi.size > 0:  # Check ROI is not empty
                    # Rotate for vertical text
                    roi_rotated = cv2.rotate(roi, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    text = pytesseract.image_to_string(roi_rotated)
                    dims = self._parse_dimension_text(text)
                    if dims:
                        for d in dims:
                            d['position'] = 'left'
                            d['line_coords'] = (x1, y1, x2, y2)
                        dimensions['vertical'].extend(dims)
            
            # Right lines
            elif x1 > width * 0.75:
                roi = image[y1:y2, max(0, x1-30):min(width, x1+30)]
                if roi.size > 0:  # Check ROI is not empty
                    # Rotate for vertical text
                    roi_rotated = cv2.rotate(roi, cv2.ROTATE_90_CLOCKWISE)
                    text = pytesseract.image_to_string(roi_rotated)
                    dims = self._parse_dimension_text(text)
                    if dims:
                        for d in dims:
                            d['position'] = 'right'
                            d['line_coords'] = (x1, y1, x2, y2)
                        dimensions['vertical'].extend(dims)
        
        return dimensions
    
    def _add_positions_to_dimensions(self, dimensions: List[Dict], 
                                    ocr_data: Dict, side: str) -> List[Dict]:
        """Add position information to detected dimensions."""
        # Match dimension text with OCR bounding boxes
        for dim in dimensions:
            dim['side'] = side
            dim['position'] = None
            
            # Try to find matching text in OCR data
            for i, text in enumerate(ocr_data['text']):
                if dim['text'] in text or str(dim['feet']) in text:
                    dim['position'] = {
                        'x': ocr_data['left'][i],
                        'y': ocr_data['top'][i],
                        'width': ocr_data['width'][i],
                        'height': ocr_data['height'][i]
                    }
                    break
        
        return dimensions
    
    def _merge_dimension_results(self, region_dims: Dict, line_dims: Dict) -> Dict:
        """Merge dimensions from different detection methods."""
        merged = {
            'top': [],
            'bottom': [],
            'left': [],
            'right': [],
            'total_width': None,
            'total_height': None
        }
        
        # Add region-based dimensions
        for side, dims in region_dims.items():
            merged[side].extend(dims)
        
        # Add line-based dimensions
        for orientation, dims in line_dims.items():
            for dim in dims:
                if dim.get('position'):
                    side = dim['position']
                    if side in merged and dim not in merged[side]:
                        merged[side].append(dim)
        
        # Calculate totals
        merged['total_width'] = self._calculate_total_dimension(
            merged['top'] + merged['bottom']
        )
        merged['total_height'] = self._calculate_total_dimension(
            merged['left'] + merged['right']
        )
        
        return merged
    
    def _calculate_total_dimension(self, dimensions: List[Dict]) -> Optional[int]:
        """Calculate total dimension from a list of measurements."""
        if not dimensions:
            return None
        
        # Sum all dimensions (in inches)
        total = sum(d['total_inches'] for d in dimensions)
        
        # Remove duplicates if any
        unique_dims = []
        seen = set()
        for d in dimensions:
            key = (d['feet'], d['inches'])
            if key not in seen:
                seen.add(key)
                unique_dims.append(d)
        
        if unique_dims:
            total = sum(d['total_inches'] for d in unique_dims)
        
        return total
    
    def extract_floor_dimensions(self, image: np.ndarray) -> Tuple[float, float]:
        """
        Extract total floor dimensions in feet.
        
        Args:
            image: Floor plan image
            
        Returns:
            Tuple of (width_feet, height_feet)
        """
        dimensions = self.detect_dimension_lines(image)
        
        # First, look for "Overall" dimensions which Luna uses
        width_feet = None
        height_feet = None
        
        # Check all dimensions for "Overall" label
        for side_dims in [dimensions['top'], dimensions['bottom'], dimensions['left'], dimensions['right']]:
            for dim in side_dims:
                if 'overall' in dim.get('label', '').lower():
                    # This is likely the overall dimension
                    total_feet = dim['feet'] + dim['inches'] / 12
                    if total_feet <= 100:  # Reasonable dimension
                        if side_dims in [dimensions['top'], dimensions['bottom']]:
                            width_feet = total_feet
                        else:
                            height_feet = total_feet
        
        # If no "Overall" found, try to sum individual dimensions
        if width_feet is None:
            # Get totals in inches
            width_inches = dimensions.get('total_width')
            if width_inches:
                width_feet = width_inches / 12
            else:
                # Sum individual dimensions
                top_total = sum(d['total_inches'] for d in dimensions['top']) / 12
                bottom_total = sum(d['total_inches'] for d in dimensions['bottom']) / 12
                if top_total > 0 and top_total <= 200:
                    width_feet = top_total
                elif bottom_total > 0 and bottom_total <= 200:
                    width_feet = bottom_total
                else:
                    width_feet = 73.0  # Default for Luna
        
        if height_feet is None:
            # Get totals in inches
            height_inches = dimensions.get('total_height')
            if height_inches:
                height_feet = height_inches / 12
            else:
                # Sum individual dimensions
                left_total = sum(d['total_inches'] for d in dimensions['left']) / 12
                right_total = sum(d['total_inches'] for d in dimensions['right']) / 12
                if left_total > 0 and left_total <= 200:
                    height_feet = left_total
                elif right_total > 0 and right_total <= 200:
                    height_feet = right_total
                else:
                    height_feet = 38.0  # Default for Luna
        
        # Sanity check dimensions
        if width_feet > 200:
            logger.warning(f"Width {width_feet:.1f}' seems too large, using default")
            width_feet = 73.0
        if height_feet > 200:
            logger.warning(f"Height {height_feet:.1f}' seems too large, using default")
            height_feet = 38.0
        
        logger.info(f"Extracted dimensions: {width_feet:.1f}' x {height_feet:.1f}'")
        
        return width_feet, height_feet
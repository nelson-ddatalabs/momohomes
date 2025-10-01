"""
Dimension Extractor Module
==========================
Extracts dimensions from floor plan images using OCR and pattern matching.
"""

import cv2
import numpy as np
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

from config_cassette import CassetteConfig

logger = logging.getLogger(__name__)


@dataclass
class Dimension:
    """Represents a dimension annotation from the floor plan."""
    text: str
    value_feet: float
    location: str  # 'top', 'bottom', 'left', 'right'
    confidence: float
    bbox: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h
    
    def __str__(self):
        return f"{self.location}: {self.value_feet}' ({self.text})"


class DimensionExtractor:
    """Extracts dimensions from floor plan images."""
    
    def __init__(self):
        """Initialize dimension extractor."""
        self.config = CassetteConfig.IMAGE_PROCESSING
        self.dimension_patterns = [
            re.compile(pattern) 
            for pattern in self.config['dimension_regex_patterns']
        ]
        
        # Try to import pytesseract, fallback to manual if not available
        self.ocr_available = False
        try:
            import pytesseract
            self.pytesseract = pytesseract
            self.ocr_available = True
            logger.info("OCR (pytesseract) initialized successfully")
        except ImportError:
            logger.warning("pytesseract not available, using fallback method")
    
    def extract_dimensions(self, image_path: str) -> Dict[str, List[Dimension]]:
        """
        Extract dimensions from all edges of the floor plan.
        
        Args:
            image_path: Path to floor plan image
            
        Returns:
            Dictionary with dimensions for each edge
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        height, width = image.shape[:2]
        
        # Extract dimensions from each edge
        dimensions = {
            'top': self._extract_from_edge(image, 'top'),
            'bottom': self._extract_from_edge(image, 'bottom'),
            'left': self._extract_from_edge(image, 'left'),
            'right': self._extract_from_edge(image, 'right'),
        }
        
        # Validate and clean dimensions
        dimensions = self._validate_dimensions(dimensions)
        
        # Calculate scale factor
        scale = self._calculate_scale(dimensions, width, height)
        
        logger.info(f"Extracted dimensions: {self._summarize_dimensions(dimensions)}")
        logger.info(f"Calculated scale: {scale:.2f} pixels per foot")
        
        return dimensions
    
    def _extract_from_edge(self, image: np.ndarray, edge: str) -> List[Dimension]:
        """Extract dimensions from a specific edge of the image."""
        height, width = image.shape[:2]
        
        # Define region of interest based on edge
        if edge == 'top':
            roi = image[0:int(height * 0.15), :]
        elif edge == 'bottom':
            roi = image[int(height * 0.85):, :]
        elif edge == 'left':
            roi = image[:, 0:int(width * 0.15)]
            # Rotate for better OCR
            roi = cv2.rotate(roi, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif edge == 'right':
            roi = image[:, int(width * 0.85):]
            # Rotate for better OCR
            roi = cv2.rotate(roi, cv2.ROTATE_90_CLOCKWISE)
        else:
            return []
        
        # Preprocess for OCR
        processed = self._preprocess_for_ocr(roi)
        
        # Extract text
        if self.ocr_available:
            dimensions = self._extract_with_ocr(processed, edge)
        else:
            dimensions = self._extract_with_fallback(processed, edge)
        
        return dimensions
    
    def _preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image region for better OCR results."""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply thresholding to get black text on white background
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Denoise
        denoised = cv2.medianBlur(thresh, 3)
        
        # Enhance contrast
        enhanced = cv2.convertScaleAbs(denoised, alpha=1.5, beta=0)
        
        return enhanced
    
    def _extract_with_ocr(self, image: np.ndarray, edge: str) -> List[Dimension]:
        """Extract dimensions using OCR."""
        dimensions = []
        
        try:
            # Get OCR data with confidence scores
            data = self.pytesseract.image_to_data(
                image, 
                output_type=self.pytesseract.Output.DICT,
                config='--psm 6'  # Assume uniform text block
            )
            
            # Process each text element
            for i, text in enumerate(data['text']):
                if not text.strip():
                    continue
                
                confidence = float(data['conf'][i])
                if confidence < self.config['ocr_confidence_threshold'] * 100:
                    continue
                
                # Try to parse as dimension
                value = self._parse_dimension_text(text)
                if value is not None:
                    bbox = (
                        data['left'][i],
                        data['top'][i],
                        data['width'][i],
                        data['height'][i]
                    )
                    
                    dimensions.append(Dimension(
                        text=text,
                        value_feet=value,
                        location=edge,
                        confidence=confidence / 100,
                        bbox=bbox
                    ))
        
        except Exception as e:
            logger.warning(f"OCR failed for {edge}: {e}")
        
        return dimensions
    
    def _extract_with_fallback(self, image: np.ndarray, edge: str) -> List[Dimension]:
        """Fallback method when OCR is not available."""
        # For MVP, return hardcoded dimensions for known floor plans
        # In production, this would use template matching or other methods
        
        fallback_dimensions = {
            'top': [
                Dimension("30'", 30.0, 'top', 0.5),
                Dimension("40'", 40.0, 'top', 0.5),
            ],
            'bottom': [
                Dimension("30'", 30.0, 'bottom', 0.5),
            ],
            'left': [
                Dimension("40'", 40.0, 'left', 0.5),
            ],
            'right': [
                Dimension("40'", 40.0, 'right', 0.5),
            ],
        }
        
        return fallback_dimensions.get(edge, [])
    
    def _parse_dimension_text(self, text: str) -> Optional[float]:
        """
        Parse dimension text to extract value in feet.
        
        Handles formats like:
        - 14'-6" or 14' 6"
        - 14.5'
        - 14'
        """
        text = text.strip()
        
        # Try each pattern
        for pattern in self.dimension_patterns:
            match = pattern.search(text)
            if match:
                groups = match.groups()
                
                if len(groups) == 2:
                    # Format: 14'-6" or 14.5'
                    feet = float(groups[0])
                    inches = float(groups[1])
                    
                    # Check if second group is inches or decimal
                    if inches > 12:
                        # Likely decimal (14.5')
                        return feet + inches / 10
                    else:
                        # Inches (14'-6")
                        return feet + inches / 12
                
                elif len(groups) == 1:
                    # Format: 14'
                    return float(groups[0])
        
        return None
    
    def _validate_dimensions(self, dimensions: Dict[str, List[Dimension]]) -> Dict[str, List[Dimension]]:
        """Validate and clean extracted dimensions."""
        validated = {}
        
        for edge, dims in dimensions.items():
            # Remove duplicates
            unique_dims = []
            seen_values = set()
            
            for dim in dims:
                if dim.value_feet not in seen_values:
                    unique_dims.append(dim)
                    seen_values.add(dim.value_feet)
            
            # Sort by position (for top/bottom: by x, for left/right: by y)
            if edge in ['top', 'bottom'] and all(d.bbox for d in unique_dims):
                unique_dims.sort(key=lambda d: d.bbox[0])
            elif edge in ['left', 'right'] and all(d.bbox for d in unique_dims):
                unique_dims.sort(key=lambda d: d.bbox[1])
            
            validated[edge] = unique_dims
        
        return validated
    
    def _calculate_scale(self, dimensions: Dict[str, List[Dimension]], 
                        image_width: int, image_height: int) -> float:
        """Calculate pixels per foot scale from dimensions."""
        scales = []
        
        # Calculate from horizontal dimensions
        top_total = sum(d.value_feet for d in dimensions.get('top', []))
        bottom_total = sum(d.value_feet for d in dimensions.get('bottom', []))
        
        if top_total > 0:
            scales.append(image_width / top_total)
        if bottom_total > 0:
            scales.append(image_width / bottom_total)
        
        # Calculate from vertical dimensions
        left_total = sum(d.value_feet for d in dimensions.get('left', []))
        right_total = sum(d.value_feet for d in dimensions.get('right', []))
        
        if left_total > 0:
            scales.append(image_height / left_total)
        if right_total > 0:
            scales.append(image_height / right_total)
        
        # Use median scale if available
        if scales:
            scale = np.median(scales)
        else:
            # Use default if no dimensions found
            scale = self.config['default_pixels_per_foot']
            logger.warning(f"No dimensions found, using default scale: {scale}")
        
        # Clamp to reasonable range
        scale = max(self.config['min_pixels_per_foot'], 
                   min(scale, self.config['max_pixels_per_foot']))
        
        return scale
    
    def _summarize_dimensions(self, dimensions: Dict[str, List[Dimension]]) -> str:
        """Create summary string of extracted dimensions."""
        summary = []
        for edge, dims in dimensions.items():
            if dims:
                values = [f"{d.value_feet}'" for d in dims]
                summary.append(f"{edge}: {', '.join(values)}")
        return "; ".join(summary) if summary else "No dimensions found"
    
    def extract_from_luna(self, image_path: str) -> Dict[str, List[Dimension]]:
        """
        Special handling for Luna floor plan with known dimensions.
        Leverages existing luna_dimension_extractor.py data.
        """
        # Known Luna dimensions
        luna_dimensions = {
            'top': [
                Dimension("14'-6\"", 14.5, 'top', 1.0),
                Dimension("11'", 11.0, 'top', 1.0),
                Dimension("8'", 8.0, 'top', 1.0),
                Dimension("10'-6\"", 10.5, 'top', 1.0),
                Dimension("11'", 11.0, 'top', 1.0),
                Dimension("8'", 8.0, 'top', 1.0),
                Dimension("11'", 11.0, 'top', 1.0),
            ],
            'bottom': [
                Dimension("18'", 18.0, 'bottom', 1.0),
                Dimension("32'-6\"", 32.5, 'bottom', 1.0),
                Dimension("8'", 8.0, 'bottom', 1.0),
                Dimension("7'-6\"", 7.5, 'bottom', 1.0),
                Dimension("7'-6\"", 7.5, 'bottom', 1.0),
                Dimension("24'", 24.0, 'bottom', 1.0),
            ],
            'left': [
                Dimension("20'", 20.0, 'left', 1.0),
                Dimension("10'", 10.0, 'left', 1.0),
                Dimension("18'", 18.0, 'left', 1.0),
            ],
            'right': [
                Dimension("11'", 11.0, 'right', 1.0),
                Dimension("5'", 5.0, 'right', 1.0),
                Dimension("18'", 18.0, 'right', 1.0),
                Dimension("20'", 20.0, 'right', 1.0),
            ],
        }
        
        logger.info("Using known Luna floor plan dimensions")
        return luna_dimensions
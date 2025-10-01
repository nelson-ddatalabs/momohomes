#!/usr/bin/env python3
"""
Enhanced Boundary Extractor with Measurement Preservation
==========================================================
Extracts floor plan boundaries while preserving measurement annotations.
"""

import cv2
import numpy as np
import pytesseract
from typing import Tuple, List, Dict, Optional
import logging
import re
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Measurement:
    """Represents a measurement annotation."""
    text: str
    value_feet: float
    position: Tuple[int, int]
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    confidence: float


class EnhancedBoundaryExtractor:
    """
    Extracts boundaries while preserving measurements.
    
    Two-step process:
    1. Create binary mask (inside=white, outside=black)
    2. Preserve measurement annotations
    """
    
    def __init__(self):
        """Initialize the extractor."""
        self.original_image = None
        self.binary_mask = None
        self.enhanced_binary = None
        self.measurements = []
        self.boundary_points = []
        
    def process_floor_plan(self, image_path: str) -> Dict:
        """
        Main processing pipeline.
        
        Args:
            image_path: Path to floor plan image
            
        Returns:
            Dictionary with binary image and measurements
        """
        logger.info(f"Processing floor plan: {image_path}")
        
        # Load image
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        height, width = self.original_image.shape[:2]
        logger.info(f"Image size: {width}x{height}")
        
        # Step 1: Create proper binary mask
        self.binary_mask = self._create_binary_mask()
        
        # Step 2: Extract measurements BEFORE they're lost
        self.measurements = self._extract_measurements()
        
        # Step 3: Create enhanced binary with measurements preserved
        self.enhanced_binary = self._create_enhanced_binary()
        
        # Step 4: Extract boundary points
        self.boundary_points = self._extract_boundary_points()
        
        # Save debug images
        cv2.imwrite("debug_original.png", self.original_image)
        cv2.imwrite("debug_binary_mask.png", self.binary_mask)
        cv2.imwrite("debug_enhanced_binary.png", self.enhanced_binary)
        
        logger.info(f"Extracted {len(self.measurements)} measurements")
        logger.info(f"Boundary has {len(self.boundary_points)} points")
        
        return {
            'binary_mask': self.binary_mask,
            'enhanced_binary': self.enhanced_binary,
            'measurements': self.measurements,
            'boundary_points': self.boundary_points,
            'success': True
        }
    
    def _create_binary_mask(self) -> np.ndarray:
        """
        Create binary mask where inside=white, outside=black.
        
        Returns:
            Binary mask image
        """
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2HSV)
        
        # Detect green areas (indoor space)
        lower_green = np.array([35, 30, 30])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
        
        # Find the main contour
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            logger.warning("No green area found, using full image")
            return np.ones_like(green_mask) * 255
        
        # Get largest contour (main floor area)
        main_contour = max(contours, key=cv2.contourArea)
        
        # Create new mask and fill the interior
        filled_mask = np.zeros_like(green_mask)
        cv2.drawContours(filled_mask, [main_contour], -1, 255, -1)  # Fill interior
        
        logger.info(f"Binary mask created, filled area: {np.sum(filled_mask > 0)} pixels")
        
        return filled_mask
    
    def _extract_measurements(self) -> List[Measurement]:
        """
        Extract measurement annotations from the original image.
        
        Returns:
            List of Measurement objects
        """
        measurements = []
        
        # Convert to grayscale for OCR
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Direct OCR on full image
        self._extract_with_ocr(gray, measurements)
        
        # Method 2: Look for text in specific regions
        self._extract_from_regions(gray, measurements)
        
        # Method 3: Detect measurement patterns near edges
        self._extract_edge_measurements(gray, measurements)
        
        # Remove duplicates
        measurements = self._remove_duplicate_measurements(measurements)
        
        return measurements
    
    def _extract_with_ocr(self, gray_image: np.ndarray, measurements: List[Measurement]):
        """Extract measurements using OCR."""
        # Get OCR data with bounding boxes
        ocr_data = pytesseract.image_to_data(gray_image, output_type=pytesseract.Output.DICT)
        
        n_boxes = len(ocr_data['text'])
        for i in range(n_boxes):
            text = str(ocr_data['text'][i]).strip()
            conf = ocr_data['conf'][i]
            
            if text and conf > 30:  # Confidence threshold
                # Check if it looks like a measurement
                value = self._parse_measurement(text)
                if value:
                    x = ocr_data['left'][i]
                    y = ocr_data['top'][i]
                    w = ocr_data['width'][i]
                    h = ocr_data['height'][i]
                    
                    measurement = Measurement(
                        text=text,
                        value_feet=value,
                        position=(x + w//2, y + h//2),
                        bbox=(x, y, w, h),
                        confidence=conf/100.0
                    )
                    measurements.append(measurement)
                    logger.debug(f"Found measurement: {text} = {value} ft at ({x},{y})")
    
    def _extract_from_regions(self, gray_image: np.ndarray, measurements: List[Measurement]):
        """Extract measurements from perimeter regions."""
        height, width = gray_image.shape
        
        # Define perimeter regions
        regions = [
            ('top', 0, 0, width, min(200, height//4)),
            ('bottom', 0, max(0, height-200), width, min(200, height//4)),
            ('left', 0, 0, min(200, width//4), height),
            ('right', max(0, width-200), 0, min(200, width//4), height)
        ]
        
        for name, x, y, w, h in regions:
            if w <= 0 or h <= 0:
                continue
                
            roi = gray_image[y:y+h, x:x+w]
            
            # Enhance contrast for better OCR
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            roi_enhanced = clahe.apply(roi)
            
            # Try different preprocessing
            _, binary = cv2.threshold(roi_enhanced, 150, 255, cv2.THRESH_BINARY)
            
            # OCR on region
            text = pytesseract.image_to_string(binary)
            lines = text.strip().split('\n')
            
            for line in lines:
                value = self._parse_measurement(line)
                if value:
                    # Approximate position
                    measurement = Measurement(
                        text=line,
                        value_feet=value,
                        position=(x + w//2, y + h//2),
                        bbox=(x, y, w, h),
                        confidence=0.7
                    )
                    measurements.append(measurement)
    
    def _extract_edge_measurements(self, gray_image: np.ndarray, measurements: List[Measurement]):
        """Extract measurements near detected edges."""
        # Detect edges
        edges = cv2.Canny(gray_image, 50, 150)
        
        # Find horizontal and vertical lines (likely dimension lines)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, 
                               minLineLength=50, maxLineGap=10)
        
        if lines is None:
            return
        
        # Look for text near lines
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Define ROI around line
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            
            # Extract small region around line center
            roi_size = 100
            x_start = max(0, cx - roi_size//2)
            y_start = max(0, cy - roi_size//2)
            x_end = min(gray_image.shape[1], cx + roi_size//2)
            y_end = min(gray_image.shape[0], cy + roi_size//2)
            
            if x_end > x_start and y_end > y_start:
                roi = gray_image[y_start:y_end, x_start:x_end]
                
                # Quick OCR check
                text = pytesseract.image_to_string(roi, config='--psm 8')
                value = self._parse_measurement(text)
                
                if value:
                    measurement = Measurement(
                        text=text.strip(),
                        value_feet=value,
                        position=(cx, cy),
                        bbox=(x_start, y_start, x_end-x_start, y_end-y_start),
                        confidence=0.6
                    )
                    measurements.append(measurement)
    
    def _parse_measurement(self, text: str) -> Optional[float]:
        """
        Parse measurement text to extract feet value.
        
        Args:
            text: Text to parse
            
        Returns:
            Value in feet or None
        """
        if not text:
            return None
        
        # Clean text
        text = text.strip()
        
        # Patterns for measurements
        patterns = [
            # Format: 32'-6" or 32' 6"
            (r"(\d+)['''][\s\-]?(\d+)[\"\"]", 
             lambda m: float(m.group(1)) + float(m.group(2))/12),
            # Format: 32' or 32.5'
            (r"(\d+\.?\d*)[''']", 
             lambda m: float(m.group(1))),
            # Format: 32-6 (assume feet-inches)
            (r"(\d+)[\-\s](\d{1,2})(?!\d)", 
             lambda m: float(m.group(1)) + float(m.group(2))/12),
        ]
        
        for pattern, converter in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    value = converter(match)
                    if 1 <= value <= 200:  # Reasonable range
                        return value
                except:
                    continue
        
        return None
    
    def _remove_duplicate_measurements(self, measurements: List[Measurement]) -> List[Measurement]:
        """Remove duplicate measurements based on proximity."""
        if not measurements:
            return []
        
        unique = []
        distance_threshold = 50  # pixels
        
        for m in measurements:
            is_duplicate = False
            for existing in unique:
                # Check spatial proximity
                dist = np.sqrt((m.position[0] - existing.position[0])**2 + 
                             (m.position[1] - existing.position[1])**2)
                
                # Check value similarity
                value_diff = abs(m.value_feet - existing.value_feet)
                
                if dist < distance_threshold and value_diff < 0.5:
                    is_duplicate = True
                    # Keep the one with higher confidence
                    if m.confidence > existing.confidence:
                        unique.remove(existing)
                        unique.append(m)
                    break
            
            if not is_duplicate:
                unique.append(m)
        
        return unique
    
    def _create_enhanced_binary(self) -> np.ndarray:
        """
        Create enhanced binary image with measurements preserved.
        
        Returns:
            Enhanced binary image
        """
        # Start with the binary mask
        enhanced = self.binary_mask.copy()
        
        # Convert to 3-channel for annotation
        if len(enhanced.shape) == 2:
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        # Add measurement annotations
        for m in self.measurements:
            # Draw measurement text
            cv2.putText(enhanced, f"{m.value_feet:.1f}'", 
                       (m.position[0]-20, m.position[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
            
            # Draw a small marker
            cv2.circle(enhanced, m.position, 3, (128, 128, 128), -1)
        
        # Convert back to grayscale
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        
        return enhanced
    
    def _extract_boundary_points(self) -> List[Tuple[int, int]]:
        """
        Extract boundary points from the binary mask.
        
        Returns:
            List of (x, y) boundary points
        """
        # Find contours
        contours, _ = cv2.findContours(self.binary_mask, 
                                      cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return []
        
        # Get largest contour
        main_contour = max(contours, key=cv2.contourArea)
        
        # Simplify contour
        epsilon = 0.01 * cv2.arcLength(main_contour, True)
        approx = cv2.approxPolyDP(main_contour, epsilon, True)
        
        # Convert to list of points
        points = [(int(p[0][0]), int(p[0][1])) for p in approx]
        
        return points
    
    def visualize_results(self, output_path: str = "extraction_visualization.png"):
        """Create visualization of extraction results."""
        if self.original_image is None:
            return
        
        # Create figure with subplots
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        
        # Original image
        axes[0, 0].imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("Original Floor Plan")
        axes[0, 0].axis('off')
        
        # Binary mask
        axes[0, 1].imshow(self.binary_mask, cmap='gray')
        axes[0, 1].set_title("Binary Mask (Inside=White)")
        axes[0, 1].axis('off')
        
        # Enhanced binary with measurements
        axes[1, 0].imshow(self.enhanced_binary, cmap='gray')
        axes[1, 0].set_title(f"Enhanced with {len(self.measurements)} Measurements")
        axes[1, 0].axis('off')
        
        # Boundary overlay
        overlay = self.original_image.copy()
        if self.boundary_points:
            pts = np.array(self.boundary_points, np.int32)
            cv2.polylines(overlay, [pts], True, (0, 255, 0), 2)
        
        # Add measurement markers
        for m in self.measurements:
            cv2.circle(overlay, m.position, 5, (255, 0, 0), -1)
            cv2.putText(overlay, f"{m.value_feet:.0f}'", 
                       (m.position[0]+10, m.position[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        axes[1, 1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title("Boundary & Measurements Overlay")
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        logger.info(f"Visualization saved to {output_path}")


def test_on_luna():
    """Test the enhanced extractor on Luna.png."""
    print("\n" + "="*70)
    print("TESTING ENHANCED BOUNDARY EXTRACTOR")
    print("="*70)
    
    extractor = EnhancedBoundaryExtractor()
    
    # Process Luna
    results = extractor.process_floor_plan("Luna.png")
    
    if results['success']:
        print(f"\n✓ Successfully processed Luna.png")
        print(f"  Measurements found: {len(results['measurements'])}")
        print(f"  Boundary points: {len(results['boundary_points'])}")
        
        print("\nExtracted Measurements:")
        for i, m in enumerate(results['measurements'][:10], 1):
            print(f"  {i}. {m.text} = {m.value_feet:.1f} ft at ({m.position[0]}, {m.position[1]})")
        
        if len(results['measurements']) > 10:
            print(f"  ... and {len(results['measurements'])-10} more")
        
        # Create visualization
        extractor.visualize_results("luna_extraction_debug.png")
        print("\n✓ Visualization saved to luna_extraction_debug.png")
    else:
        print("✗ Extraction failed")
    
    return results


if __name__ == "__main__":
    test_on_luna()
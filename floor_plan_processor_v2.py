#!/usr/bin/env python3
"""
Floor Plan Processor V2
========================
Correctly handles floor plans with white measurements on black background.
"""

import cv2
import numpy as np
import pytesseract
from pytesseract import Output
from typing import Tuple, List, Dict, Optional
import logging
import re
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EdgeMeasurement:
    """Measurement associated with an edge."""
    text: str
    value_feet: float
    position: Tuple[int, int]
    edge_side: str  # 'top', 'bottom', 'left', 'right'
    confidence: float


class FloorPlanProcessorV2:
    """
    Processes floor plans with white measurements on black background.
    
    Key steps:
    1. Extract green area as indoor space
    2. Preserve white measurement annotations
    3. Create clean binary with measurements
    """
    
    def __init__(self):
        """Initialize processor."""
        self.original = None
        self.green_mask = None
        self.white_mask = None
        self.binary_with_measurements = None
        self.measurements = []
        self.boundary_contour = None
        
    def process(self, image_path: str) -> Dict:
        """
        Main processing pipeline.
        
        Args:
            image_path: Path to floor plan image
            
        Returns:
            Processing results
        """
        logger.info(f"Processing: {image_path}")
        
        # Load image
        self.original = cv2.imread(image_path)
        if self.original is None:
            raise ValueError(f"Cannot load: {image_path}")
        
        h, w = self.original.shape[:2]
        logger.info(f"Image size: {w}x{h}")
        
        # Step 1: Extract green area (indoor space)
        self.green_mask = self._extract_green_area()
        
        # Step 2: Extract white pixels (measurements)
        self.white_mask = self._extract_white_pixels()
        
        # Step 3: Find boundary of green area
        self.boundary_contour = self._find_boundary()
        
        # Step 4: Extract measurements from white pixels
        self.measurements = self._extract_measurements()
        
        # Step 5: Create final binary image
        self.binary_with_measurements = self._create_final_binary()
        
        # Save debug outputs
        self._save_debug_images()
        
        logger.info(f"Found {len(self.measurements)} measurements")
        
        return {
            'success': True,
            'green_mask': self.green_mask,
            'white_mask': self.white_mask,
            'binary_final': self.binary_with_measurements,
            'measurements': self.measurements,
            'boundary_points': self._get_boundary_points()
        }
    
    def _extract_green_area(self) -> np.ndarray:
        """
        Extract green pixels (indoor living space).
        
        Returns:
            Binary mask of green area
        """
        # Convert to HSV
        hsv = cv2.cvtColor(self.original, cv2.COLOR_BGR2HSV)
        
        # Green range (tuned for floor plans)
        lower_green = np.array([40, 40, 40])  # Adjusted for floor plan green
        upper_green = np.array([80, 255, 255])
        
        # Create mask
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Clean up
        kernel = np.ones((5, 5), np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
        
        # Fill holes
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Fill the largest contour
            largest = max(contours, key=cv2.contourArea)
            filled = np.zeros_like(green_mask)
            cv2.drawContours(filled, [largest], -1, 255, -1)
            green_mask = filled
        
        logger.info(f"Green area: {np.sum(green_mask > 0)} pixels")
        
        return green_mask
    
    def _extract_white_pixels(self) -> np.ndarray:
        """
        Extract white pixels (measurements and dimension lines).
        
        Returns:
            Binary mask of white pixels
        """
        # Convert to grayscale
        gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        
        # Threshold for white pixels
        # White text/lines should be > 200 in grayscale
        _, white_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Remove noise
        kernel = np.ones((2, 2), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        
        logger.info(f"White pixels: {np.sum(white_mask > 0)}")
        
        return white_mask
    
    def _find_boundary(self):
        """
        Find the boundary contour of the green area.
        
        Returns:
            Main boundary contour
        """
        contours, _ = cv2.findContours(self.green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            logger.warning("No boundary found")
            return None
        
        # Get largest contour
        boundary = max(contours, key=cv2.contourArea)
        
        # Simplify
        epsilon = 0.01 * cv2.arcLength(boundary, True)
        boundary = cv2.approxPolyDP(boundary, epsilon, True)
        
        logger.info(f"Boundary has {len(boundary)} points")
        
        return boundary
    
    def _extract_measurements(self) -> List[EdgeMeasurement]:
        """
        Extract measurements from white pixels using OCR.
        
        Returns:
            List of measurements
        """
        measurements = []
        
        # Create image with only white text for OCR
        ocr_image = np.zeros_like(self.original)
        ocr_image[self.white_mask > 0] = self.original[self.white_mask > 0]
        
        # Convert to grayscale for OCR
        gray_ocr = cv2.cvtColor(ocr_image, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Direct OCR on white pixels
        data = pytesseract.image_to_data(gray_ocr, output_type=Output.DICT)
        
        n_boxes = len(data['text'])
        for i in range(n_boxes):
            text = str(data['text'][i]).strip()
            conf = data['conf'][i]
            
            if text and conf > 30:
                # Parse measurement
                value = self._parse_measurement(text)
                if value:
                    x = data['left'][i] + data['width'][i] // 2
                    y = data['top'][i] + data['height'][i] // 2
                    
                    # Determine which edge this belongs to
                    edge = self._determine_edge(x, y)
                    
                    measurement = EdgeMeasurement(
                        text=text,
                        value_feet=value,
                        position=(x, y),
                        edge_side=edge,
                        confidence=conf / 100.0
                    )
                    measurements.append(measurement)
                    logger.debug(f"Found: {text} = {value}' on {edge} edge")
        
        # Method 2: Look for measurements in specific regions
        self._extract_edge_measurements(gray_ocr, measurements)
        
        # Remove duplicates
        measurements = self._remove_duplicates(measurements)
        
        return measurements
    
    def _extract_edge_measurements(self, gray_image: np.ndarray, measurements: List[EdgeMeasurement]):
        """Extract measurements from edge regions."""
        h, w = gray_image.shape
        
        # Define edge regions (where measurements typically appear)
        regions = [
            ('top', 0, 0, w, min(150, h//4)),
            ('bottom', 0, max(0, h-150), w, min(150, h//4)),
            ('left', 0, 0, min(150, w//4), h),
            ('right', max(0, w-150), 0, min(150, w//4), h)
        ]
        
        for edge_name, x, y, width, height in regions:
            if width <= 0 or height <= 0:
                continue
            
            roi = gray_image[y:y+height, x:x+width]
            
            # OCR on region
            text = pytesseract.image_to_string(roi)
            
            # Parse all measurements in text
            for line in text.split('\n'):
                value = self._parse_measurement(line)
                if value:
                    # Calculate position in full image
                    pos_x = x + width // 2
                    pos_y = y + height // 2
                    
                    measurement = EdgeMeasurement(
                        text=line.strip(),
                        value_feet=value,
                        position=(pos_x, pos_y),
                        edge_side=edge_name,
                        confidence=0.7
                    )
                    
                    # Check if not duplicate
                    is_dup = False
                    for m in measurements:
                        if abs(m.value_feet - value) < 0.5 and m.edge_side == edge_name:
                            is_dup = True
                            break
                    
                    if not is_dup:
                        measurements.append(measurement)
    
    def _parse_measurement(self, text: str) -> Optional[float]:
        """
        Parse measurement text to extract feet value.
        
        Handles formats like:
        - 14'-6" or 14' 6"
        - 11' Bdrm 2
        - 78' Overall
        """
        if not text:
            return None
        
        # Patterns for measurements
        patterns = [
            # Feet and inches: 14'-6", 14' 6"
            (r"(\d+)['''][\s\-]?(\d+)[\"\"]", 
             lambda m: float(m.group(1)) + float(m.group(2))/12),
            # Just feet: 11' or 78'
            (r"(\d+)[''']", 
             lambda m: float(m.group(1))),
            # Feet-inches without symbols: 14-6
            (r"(\d+)[\-\s](\d{1,2})(?!\d)", 
             lambda m: float(m.group(1)) + float(m.group(2))/12 if int(m.group(2)) < 12 else float(m.group(1))),
        ]
        
        for pattern, converter in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    value = converter(match)
                    if 2 <= value <= 200:  # Reasonable range
                        return value
                except:
                    continue
        
        return None
    
    def _determine_edge(self, x: int, y: int) -> str:
        """Determine which edge a measurement belongs to."""
        h, w = self.original.shape[:2]
        
        # Distance from edges
        dist_top = y
        dist_bottom = h - y
        dist_left = x
        dist_right = w - x
        
        # Find minimum distance
        min_dist = min(dist_top, dist_bottom, dist_left, dist_right)
        
        if min_dist == dist_top:
            return 'top'
        elif min_dist == dist_bottom:
            return 'bottom'
        elif min_dist == dist_left:
            return 'left'
        else:
            return 'right'
    
    def _remove_duplicates(self, measurements: List[EdgeMeasurement]) -> List[EdgeMeasurement]:
        """Remove duplicate measurements."""
        unique = []
        
        for m in measurements:
            is_dup = False
            for existing in unique:
                # Check if same value and close position
                if (abs(m.value_feet - existing.value_feet) < 0.5 and
                    abs(m.position[0] - existing.position[0]) < 50 and
                    abs(m.position[1] - existing.position[1]) < 50):
                    is_dup = True
                    # Keep higher confidence
                    if m.confidence > existing.confidence:
                        unique.remove(existing)
                        unique.append(m)
                    break
            
            if not is_dup:
                unique.append(m)
        
        return unique
    
    def _create_final_binary(self) -> np.ndarray:
        """
        Create final binary image:
        - Indoor space (green) -> White
        - Outside -> Black
        - Measurements preserved
        """
        # Start with green mask as base
        binary = self.green_mask.copy()
        
        # Add white measurement pixels
        # Only add white pixels that are outside the green area
        measurement_pixels = np.logical_and(self.white_mask > 0, self.green_mask == 0)
        binary[measurement_pixels] = 255
        
        return binary
    
    def _get_boundary_points(self) -> List[Tuple[int, int]]:
        """Get boundary points as list."""
        if self.boundary_contour is None:
            return []
        
        points = [(int(p[0][0]), int(p[0][1])) for p in self.boundary_contour]
        return points
    
    def _save_debug_images(self):
        """Save debug images for inspection."""
        cv2.imwrite("debug_1_green_mask.png", self.green_mask)
        cv2.imwrite("debug_2_white_mask.png", self.white_mask)
        cv2.imwrite("debug_3_final_binary.png", self.binary_with_measurements)
        
        # Create overlay visualization
        overlay = self.original.copy()
        if self.boundary_contour is not None:
            cv2.drawContours(overlay, [self.boundary_contour], -1, (0, 255, 0), 2)
        
        # Mark measurements
        for m in self.measurements:
            color = {
                'top': (255, 0, 0),      # Blue
                'bottom': (0, 255, 255), # Yellow
                'left': (255, 0, 255),   # Magenta
                'right': (0, 255, 0)     # Green
            }.get(m.edge_side, (255, 255, 255))
            
            cv2.circle(overlay, m.position, 5, color, -1)
            cv2.putText(overlay, f"{m.value_feet:.0f}'", 
                       (m.position[0] + 10, m.position[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        cv2.imwrite("debug_4_overlay.png", overlay)
        logger.info("Debug images saved")
    
    def visualize_results(self):
        """Create comprehensive visualization."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original
        axes[0, 0].imshow(cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("Original Floor Plan")
        axes[0, 0].axis('off')
        
        # Green mask
        axes[0, 1].imshow(self.green_mask, cmap='gray')
        axes[0, 1].set_title("Green Area (Indoor Space)")
        axes[0, 1].axis('off')
        
        # White mask
        axes[0, 2].imshow(self.white_mask, cmap='gray')
        axes[0, 2].set_title("White Pixels (Measurements)")
        axes[0, 2].axis('off')
        
        # Final binary
        axes[1, 0].imshow(self.binary_with_measurements, cmap='gray')
        axes[1, 0].set_title("Final Binary with Measurements")
        axes[1, 0].axis('off')
        
        # Measurements by edge
        edge_viz = np.zeros_like(self.original)
        colors = {
            'top': (255, 0, 0),
            'bottom': (0, 255, 255),
            'left': (255, 0, 255),
            'right': (0, 255, 0)
        }
        
        for m in self.measurements:
            color = colors.get(m.edge_side, (255, 255, 255))
            cv2.circle(edge_viz, m.position, 10, color, -1)
        
        axes[1, 1].imshow(cv2.cvtColor(edge_viz, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title(f"Measurements by Edge ({len(self.measurements)} total)")
        axes[1, 1].axis('off')
        
        # Summary text
        axes[1, 2].axis('off')
        summary = f"Summary:\n"
        summary += f"Image: {self.original.shape[1]}x{self.original.shape[0]}\n"
        summary += f"Indoor area: {np.sum(self.green_mask > 0)} pixels\n"
        summary += f"Measurements found: {len(self.measurements)}\n\n"
        
        # Group by edge
        edge_counts = {'top': 0, 'bottom': 0, 'left': 0, 'right': 0}
        for m in self.measurements:
            edge_counts[m.edge_side] = edge_counts.get(m.edge_side, 0) + 1
        
        summary += "By Edge:\n"
        for edge, count in edge_counts.items():
            summary += f"  {edge}: {count} measurements\n"
        
        summary += "\nMeasurement Values:\n"
        for i, m in enumerate(self.measurements[:10], 1):
            summary += f"  {i}. {m.text} = {m.value_feet:.1f}' ({m.edge_side})\n"
        
        axes[1, 2].text(0.1, 0.9, summary, fontsize=10, verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig("floor_plan_analysis.png", dpi=150)
        plt.close()
        
        logger.info("Visualization saved to floor_plan_analysis.png")


def test_on_conditioned_plans():
    """Test on all conditioned floor plans."""
    print("\n" + "="*70)
    print("TESTING FLOOR PLAN PROCESSOR V2")
    print("="*70)
    
    # Test on Luna-Conditioned
    processor = FloorPlanProcessorV2()
    
    test_file = "/Users/nelsondsouza/Dropbox (Personal)/Mac/Documents/products/momohomes/floorplans/Luna-Conditioned.png"
    
    if not Path(test_file).exists():
        print(f"Error: {test_file} not found")
        return
    
    results = processor.process(test_file)
    
    if results['success']:
        print(f"\n✓ Successfully processed Luna-Conditioned.png")
        print(f"  Measurements found: {len(results['measurements'])}")
        print(f"  Boundary points: {len(results['boundary_points'])}")
        
        print("\nExtracted Measurements:")
        for m in results['measurements']:
            print(f"  {m.edge_side:6s}: {m.text:20s} = {m.value_feet:5.1f} ft")
        
        # Create visualization
        processor.visualize_results()
        print("\n✓ Visualization saved")
        
        # Check against expected
        print("\nValidation:")
        print(f"  Expected area: 2733.25 sq ft")
        print(f"  Expected perimeter: 265 ft")
        
        # Calculate from measurements
        top_measurements = [m for m in results['measurements'] if m.edge_side == 'top']
        print(f"  Top edge measurements: {len(top_measurements)}")
    else:
        print("✗ Processing failed")


if __name__ == "__main__":
    test_on_conditioned_plans()
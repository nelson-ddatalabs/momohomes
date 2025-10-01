#!/usr/bin/env python3
"""
Integrated Floor Plan Measurement System
=========================================
Combines OCR extraction, binary conversion, and edge association.
"""

import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import re
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Edge:
    """Represents an edge of the floor plan."""
    start: Tuple[int, int]
    end: Tuple[int, int]
    measurement: Optional[float] = None
    orientation: str = ""  # 'horizontal' or 'vertical'
    
    def length_pixels(self) -> float:
        """Calculate pixel length of edge."""
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        return np.sqrt(dx**2 + dy**2)
    
    def midpoint(self) -> Tuple[int, int]:
        """Get midpoint of edge."""
        return (
            (self.start[0] + self.end[0]) // 2,
            (self.start[1] + self.end[1]) // 2
        )


class IntegratedMeasurementSystem:
    """Extract measurements and associate with floor plan edges."""
    
    def __init__(self):
        self.edges = []
        self.measurements = []
        self.scale_factor = None
        
    def process_floor_plan(self, image_path: str) -> Dict:
        """
        Complete processing pipeline for floor plan.
        
        Returns:
            Dict with edges, measurements, and binary image
        """
        logger.info(f"Processing {image_path}")
        
        # Load original image
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Cannot load {image_path}")
            return {}
        
        # Step 1: Extract measurements using OCR
        self.measurements = self._extract_measurements(img)
        logger.info(f"Found {len(self.measurements)} measurements")
        
        # Step 2: Create binary image preserving measurements
        binary_img, white_mask = self._create_binary_with_measurements(img)
        
        # Step 3: Find edges from binary image
        self.edges = self._find_edges(binary_img)
        logger.info(f"Found {len(self.edges)} edges")
        
        # Step 4: Associate measurements with edges
        self._associate_measurements_to_edges()
        
        # Step 5: Calculate scale factor
        self._calculate_scale_factor()
        
        # Step 6: Generate output
        result = self._generate_output(binary_img)
        
        # Save debug images
        self._save_debug_images(img, binary_img, white_mask)
        
        return result
    
    def _extract_measurements(self, img: np.ndarray) -> List[Dict]:
        """Extract white text measurements using multiple methods."""
        measurements = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Direct white text extraction
        _, white_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Find text regions
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter for text-like regions
            if 10 < w < 200 and 5 < h < 50:
                roi = gray[y:y+h, x:x+w]
                
                # Invert for OCR (white text → black)
                roi_inv = cv2.bitwise_not(roi)
                
                # OCR on region
                try:
                    text = pytesseract.image_to_string(roi_inv, config='--psm 8')
                    value = self._parse_measurement(text)
                    
                    if value:
                        measurements.append({
                            'text': text.strip(),
                            'value': value,
                            'x': x + w//2,
                            'y': y + h//2,
                            'bbox': (x, y, w, h)
                        })
                        logger.debug(f"Found: {text.strip()} = {value}' at ({x+w//2}, {y+h//2})")
                except:
                    pass
        
        # Method 2: Full image OCR with inversion
        inverted = cv2.bitwise_not(gray)
        data = pytesseract.image_to_data(inverted, config='--psm 11', output_type=Output.DICT)
        
        for i in range(len(data['text'])):
            text = str(data['text'][i]).strip()
            conf = data['conf'][i]
            
            if text and conf > 30:
                value = self._parse_measurement(text)
                if value:
                    measurements.append({
                        'text': text,
                        'value': value,
                        'x': data['left'][i] + data['width'][i]//2,
                        'y': data['top'][i] + data['height'][i]//2,
                        'bbox': (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
                    })
        
        # Remove duplicates
        unique = self._deduplicate_measurements(measurements)
        
        return unique
    
    def _parse_measurement(self, text: str) -> Optional[float]:
        """Parse measurement from text."""
        if not text:
            return None
        
        text = text.strip()
        
        # Patterns for feet and inches
        patterns = [
            # 14'-6" or 14' 6"
            (r"(\d+)['''][\s\-]?(\d+)[\"\"]", 
             lambda m: float(m.group(1)) + float(m.group(2))/12),
            # 11' or 78'
            (r"(\d+)[''']", 
             lambda m: float(m.group(1))),
            # 14-6 (interpret as feet-inches)
            (r"(\d+)[\-\s](\d{1,2})(?!\d)", 
             lambda m: float(m.group(1)) + float(m.group(2))/12 if int(m.group(2)) < 12 else None),
        ]
        
        for pattern, converter in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    value = converter(match)
                    if value and 2 <= value <= 200:  # Reasonable range for room dimensions
                        return value
                except:
                    continue
        
        return None
    
    def _create_binary_with_measurements(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create binary image: green areas → white, outside → black.
        Preserve white measurement text.
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Extract green areas (indoor space)
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([80, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Extract white areas (measurements)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, white_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Create binary: green areas become white
        binary = np.zeros_like(gray)
        binary[green_mask > 0] = 255
        
        # Don't overwrite white measurement pixels
        # They stay as part of the black background
        
        return binary, white_mask
    
    def _find_edges(self, binary_img: np.ndarray) -> List[Edge]:
        """Find edges of the indoor space."""
        edges = []
        
        # Find contours
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return edges
        
        # Get largest contour (main floor area)
        main_contour = max(contours, key=cv2.contourArea)
        
        # Approximate polygon
        epsilon = 0.01 * cv2.arcLength(main_contour, True)
        approx = cv2.approxPolyDP(main_contour, epsilon, True)
        
        # Create edges from polygon points
        for i in range(len(approx)):
            start = tuple(approx[i][0])
            end = tuple(approx[(i+1) % len(approx)][0])
            
            # Determine orientation
            dx = abs(end[0] - start[0])
            dy = abs(end[1] - start[1])
            orientation = "horizontal" if dx > dy else "vertical"
            
            edge = Edge(start=start, end=end, orientation=orientation)
            edges.append(edge)
        
        return edges
    
    def _associate_measurements_to_edges(self):
        """Associate extracted measurements with edges."""
        for measurement in self.measurements:
            mx, my = measurement['x'], measurement['y']
            best_edge = None
            min_dist = float('inf')
            
            for edge in self.edges:
                # Distance from measurement to edge midpoint
                ex, ey = edge.midpoint()
                dist = np.sqrt((mx - ex)**2 + (my - ey)**2)
                
                # Also check perpendicular distance to edge line
                # This helps associate measurements that are offset from edges
                perp_dist = self._perpendicular_distance_to_edge(mx, my, edge)
                
                # Combined distance metric
                combined_dist = dist * 0.5 + perp_dist * 0.5
                
                if combined_dist < min_dist and combined_dist < 100:  # Within 100 pixels
                    min_dist = combined_dist
                    best_edge = edge
            
            if best_edge and not best_edge.measurement:
                best_edge.measurement = measurement['value']
                logger.debug(f"Associated {measurement['value']}' to edge at {best_edge.midpoint()}")
    
    def _perpendicular_distance_to_edge(self, px: int, py: int, edge: Edge) -> float:
        """Calculate perpendicular distance from point to edge line."""
        x1, y1 = edge.start
        x2, y2 = edge.end
        
        # Line equation: ax + by + c = 0
        a = y2 - y1
        b = x1 - x2
        c = x2*y1 - x1*y2
        
        # Perpendicular distance
        dist = abs(a*px + b*py + c) / np.sqrt(a**2 + b**2)
        
        return dist
    
    def _calculate_scale_factor(self):
        """Calculate scale factor from measured edges."""
        scale_factors = []
        
        for edge in self.edges:
            if edge.measurement:
                pixel_length = edge.length_pixels()
                if pixel_length > 0:
                    scale = edge.measurement / pixel_length  # feet per pixel
                    scale_factors.append(scale)
        
        if scale_factors:
            self.scale_factor = np.median(scale_factors)
            logger.info(f"Scale factor: {self.scale_factor:.4f} feet/pixel")
    
    def _deduplicate_measurements(self, measurements: List[Dict]) -> List[Dict]:
        """Remove duplicate measurements."""
        unique = []
        seen = set()
        
        for m in measurements:
            # Create key from value and approximate position
            key = (round(m['value'], 1), m['x']//50, m['y']//50)
            
            if key not in seen:
                seen.add(key)
                unique.append(m)
        
        return unique
    
    def _generate_output(self, binary_img: np.ndarray) -> Dict:
        """Generate output dictionary."""
        # Convert edges to dictionary format
        edges_data = []
        for i, edge in enumerate(self.edges):
            edges_data.append({
                'id': i,
                'start': edge.start,
                'end': edge.end,
                'measurement': edge.measurement,
                'orientation': edge.orientation,
                'length_pixels': edge.length_pixels()
            })
        
        # Calculate total measured perimeter
        perimeter = sum(e.measurement for e in self.edges if e.measurement) if self.edges else 0
        
        return {
            'edges': edges_data,
            'measurements': self.measurements,
            'scale_factor': self.scale_factor,
            'perimeter_feet': perimeter,
            'num_edges': len(self.edges),
            'num_measurements': len(self.measurements),
            'binary_image': binary_img
        }
    
    def _save_debug_images(self, original: np.ndarray, binary: np.ndarray, white_mask: np.ndarray):
        """Save debug images."""
        # Draw edges and measurements on original
        debug_img = original.copy()
        
        # Draw edges
        for edge in self.edges:
            cv2.line(debug_img, edge.start, edge.end, (0, 255, 0), 2)
            
            # Draw measurement if available
            if edge.measurement:
                mx, my = edge.midpoint()
                cv2.putText(debug_img, f"{edge.measurement:.1f}'", 
                           (mx-20, my-10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (255, 255, 0), 2)
        
        # Draw measurement locations
        for m in self.measurements:
            cv2.circle(debug_img, (m['x'], m['y']), 5, (0, 0, 255), -1)
        
        cv2.imwrite("debug_integrated_overlay.png", debug_img)
        cv2.imwrite("debug_integrated_binary.png", binary)
        cv2.imwrite("debug_integrated_white_mask.png", white_mask)
        
        logger.info("Saved debug images")


def test_integration():
    """Test the integrated system."""
    system = IntegratedMeasurementSystem()
    
    test_file = "/Users/nelsondsouza/Dropbox (Personal)/Mac/Documents/products/momohomes/floorplans/Luna-Conditioned.png"
    
    result = system.process_floor_plan(test_file)
    
    print("\n" + "="*70)
    print("INTEGRATED MEASUREMENT SYSTEM TEST")
    print("="*70)
    
    print(f"\nEdges found: {result.get('num_edges', 0)}")
    print(f"Measurements found: {result.get('num_measurements', 0)}")
    print(f"Scale factor: {result.get('scale_factor', 0):.4f} feet/pixel")
    print(f"Perimeter: {result.get('perimeter_feet', 0):.1f} feet")
    
    print("\nEdges with measurements:")
    for edge in result.get('edges', []):
        if edge['measurement']:
            print(f"  Edge {edge['id']}: {edge['measurement']:.1f}' ({edge['orientation']})")
    
    print("\nAll measurements extracted:")
    for m in result.get('measurements', [])[:10]:  # Show first 10
        print(f"  {m['text']} = {m['value']:.1f}' at ({m['x']}, {m['y']})")
    
    # Save result to JSON
    json_result = {k: v for k, v in result.items() if k != 'binary_image'}
    with open("measurement_result.json", "w") as f:
        json.dump(json_result, f, indent=2, default=str)
    
    print("\nResults saved to measurement_result.json")
    
    return result


if __name__ == "__main__":
    test_integration()
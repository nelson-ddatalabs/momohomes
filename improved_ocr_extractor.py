#!/usr/bin/env python3
"""
Improved OCR Extractor for White Text
======================================
Specialized OCR for white text on black background floor plans.
"""

import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import re
from typing import List, Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class ImprovedOCRExtractor:
    """Extract white text measurements from floor plans."""
    
    def extract_white_text(self, image_path: str) -> List[Dict]:
        """
        Extract white text measurements from floor plan.
        
        Args:
            image_path: Path to floor plan image
            
        Returns:
            List of extracted measurements
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Cannot load {image_path}")
            return []
        
        measurements = []
        
        # Method 1: Invert for OCR (white text → black text on white bg)
        measurements.extend(self._method_inverted(img))
        
        # Method 2: Direct white pixel extraction with dilation
        measurements.extend(self._method_white_extraction(img))
        
        # Method 3: Edge-based text detection
        measurements.extend(self._method_edge_text(img))
        
        # Method 4: Threshold variations
        measurements.extend(self._method_threshold_variations(img))
        
        # Remove duplicates
        unique = self._deduplicate(measurements)
        
        logger.info(f"Total unique measurements: {len(unique)}")
        
        return unique
    
    def _method_inverted(self, img: np.ndarray) -> List[Dict]:
        """Method 1: Invert image for standard OCR."""
        logger.debug("Method 1: Inverted OCR")
        measurements = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Invert (white text becomes black)
        inverted = cv2.bitwise_not(gray)
        
        # Threshold to clean up
        _, thresh = cv2.threshold(inverted, 180, 255, cv2.THRESH_BINARY)
        
        # Save for debug
        cv2.imwrite("debug_ocr_inverted.png", thresh)
        
        # OCR with different configs
        configs = [
            '--psm 11',  # Sparse text
            '--psm 6',   # Uniform block
            '--psm 12',  # Sparse text with OSD
        ]
        
        for config in configs:
            try:
                data = pytesseract.image_to_data(thresh, config=config, output_type=Output.DICT)
                
                for i in range(len(data['text'])):
                    text = str(data['text'][i]).strip()
                    conf = data['conf'][i]
                    
                    if text and conf > 20:  # Lower threshold
                        value = self._parse_measurement(text)
                        if value:
                            measurements.append({
                                'text': text,
                                'value': value,
                                'x': data['left'][i] + data['width'][i]//2,
                                'y': data['top'][i] + data['height'][i]//2,
                                'conf': conf,
                                'method': 'inverted'
                            })
                            logger.debug(f"Found (inverted): {text} = {value}")
            except Exception as e:
                logger.debug(f"OCR failed with config {config}: {e}")
        
        return measurements
    
    def _method_white_extraction(self, img: np.ndarray) -> List[Dict]:
        """Method 2: Extract and enhance white pixels."""
        logger.debug("Method 2: White pixel extraction")
        measurements = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Extract white pixels
        _, white_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Dilate to connect text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        dilated = cv2.dilate(white_mask, kernel, iterations=1)
        
        # Create black background with white text
        ocr_img = np.zeros_like(gray)
        ocr_img[dilated > 0] = 255
        
        # Invert for OCR
        ocr_img = cv2.bitwise_not(ocr_img)
        
        cv2.imwrite("debug_ocr_white_extracted.png", ocr_img)
        
        # OCR
        text = pytesseract.image_to_string(ocr_img)
        
        # Parse line by line
        for line in text.split('\n'):
            value = self._parse_measurement(line)
            if value:
                measurements.append({
                    'text': line.strip(),
                    'value': value,
                    'x': 0,  # Position unknown with this method
                    'y': 0,
                    'conf': 50,
                    'method': 'white_extraction'
                })
                logger.debug(f"Found (white): {line.strip()} = {value}")
        
        return measurements
    
    def _method_edge_text(self, img: np.ndarray) -> List[Dict]:
        """Method 3: Look for text along edges."""
        logger.debug("Method 3: Edge text detection")
        measurements = []
        
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Define edge regions with larger bands
        regions = [
            ('top', 0, 0, w, min(250, h//3)),
            ('bottom', 0, max(0, h-250), w, min(250, h//3)),
            ('left', 0, 0, min(250, w//3), h),
            ('right', max(0, w-250), 0, min(250, w//3), h)
        ]
        
        for edge, x, y, width, height in regions:
            if width <= 0 or height <= 0:
                continue
            
            # Extract region
            roi = gray[y:y+height, x:x+width]
            
            # Invert ROI
            roi_inv = cv2.bitwise_not(roi)
            
            # Threshold
            _, roi_thresh = cv2.threshold(roi_inv, 180, 255, cv2.THRESH_BINARY)
            
            # OCR
            text = pytesseract.image_to_string(roi_thresh)
            
            # Parse measurements
            for line in text.split('\n'):
                value = self._parse_measurement(line)
                if value:
                    measurements.append({
                        'text': line.strip(),
                        'value': value,
                        'x': x + width//2,
                        'y': y + height//2,
                        'conf': 60,
                        'method': f'edge_{edge}'
                    })
                    logger.debug(f"Found ({edge}): {line.strip()} = {value}")
        
        return measurements
    
    def _method_threshold_variations(self, img: np.ndarray) -> List[Dict]:
        """Method 4: Try different threshold values."""
        logger.debug("Method 4: Threshold variations")
        measurements = []
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Try different thresholds
        thresholds = [180, 190, 200, 210, 220]
        
        for thresh_val in thresholds:
            _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
            
            # Invert
            thresh_inv = cv2.bitwise_not(thresh)
            
            # Quick OCR
            text = pytesseract.image_to_string(thresh_inv, config='--psm 11')
            
            # Look for first valid measurement
            for line in text.split('\n')[:5]:  # Check first 5 lines
                value = self._parse_measurement(line)
                if value:
                    measurements.append({
                        'text': line.strip(),
                        'value': value,
                        'x': 0,
                        'y': 0,
                        'conf': 40,
                        'method': f'threshold_{thresh_val}'
                    })
                    break  # One per threshold
        
        return measurements
    
    def _parse_measurement(self, text: str) -> Optional[float]:
        """Parse measurement from text."""
        if not text:
            return None
        
        # Clean text
        text = text.strip()
        
        # Patterns
        patterns = [
            # 14'-6" or 14' 6"
            (r"(\d+)['''][\s\-]?(\d+)[\"\"]", 
             lambda m: float(m.group(1)) + float(m.group(2))/12),
            # 11' or 78'
            (r"(\d+)[''']", 
             lambda m: float(m.group(1))),
            # 14-6 or 14 6
            (r"(\d+)[\-\s](\d{1,2})(?!\d)", 
             lambda m: float(m.group(1)) + float(m.group(2))/12 if int(m.group(2)) < 12 else None),
        ]
        
        for pattern, converter in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    value = converter(match)
                    if value and 2 <= value <= 200:
                        return value
                except:
                    continue
        
        return None
    
    def _deduplicate(self, measurements: List[Dict]) -> List[Dict]:
        """Remove duplicate measurements."""
        unique = []
        seen_values = set()
        
        for m in measurements:
            # Create key from value and rough position
            key = (round(m['value'], 1), m['method'][:4])
            
            if key not in seen_values:
                seen_values.add(key)
                unique.append(m)
        
        return unique
    
    def test_all_methods(self, image_path: str):
        """Test and report all methods."""
        print("\n" + "="*70)
        print("OCR EXTRACTION TEST")
        print("="*70)
        
        measurements = self.extract_white_text(image_path)
        
        print(f"\nTotal measurements found: {len(measurements)}")
        
        # Group by method
        by_method = {}
        for m in measurements:
            method = m['method']
            if method not in by_method:
                by_method[method] = []
            by_method[method].append(m)
        
        print("\nBy extraction method:")
        for method, items in by_method.items():
            print(f"  {method}: {len(items)} measurements")
            for item in items[:3]:  # Show first 3
                print(f"    - {item['text']} = {item['value']:.1f}'")
        
        print("\nAll unique values found:")
        all_values = sorted(set(m['value'] for m in measurements))
        print(f"  {all_values}")
        
        return measurements


if __name__ == "__main__":
    extractor = ImprovedOCRExtractor()
    
    # Test on Luna-Conditioned
    test_file = "/Users/nelsondsouza/Dropbox (Personal)/Mac/Documents/products/momohomes/floorplans/Luna-Conditioned.png"
    
    measurements = extractor.test_all_methods(test_file)
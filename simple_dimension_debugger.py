#!/usr/bin/env python3
"""
Simple Dimension Extraction Debugger
=====================================
Focused debugging tool for dimension extraction issues.
"""

import cv2
import numpy as np
import pytesseract
from pathlib import Path
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def debug_luna_dimensions():
    """Debug dimension extraction on Luna.png."""
    
    # Load image
    image = cv2.imread("Luna.png")
    if image is None:
        logger.error("Failed to load Luna.png")
        return
    
    height, width = image.shape[:2]
    logger.info(f"Image size: {width}x{height}")
    
    # Create output directory
    output_dir = Path("dimension_debug")
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*70)
    print("DIMENSION EXTRACTION DEBUG")
    print("="*70)
    
    # 1. Look at the actual dimension text on the image
    print("\n1. ANALYZING DIMENSION TEXT AREAS")
    print("-" * 40)
    
    # Define perimeter bands to check for dimensions
    # These are the areas where dimensions typically appear
    band_width = 150  # Increased band width to capture dimensions
    
    regions = {
        'top': (0, 0, width, band_width),
        'bottom': (0, height - band_width, width, band_width),
        'left': (0, 0, band_width, height),
        'right': (width - band_width, 0, band_width, height)
    }
    
    all_dimensions = []
    
    for edge_name, (x, y, w, h) in regions.items():
        print(f"\nAnalyzing {edge_name} edge ({x},{y}) {w}x{h}:")
        
        # Extract ROI
        roi = image[y:y+h, x:x+w]
        
        # Save ROI for inspection
        roi_path = output_dir / f"{edge_name}_edge.png"
        cv2.imwrite(str(roi_path), roi)
        
        # Try multiple preprocessing methods
        dimensions_found = []
        
        # Method 1: Direct OCR on color image
        text1 = pytesseract.image_to_string(roi, config='--psm 11')
        dims1 = extract_dimensions_from_text(text1)
        if dims1:
            print(f"  Method 1 (direct): {dims1}")
            dimensions_found.extend(dims1)
        
        # Method 2: Grayscale with threshold
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        cv2.imwrite(str(output_dir / f"{edge_name}_binary.png"), binary)
        
        text2 = pytesseract.image_to_string(binary, config='--psm 11')
        dims2 = extract_dimensions_from_text(text2)
        if dims2:
            print(f"  Method 2 (binary): {dims2}")
            dimensions_found.extend(dims2)
        
        # Method 3: Inverted binary
        binary_inv = cv2.bitwise_not(binary)
        cv2.imwrite(str(output_dir / f"{edge_name}_binary_inv.png"), binary_inv)
        
        text3 = pytesseract.image_to_string(binary_inv, config='--psm 11')
        dims3 = extract_dimensions_from_text(text3)
        if dims3:
            print(f"  Method 3 (inverted): {dims3}")
            dimensions_found.extend(dims3)
        
        # Method 4: Enhanced contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        cv2.imwrite(str(output_dir / f"{edge_name}_enhanced.png"), enhanced)
        
        text4 = pytesseract.image_to_string(enhanced, config='--psm 11')
        dims4 = extract_dimensions_from_text(text4)
        if dims4:
            print(f"  Method 4 (enhanced): {dims4}")
            dimensions_found.extend(dims4)
        
        # Remove duplicates and store
        unique_dims = list(set(dimensions_found))
        if unique_dims:
            all_dimensions.append({
                'edge': edge_name,
                'dimensions': unique_dims
            })
            print(f"  Total unique: {unique_dims}")
        else:
            print(f"  No dimensions found")
    
    # 2. Try OCR on the full image
    print("\n2. FULL IMAGE OCR")
    print("-" * 40)
    
    gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text_full = pytesseract.image_to_string(gray_full, config='--psm 3')
    dims_full = extract_dimensions_from_text(text_full)
    print(f"Dimensions found in full image: {dims_full}")
    
    # 3. Look specifically for the legend area
    print("\n3. LEGEND AREA ANALYSIS")
    print("-" * 40)
    
    # Legend is typically at the bottom
    legend_roi = image[int(height * 0.85):, :]
    cv2.imwrite(str(output_dir / "legend_area.png"), legend_roi)
    
    legend_text = pytesseract.image_to_string(legend_roi)
    print(f"Legend text: {legend_text[:200]}...")
    
    # Look for area in legend
    area_match = re.search(r'(\d+\.?\d*)\s*square\s*ft', legend_text, re.IGNORECASE)
    if area_match:
        print(f"Legend area: {area_match.group(1)} sq ft")
    
    # 4. Manually look at specific coordinates where dimensions should be
    print("\n4. TARGETED DIMENSION EXTRACTION")
    print("-" * 40)
    
    # Based on Luna.png, dimensions appear to be at specific locations
    # Let's target those areas more precisely
    
    # Top edge dimensions (appear to be multiple segments)
    top_segments = [
        (300, 0, 200, 100),   # First segment
        (600, 0, 200, 100),   # Second segment
        (900, 0, 200, 100),   # Third segment
        (1200, 0, 200, 100),  # Fourth segment
        (1500, 0, 200, 100),  # Fifth segment
        (1800, 0, 200, 100),  # Sixth segment
        (2100, 0, 200, 100),  # Seventh segment
        (2400, 0, 200, 100),  # Eighth segment
    ]
    
    print("\nTop edge segments:")
    for i, (sx, sy, sw, sh) in enumerate(top_segments):
        if sx + sw <= width and sy + sh <= height:
            seg_roi = image[sy:sy+sh, sx:sx+sw]
            seg_text = pytesseract.image_to_string(seg_roi, config='--psm 8')
            seg_dims = extract_dimensions_from_text(seg_text)
            if seg_dims:
                print(f"  Segment {i+1}: {seg_dims}")
    
    # 5. Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    total_dims = sum(len(d['dimensions']) for d in all_dimensions)
    print(f"Total dimensions extracted: {total_dims}")
    
    for edge_data in all_dimensions:
        print(f"\n{edge_data['edge'].capitalize()} edge:")
        for dim in edge_data['dimensions']:
            print(f"  - {dim}")
    
    print("\nExpected for Luna.png:")
    print("  - Total area: 2733.25 sq ft")
    print("  - Approximate dimensions: 73' x 38'")
    print("  - Should have 8-12 perimeter dimensions")
    
    if total_dims < 8:
        print("\n⚠ WARNING: Insufficient dimensions extracted!")
        print("\nPossible issues:")
        print("1. Dimension text may be too small or unclear")
        print("2. Text may be at an angle")
        print("3. Background interference")
        print("4. Need custom OCR training for architectural drawings")
        
        print("\nRecommended approach:")
        print("1. Use line detection to find dimension lines")
        print("2. Target specific areas where dimensions appear")
        print("3. Use multiple OCR passes with different preprocessing")
        print("4. Consider manual annotation for initial setup")
    
    print(f"\nDebug images saved to: {output_dir}/")
    
    return all_dimensions


def extract_dimensions_from_text(text):
    """Extract dimension patterns from OCR text."""
    if not text:
        return []
    
    dimensions = []
    
    # Multiple patterns to catch different formats
    patterns = [
        # Feet and inches: 7'-5", 7' 5", 7'5"
        r"(\d+)['''][\s\-]?(\d+)[\"\"]",
        # Just feet: 7', 10.5'
        r"(\d+\.?\d*)['''](?!\s*\d)",
        # Feet-inches without symbols: 7-5, 7 5
        r"(\d+)[\s\-](\d{1,2})(?!\d)",
        # Decimal feet: 7.5
        r"(\d+\.\d+)(?!\d)",
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if isinstance(match, tuple):
                if len(match) == 2:
                    # Feet and inches
                    feet = match[0]
                    inches = match[1]
                    dimensions.append(f"{feet}'-{inches}\"")
                else:
                    dimensions.append(str(match))
            else:
                dimensions.append(str(match))
    
    # Also look for simple numbers that might be dimensions
    # (in context of architectural drawings)
    number_pattern = r'\b(\d{1,2})\b'
    numbers = re.findall(number_pattern, text)
    for num in numbers:
        if 3 <= int(num) <= 50:  # Reasonable room dimensions in feet
            dimensions.append(f"{num}'")
    
    return list(set(dimensions))  # Remove duplicates


if __name__ == "__main__":
    debug_luna_dimensions()
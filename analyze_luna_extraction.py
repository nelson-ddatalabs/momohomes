#!/usr/bin/env python3
"""
Analyze Luna Floor Plan Extraction
===================================
Step-by-step extraction with visualization.
"""

import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import re
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def analyze_luna_extraction():
    """Sequential analysis of Luna floor plan."""
    
    image_path = "/Users/nelsondsouza/Dropbox (Personal)/Mac/Documents/products/momohomes/floorplans/Luna-Conditioned.png"
    
    print("\n" + "="*70)
    print("STEP-BY-STEP LUNA EXTRACTION ANALYSIS")
    print("="*70)
    
    # Step 1: Load image
    print("\n1. LOADING IMAGE")
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    print(f"   Image size: {w} x {h} pixels")
    
    # Step 2: Extract green area (indoor space)
    print("\n2. EXTRACTING GREEN AREA (INDOOR SPACE)")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    green_area = cv2.countNonZero(green_mask)
    print(f"   Green pixels: {green_area} ({green_area*100/(w*h):.1f}% of image)")
    
    # Step 3: Extract white text areas
    print("\n3. EXTRACTING WHITE TEXT AREAS")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, white_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    white_area = cv2.countNonZero(white_mask)
    print(f"   White pixels: {white_area} ({white_area*100/(w*h):.1f}% of image)")
    
    # Step 4: Find text regions
    print("\n4. FINDING TEXT REGIONS")
    # Dilate white areas to connect text
    kernel = np.ones((3, 3), np.uint8)
    white_dilated = cv2.dilate(white_mask, kernel, iterations=2)
    
    # Find contours of text regions
    contours, _ = cv2.findContours(white_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    text_regions = []
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Filter for text-like regions
        aspect_ratio = w / h if h > 0 else 0
        if 10 < w < 500 and 5 < h < 100 and 0.5 < aspect_ratio < 20:
            text_regions.append((x, y, w, h))
    
    print(f"   Found {len(text_regions)} potential text regions")
    
    # Step 5: OCR on each region
    print("\n5. PERFORMING OCR ON TEXT REGIONS")
    measurements = []
    
    for i, (x, y, w, h) in enumerate(text_regions):
        # Extract region
        roi = gray[y:y+h, x:x+w]
        
        # Invert for OCR (white text becomes black)
        roi_inv = cv2.bitwise_not(roi)
        
        # Try OCR
        try:
            text = pytesseract.image_to_string(roi_inv, config='--psm 8')
            text = text.strip()
            
            if text:
                # Parse measurement
                patterns = [
                    (r"(\d+)'", lambda m: float(m.group(1))),  # 78'
                    (r"(\d+)['-](\d+)[\"\"]?", lambda m: float(m.group(1)) + float(m.group(2))/12),  # 14'-6"
                ]
                
                value = None
                for pattern, converter in patterns:
                    match = re.search(pattern, text)
                    if match:
                        try:
                            value = converter(match)
                            if 2 <= value <= 200:
                                break
                        except:
                            pass
                
                if value or text:  # Keep even if no value parsed
                    measurements.append({
                        'text': text,
                        'value': value,
                        'x': x + w//2,
                        'y': y + h//2,
                        'bbox': (x, y, w, h)
                    })
                    print(f"   Region {i}: '{text}' → {value}' at ({x+w//2}, {y+h//2})")
        except Exception as e:
            print(f"   Region {i}: OCR failed - {e}")
    
    # Step 6: Find edges of green area
    print("\n6. FINDING EDGES OF GREEN AREA")
    # Find contours of green area
    green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if green_contours:
        # Get largest contour
        main_contour = max(green_contours, key=cv2.contourArea)
        
        # Approximate polygon
        epsilon = 0.01 * cv2.arcLength(main_contour, True)
        approx = cv2.approxPolyDP(main_contour, epsilon, True)
        
        print(f"   Main contour: {len(approx)} vertices")
        
        edges = []
        for i in range(len(approx)):
            p1 = tuple(approx[i][0])
            p2 = tuple(approx[(i+1) % len(approx)][0])
            
            # Calculate edge properties
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = np.sqrt(dx**2 + dy**2)
            
            # Determine orientation
            if abs(dx) > abs(dy):
                orientation = "horizontal"
                position = "top" if p1[1] < h/2 else "bottom"
            else:
                orientation = "vertical"
                position = "left" if p1[0] < w/2 else "right"
            
            edges.append({
                'start': p1,
                'end': p2,
                'length_pixels': length,
                'orientation': orientation,
                'position': position
            })
            
            print(f"   Edge {i}: {orientation} {position}, {length:.0f} pixels")
    
    # Step 7: Create visualization
    print("\n7. CREATING VISUALIZATION")
    vis = img.copy()
    
    # Draw green area outline
    cv2.drawContours(vis, [approx], -1, (0, 255, 0), 3)
    
    # Draw edge numbers
    for i, edge in enumerate(edges):
        mid_x = (edge['start'][0] + edge['end'][0]) // 2
        mid_y = (edge['start'][1] + edge['end'][1]) // 2
        cv2.putText(vis, f"E{i}", (mid_x-10, mid_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Draw text regions and measurements
    for i, m in enumerate(measurements):
        x, y, w, h = m['bbox']
        # Draw box around text region
        cv2.rectangle(vis, (x, y), (x+w, y+h), (255, 255, 0), 2)
        
        # Draw measurement value if parsed
        if m['value']:
            cv2.putText(vis, f"{m['value']:.1f}'", (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
    # Draw measurement indicators
    for m in measurements:
        cv2.circle(vis, (m['x'], m['y']), 5, (0, 0, 255), -1)
    
    # Add legend
    legend_y = 50
    cv2.putText(vis, "EXTRACTION RESULTS:", (50, legend_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(vis, f"Green edges: {len(edges)}", (50, legend_y+30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(vis, f"Text regions: {len(text_regions)}", (50, legend_y+60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(vis, f"Measurements: {len(measurements)}", (50, legend_y+90),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Save visualization
    output_path = "luna_extraction_visualization.png"
    cv2.imwrite(output_path, vis)
    print(f"\n   Saved to: {output_path}")
    
    # Step 8: Analysis summary
    print("\n8. EXTRACTION SUMMARY")
    print("="*70)
    print(f"   Total edges found: {len(edges)}")
    print(f"   Total measurements found: {len(measurements)}")
    
    print("\n   Measurements extracted:")
    for m in measurements:
        print(f"   - '{m['text']}' = {m['value']}' at ({m['x']}, {m['y']})")
    
    print("\n   Edge positions:")
    edge_positions = {}
    for i, e in enumerate(edges):
        key = f"{e['orientation']} {e['position']}"
        if key not in edge_positions:
            edge_positions[key] = []
        edge_positions[key].append(i)
    
    for pos, indices in edge_positions.items():
        print(f"   - {pos}: edges {indices}")
    
    print("\n   PROBLEMS IDENTIFIED:")
    print("   1. Not all white text is being detected")
    print("   2. Edge detection is following green contour, not actual building perimeter")
    print("   3. Measurements not properly associated with edges")
    print("   4. Missing overall dimensions (78' and 40'-6\")")
    
    return measurements, edges


if __name__ == "__main__":
    analyze_luna_extraction()
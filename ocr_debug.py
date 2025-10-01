#!/usr/bin/env python3
"""
OCR Debug Tool
==============
Debug what OCR is actually returning from Luna.png.
"""

import cv2
import pytesseract
from pytesseract import Output
import re

def debug_ocr_output():
    """Debug OCR output to understand what's being returned."""
    
    print("\n" + "="*70)
    print("OCR DEBUG - Luna.png")
    print("="*70)
    
    # Load image
    image = cv2.imread("Luna.png")
    if image is None:
        print("Error: Cannot load Luna.png")
        return
    
    print(f"\nImage size: {image.shape[1]}x{image.shape[0]}")
    
    # 1. Simple string extraction
    print("\n1. SIMPLE STRING EXTRACTION")
    print("-" * 40)
    
    text = pytesseract.image_to_string(image)
    lines = [l for l in text.split('\n') if l.strip()]
    
    print(f"Found {len(lines)} lines of text")
    print("\nFirst 10 lines:")
    for i, line in enumerate(lines[:10]):
        print(f"  {i+1}: {line}")
    
    # Look for dimension patterns
    print("\nDimension-like text found:")
    dim_pattern = r"\d+['''][\s\-]?\d+[\"\"]|\d+[''']|\d+\-\d+"
    for line in lines:
        if re.search(dim_pattern, line):
            print(f"  - {line}")
    
    # 2. Data extraction with bounding boxes
    print("\n2. DATA EXTRACTION WITH BOUNDING BOXES")
    print("-" * 40)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Get OCR data
    ocr_data = pytesseract.image_to_data(gray, output_type=Output.DICT)
    
    print(f"Total text boxes: {len(ocr_data['text'])}")
    
    # Filter non-empty text
    valid_texts = []
    for i in range(len(ocr_data['text'])):
        text = str(ocr_data['text'][i]).strip()
        if text:
            valid_texts.append({
                'text': text,
                'x': ocr_data['left'][i],
                'y': ocr_data['top'][i],
                'width': ocr_data['width'][i],
                'height': ocr_data['height'][i],
                'conf': ocr_data['conf'][i]
            })
    
    print(f"Non-empty text boxes: {len(valid_texts)}")
    
    # Show first 20 valid texts
    print("\nFirst 20 text detections:")
    for i, item in enumerate(valid_texts[:20]):
        print(f"  {i+1}: '{item['text']}' at ({item['x']},{item['y']}) conf={item['conf']}")
    
    # Look for dimension patterns in detected text
    print("\nDimension patterns in OCR data:")
    dimension_texts = []
    
    patterns = [
        (r"(\d+)['''][\s\-]?(\d+)[\"\"]", "feet-inches"),  # 32'-6"
        (r"(\d+)[''']", "feet"),  # 32'
        (r"(\d+)\-(\d+)", "hyphenated"),  # 32-6
        (r"^\d+$", "number")  # Just numbers
    ]
    
    for item in valid_texts:
        text = item['text']
        for pattern, pattern_name in patterns:
            if re.search(pattern, text):
                dimension_texts.append({
                    'text': text,
                    'pattern': pattern_name,
                    'x': item['x'],
                    'y': item['y'],
                    'conf': item['conf']
                })
                break
    
    print(f"\nFound {len(dimension_texts)} dimension-like texts:")
    for dim in dimension_texts[:20]:
        print(f"  '{dim['text']}' ({dim['pattern']}) at ({dim['x']},{dim['y']}) conf={dim['conf']}")
    
    # 3. Check specific regions
    print("\n3. CHECKING SPECIFIC REGIONS")
    print("-" * 40)
    
    # Check top region
    top_region = image[0:200, :]
    top_text = pytesseract.image_to_string(top_region)
    print(f"\nTop region text: {top_text[:200]}...")
    
    # Check if dimensions are at specific offsets
    print("\n4. LOOKING FOR DIMENSION LINES")
    print("-" * 40)
    
    # Typical dimension line offset from building edge
    offsets = [50, 100, 150, 200, 250]
    
    for offset in offsets:
        print(f"\nChecking at offset {offset}px from top:")
        roi = image[offset:offset+50, :]
        roi_text = pytesseract.image_to_string(roi)
        if roi_text.strip():
            print(f"  Found: {roi_text[:100]}")
    
    # 5. Save debug visualization
    print("\n5. CREATING DEBUG VISUALIZATION")
    print("-" * 40)
    
    vis_img = image.copy()
    
    # Draw all dimension text locations
    for dim in dimension_texts:
        x, y = dim['x'], dim['y']
        cv2.rectangle(vis_img, (x-5, y-5), (x+50, y+20), (0, 255, 0), 2)
        cv2.putText(vis_img, dim['text'], (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    cv2.imwrite("ocr_debug_visualization.png", vis_img)
    print("Saved visualization to ocr_debug_visualization.png")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total text found: {len(valid_texts)}")
    print(f"Dimension-like texts: {len(dimension_texts)}")
    
    if dimension_texts:
        print("\nDimension values found:")
        unique_dims = set(d['text'] for d in dimension_texts)
        for dim in sorted(unique_dims):
            print(f"  - {dim}")
    
    return dimension_texts


if __name__ == "__main__":
    debug_ocr_output()
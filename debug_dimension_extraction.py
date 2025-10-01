#!/usr/bin/env python3
"""
Dimension Extraction Debugger
==============================
Debug tool to visualize dimension detection on floor plans.
"""

import cv2
import numpy as np
import pytesseract
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class DimensionDebugger:
    """Debug dimension extraction issues."""
    
    def __init__(self):
        """Initialize debugger."""
        self.image = None
        self.dimensions_found = []
        self.debug_images = {}
        
    def analyze_luna(self, image_path: str = "Luna.png"):
        """
        Comprehensive analysis of Luna.png dimension extraction.
        
        Args:
            image_path: Path to Luna.png
        """
        logger.info(f"Analyzing: {image_path}")
        
        # Load image
        self.image = cv2.imread(image_path)
        if self.image is None:
            logger.error(f"Failed to load {image_path}")
            return
        
        height, width = self.image.shape[:2]
        logger.info(f"Image dimensions: {width}x{height}")
        
        # Create figure for visualization
        fig, axes = plt.subplots(3, 3, figsize=(20, 20))
        fig.suptitle("Luna.png Dimension Extraction Debug", fontsize=16)
        
        # 1. Original image
        axes[0, 0].imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis('off')
        
        # 2. Green mask for indoor area
        green_mask = self._detect_green_area()
        axes[0, 1].imshow(green_mask, cmap='gray')
        axes[0, 1].set_title("Green Indoor Area Mask")
        axes[0, 1].axis('off')
        
        # 3. Edge detection
        edges = self._detect_edges()
        axes[0, 2].imshow(edges, cmap='gray')
        axes[0, 2].set_title("Edge Detection (Canny)")
        axes[0, 2].axis('off')
        
        # 4. Text detection areas
        text_regions = self._detect_text_regions()
        axes[1, 0].imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title(f"Text Regions Found: {len(text_regions)}")
        for region in text_regions:
            x, y, w, h = region
            rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                    edgecolor='red', facecolor='none')
            axes[1, 0].add_patch(rect)
        axes[1, 0].axis('off')
        
        # 5. OCR results on each edge
        edge_dims = self._extract_edge_dimensions()
        axes[1, 1].imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title("Dimension Extraction by Edge")
        
        # Draw edge regions
        colors = {'top': 'red', 'right': 'green', 'bottom': 'blue', 'left': 'yellow'}
        for edge, dims in edge_dims.items():
            color = colors.get(edge, 'white')
            for dim in dims:
                x, y, text = dim
                axes[1, 1].scatter(x, y, c=color, s=100, marker='o')
                axes[1, 1].text(x, y+20, f"{edge}: {text}", fontsize=8, color=color)
        axes[1, 1].axis('off')
        
        # 6. Dimension lines detected
        dimension_lines = self._detect_dimension_lines()
        axes[1, 2].imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        axes[1, 2].set_title(f"Dimension Lines: {len(dimension_lines)}")
        
        for line in dimension_lines:
            x1, y1, x2, y2 = line[0]
            axes[1, 2].plot([x1, x2], [y1, y2], 'r-', linewidth=1)
        axes[1, 2].axis('off')
        
        # 7. OCR on full image
        full_ocr = self._perform_full_ocr()
        axes[2, 0].text(0.1, 0.9, "Full OCR Results:", fontsize=12, weight='bold')
        ocr_text = "\n".join(full_ocr[:20])  # Show first 20 lines
        axes[2, 0].text(0.1, 0.8, ocr_text, fontsize=8, verticalalignment='top')
        axes[2, 0].set_title("OCR Text Extraction")
        axes[2, 0].axis('off')
        
        # 8. Perimeter region analysis
        perimeter_analysis = self._analyze_perimeter_regions()
        axes[2, 1].imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        axes[2, 1].set_title("Perimeter Regions (50px band)")
        
        # Draw perimeter bands
        # Top band
        axes[2, 1].add_patch(patches.Rectangle((0, 0), width, 50, 
                                              linewidth=2, edgecolor='red', 
                                              facecolor='red', alpha=0.3))
        # Bottom band
        axes[2, 1].add_patch(patches.Rectangle((0, height-50), width, 50,
                                              linewidth=2, edgecolor='blue',
                                              facecolor='blue', alpha=0.3))
        # Left band
        axes[2, 1].add_patch(patches.Rectangle((0, 0), 50, height,
                                              linewidth=2, edgecolor='yellow',
                                              facecolor='yellow', alpha=0.3))
        # Right band
        axes[2, 1].add_patch(patches.Rectangle((width-50, 0), 50, height,
                                              linewidth=2, edgecolor='green',
                                              facecolor='green', alpha=0.3))
        axes[2, 1].axis('off')
        
        # 9. Summary statistics
        summary = self._generate_summary()
        axes[2, 2].text(0.1, 0.9, "SUMMARY:", fontsize=14, weight='bold')
        axes[2, 2].text(0.1, 0.8, summary, fontsize=10, verticalalignment='top')
        axes[2, 2].axis('off')
        
        # Save debug visualization
        output_path = "debug_dimension_extraction.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Debug visualization saved to: {output_path}")
        
        # Also save individual debug images
        self._save_individual_debugs()
        
        # Close plot to free memory
        plt.close()
        
        return {
            'dimensions_found': self.dimensions_found,
            'edge_dims': edge_dims,
            'text_regions': len(text_regions),
            'dimension_lines': len(dimension_lines)
        }
    
    def _detect_green_area(self) -> np.ndarray:
        """Detect green indoor area."""
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 30, 30])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Clean up
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        logger.info(f"Green area pixels: {np.sum(mask > 0)}")
        return mask
    
    def _detect_edges(self) -> np.ndarray:
        """Detect edges in image."""
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        logger.info(f"Edge pixels: {np.sum(edges > 0)}")
        return edges
    
    def _detect_text_regions(self) -> List[Tuple[int, int, int, int]]:
        """Detect potential text regions."""
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Use MSER to detect text regions
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray)
        
        text_regions = []
        for region in regions:
            x, y, w, h = cv2.boundingRect(region)
            # Filter for reasonable text size
            if 10 < w < 200 and 5 < h < 50:
                text_regions.append((x, y, w, h))
        
        logger.info(f"Text regions detected: {len(text_regions)}")
        return text_regions
    
    def _extract_edge_dimensions(self) -> Dict[str, List]:
        """Extract dimensions from each edge."""
        height, width = self.image.shape[:2]
        edge_dims = {
            'top': [],
            'right': [],
            'bottom': [],
            'left': []
        }
        
        # Define edge regions (50 pixel bands)
        regions = {
            'top': (0, 0, width, 100),
            'bottom': (0, height-100, width, 100),
            'left': (0, 0, 100, height),
            'right': (width-100, 0, 100, height)
        }
        
        for edge, (x, y, w, h) in regions.items():
            roi = self.image[y:y+h, x:x+w]
            
            # Apply OCR
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Try different preprocessing
            _, binary = cv2.threshold(gray_roi, 150, 255, cv2.THRESH_BINARY)
            
            # OCR with different configs
            configs = [
                '--psm 6',  # Uniform block of text
                '--psm 8',  # Single word
                '--psm 11', # Sparse text
                '--psm 12', # Sparse text with OSD
            ]
            
            for config in configs:
                try:
                    text = pytesseract.image_to_string(binary, config=config)
                    lines = text.strip().split('\n')
                    
                    for line in lines:
                        # Look for dimension patterns
                        import re
                        # Match patterns like "7'-5"" or "7' 5"" or "7.5'"
                        patterns = [
                            r"(\d+)[''][\s-]?(\d+)[\"\"]",  # feet and inches
                            r"(\d+\.?\d*)['']",  # just feet
                            r"(\d+)[\s-]?(\d+)"  # numbers without symbols
                        ]
                        
                        for pattern in patterns:
                            matches = re.findall(pattern, line)
                            if matches:
                                for match in matches:
                                    # Calculate position in original image
                                    cx = x + w//2
                                    cy = y + h//2
                                    edge_dims[edge].append((cx, cy, str(match)))
                                    self.dimensions_found.append({
                                        'edge': edge,
                                        'text': str(match),
                                        'position': (cx, cy)
                                    })
                except Exception as e:
                    logger.debug(f"OCR failed for {edge} with config {config}: {e}")
        
        # Log results
        for edge, dims in edge_dims.items():
            logger.info(f"{edge} edge: {len(dims)} dimensions found")
            for dim in dims:
                logger.debug(f"  {edge}: {dim[2]}")
        
        return edge_dims
    
    def _detect_dimension_lines(self) -> List:
        """Detect dimension lines using Hough transform."""
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100,
                               minLineLength=50, maxLineGap=10)
        
        if lines is not None:
            logger.info(f"Dimension lines detected: {len(lines)}")
            return lines
        return []
    
    def _perform_full_ocr(self) -> List[str]:
        """Perform OCR on entire image."""
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # OCR
        text = pytesseract.image_to_string(enhanced)
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        logger.info(f"Full OCR extracted {len(lines)} lines of text")
        
        # Look for dimension-like text
        import re
        dimension_pattern = r"(\d+)[''][\s-]?(\d+)?[\"\"]?"
        dimensions = []
        for line in lines:
            if re.search(dimension_pattern, line):
                dimensions.append(line)
                logger.debug(f"Dimension text found: {line}")
        
        logger.info(f"Found {len(dimensions)} dimension-like text lines")
        
        return lines
    
    def _analyze_perimeter_regions(self) -> Dict:
        """Analyze perimeter regions specifically."""
        height, width = self.image.shape[:2]
        band_width = 50
        
        analysis = {}
        
        # Analyze each edge band
        edges = {
            'top': self.image[0:band_width, :],
            'bottom': self.image[height-band_width:height, :],
            'left': self.image[:, 0:band_width],
            'right': self.image[:, width-band_width:width]
        }
        
        for edge_name, roi in edges.items():
            # Convert to grayscale
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Count non-white pixels (potential text/lines)
            _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
            non_white = np.sum(binary > 0)
            
            analysis[edge_name] = {
                'non_white_pixels': non_white,
                'percentage': non_white / (roi.shape[0] * roi.shape[1]) * 100
            }
            
            logger.info(f"{edge_name} band: {non_white} non-white pixels "
                       f"({analysis[edge_name]['percentage']:.1f}%)")
        
        return analysis
    
    def _generate_summary(self) -> str:
        """Generate summary of findings."""
        summary = []
        
        summary.append(f"Dimensions found: {len(self.dimensions_found)}")
        
        # Group by edge
        edge_counts = {}
        for dim in self.dimensions_found:
            edge = dim['edge']
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
        
        summary.append("\nBy Edge:")
        for edge in ['top', 'right', 'bottom', 'left']:
            count = edge_counts.get(edge, 0)
            summary.append(f"  {edge}: {count} dimensions")
        
        # Expected vs actual
        summary.append("\nExpected for Luna.png:")
        summary.append("  ~8-12 perimeter dimensions")
        summary.append("  Area: 2733.25 sq ft")
        summary.append("  Approx: 73' x 38'")
        
        summary.append("\nIssues Identified:")
        if len(self.dimensions_found) < 8:
            summary.append("  ⚠ Too few dimensions extracted")
            summary.append("  ⚠ OCR may need preprocessing")
            summary.append("  ⚠ Text may be too small/unclear")
        
        return "\n".join(summary)
    
    def _save_individual_debugs(self):
        """Save individual debug images."""
        output_dir = Path("debug_output")
        output_dir.mkdir(exist_ok=True)
        
        # Save preprocessed versions
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(str(output_dir / "gray.png"), gray)
        
        # Enhanced contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        cv2.imwrite(str(output_dir / "enhanced.png"), enhanced)
        
        # Binary threshold
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        cv2.imwrite(str(output_dir / "binary.png"), binary)
        
        # Edges
        edges = cv2.Canny(gray, 50, 150)
        cv2.imwrite(str(output_dir / "edges.png"), edges)
        
        logger.info(f"Individual debug images saved to {output_dir}/")


def main():
    """Run dimension extraction debugging."""
    debugger = DimensionDebugger()
    results = debugger.analyze_luna("Luna.png")
    
    print("\n" + "="*70)
    print("DIMENSION EXTRACTION DEBUG COMPLETE")
    print("="*70)
    print(f"Total dimensions found: {len(results['dimensions_found'])}")
    print(f"Text regions detected: {results['text_regions']}")
    print(f"Dimension lines detected: {results['dimension_lines']}")
    
    if len(results['dimensions_found']) < 8:
        print("\n⚠ WARNING: Insufficient dimensions extracted!")
        print("Recommended fixes:")
        print("1. Improve image preprocessing (contrast, sharpening)")
        print("2. Use targeted ROI extraction for dimension text")
        print("3. Train custom OCR model for architectural drawings")
        print("4. Manual dimension annotation as fallback")


if __name__ == "__main__":
    main()
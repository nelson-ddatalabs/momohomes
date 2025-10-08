"""
Simplified edge detection processor for web application
Extracts floor plan edges and generates labeled visualization
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict
import sys
import os
from pathlib import Path

# Add parent directory to access main modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from cardinal_edge_detector import CardinalEdge


class EdgeProcessor:
    """Process floor plan images to detect and label edges"""

    def __init__(self):
        pass

    def process_image(self, image_path: str, output_path: str) -> Dict:
        """
        Process image to detect edges and generate labeled visualization

        Args:
            image_path: Path to input floor plan image
            output_path: Path to save labeled output image

        Returns:
            Dict with edge_count and edge_data
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Failed to read image")

            # Detect edges using simple contour detection
            edges = self._detect_edges(image)

            # Generate labeled visualization
            self._generate_labeled_image(image, edges, output_path)

            return {
                'success': True,
                'edge_count': len(edges),
                'edges': [{'id': i+1, 'length': e['pixel_length']} for i, e in enumerate(edges)]
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def _detect_edges(self, image: np.ndarray) -> List[Dict]:
        """
        Detect edges in floor plan using contour detection

        Args:
            image: Input image as numpy array

        Returns:
            List of edge dictionaries
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Detect edges using Canny
        edges_img = cv2.Canny(blurred, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            # Fallback: create simple rectangular boundary
            h, w = image.shape[:2]
            return [
                {'pixel_length': w, 'start': (0, 0), 'end': (w, 0)},      # Top
                {'pixel_length': h, 'start': (w, 0), 'end': (w, h)},      # Right
                {'pixel_length': w, 'start': (w, h), 'end': (0, h)},      # Bottom
                {'pixel_length': h, 'start': (0, h), 'end': (0, 0)}       # Left
            ]

        # Get largest contour (main floor plan boundary)
        largest_contour = max(contours, key=cv2.contourArea)

        # Approximate contour to polygon
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)

        # Convert polygon to edges
        edges = []
        points = approx.reshape(-1, 2)

        for i in range(len(points)):
            start = tuple(points[i])
            end = tuple(points[(i + 1) % len(points)])

            # Calculate edge length
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            length = np.sqrt(dx*dx + dy*dy)

            edges.append({
                'pixel_length': length,
                'start': start,
                'end': end
            })

        return edges

    def _generate_labeled_image(self, image: np.ndarray, edges: List[Dict], output_path: str):
        """
        Generate visualization with numbered edges

        Args:
            image: Input image
            edges: List of detected edges
            output_path: Path to save output
        """
        # Create copy of image
        labeled = image.copy()

        # Colors
        edge_color = (0, 255, 0)      # Green for edges
        text_color = (255, 255, 255)  # White text
        bg_color = (0, 100, 0)        # Dark green background for numbers

        # Draw edges and labels
        for i, edge in enumerate(edges):
            edge_num = i + 1
            start = edge['start']
            end = edge['end']

            # Draw edge line (thicker)
            cv2.line(labeled, start, end, edge_color, 4)

            # Calculate midpoint for label
            mid_x = int((start[0] + end[0]) / 2)
            mid_y = int((start[1] + end[1]) / 2)

            # Draw label background circle
            cv2.circle(labeled, (mid_x, mid_y), 25, bg_color, -1)
            cv2.circle(labeled, (mid_x, mid_y), 25, edge_color, 2)

            # Draw edge number
            text = str(edge_num)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2

            # Get text size for centering
            (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_x = mid_x - text_w // 2
            text_y = mid_y + text_h // 2

            cv2.putText(labeled, text, (text_x, text_y), font, font_scale, text_color, thickness)

        # Add title
        cv2.putText(labeled, "Floor Plan Edges - Enter measurements for each numbered edge",
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Save labeled image
        cv2.imwrite(output_path, labeled)

"""
Cardinal Edge Detection Processor for Web Application
======================================================
Uses the same robust cardinal system as run_cassette_system.py
Extracts floor plan edges using green color detection and cardinal directions
"""

import cv2
import numpy as np
from typing import List, Dict
import sys
from pathlib import Path

# Add parent directory to access main modules
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Add alternate parent directory
alt_parent = "/Users/nelsondsouza/Documents/products/momohomes"
if alt_parent not in sys.path:
    sys.path.insert(0, alt_parent)

from enhanced_binary_converter import EnhancedBinaryConverter
from cardinal_edge_detector import CardinalEdgeDetector, CardinalEdge


class EdgeProcessor:
    """Process floor plan images using cardinal edge detection system"""

    def __init__(self):
        """Initialize with cardinal system components"""
        self.binary_converter = EnhancedBinaryConverter()
        self.edge_detector = CardinalEdgeDetector(simplification_epsilon=5.0)

    def process_image(self, image_path: str, output_path: str, binary_output_path: str = None) -> Dict:
        """
        Process image using cardinal edge detection system

        Args:
            image_path: Path to input floor plan image
            output_path: Path to save labeled output image
            binary_output_path: Optional path to save binary image

        Returns:
            Dict with edge_count, edges list, and binary_path
        """
        try:
            # Step 1: Convert to binary using green extraction
            binary_image = self.binary_converter.convert_to_binary(image_path)
            if binary_image is None:
                raise ValueError("Failed to convert image to binary")

            # Save binary image if path provided
            if binary_output_path:
                self.binary_converter.save_binary(binary_image, binary_output_path)

            # Step 2: Detect cardinal edges
            cardinal_edges = self.edge_detector.detect_cardinal_edges(binary_image)

            if not cardinal_edges:
                raise ValueError("No cardinal edges detected")

            # Step 3: Generate labeled visualization
            self._generate_labeled_image(image_path, binary_image, cardinal_edges, output_path)

            # Step 4: Convert to dictionary format for webapp
            # Convert numpy int32 to Python int for JSON serialization
            edges_data = []
            for i, edge in enumerate(cardinal_edges):
                edges_data.append({
                    'id': i + 1,
                    'pixel_length': float(edge.pixel_length),
                    'cardinal_direction': edge.cardinal_direction,
                    'start': (int(edge.start[0]), int(edge.start[1])),
                    'end': (int(edge.end[0]), int(edge.end[1]))
                })

            return {
                'success': True,
                'edge_count': len(cardinal_edges),
                'edges': edges_data,
                'cardinal_edges': cardinal_edges,  # Keep original objects for optimization
                'binary_path': binary_output_path
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def _generate_labeled_image(self, original_image_path: str, binary_image: np.ndarray,
                                edges: List[CardinalEdge], output_path: str):
        """
        Generate visualization with numbered edges and cardinal directions

        Args:
            original_image_path: Path to original image
            binary_image: Binary image from conversion
            edges: List of CardinalEdge objects
            output_path: Path to save output
        """
        # Load original image for better visualization
        original = cv2.imread(original_image_path)
        if original is None:
            # Fallback to binary if original fails
            original = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

        labeled = original.copy()

        # Colors for cardinal directions
        direction_colors = {
            'N': (255, 0, 0),    # Blue for North
            'S': (0, 255, 255),  # Yellow for South
            'E': (0, 255, 0),    # Green for East
            'W': (255, 0, 255)   # Magenta for West
        }

        # Draw edges with cardinal direction colors
        for i, edge in enumerate(edges):
            edge_num = i + 1
            start = edge.start
            end = edge.end
            direction = edge.cardinal_direction
            color = direction_colors.get(direction, (0, 255, 0))

            # Draw edge line (thicker)
            cv2.line(labeled, start, end, color, 4)

            # Calculate midpoint for label
            mid_x = int((start[0] + end[0]) / 2)
            mid_y = int((start[1] + end[1]) / 2)

            # Draw label background circle
            cv2.circle(labeled, (mid_x, mid_y), 30, (50, 50, 50), -1)
            cv2.circle(labeled, (mid_x, mid_y), 30, color, 2)

            # Draw edge number and direction
            text = f"{edge_num}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.9
            thickness = 2

            # Get text size for centering
            (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_x = mid_x - text_w // 2
            text_y = mid_y + text_h // 2

            # Draw number
            cv2.putText(labeled, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

            # Draw direction indicator (small letter below)
            dir_text = direction
            dir_font_scale = 0.4
            (dir_w, dir_h), _ = cv2.getTextSize(dir_text, font, dir_font_scale, 1)
            dir_x = mid_x - dir_w // 2
            dir_y = mid_y + 20
            cv2.putText(labeled, dir_text, (dir_x, dir_y), font, dir_font_scale, (255, 255, 255), 1)

        # Add title with cardinal directions legend
        title_y = 30
        cv2.putText(labeled, "Cardinal Edge Detection - Enter measurements for each edge",
                   (20, title_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Add legend
        legend_y = 60
        legend_x = 20
        cv2.putText(labeled, "N=North", (legend_x, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, direction_colors['N'], 2)
        cv2.putText(labeled, "S=South", (legend_x + 100, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, direction_colors['S'], 2)
        cv2.putText(labeled, "E=East", (legend_x + 200, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, direction_colors['E'], 2)
        cv2.putText(labeled, "W=West", (legend_x + 290, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, direction_colors['W'], 2)

        # Save labeled image
        cv2.imwrite(output_path, labeled)

#!/usr/bin/env python3
"""
Interactive Cardinal Edge Mapper with Visual Display
=====================================================
Shows the numbered edges visualization to the user while collecting measurements.
Uses matplotlib to display the image alongside the terminal.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from cardinal_edge_detector import CardinalEdge

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InteractiveCardinalMapper:
    """Interactive mapper that displays edges while collecting measurements."""

    def __init__(self, edges: List[CardinalEdge], image: np.ndarray):
        """
        Initialize interactive mapper.

        Args:
            edges: List of CardinalEdge objects
            image: Binary or original image for display
        """
        self.edges = edges
        self.image = image
        self.measurements = {}
        self.scale_factor = None
        self.display_image = None
        self.fig = None
        self.ax = None

    def show_edges_with_matplotlib(self, save_path: str = None) -> None:
        """
        Display numbered edges using matplotlib for interactive viewing.

        Args:
            save_path: Optional path to save the visualization
        """
        # Create display image
        if len(self.image.shape) == 2:
            display = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        else:
            display = self.image.copy()

        # Direction colors
        colors = {
            'E': (0, 0, 255),    # Red in BGR
            'W': (255, 0, 255),  # Magenta in BGR
            'N': (0, 255, 0),    # Green in BGR
            'S': (255, 0, 0)     # Blue in BGR
        }

        # Draw edges with numbers
        for i, edge in enumerate(self.edges):
            color = colors[edge.cardinal_direction]

            # Draw edge line
            cv2.line(display, edge.start, edge.end, color, 3)

            # Calculate midpoint
            mid_x = (edge.start[0] + edge.end[0]) // 2
            mid_y = (edge.start[1] + edge.end[1]) // 2

            # Create label with edge number and direction
            label = f"{i}:{edge.cardinal_direction}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2

            # Get text size for background
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)

            # Draw white background for text
            cv2.rectangle(display,
                         (mid_x - text_width // 2 - 5, mid_y - text_height // 2 - 5),
                         (mid_x + text_width // 2 + 5, mid_y + text_height // 2 + 5),
                         (255, 255, 255), -1)

            # Draw text
            cv2.putText(display, label,
                       (mid_x - text_width // 2, mid_y + text_height // 2),
                       font, font_scale, (0, 0, 0), thickness)

            # Draw arrow to show direction
            arrow_length = 30
            arrow_color = (255, 255, 0)  # Yellow for arrows

            # Calculate arrow position
            if edge.cardinal_direction == 'E':
                arrow_start = (edge.end[0] - arrow_length, edge.end[1])
                arrow_end = edge.end
            elif edge.cardinal_direction == 'W':
                arrow_start = (edge.start[0] + arrow_length, edge.start[1])
                arrow_end = edge.start
            elif edge.cardinal_direction == 'N':
                arrow_start = (edge.start[0], edge.start[1] + arrow_length)
                arrow_end = edge.start
            else:  # South
                arrow_start = (edge.end[0], edge.end[1] - arrow_length)
                arrow_end = edge.end

            cv2.arrowedLine(display, arrow_start, arrow_end, arrow_color, 2, tipLength=0.3)

        # Add title and legend
        cv2.putText(display, "NUMBERED EDGES - Cardinal Directions",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Add legend
        legend_y = 60
        cv2.putText(display, "Colors: E=Red, W=Magenta, N=Green, S=Blue",
                   (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Add instruction
        cv2.putText(display, "Measure each edge in FEET going CLOCKWISE from Edge 0",
                   (10, legend_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Convert BGR to RGB for matplotlib
        display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        self.display_image = display_rgb

        # Save if requested
        if save_path:
            cv2.imwrite(save_path, display)
            logger.info(f"Edges visualization saved to: {save_path}")

        # Create matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.ax.imshow(display_rgb)
        self.ax.set_title("Floor Plan with Numbered Edges - Keep this window open while measuring!")
        self.ax.axis('off')

        # Add text box with instructions
        textstr = '\n'.join([
            'Instructions:',
            '1. Note edge numbers and directions',
            '2. Measure each edge in FEET',
            '3. Enter measurements CLOCKWISE',
            '4. Start from Edge 0',
            '',
            'For perfect closure:',
            '• Sum of East = Sum of West',
            '• Sum of North = Sum of South'
        ])

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        self.ax.text(0.02, 0.98, textstr, transform=self.ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)

        # Show the figure
        plt.tight_layout()
        plt.ion()  # Interactive mode
        plt.show()
        plt.pause(0.1)  # Allow the window to appear

    def collect_measurements_interactive(self) -> Dict[int, float]:
        """
        Collect measurements while showing the visualization.

        Returns:
            Dictionary mapping edge index to measurement in feet
        """
        # First show the visualization
        print("\n" + "="*70)
        print("OPENING VISUALIZATION WINDOW")
        print("="*70)
        print("\nA window should open showing numbered edges with cardinal directions.")
        print("Keep this window visible while entering measurements.\n")

        self.show_edges_with_matplotlib(save_path="numbered_cardinal_edges.png")

        input("Press Enter when you can see the visualization window...")

        print("\n" + "="*70)
        print("CARDINAL EDGE MEASUREMENT COLLECTION")
        print("="*70)
        print("\nRefer to the visualization window and enter measurements in FEET.")
        print("Go CLOCKWISE starting from Edge 0.\n")

        # Collect measurements for each edge
        for i, edge in enumerate(self.edges):
            # Highlight current edge in the visualization
            self.highlight_current_edge(i)

            while True:
                try:
                    # Show edge info
                    direction_names = {
                        'E': "East (→)",
                        'W': "West (←)",
                        'N': "North (↑)",
                        'S': "South (↓)"
                    }

                    print(f"\n{'='*40}")
                    print(f"EDGE {i}: {direction_names[edge.cardinal_direction]}")
                    print(f"{'='*40}")
                    print(f"  Direction: {edge.cardinal_direction}")
                    print(f"  Pixel length: {edge.pixel_length:.1f}")
                    print(f"  Start point: {edge.start}")
                    print(f"  End point: {edge.end}")

                    # Get measurement
                    measurement = input(f"\n  Enter measurement for Edge {i} (feet): ").strip()

                    # Parse measurement
                    value = float(measurement)

                    if value < 0:
                        print("  ⚠️  Measurement must be positive.")
                        continue

                    self.measurements[i] = value
                    edge.measurement = value
                    print(f"  ✓ Recorded: {value} feet {edge.cardinal_direction}")
                    break

                except ValueError:
                    print("  ⚠️  Invalid input. Please enter a number.")

        # Close the matplotlib window
        plt.close(self.fig)

        return self.measurements

    def highlight_current_edge(self, edge_index: int):
        """
        Highlight the current edge being measured in the visualization.

        Args:
            edge_index: Index of edge to highlight
        """
        if self.fig is None or self.ax is None:
            return

        # Clear previous highlights
        self.ax.clear()
        self.ax.imshow(self.display_image)
        self.ax.set_title(f"Measuring Edge {edge_index} - {self.edges[edge_index].cardinal_direction}")
        self.ax.axis('off')

        # Draw a highlight box around the current edge
        edge = self.edges[edge_index]
        x_coords = [edge.start[0], edge.end[0]]
        y_coords = [edge.start[1], edge.end[1]]

        # Create a bounding box around the edge
        margin = 20
        x_min = min(x_coords) - margin
        x_max = max(x_coords) + margin
        y_min = min(y_coords) - margin
        y_max = max(y_coords) + margin

        # Draw highlight rectangle
        rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                        linewidth=3, edgecolor='yellow', facecolor='none')
        self.ax.add_patch(rect)

        # Add arrow pointing to the edge
        mid_x = (edge.start[0] + edge.end[0]) / 2
        mid_y = (edge.start[1] + edge.end[1]) / 2

        self.ax.annotate(f'MEASURING THIS EDGE #{edge_index}',
                        xy=(mid_x, mid_y),
                        xytext=(mid_x, mid_y - 100),
                        arrowprops=dict(arrowstyle='->', color='yellow', lw=2),
                        fontsize=12, color='yellow', weight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7))

        # Update the display
        plt.draw()
        plt.pause(0.1)

    def calculate_scale_factor(self) -> float:
        """
        Calculate average scale factor from measurements.

        Returns:
            Scale factor in feet per pixel
        """
        if not self.measurements:
            logger.error("No measurements to calculate scale factor")
            return None

        scale_factors = []

        for i, edge in enumerate(self.edges):
            if i in self.measurements and edge.pixel_length > 0:
                scale = self.measurements[i] / edge.pixel_length
                scale_factors.append(scale)
                logger.debug(f"Edge {i}: {self.measurements[i]}ft / {edge.pixel_length:.1f}px = {scale:.4f} ft/px")

        if scale_factors:
            self.scale_factor = np.median(scale_factors)
            logger.info(f"Scale factor: {self.scale_factor:.4f} feet/pixel")

        return self.scale_factor

    def verify_closure(self) -> Tuple[bool, float, str]:
        """
        Verify polygon closure with cardinal directions.

        Returns:
            Tuple of (is_closed, error_feet, debug_message)
        """
        # Calculate sums for each direction
        sums = {'E': 0, 'W': 0, 'N': 0, 'S': 0}

        for i, edge in enumerate(self.edges):
            if i in self.measurements:
                sums[edge.cardinal_direction] += self.measurements[i]

        # Check balance
        e_w_diff = abs(sums['E'] - sums['W'])
        n_s_diff = abs(sums['N'] - sums['S'])

        # Build debug message
        msg = "Cardinal Direction Balance:\n"
        msg += f"  East total:  {sums['E']:.1f} ft\n"
        msg += f"  West total:  {sums['W']:.1f} ft\n"
        msg += f"  Difference:  {e_w_diff:.1f} ft {'✓' if e_w_diff < 0.1 else '✗'}\n\n"
        msg += f"  North total: {sums['N']:.1f} ft\n"
        msg += f"  South total: {sums['S']:.1f} ft\n"
        msg += f"  Difference:  {n_s_diff:.1f} ft {'✓' if n_s_diff < 0.1 else '✗'}\n"

        # Calculate total error
        error = np.sqrt(e_w_diff**2 + n_s_diff**2)
        is_closed = error < 0.1

        if is_closed:
            msg += f"\n✓ PERFECT CLOSURE with cardinal directions!"
        else:
            msg += f"\n✗ Closure error: {error:.2f} feet"
            msg += f"\n   Adjust measurements so E=W and N=S"

        return is_closed, error, msg

    def display_summary(self):
        """Display a summary of all measurements."""
        print("\n" + "="*70)
        print("MEASUREMENT SUMMARY")
        print("="*70)

        # Calculate direction sums
        sums = {'E': 0, 'W': 0, 'N': 0, 'S': 0}

        print("\nEdge Measurements:")
        for i, edge in enumerate(self.edges):
            if i in self.measurements:
                print(f"  Edge {i:2d}: {edge.cardinal_direction} = {self.measurements[i]:6.1f} ft")
                sums[edge.cardinal_direction] += self.measurements[i]

        print("\nDirection Totals:")
        print(f"  East (E):  {sums['E']:6.1f} ft")
        print(f"  West (W):  {sums['W']:6.1f} ft")
        print(f"  North (N): {sums['N']:6.1f} ft")
        print(f"  South (S): {sums['S']:6.1f} ft")

        # Verify closure
        is_closed, error, msg = self.verify_closure()
        print("\n" + msg)
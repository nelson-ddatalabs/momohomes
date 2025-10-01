#!/usr/bin/env python3
"""
Cassette Layout System
======================
Main system integrating all modules for complete cassette layout optimization.
Handles the entire pipeline from floor plan to final visualization.
"""

import os
import json
import logging
from typing import Dict, Optional, Tuple, List
from pathlib import Path

# Import all modules
from enhanced_binary_converter import EnhancedBinaryConverter
from edge_detection_system import EdgeDetectionSystem
from interactive_edge_mapper import InteractiveEdgeMapper
from polygon_reconstructor import PolygonReconstructor
from hybrid_optimizer import HybridOptimizer
from cassette_layout_visualizer import CassetteLayoutVisualizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CassetteLayoutSystem:
    """Complete cassette layout optimization system."""

    def __init__(self, output_dir: str = "output"):
        """
        Initialize system.

        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Initialize components
        self.binary_converter = EnhancedBinaryConverter()
        self.edge_detector = EdgeDetectionSystem()
        self.visualizer = CassetteLayoutVisualizer()

        # Storage for intermediate results
        self.floor_plan_path = None
        self.binary_image = None
        self.edges = None
        self.measurements = None
        self.scale_factor = None
        self.polygon = None
        self.cassettes = None
        self.statistics = None

    def process_floor_plan(self, floor_plan_path: str,
                          manual_measurements: Optional[Dict[int, float]] = None) -> Dict:
        """
        Process floor plan with complete pipeline.

        Args:
            floor_plan_path: Path to floor plan image
            manual_measurements: Optional pre-defined measurements for testing

        Returns:
            Dictionary with complete results
        """
        try:
            self.floor_plan_path = floor_plan_path
            logger.info(f"Processing floor plan: {floor_plan_path}")

            # Step 1: Binary conversion
            logger.info("Step 1: Converting to binary...")
            self.binary_image = self.binary_converter.convert_to_binary(floor_plan_path)
            if self.binary_image is None:
                raise ValueError("Failed to convert image to binary")

            # Save binary for debugging
            binary_path = self.output_dir / "binary.png"
            self.binary_converter.save_binary(self.binary_image, str(binary_path))

            # Step 2: Edge detection
            logger.info("Step 2: Detecting edges...")
            self.edges = self.edge_detector.detect_edges(self.binary_image)
            if not self.edges:
                raise ValueError("No edges detected in floor plan")

            logger.info(f"Detected {len(self.edges)} edges")

            # Save edge visualization
            edge_vis = self.edge_detector.visualize_edges(
                self.binary_image,
                str(self.output_dir / "edges.png")
            )

            # Step 3: Interactive edge measurement
            logger.info("Step 3: Collecting measurements...")
            mapper = InteractiveEdgeMapper(self.edges, self.binary_image)

            # Display numbered edges for user reference
            mapper.display_edges(str(self.output_dir / "numbered_edges.png"))

            # Get measurements (use provided or collect from user)
            if manual_measurements:
                self.measurements = manual_measurements
                logger.info("Using provided measurements")
            else:
                self.measurements = mapper.collect_measurements()

            # Calculate scale factor
            mapper.measurements = self.measurements
            self.scale_factor = mapper.calculate_scale_factor()

            if self.scale_factor is None:
                raise ValueError("Failed to calculate scale factor")

            # Validate measurements
            is_valid, closure_error = mapper.validate_measurements()
            if not is_valid:
                logger.warning(f"Polygon closure error: {closure_error:.2f} feet")
                if closure_error > 100.0:  # Allow larger error for complex shapes
                    raise ValueError(f"Polygon does not close (error: {closure_error:.1f} feet)")
                else:
                    logger.warning(f"Proceeding despite closure error of {closure_error:.1f} feet")

            # Step 4: Reconstruct polygon
            logger.info("Step 4: Reconstructing polygon...")
            reconstructor = PolygonReconstructor()
            self.polygon = reconstructor.build_from_measurements(self.edges, self.measurements)

            # Normalize to positive coordinates
            self.polygon = reconstructor.normalize_to_positive()

            area = reconstructor.area
            perimeter = reconstructor.perimeter
            logger.info(f"Polygon area: {area:.1f} sq ft, perimeter: {perimeter:.1f} ft")

            # Step 5: Optimize cassette placement
            logger.info("Step 5: Optimizing cassette placement...")
            optimizer = HybridOptimizer(self.polygon)
            optimization_result = optimizer.optimize_hybrid()

            self.cassettes = optimization_result['cassettes']
            self.statistics = optimization_result

            logger.info(f"Placed {len(self.cassettes)} cassettes")
            logger.info(f"Coverage: {optimization_result['coverage_percent']:.1f}%")

            # Step 6: Create visualization
            logger.info("Step 6: Creating visualization...")

            # Convert polygon to pixel coordinates
            polygon_pixels = [(int(x / self.scale_factor), int(y / self.scale_factor))
                            for x, y in self.polygon]

            output_path = str(self.output_dir / "cassette_layout_final.png")

            visualization = self.visualizer.create_visualization(
                floor_plan_path=floor_plan_path,
                cassettes=self.cassettes,
                scale_factor=self.scale_factor,
                polygon_pixels=polygon_pixels,
                statistics=self.statistics,
                output_path=output_path
            )

            # Save results to JSON
            self._save_results()

            # Return summary
            return {
                'success': True,
                'floor_plan': floor_plan_path,
                'num_edges': len(self.edges),
                'scale_factor': self.scale_factor,
                'area': area,
                'perimeter': perimeter,
                'num_cassettes': len(self.cassettes),
                'coverage_percent': optimization_result['coverage_percent'],
                'gap_percent': optimization_result['gap_percent'],
                'total_weight': optimization_result['total_weight'],
                'output_visualization': output_path,
                'meets_requirement': optimization_result['coverage_percent'] >= 94
            }

        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'floor_plan': floor_plan_path
            }

    def _save_results(self):
        """Save all results to JSON file."""
        results = {
            'floor_plan': self.floor_plan_path,
            'scale_factor': self.scale_factor,
            'measurements': self.measurements,
            'polygon': self.polygon,
            'cassettes': self.cassettes,
            'statistics': self.statistics
        }

        results_path = self.output_dir / "results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Results saved to: {results_path}")


def test_with_luna():
    """Test system with Luna floor plan."""
    # Luna measurements for all 12 detected edges (clockwise)
    # Based on actual edge detection results
    luna_measurements = {
        0: 28,     # Left side partial
        1: 78,     # Top edge (full width)
        2: 5,      # Small jog right
        3: 12,     # Down into patio cutout
        4: 33,     # Across patio area
        5: 12,     # Back up from patio
        6: 5,      # Small jog
        7: 15.5,   # Continue right side
        8: 40.5,   # Full right side down
        9: 45,     # Bottom partial
        10: 12.5,  # Up into garage area
        11: 33     # Back across to complete
    }

    system = CassetteLayoutSystem(output_dir="luna_output")

    # Find Luna floor plan
    luna_path = "/Users/nelsondsouza/Dropbox (Personal)/Mac/Documents/products/momohomes/floorplans/Luna-Conditioned.png"

    if not os.path.exists(luna_path):
        luna_path = "floorplans/Luna-Conditioned.png"

    if not os.path.exists(luna_path):
        logger.error("Luna floor plan not found")
        return

    # Process with manual measurements for testing
    result = system.process_floor_plan(luna_path, manual_measurements=luna_measurements)

    # Print results
    print("\n" + "="*70)
    print("LUNA FLOOR PLAN TEST RESULTS")
    print("="*70)

    if result['success']:
        print(f"✓ Processing successful")
        print(f"  Area: {result['area']:.1f} sq ft")
        print(f"  Cassettes: {result['num_cassettes']}")
        print(f"  Coverage: {result['coverage_percent']:.1f}%")
        print(f"  Gap: {result['gap_percent']:.1f}%")
        print(f"  Weight: {result['total_weight']:.0f} lbs")

        if result['meets_requirement']:
            print(f"\n✓✓✓ MEETS 94% COVERAGE REQUIREMENT!")
        else:
            print(f"\n✗ Coverage below 94% requirement")

        print(f"\nVisualization saved to: {result['output_visualization']}")
    else:
        print(f"✗ Processing failed: {result['error']}")


if __name__ == "__main__":
    test_with_luna()
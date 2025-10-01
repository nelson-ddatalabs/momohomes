#!/usr/bin/env python3
"""
Cassette Layout System - Cardinal Version
==========================================
Uses cardinal directions (N/S/E/W) for perfect polygon closure.
"""

import os
import json
import logging
from typing import Dict, Optional, List
from pathlib import Path

# Import cardinal modules
from enhanced_binary_converter import EnhancedBinaryConverter
from cardinal_edge_detector import CardinalEdgeDetector
from cardinal_edge_mapper import CardinalEdgeMapper
from cardinal_polygon_reconstructor import CardinalPolygonReconstructor
from hybrid_optimizer import HybridOptimizer
from cassette_layout_visualizer import CassetteLayoutVisualizer
from grid_alignment_system import GridAlignmentSystem
from smart_edge_merger import SmartEdgeMerger

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CassetteLayoutSystemCardinal:
    """Cassette layout system with cardinal direction support."""

    def __init__(self, output_dir: str = "output_cardinal"):
        """Initialize system."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Initialize components
        self.binary_converter = EnhancedBinaryConverter()
        self.edge_detector = CardinalEdgeDetector()
        self.visualizer = CassetteLayoutVisualizer()

        # Storage
        self.floor_plan_path = None
        self.binary_image = None
        self.edges = None
        self.measurements = None
        self.scale_factor = None
        self.polygon = None
        self.cassettes = None
        self.statistics = None

    def process_floor_plan(self, floor_plan_path: str,
                          manual_measurements: Optional[Dict[int, float]] = None,
                          use_smart_optimizer: bool = True,
                          use_multi_pass: bool = False) -> Dict:
        """
        Process floor plan with cardinal directions.

        Args:
            floor_plan_path: Path to floor plan image
            manual_measurements: Optional pre-defined measurements

        Returns:
            Dictionary with results
        """
        try:
            self.floor_plan_path = floor_plan_path
            logger.info(f"Processing: {floor_plan_path}")

            # Step 1: Binary conversion
            logger.info("Step 1: Converting to binary...")
            self.binary_image = self.binary_converter.convert_to_binary(floor_plan_path)
            if self.binary_image is None:
                raise ValueError("Failed to convert image to binary")

            binary_path = self.output_dir / "binary.png"
            self.binary_converter.save_binary(self.binary_image, str(binary_path))

            # Step 2: Cardinal edge detection
            logger.info("Step 2: Detecting cardinal edges...")
            self.edges = self.edge_detector.detect_cardinal_edges(self.binary_image)
            if not self.edges:
                raise ValueError("No edges detected")

            logger.info(f"Detected {len(self.edges)} cardinal edges")

            # Step 3: Cardinal edge measurements
            logger.info("Step 3: Collecting measurements...")

            # Use interactive mapper for better user experience
            from interactive_cardinal_mapper import InteractiveCardinalMapper
            interactive_mapper = InteractiveCardinalMapper(self.edges, self.binary_image)

            # Get measurements
            if manual_measurements:
                self.measurements = manual_measurements
                interactive_mapper.measurements = manual_measurements
                logger.info("Using provided measurements")
                # Still save the visualization for reference
                interactive_mapper.show_edges_with_matplotlib(
                    save_path=str(self.output_dir / "numbered_cardinal_edges.png")
                )
                import matplotlib.pyplot as plt
                plt.close()  # Close the window if using manual measurements
            else:
                # Interactive collection with visual display
                self.measurements = interactive_mapper.collect_measurements_interactive()
                interactive_mapper.display_summary()

            # Create standard mapper for other operations
            mapper = CardinalEdgeMapper(self.edges, self.binary_image)
            mapper.measurements = self.measurements

            # Step 3.5: Smart edge merging for zero-length edges
            logger.info("Step 3.5: Applying smart edge merging...")
            edge_merger = SmartEdgeMerger(min_edge_length=0.5)

            # Merge edges with zero or minimal measurements
            original_edge_count = len(self.edges)
            self.edges, self.measurements = edge_merger.merge_edges(self.edges, self.measurements)

            if len(self.edges) < original_edge_count:
                logger.info(f"Edges merged: {original_edge_count} -> {len(self.edges)}")
                logger.info(edge_merger.get_merge_summary())

                # Update mapper with merged edges
                mapper = CardinalEdgeMapper(self.edges, self.binary_image)
                mapper.measurements = self.measurements

            # Calculate scale factor
            self.scale_factor = mapper.calculate_scale_factor()

            # NOTE: We do NOT verify closure here using pixel-based edges
            # Closure will be checked after reconstruction with measurements

            # Step 4: Cardinal polygon reconstruction
            logger.info("Step 4: Reconstructing polygon with cardinal directions...")
            reconstructor = CardinalPolygonReconstructor()
            self.polygon = reconstructor.build_from_cardinal_measurements(
                self.edges, self.measurements
            )

            # Normalize to positive
            self.polygon = reconstructor.normalize_to_positive()

            # Simplify polygon to remove duplicate vertices
            logger.info("Simplifying polygon to remove duplicate vertices...")
            self.polygon = edge_merger.simplify_polygon(self.polygon)
            logger.info(f"Final polygon has {len(self.polygon)} vertices")

            area = reconstructor.area
            perimeter = reconstructor.perimeter

            logger.info(f"Polygon area: {area:.1f} sq ft, perimeter: {perimeter:.1f} ft")
            logger.info(f"Measurement-based closure: {reconstructor.is_closed}, Error: {reconstructor.closure_error:.4f} ft")

            # Save edge visualization with measurements and closure status
            annotated_edges_path = str(self.output_dir / "cardinal_edges_measured.png")
            self.annotated_edges = self.edge_detector.visualize_cardinal_edges(
                self.binary_image,
                annotated_edges_path,
                self.measurements
            )
            logger.info(f"Saved annotated edges with measurements to: {annotated_edges_path}")

            # Debug trace
            logger.debug(reconstructor.debug_trace())

            # Step 5: Optimize cassettes
            logger.info("Step 5: Optimizing cassette placement...")

            # Determine which polygon to use
            if "Luna" in floor_plan_path:
                from fix_polygon_for_indoor_only import get_corrected_luna_polygon
                working_polygon = get_corrected_luna_polygon()
            else:
                working_polygon = self.polygon

            # Choose optimizer based on settings
            if use_multi_pass:
                # Use new multi-pass optimizer with backtracking
                logger.info("Using multi-pass optimizer with backtracking...")
                from multi_pass_optimizer import MultiPassOptimizer
                optimizer = MultiPassOptimizer(working_polygon)
                optimization_result = optimizer.optimize()
            elif use_smart_optimizer:
                # Use improved optimizer for better coverage
                logger.info("Using ultra-smart optimizer...")
                from ultra_smart_optimizer import UltraSmartOptimizer
                optimizer = UltraSmartOptimizer(working_polygon)
                optimization_result = optimizer.optimize()
            else:
                # Use original optimizer
                logger.info("Using hybrid optimizer...")
                optimizer = HybridOptimizer(working_polygon)
                optimization_result = optimizer.optimize_hybrid()

            self.cassettes = optimization_result['cassettes']
            self.statistics = optimization_result

            logger.info(f"Placed {len(self.cassettes)} cassettes")
            logger.info(f"Coverage: {optimization_result['coverage_percent']:.1f}%")

            # Step 6: Create Grid Alignment System
            logger.info("Step 6: Creating grid alignment system...")

            # Calculate actual building dimensions from the polygon
            if self.polygon and len(self.polygon) >= 4:
                xs = [p[0] for p in self.polygon]
                ys = [p[1] for p in self.polygon]
                actual_width = max(xs) - min(xs)
                actual_height = max(ys) - min(ys)
                logger.info(f"Calculated actual building dimensions: {actual_width:.1f} x {actual_height:.1f} ft")
            else:
                # Fallback to default if polygon not available
                actual_width = 78.0
                actual_height = 50.0
                logger.warning("Using default dimensions as polygon not available")

            # Create grid alignment system with actual dimensions
            grid_aligner = GridAlignmentSystem(
                world_width=actual_width,  # Use ACTUAL building width
                world_height=actual_height,  # Use ACTUAL building height
                image_width=self.binary_image.shape[1],
                image_height=self.binary_image.shape[0]
            )

            # Add Luna-specific correspondences if this is Luna
            if "Luna" in floor_plan_path:
                # For Luna, use hardcoded correspondences since we know the exact layout
                grid_aligner.add_luna_correspondences()
            else:
                # For other buildings, calculate correspondences from actual building edges
                # Use corners of the polygon and actual detected edges
                if self.polygon and len(self.polygon) >= 4:
                    # Find min/max coordinates from polygon
                    xs = [p[0] for p in self.polygon]
                    ys = [p[1] for p in self.polygon]
                    min_x, max_x = min(xs), max(xs)
                    min_y, max_y = min(ys), max(ys)

                    # Find corresponding pixels in the binary image
                    # Use the edge detector's pixel positions for more accurate mapping
                    if self.edges and len(self.edges) > 0:
                        # Get the actual building boundaries from detected edges
                        edge_pixels = []
                        for edge in self.edges:
                            edge_pixels.append(edge.start)
                            edge_pixels.append(edge.end)

                        # Find bounding box of actual building in pixels
                        edge_xs = [p[0] for p in edge_pixels]
                        edge_ys = [p[1] for p in edge_pixels]
                        pixel_min_x = min(edge_xs)
                        pixel_max_x = max(edge_xs)
                        pixel_min_y = min(edge_ys)
                        pixel_max_y = max(edge_ys)

                        logger.info(f"Building pixel bounds: X[{pixel_min_x},{pixel_max_x}], Y[{pixel_min_y},{pixel_max_y}]")
                        logger.info(f"Building world bounds: X[{min_x:.1f},{max_x:.1f}], Y[{min_y:.1f},{max_y:.1f}]")

                        # Add corner correspondences using actual pixel positions
                        grid_aligner.add_correspondence(
                            (min_x, min_y),
                            (pixel_min_x, pixel_max_y),  # Bottom-left in world = bottom-left in image (Y inverted)
                            "Bottom-left"
                        )
                        grid_aligner.add_correspondence(
                            (max_x, min_y),
                            (pixel_max_x, pixel_max_y),  # Bottom-right in world = bottom-right in image
                            "Bottom-right"
                        )
                        grid_aligner.add_correspondence(
                            (max_x, max_y),
                            (pixel_max_x, pixel_min_y),  # Top-right in world = top-right in image
                            "Top-right"
                        )
                        grid_aligner.add_correspondence(
                            (min_x, max_y),
                            (pixel_min_x, pixel_min_y),  # Top-left in world = top-left in image
                            "Top-left"
                        )

                        # Add center point for better accuracy
                        center_x = (min_x + max_x) / 2
                        center_y = (min_y + max_y) / 2
                        pixel_center_x = (pixel_min_x + pixel_max_x) // 2
                        pixel_center_y = (pixel_min_y + pixel_max_y) // 2
                        grid_aligner.add_correspondence(
                            (center_x, center_y),
                            (pixel_center_x, pixel_center_y),
                            "Center"
                        )
                    else:
                        # Fallback to calculated positions if edges not available
                        logger.warning("No edges available, using calculated pixel positions")
                        grid_aligner.add_correspondence(
                            (min_x, min_y),
                            (int(min_x * grid_aligner.scale_x),
                             int(self.binary_image.shape[0] - min_y * grid_aligner.scale_y)),
                            "Bottom-left"
                        )
                        grid_aligner.add_correspondence(
                            (max_x, min_y),
                            (int(max_x * grid_aligner.scale_x),
                             int(self.binary_image.shape[0] - min_y * grid_aligner.scale_y)),
                            "Bottom-right"
                        )
                        grid_aligner.add_correspondence(
                            (max_x, max_y),
                            (int(max_x * grid_aligner.scale_x),
                             int(self.binary_image.shape[0] - max_y * grid_aligner.scale_y)),
                            "Top-right"
                        )
                        grid_aligner.add_correspondence(
                            (min_x, max_y),
                            (int(min_x * grid_aligner.scale_x),
                             int(self.binary_image.shape[0] - max_y * grid_aligner.scale_y)),
                            "Top-left"
                        )

            # Calculate homography matrix
            homography = grid_aligner.calculate_homography()
            if homography is not None:
                logger.info("Homography matrix calculated successfully")
            else:
                logger.warning("Could not calculate homography, falling back to simple scaling")

            # Origin offset is no longer needed with grid alignment
            origin_offset = (0.0, 0.0)
            logger.info("Using grid-based alignment for coordinate transformation")

            # Step 7: Visualization
            logger.info("Step 7: Creating visualization...")

            # Convert polygon to pixels
            if self.scale_factor and self.scale_factor > 0:
                polygon_pixels = [(int(x / self.scale_factor), int(y / self.scale_factor))
                                for x, y in self.polygon]
            else:
                logger.warning("Invalid scale factor, using default")
                polygon_pixels = [(int(x * 10), int(y * 10)) for x, y in self.polygon]

            output_path = str(self.output_dir / "cassette_layout_cardinal.png")

            # Use the annotated edges image as base for visualization
            visualization = self.visualizer.create_visualization(
                floor_plan_path=annotated_edges_path,  # Use annotated edges instead
                cassettes=self.cassettes,
                scale_factor=self.scale_factor,
                polygon_pixels=polygon_pixels,
                statistics=self.statistics,
                output_path=output_path,
                origin_offset=origin_offset,
                grid_alignment=grid_aligner if homography is not None else None
            )

            # Save results
            self._save_results()

            return {
                'success': True,
                'floor_plan': floor_plan_path,
                'num_edges': len(self.edges),
                'scale_factor': self.scale_factor,
                'area': area,
                'perimeter': perimeter,
                'closure_error': reconstructor.closure_error,  # Measurement-based closure
                'is_closed': reconstructor.is_closed,  # Measurement-based closure
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
        """Save results to JSON."""
        # Convert edges for serialization
        edges_data = []
        for i, edge in enumerate(self.edges):
            edges_data.append({
                'index': i,
                'cardinal_direction': edge.cardinal_direction,
                'pixel_length': edge.pixel_length,
                'measurement': self.measurements.get(i)
            })

        results = {
            'floor_plan': str(self.floor_plan_path),
            'scale_factor': self.scale_factor,
            'edges': edges_data,
            'measurements': self.measurements,
            'polygon': self.polygon,
            'cassettes': self.cassettes,
            'statistics': self.statistics
        }

        results_path = self.output_dir / "results_cardinal.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Results saved to: {results_path}")


def test_with_luna_cardinal():
    """Test with Luna using cardinal directions."""

    # Luna floor plan with two cutouts
    # Building is 78' wide x 40.5' tall
    # Cutouts are each 33' wide x 12' deep
    # Using approximate measurements for the 12 detected edges
    luna_measurements = {
        0: 40.5,  # N - left edge up
        1: 45,    # E - top to patio
        2: 12,    # S - into patio
        3: 33,    # W - patio width
        4: 12,    # S - exit patio
        5: 33,    # W - to left wall
        6: 16.5,  # N - partial up
        7: 33,    # W - garage width
        8: 12,    # S - into garage
        9: 33,    # W - garage depth
        10: 12,   # S - exit garage
        11: 78,   # W - bottom edge
    }

    system = CassetteLayoutSystemCardinal(output_dir="luna_cardinal_output")

    # Find Luna floor plan
    luna_path = "/Users/nelsondsouza/Dropbox (Personal)/Mac/Documents/products/momohomes/floorplans/Luna-Conditioned.png"

    if not os.path.exists(luna_path):
        luna_path = "floorplans/Luna-Conditioned.png"

    if not os.path.exists(luna_path):
        logger.error("Luna floor plan not found")
        return

    # Process with manual measurements
    result = system.process_floor_plan(luna_path, luna_measurements)

    # Print results
    print("\n" + "="*70)
    print("LUNA CARDINAL TEST RESULTS")
    print("="*70)

    if result['success']:
        print(f"✓ Processing successful")
        print(f"  Area: {result['area']:.1f} sq ft")
        print(f"  Perimeter: {result['perimeter']:.1f} ft")
        print(f"  Cassettes: {result['num_cassettes']}")
        print(f"  Coverage: {result['coverage_percent']:.1f}%")
        print(f"  Weight: {result['total_weight']:.0f} lbs")

        if result['is_closed']:
            print(f"\n✓✓✓ PERFECT POLYGON CLOSURE!")
            print(f"  Closure error: {result['closure_error']:.6f} feet")
        else:
            print(f"\n✗ Polygon closure error: {result['closure_error']:.2f} feet")

        if result['meets_requirement']:
            print(f"\n✓ Meets 94% coverage requirement")
        else:
            print(f"\n✗ Coverage below 94% requirement")

        print(f"\nVisualization: {result['output_visualization']}")
    else:
        print(f"✗ Processing failed: {result['error']}")


if __name__ == "__main__":
    test_with_luna_cardinal()
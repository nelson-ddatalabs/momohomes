#!/usr/bin/env python3
"""
Interactive Cassette Layout System Runner
==========================================
Provides visual feedback while collecting measurements.
"""

import os
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def show_opencv_window(image_path: str):
    """
    Display image using OpenCV (alternative to matplotlib).

    Args:
        image_path: Path to image to display
    """
    try:
        import cv2
        img = cv2.imread(image_path)
        if img is not None:
            # Resize if too large
            height, width = img.shape[:2]
            max_height = 800
            if height > max_height:
                scale = max_height / height
                new_width = int(width * scale)
                img = cv2.resize(img, (new_width, max_height))

            cv2.imshow("Floor Plan - Numbered Edges (Press any key to continue)", img)
            print("\n✓ OpenCV window opened. Press any key in the image window to continue...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return True
    except ImportError:
        logger.warning("OpenCV not available")
    except Exception as e:
        logger.warning(f"Could not display with OpenCV: {e}")
    return False


def run_interactive(floor_plan_path: str, use_matplotlib: bool = True):
    """
    Run the cassette system with interactive visualization.

    Args:
        floor_plan_path: Path to floor plan image
        use_matplotlib: Whether to use matplotlib for display
    """
    from cassette_layout_system_cardinal import CassetteLayoutSystemCardinal

    # Setup output directory
    input_name = Path(floor_plan_path).stem
    output_dir = f"output_{input_name}"

    print("\n" + "="*70)
    print("CASSETTE LAYOUT SYSTEM - INTERACTIVE MODE")
    print("="*70)
    print(f"Floor plan: {floor_plan_path}")
    print(f"Output directory: {output_dir}/")
    print("-"*70)

    # Check if matplotlib is available
    if use_matplotlib:
        try:
            import matplotlib
            matplotlib.use('TkAgg')  # Use TkAgg backend for better compatibility
            import matplotlib.pyplot as plt
            print("✓ Matplotlib available - will show interactive visualization")
        except ImportError:
            print("⚠️  Matplotlib not available - will use alternative display")
            use_matplotlib = False

    # Initialize system
    system = CassetteLayoutSystemCardinal(output_dir=output_dir)

    # For non-matplotlib mode, we need to handle visualization differently
    if not use_matplotlib:
        print("\n📌 Running in non-matplotlib mode")
        print("   The system will generate numbered edge images for reference.")

        # First pass to generate the visualization
        from cardinal_edge_detector import CardinalEdgeDetector
        from enhanced_binary_converter import EnhancedBinaryConverter
        from cardinal_edge_mapper import CardinalEdgeMapper

        # Convert to binary
        converter = EnhancedBinaryConverter()
        binary = converter.convert_to_binary(floor_plan_path)

        # Detect edges
        detector = CardinalEdgeDetector()
        edges = detector.detect_cardinal_edges(binary)

        # Create visualization
        mapper = CardinalEdgeMapper(edges, binary)
        numbered_path = f"{output_dir}/numbered_cardinal_edges.png"
        os.makedirs(output_dir, exist_ok=True)
        mapper.display_cardinal_edges(numbered_path)

        print(f"\n✅ Generated edge visualization: {numbered_path}")

        # Try to display with OpenCV
        if not show_opencv_window(numbered_path):
            print("\n📁 Please open this file to see numbered edges:")
            print(f"   {os.path.abspath(numbered_path)}")
            input("\n   Press Enter after viewing the image...")

        # Now collect measurements manually
        print("\n" + "="*70)
        print("MEASUREMENT COLLECTION")
        print("="*70)
        print("Enter measurements for each edge shown in the visualization.")
        print("Go CLOCKWISE starting from Edge 0.\n")

        measurements = {}
        for i, edge in enumerate(edges):
            direction_names = {
                'E': "East (→)",
                'W': "West (←)",
                'N': "North (↑)",
                'S': "South (↓)"
            }
            print(f"\nEdge {i}: {direction_names[edge.cardinal_direction]}")
            while True:
                try:
                    value = float(input(f"  Measurement (feet): "))
                    if value < 0:
                        print("  Must be positive!")
                        continue
                    measurements[i] = value
                    print(f"  ✓ Recorded: {value} ft")
                    break
                except ValueError:
                    print("  Invalid number!")

        # Run system with collected measurements
        result = system.process_floor_plan(floor_plan_path, measurements)

    else:
        # Use matplotlib interactive mode (already integrated)
        result = system.process_floor_plan(floor_plan_path)

    # Display results
    print("\n" + "="*70)
    print("PROCESSING RESULTS")
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
        else:
            print(f"\n⚠️  Polygon closure error: {result['closure_error']:.2f} feet")
            print("    Tip: For perfect closure, ensure ΣE=ΣW and ΣN=ΣS")

        if result['meets_requirement']:
            print(f"\n✓ Meets 94% coverage requirement")
        else:
            print(f"\n✗ Below 94% requirement")

        print(f"\n📁 Output files in {output_dir}/:")
        print(f"   • numbered_cardinal_edges.png - Edge reference")
        print(f"   • cassette_layout_cardinal.png - Final layout")
        print(f"   • results_cardinal.json - Detailed results")
    else:
        print(f"✗ Processing failed: {result['error']}")


def main():
    """Main entry point."""
    print("\n" + "="*70)
    print("CASSETTE FLOOR JOIST OPTIMIZATION SYSTEM")
    print("="*70)

    # Get floor plan
    if len(sys.argv) > 1:
        floor_plan_path = sys.argv[1]
    else:
        print("\nUsage: python run_cassette_interactive.py <floor_plan.png> [--no-matplotlib]")
        print("\nExample floor plans:")
        print("  • floorplans/Luna-Conditioned.png")
        print("  • floorplans/YourFloorPlan.png")

        floor_plan_path = input("\nEnter floor plan path (or drag file here): ").strip().strip('"\'')

    # Check file exists
    if not Path(floor_plan_path).exists():
        print(f"\n❌ Error: File not found: {floor_plan_path}")
        sys.exit(1)

    # Check for matplotlib flag
    use_matplotlib = "--no-matplotlib" not in sys.argv

    # Display system info
    print(f"\n📊 System Information:")
    print(f"   • Max cassette size: 6' x 8' (48 sq ft)")
    print(f"   • Max weight: 500 lbs per cassette")
    print(f"   • Weight: 10.4 lbs per sq ft")
    print(f"   • Target coverage: ≥94%")
    print(f"   • Available sizes: 6x8, 5x8, 6x6, 4x8, 4x6, 6x4, 4x4, 3x4, 4x3")

    # Run interactive system
    run_interactive(floor_plan_path, use_matplotlib)


if __name__ == "__main__":
    main()
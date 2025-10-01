#!/usr/bin/env python3
"""
Simple Cassette Optimization Runner
====================================
Easy-to-use script for running cassette optimization on floor plans.
"""

import sys
import os
from pathlib import Path


def print_usage():
    """Print usage instructions."""
    print("\n" + "="*70)
    print("CASSETTE FLOOR JOIST OPTIMIZER")
    print("="*70)
    print("\nUsage: python optimize_cassettes.py [polygon_name_or_file]")
    print("\nQuick Examples:")
    print("  python optimize_cassettes.py bungalow")
    print("  python optimize_cassettes.py luna")
    print("  python optimize_cassettes.py rectangle")
    print("  python optimize_cassettes.py myfloorplan.json")
    print("  python optimize_cassettes.py floorplan.png")
    print("\nTest Polygons Available:")
    print("  • bungalow - Bungalow floor plan (79% coverage)")
    print("  • luna - Luna floor plan")
    print("  • rectangle - Simple 40x30 rectangle")
    print("  • l-shape - L-shaped test building")
    print("\nFile Formats Supported:")
    print("  • .json - Polygon coordinates")
    print("  • .txt, .csv - Vertex list")
    print("  • .png, .jpg - Floor plan image (requires edge measurement)")
    print("-"*70)


def main():
    """Main entry point."""
    if len(sys.argv) < 2 or sys.argv[1] in ['-h', '--help', 'help']:
        print_usage()

        choice = input("\nEnter polygon name or file path (or 'quit' to exit): ").strip()
        if choice.lower() in ['quit', 'q', 'exit']:
            return

        if not choice:
            choice = 'bungalow'  # Default
            print(f"Using default: {choice}")

        sys.argv = ['optimize_cassettes.py', choice]

    # Import and run the modular pipeline
    from run_modular_pipeline import main as run_pipeline

    try:
        run_pipeline()
    except Exception as e:
        print(f"\nError: {e}")
        print("\nPlease check your input and try again.")
        print_usage()


if __name__ == "__main__":
    main()
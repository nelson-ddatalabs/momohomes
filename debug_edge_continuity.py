#!/usr/bin/env python3
"""
Debug Edge Continuity
=====================
Check if edges are properly connected in sequence.
"""

from cassette_layout_system_cardinal import CassetteLayoutSystemCardinal
from cardinal_edge_detector import CardinalEdgeDetector
from enhanced_binary_converter import EnhancedBinaryConverter
import os

def check_edge_continuity():
    """Check if edges form a continuous chain."""

    # Process Luna floor plan
    luna_path = "/Users/nelsondsouza/Dropbox (Personal)/Mac/Documents/products/momohomes/floorplans/Luna-Conditioned.png"

    if not os.path.exists(luna_path):
        print("Luna floor plan not found")
        return

    # Convert to binary
    converter = EnhancedBinaryConverter()
    binary = converter.convert_to_binary(luna_path)

    # Detect edges
    detector = CardinalEdgeDetector()
    edges = detector.detect_cardinal_edges(binary)

    print("="*70)
    print("EDGE CONTINUITY CHECK")
    print("="*70)
    print(f"Total edges detected: {len(edges)}")
    print()

    # Check continuity
    discontinuities = []

    for i in range(len(edges)):
        current_edge = edges[i]
        next_edge = edges[(i + 1) % len(edges)]

        # Check if current edge's end connects to next edge's start
        end_x, end_y = current_edge.end
        start_x, start_y = next_edge.start

        gap = ((end_x - start_x)**2 + (end_y - start_y)**2)**0.5

        print(f"Edge {i:2d} ({current_edge.cardinal_direction}):")
        print(f"  Start: {current_edge.start}")
        print(f"  End:   {current_edge.end}")
        print(f"  Next edge start: {next_edge.start}")
        print(f"  Gap to next: {gap:.1f} pixels")

        if gap > 1.0:  # Allow 1 pixel tolerance
            discontinuities.append((i, gap))
            print(f"  ⚠️ DISCONTINUITY! Gap of {gap:.1f} pixels")
        else:
            print(f"  ✓ Connected")
        print()

    # Summary
    print("="*70)
    print("SUMMARY")
    print("="*70)

    if discontinuities:
        print(f"❌ Found {len(discontinuities)} discontinuities:")
        for edge_idx, gap in discontinuities:
            print(f"   After edge {edge_idx}: {gap:.1f} pixel gap")
        print("\nThis explains the closure error!")
        print("The edges aren't forming a continuous chain.")
    else:
        print("✓ All edges are properly connected")
        print("The issue must be in direction assignment or coordinate system")

    # Check if it's truly clockwise
    print("\n" + "="*70)
    print("CLOCKWISE CHECK")
    print("="*70)

    # Calculate signed area
    signed_area = 0
    for edge in edges:
        signed_area += (edge.end[0] - edge.start[0]) * (edge.end[1] + edge.start[1])

    print(f"Signed area: {signed_area}")
    if signed_area < 0:
        print("✓ Edges are clockwise")
    else:
        print("❌ Edges are counter-clockwise (should have been reversed)")

    # Check direction assignments
    print("\n" + "="*70)
    print("DIRECTION ASSIGNMENT CHECK")
    print("="*70)

    for i, edge in enumerate(edges):
        dx = edge.end[0] - edge.start[0]
        dy = edge.end[1] - edge.start[1]

        # What direction should this be?
        if abs(dx) > abs(dy):
            # Primarily horizontal
            expected = 'E' if dx > 0 else 'W'
        else:
            # Primarily vertical
            expected = 'S' if dy > 0 else 'N'  # Note: Y increases downward in images

        actual = edge.cardinal_direction

        print(f"Edge {i:2d}: dx={dx:6.1f}, dy={dy:6.1f}")
        print(f"  Expected: {expected}, Actual: {actual}", end="")
        if expected != actual:
            print(" ❌ MISMATCH!")
        else:
            print(" ✓")

if __name__ == "__main__":
    check_edge_continuity()
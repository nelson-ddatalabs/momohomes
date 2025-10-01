#!/usr/bin/env python3
"""
Fix Polygon for Indoor Only
============================
Ensures polygon represents only indoor living space, excluding garage and patio.
"""

def get_corrected_luna_polygon():
    """
    Get the correct polygon for Luna that excludes cutouts.

    The Luna floor plan has:
    - Main living area: 78' wide total
    - Bottom-left garage cutout: 15' x 14' (excluded)
    - Top-right patio cutout: 7.5' x 9' (excluded)

    Returns polygon vertices that trace ONLY the indoor space.
    """
    # Start from bottom-right corner of living space (not garage)
    # Trace clockwise around INDOOR areas only

    # Method: Create polygon that explicitly excludes cutouts
    # This creates an L-shaped polygon

    vertices = [
        # Start at bottom-right of living area
        (15.0, 0.0),   # Bottom-right corner of house (not garage)
        (15.0, 14.0),  # Up to where garage meets house
        (0.0, 14.0),   # Left to actual left wall of living space
        (0.0, 50.0),   # Up to top-left corner
        (78.0, 50.0),  # Right to top-right corner
        (78.0, 34.0),  # Down to patio
        (70.5, 34.0),  # Left into patio cutout
        (70.5, 25.0),  # Down through patio
        (55.5, 25.0),  # Left past patio
        (55.5, 29.0),  # Up a bit
        (47.5, 29.0),  # Left more
        (47.5, 14.0),  # Down to bottom level
        (15.0, 14.0),  # Left to starting area
        (15.0, 0.0),   # Down to close polygon
    ]

    # Remove duplicate closing vertex if present
    if vertices[-1] == vertices[0]:
        vertices = vertices[:-1]

    return vertices


def validate_indoor_point(x, y):
    """
    Check if a point is in the indoor living space.

    Excludes:
    - Bottom-left garage: x < 15 and y < 14
    - Top-right patio: x > 70.5 and 25 < y < 34
    """
    # Garage cutout
    if x < 15 and y < 14:
        return False, "In garage cutout"

    # Patio cutout
    if x > 70.5 and 25 < y < 34:
        return False, "In patio cutout"

    # Check general bounds
    if x < 0 or x > 78 or y < 0 or y > 50:
        return False, "Outside building bounds"

    # Additional checks for middle cutouts
    if 47.5 < x < 55.5 and 14 < y < 25:
        return False, "In middle-right cutout"

    if 47.5 < x < 55.5 and 29 < y < 50:
        return False, "In upper-middle cutout"

    return True, "Valid indoor location"


def get_indoor_only_polygon_from_measurements():
    """
    Build polygon using measurements but ensuring indoor-only representation.
    """
    # Use the balanced measurements
    measurements = {
        0: 50.0,   # N - left wall up (but start at y=14, not y=0)
        1: 78.0,   # E - top
        2: 16.0,   # S - into patio area
        3: 7.5,    # W - patio indent
        4: 9.0,    # S - through patio
        5: 15.0,   # W - past patio
        6: 4.0,    # N - small up
        7: 8.0,    # W - more left
        8: 15.0,   # S - down
        9: 32.5,   # W - across to left side
        10: 14.0,  # S - down to garage level
        11: 15.0   # W - to garage edge (not into garage)
    }

    # Start at the corner where living space meets garage
    # This is at (15, 14) not (0, 0)
    x, y = 15.0, 14.0
    vertices = [(x, y)]

    # Apply movements but adjust for indoor-only
    directions = ['N', 'E', 'S', 'W', 'S', 'W', 'N', 'W', 'S', 'W', 'S', 'W']

    # First we go left to x=0 (edge 11 reversed)
    x = 0.0
    vertices.append((x, y))

    # Then up the left wall (edge 0)
    y = 50.0
    vertices.append((x, y))

    # Continue with remaining edges...
    # East across top
    x = 78.0
    vertices.append((x, y))

    # Down into patio cutout
    y = 34.0
    vertices.append((x, y))

    # Left
    x = 70.5
    vertices.append((x, y))

    # Down through patio
    y = 25.0
    vertices.append((x, y))

    # Left past patio
    x = 55.5
    vertices.append((x, y))

    # Up slightly
    y = 29.0
    vertices.append((x, y))

    # Left more
    x = 47.5
    vertices.append((x, y))

    # Down to bottom
    y = 14.0
    vertices.append((x, y))

    # Back to start
    x = 15.0
    vertices.append((x, y))

    return vertices


if __name__ == "__main__":
    print("CORRECTED POLYGON FOR INDOOR SPACE ONLY")
    print("="*70)

    vertices = get_corrected_luna_polygon()

    print("\nVertices (indoor space only):")
    for i, (x, y) in enumerate(vertices):
        print(f"  V{i:2d}: ({x:5.1f}, {y:5.1f})")

    print("\nValidation Tests:")
    test_points = [
        (7, 7),      # In garage - should be INVALID
        (20, 20),    # In main living - should be VALID
        (75, 30),    # In patio - should be INVALID
        (50, 20),    # In middle cutout - should be INVALID
        (30, 30),    # In living area - should be VALID
    ]

    for x, y in test_points:
        valid, reason = validate_indoor_point(x, y)
        status = "✓ VALID" if valid else "✗ INVALID"
        print(f"  Point ({x:2d},{y:2d}): {status} - {reason}")
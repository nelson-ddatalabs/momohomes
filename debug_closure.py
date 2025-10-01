#!/usr/bin/env python3
"""
Debug Polygon Closure Issue
===========================
Trace through the exact polygon construction with balanced measurements.
"""

def trace_polygon_with_measurements():
    """Trace polygon construction step by step."""

    # Your provided measurements that balance perfectly
    measurements = {
        0: 50.0,   # N
        1: 78.0,   # E
        2: 16.0,   # S
        3: 7.5,    # W
        4: 9.0,    # S
        5: 15.0,   # W
        6: 4.0,    # N
        7: 8.0,    # W
        8: 15.0,   # S
        9: 32.5,   # W
        10: 14.0,  # S
        11: 15.0   # W
    }

    # Edge directions as shown in the image
    directions = ['N', 'E', 'S', 'W', 'S', 'W', 'N', 'W', 'S', 'W', 'S', 'W']

    # Verify balance
    north_sum = measurements[0] + measurements[6]  # 50 + 4 = 54
    south_sum = measurements[2] + measurements[4] + measurements[8] + measurements[10]  # 16 + 9 + 15 + 14 = 54
    east_sum = measurements[1]  # 78
    west_sum = measurements[3] + measurements[5] + measurements[7] + measurements[9] + measurements[11]  # 7.5 + 15 + 8 + 32.5 + 15 = 78

    print("Balance Check:")
    print(f"  North: {north_sum:.1f} ft")
    print(f"  South: {south_sum:.1f} ft")
    print(f"  East:  {east_sum:.1f} ft")
    print(f"  West:  {west_sum:.1f} ft")
    print(f"  N-S Balance: {north_sum - south_sum:.1f} ft (should be 0)")
    print(f"  E-W Balance: {east_sum - west_sum:.1f} ft (should be 0)")
    print()

    # Trace polygon with PERFECT cardinal directions
    x, y = 0.0, 0.0
    vertices = [(x, y)]

    print("Polygon Trace (using PERFECT cardinals):")
    print(f"Start: ({x:.1f}, {y:.1f})")

    for i in range(12):
        direction = directions[i]
        distance = measurements[i]

        # Perfect cardinal movements
        if direction == 'E':
            x += distance
        elif direction == 'W':
            x -= distance
        elif direction == 'N':
            y += distance
        elif direction == 'S':
            y -= distance

        vertices.append((x, y))
        print(f"Edge {i:2d}: {direction} {distance:5.1f}ft -> ({x:6.1f}, {y:6.1f})")

    # Check closure
    first = vertices[0]
    last = vertices[-1]
    error = ((last[0] - first[0])**2 + (last[1] - first[1])**2)**0.5

    print(f"\nClosure Check:")
    print(f"  First vertex: ({first[0]:.1f}, {first[1]:.1f})")
    print(f"  Last vertex:  ({last[0]:.1f}, {last[1]:.1f})")
    print(f"  Error: {error:.4f} ft")

    if error < 0.001:
        print("  ✓ PERFECT CLOSURE!")
    else:
        print(f"  ✗ Closure error: {error:.2f} ft")

    return vertices, error

def investigate_edge_detection_issue():
    """Investigate why balanced measurements don't close in the system."""

    print("\n" + "="*70)
    print("INVESTIGATING CLOSURE ERROR")
    print("="*70)

    vertices, error = trace_polygon_with_measurements()

    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)

    if error < 0.001:
        print("Mathematics proves the polygon MUST close with these measurements.")
        print("\nPossible bugs in the system:")
        print("1. Edge ordering might not be truly clockwise")
        print("2. Some edges might have wrong cardinal direction assigned")
        print("3. The edges list might have duplicates or be missing edges")
        print("4. The coordinate system might be inverted (Y-axis direction)")
        print("5. The edge reversal in _ensure_clockwise might be corrupting directions")
    else:
        print("Even with perfect math, there's an error. Check the measurements.")

    # Let's also check if edges are continuous
    print("\n" + "="*70)
    print("EDGE CONTINUITY CHECK")
    print("="*70)

    print("\nFor proper edge continuity:")
    print("- Each edge's END should connect to the next edge's START")
    print("- The last edge's END should connect to the first edge's START")
    print("\nIn the actual system, this needs to be verified in the edge detection.")

if __name__ == "__main__":
    investigate_edge_detection_issue()
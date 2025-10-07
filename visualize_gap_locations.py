#!/usr/bin/env python3
"""
Visualize WHERE the gaps actually are
"""

import json
from pathlib import Path
from shapely.geometry import box, Polygon
from shapely.ops import unary_union
import numpy as np
import cv2

# Load results
results_path = Path('output_Umbra_XL_fill/results_fill.json')
with open(results_path, 'r') as f:
    results = json.load(f)

cassettes = results['cassettes']
polygon_coords = results['polygon']

polygon = Polygon(polygon_coords)

# Get bounds
xs = [p[0] for p in polygon_coords]
ys = [p[1] for p in polygon_coords]
min_x, max_x = min(xs), max(xs)
min_y, max_y = min(ys), max(ys)

# Create visualization
scale = 15  # pixels per foot
margin = 50
width = int((max_x - min_x) * scale) + 2 * margin
height = int((max_y - min_y) * scale) + 2 * margin

image = np.ones((height, width, 3), dtype=np.uint8) * 255

def to_pixel(x, y):
    px = int((x - min_x) * scale) + margin
    py = int((max_y - y) * scale) + margin  # Flip Y
    return px, py

# Title
cv2.putText(image, "GAP LOCATION ANALYSIS", (20, 30),
           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

# Draw polygon boundary (blue)
poly_pts = [to_pixel(p[0], p[1]) for p in polygon_coords]
poly_pts = np.array(poly_pts, dtype=np.int32)
cv2.polylines(image, [poly_pts], True, (255, 0, 0), 3)

# Calculate cassette union
cassette_geoms = [
    box(c['x'], c['y'], c['x'] + c['width'], c['y'] + c['height'])
    for c in cassettes
]
cassette_union = unary_union(cassette_geoms)

# Calculate gap
gap_geom = polygon.difference(cassette_union)

# Draw gap regions in RED
if hasattr(gap_geom, 'geoms'):
    gap_pieces = list(gap_geom.geoms)
else:
    gap_pieces = [gap_geom]

for gap in gap_pieces:
    # Get exterior coordinates
    gap_coords = list(gap.exterior.coords)
    gap_pts = [to_pixel(p[0], p[1]) for p in gap_coords]
    gap_pts = np.array(gap_pts, dtype=np.int32)
    cv2.fillPoly(image, [gap_pts], (0, 0, 255))  # Red

# Draw cassettes (semi-transparent green)
for cassette in cassettes:
    x, y = cassette['x'], cassette['y']
    w, h = cassette['width'], cassette['height']

    p1 = to_pixel(x, y + h)
    p2 = to_pixel(x + w, y)

    # Semi-transparent overlay
    overlay = image.copy()
    cv2.rectangle(overlay, p1, p2, (0, 255, 0), -1)
    cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)

    # Border
    cv2.rectangle(image, p1, p2, (0, 150, 0), 2)

# Add legend
legend_y = height - 120
cv2.rectangle(image, (20, legend_y), (250, height - 20), (240, 240, 240), -1)
cv2.putText(image, "LEGEND", (30, legend_y + 25),
           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

cv2.rectangle(image, (30, legend_y + 35), (55, legend_y + 55), (0, 255, 0), -1)
cv2.putText(image, "Cassettes", (65, legend_y + 50),
           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

cv2.rectangle(image, (30, legend_y + 60), (55, legend_y + 80), (0, 0, 255), -1)
cv2.putText(image, "GAP (to be filled)", (65, legend_y + 75),
           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

cv2.rectangle(image, (30, legend_y + 85), (55, legend_y + 105), (255, 255, 255), -1)
cv2.rectangle(image, (30, legend_y + 85), (55, legend_y + 105), (255, 0, 0), 2)
cv2.putText(image, "Polygon boundary", (65, legend_y + 100),
           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

# Add statistics
stats_y = 60
cv2.putText(image, f"Total area: {polygon.area:.1f} sq ft", (20, stats_y),
           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
cv2.putText(image, f"Cassette coverage: {cassette_union.area:.1f} sq ft (97.8%)", (20, stats_y + 20),
           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
cv2.putText(image, f"GAP (red areas): {gap_geom.area:.1f} sq ft (2.2%)", (20, stats_y + 40),
           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Save
cv2.imwrite('gap_visualization.png', image)
print("Gap visualization saved to: gap_visualization.png")

# Detailed gap analysis
print("\n" + "=" * 80)
print("DETAILED GAP ANALYSIS")
print("=" * 80)

print(f"\nTotal gap: {gap_geom.area:.1f} sq ft")
print(f"Number of gap regions: {len(gap_pieces)}")

# Analyze each cassette's gaps
print("\n" + "=" * 80)
print("CHECKING GAPS BETWEEN ADJACENT CASSETTES")
print("=" * 80)

# Check gaps between specific adjacent cassettes
test_pairs = [
    (0, 1),  # Should be adjacent horizontally
    (0, 2),  # Should be adjacent vertically
    (1, 3),  # Should be adjacent vertically
]

for i, j in test_pairs:
    c1 = cassettes[i]
    c2 = cassettes[j]

    geom1 = box(c1['x'], c1['y'], c1['x'] + c1['width'], c1['y'] + c1['height'])
    geom2 = box(c2['x'], c2['y'], c2['x'] + c2['width'], c2['y'] + c2['height'])

    # Check if they touch or overlap
    touches = geom1.touches(geom2)
    overlaps = geom1.overlaps(geom2)
    distance = geom1.distance(geom2)

    print(f"\nCassette {i} ({c1['size']} at {c1['x']:.1f},{c1['y']:.1f})")
    print(f"  vs")
    print(f"Cassette {j} ({c2['size']} at {c2['x']:.1f},{c2['y']:.1f})")
    print(f"  Touches: {touches}")
    print(f"  Overlaps: {overlaps}")
    print(f"  Distance: {distance:.4f} ft")

    if distance > 0:
        print(f"  → GAP of {distance:.4f} ft ({distance*12:.2f} inches) between them")
    elif touches:
        print(f"  → No gap - cassettes are touching edge-to-edge")
    elif overlaps:
        print(f"  → ERROR - cassettes overlap!")

print("\n" + "=" * 80)
print("KEY FINDING:")
print("=" * 80)
print("Adjacent cassettes should be touching edge-to-edge (distance = 0).")
print("If they are, then gaps exist ONLY at polygon boundaries,")
print("NOT between cassettes.")

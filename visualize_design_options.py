#!/usr/bin/env python3
"""
Visualize different architectural design options
"""

import numpy as np
import cv2

# Create comprehensive comparison diagram
width = 1600
height = 1200
image = np.ones((height, width, 3), dtype=np.uint8) * 255

# Title
cv2.putText(image, "CASSETTE + C-CHANNEL ARCHITECTURE OPTIONS", (50, 40),
           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

# Colors
cassette_color = (200, 150, 200)  # Pink
cchannel_color = (180, 200, 240)  # Tan
gap_color = (255, 200, 200)  # Light red
boundary_color = (255, 0, 0)  # Red

# CURRENT SYSTEM
y_offset = 80
cv2.putText(image, "CURRENT SYSTEM (Touching Cassettes)", (50, y_offset),
           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

# Draw polygon boundary
poly_x, poly_y = 50, y_offset + 20
poly_w, poly_h = 600, 200
cv2.rectangle(image, (poly_x, poly_y), (poly_x + poly_w, poly_y + poly_h),
             boundary_color, 3)

# Draw 6 touching cassettes
cass_w = 90
cass_h = poly_h - 20
gap_at_edge = 10

for i in range(6):
    x = poly_x + gap_at_edge + (i * cass_w)
    y = poly_y + 10
    cv2.rectangle(image, (x, y), (x + cass_w, y + cass_h),
                 cassette_color, -1)
    cv2.rectangle(image, (x, y), (x + cass_w, y + cass_h),
                 (0, 0, 0), 2)

# Show gaps at boundaries
cv2.rectangle(image, (poly_x, poly_y), (poly_x + gap_at_edge, poly_y + poly_h),
             gap_color, -1)
cv2.rectangle(image, (poly_x + poly_w - gap_at_edge, poly_y),
             (poly_x + poly_w, poly_y + poly_h), gap_color, -1)

# Annotations
cv2.putText(image, "Cassettes touch", (poly_x + 220, poly_y + poly_h + 30),
           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 0), 1)
cv2.putText(image, "(no center gaps)", (poly_x + 220, poly_y + poly_h + 50),
           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 0), 1)

cv2.arrowedLine(image, (poly_x - 30, poly_y + poly_h//2),
               (poly_x + 5, poly_y + poly_h//2), (255, 0, 0), 2)
cv2.putText(image, "Boundary", (poly_x - 120, poly_y + poly_h//2),
           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
cv2.putText(image, "gap", (poly_x - 80, poly_y + poly_h//2 + 20),
           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

# Stats
stats_x = poly_x + poly_w + 30
cv2.putText(image, "Coverage: 97.8% (cassettes)", (stats_x, poly_y + 30),
           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
cv2.putText(image, "Gaps: At boundaries ONLY", (stats_x, poly_y + 60),
           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
cv2.putText(image, "Center gaps: 0 sq ft", (stats_x, poly_y + 90),
           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

cv2.putText(image, "With C-channel perimeters:", (stats_x, poly_y + 130),
           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
cv2.putText(image, "  Coverage: 100%", (stats_x, poly_y + 155),
           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 0), 1)
cv2.putText(image, "  But: Fills boundary gaps", (stats_x, poly_y + 180),
           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 0), 1)


# OPTION A: SPACED CASSETTES
y_offset = 360
cv2.putText(image, "OPTION A: Spaced Cassettes (With Center Gaps)", (50, y_offset),
           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

# Draw polygon boundary
poly_y = y_offset + 20
cv2.rectangle(image, (poly_x, poly_y), (poly_x + poly_w, poly_y + poly_h),
             boundary_color, 3)

# Draw 5 spaced cassettes with gaps between them
num_cass = 5
spacing = 15  # Gap between cassettes
cass_w = (poly_w - 2 * gap_at_edge - (num_cass - 1) * spacing) // num_cass

for i in range(num_cass):
    x = poly_x + gap_at_edge + (i * (cass_w + spacing))
    y = poly_y + 10
    cv2.rectangle(image, (x, y), (x + cass_w, y + cass_h),
                 cassette_color, -1)
    cv2.rectangle(image, (x, y), (x + cass_w, y + cass_h),
                 (0, 0, 0), 2)

    # Draw C-channel filler in gap (except after last cassette)
    if i < num_cass - 1:
        gap_x = x + cass_w
        cv2.rectangle(image, (gap_x, y), (gap_x + spacing, y + cass_h),
                     cchannel_color, -1)
        cv2.rectangle(image, (gap_x, y), (gap_x + spacing, y + cass_h),
                     (100, 100, 100), 1)

# Show UNFILLED gaps at boundaries
cv2.rectangle(image, (poly_x, poly_y), (poly_x + gap_at_edge, poly_y + poly_h),
             gap_color, -1)
cv2.rectangle(image, (poly_x + poly_w - gap_at_edge, poly_y),
             (poly_x + poly_w, poly_y + poly_h), gap_color, -1)

# Annotations
cv2.arrowedLine(image, (poly_x + gap_at_edge + cass_w + spacing//2, poly_y - 20),
               (poly_x + gap_at_edge + cass_w + spacing//2, y + 5), (0, 0, 255), 2)
cv2.putText(image, "C-channel filler", (poly_x + gap_at_edge + cass_w - 30, poly_y - 30),
           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
cv2.putText(image, "(in center)", (poly_x + gap_at_edge + cass_w - 20, poly_y - 15),
           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

cv2.arrowedLine(image, (poly_x + poly_w + 30, poly_y + 20),
               (poly_x + poly_w - 5, poly_y + 20), (255, 0, 0), 2)
cv2.putText(image, "UNFILLED", (poly_x + poly_w + 40, poly_y + 25),
           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

# Stats
cv2.putText(image, "Cassette coverage: ~85%", (stats_x, poly_y + 30),
           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
cv2.putText(image, "Center gaps: Filled with C-ch", (stats_x, poly_y + 60),
           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 0), 1)
cv2.putText(image, "Boundary gaps: UNFILLED", (stats_x, poly_y + 90),
           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
cv2.putText(image, "Total coverage: ~92%", (stats_x, poly_y + 120),
           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 100, 0), 1)

cv2.putText(image, "PROs:", (stats_x, poly_y + 155),
           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 0), 2)
cv2.putText(image, "  + C-ch are pure fillers", (stats_x, poly_y + 175),
           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 150, 0), 1)
cv2.putText(image, "  + Only in center", (stats_x, poly_y + 190),
           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 150, 0), 1)

cv2.putText(image, "CONs:", (stats_x + 200, poly_y + 155),
           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 0), 2)
cv2.putText(image, "  - Not 100% coverage", (stats_x + 200, poly_y + 175),
           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 0, 0), 1)
cv2.putText(image, "  - Boundary gaps unfilled", (stats_x + 200, poly_y + 190),
           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 0, 0), 1)


# OPTION B: GRID LAYOUT
y_offset = 640
cv2.putText(image, "OPTION B: Grid Layout (Uniform Spacing)", (50, y_offset),
           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

# Draw polygon boundary
poly_y = y_offset + 20
cv2.rectangle(image, (poly_x, poly_y), (poly_x + poly_w, poly_y + poly_h),
             boundary_color, 3)

# Draw grid of cassettes with uniform spacing
grid_cols = 5
grid_rows = 2
h_spacing = 12
v_spacing = 12
margin = 10

total_cass_w = poly_w - 2 * margin - (grid_cols - 1) * h_spacing
total_cass_h = poly_h - 2 * margin - (grid_rows - 1) * v_spacing
cass_w = total_cass_w // grid_cols
cass_h = total_cass_h // grid_rows

for row in range(grid_rows):
    for col in range(grid_cols):
        x = poly_x + margin + col * (cass_w + h_spacing)
        y = poly_y + margin + row * (cass_h + v_spacing)

        # Cassette
        cv2.rectangle(image, (x, y), (x + cass_w, y + cass_h),
                     cassette_color, -1)
        cv2.rectangle(image, (x, y), (x + cass_w, y + cass_h),
                     (0, 0, 0), 2)

        # Vertical C-channel filler (except last column)
        if col < grid_cols - 1:
            gap_x = x + cass_w
            cv2.rectangle(image, (gap_x, y), (gap_x + h_spacing, y + cass_h),
                         cchannel_color, -1)

        # Horizontal C-channel filler (except last row)
        if row < grid_rows - 1:
            gap_y = y + cass_h
            cv2.rectangle(image, (x, gap_y), (x + cass_w, gap_y + v_spacing),
                         cchannel_color, -1)

        # Intersection fillers (except edges)
        if col < grid_cols - 1 and row < grid_rows - 1:
            gap_x = x + cass_w
            gap_y = y + cass_h
            cv2.rectangle(image, (gap_x, gap_y),
                         (gap_x + h_spacing, gap_y + v_spacing),
                         cchannel_color, -1)

# Show UNFILLED gaps at all boundaries
# Top/bottom
cv2.rectangle(image, (poly_x, poly_y), (poly_x + poly_w, poly_y + margin),
             gap_color, -1)
cv2.rectangle(image, (poly_x, poly_y + poly_h - margin),
             (poly_x + poly_w, poly_y + poly_h), gap_color, -1)
# Left/right
cv2.rectangle(image, (poly_x, poly_y), (poly_x + margin, poly_y + poly_h),
             gap_color, -1)
cv2.rectangle(image, (poly_x + poly_w - margin, poly_y),
             (poly_x + poly_w, poly_y + poly_h), gap_color, -1)

# Annotations
cv2.putText(image, "Uniform grid with C-channel fillers between all cassettes",
           (poly_x, poly_y + poly_h + 30),
           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

# Stats
cv2.putText(image, "Cassette coverage: ~80%", (stats_x, poly_y + 30),
           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
cv2.putText(image, "Center gaps: Filled", (stats_x, poly_y + 60),
           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 0), 1)
cv2.putText(image, "Boundary gaps: UNFILLED", (stats_x, poly_y + 90),
           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
cv2.putText(image, "Total coverage: ~90%", (stats_x, poly_y + 120),
           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 100, 0), 1)

cv2.putText(image, "PROs:", (stats_x, poly_y + 155),
           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 0), 2)
cv2.putText(image, "  + Uniform spacing", (stats_x, poly_y + 175),
           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 150, 0), 1)
cv2.putText(image, "  + C-ch only in center", (stats_x, poly_y + 190),
           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 150, 0), 1)

cv2.putText(image, "CONs:", (stats_x + 200, poly_y + 155),
           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 0), 2)
cv2.putText(image, "  - Boundary gaps", (stats_x + 200, poly_y + 175),
           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 0, 0), 1)
cv2.putText(image, "  - Lower coverage", (stats_x + 200, poly_y + 190),
           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 0, 0), 1)


# COMPARISON TABLE
y_offset = 940
cv2.putText(image, "COMPARISON SUMMARY", (50, y_offset),
           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

table_data = [
    ["", "Current", "Option A", "Option B"],
    ["Cassettes touch", "YES", "NO", "NO"],
    ["Center gaps", "0 sq ft", "Yes (filled)", "Yes (filled)"],
    ["Boundary gaps", "Filled w/ C-ch", "UNFILLED", "UNFILLED"],
    ["C-ch location", "Perimeter", "Center only", "Grid center"],
    ["Coverage", "100%", "~92%", "~90%"],
    ["Meets requirement", "NO", "YES", "YES"],
]

table_x = 80
table_y = y_offset + 20
row_h = 30
col_widths = [200, 120, 120, 120]

for i, row in enumerate(table_data):
    x = table_x
    for j, cell in enumerate(row):
        # Header row
        if i == 0:
            cv2.rectangle(image, (x, table_y + i * row_h),
                         (x + col_widths[j], table_y + (i + 1) * row_h),
                         (200, 200, 200), -1)
            cv2.rectangle(image, (x, table_y + i * row_h),
                         (x + col_widths[j], table_y + (i + 1) * row_h),
                         (0, 0, 0), 1)
            cv2.putText(image, cell, (x + 5, table_y + i * row_h + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        else:
            cv2.rectangle(image, (x, table_y + i * row_h),
                         (x + col_widths[j], table_y + (i + 1) * row_h),
                         (0, 0, 0), 1)

            # Color code based on content
            color = (0, 0, 0)
            if "YES" in cell and i == len(table_data) - 1:  # Meets requirement
                color = (0, 150, 0)
            elif "NO" in cell and i == len(table_data) - 1:
                color = (200, 0, 0)

            cv2.putText(image, cell, (x + 5, table_y + i * row_h + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        x += col_widths[j]

# Legend
legend_x = 800
legend_y = y_offset + 30
cv2.putText(image, "LEGEND", (legend_x, legend_y),
           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

cv2.rectangle(image, (legend_x, legend_y + 15), (legend_x + 30, legend_y + 35),
             cassette_color, -1)
cv2.rectangle(image, (legend_x, legend_y + 15), (legend_x + 30, legend_y + 35),
             (0, 0, 0), 1)
cv2.putText(image, "Cassette", (legend_x + 40, legend_y + 30),
           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

cv2.rectangle(image, (legend_x, legend_y + 45), (legend_x + 30, legend_y + 65),
             cchannel_color, -1)
cv2.rectangle(image, (legend_x, legend_y + 45), (legend_x + 30, legend_y + 65),
             (0, 0, 0), 1)
cv2.putText(image, "C-channel filler", (legend_x + 40, legend_y + 60),
           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

cv2.rectangle(image, (legend_x, legend_y + 75), (legend_x + 30, legend_y + 95),
             gap_color, -1)
cv2.rectangle(image, (legend_x, legend_y + 75), (legend_x + 30, legend_y + 95),
             (0, 0, 0), 1)
cv2.putText(image, "Unfilled gap", (legend_x + 40, legend_y + 90),
           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

cv2.rectangle(image, (legend_x, legend_y + 105), (legend_x + 30, legend_y + 125),
             (255, 255, 255), -1)
cv2.rectangle(image, (legend_x, legend_y + 105), (legend_x + 30, legend_y + 125),
             boundary_color, 3)
cv2.putText(image, "Polygon boundary", (legend_x + 40, legend_y + 120),
           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

# Save
cv2.imwrite('design_options_comparison.png', image)
print("Design options visualization saved to: design_options_comparison.png")

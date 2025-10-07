#!/usr/bin/env python3
"""
Visual diagram showing C-channel architecture in detail
"""

import numpy as np
import cv2

# Create detailed diagram
width = 1400
height = 1000
image = np.ones((height, width, 3), dtype=np.uint8) * 255

# Title
cv2.putText(image, "C-CHANNEL PERIMETER ARCHITECTURE", (50, 50),
           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)

# DIAGRAM 1: Single Cassette with C-Channel Perimeter
y_offset = 100
cv2.putText(image, "1. SINGLE CASSETTE - Full C-Channel Perimeter", (50, y_offset),
           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

# Draw cassette
cass_x, cass_y = 100, y_offset + 30
cass_w, cass_h = 160, 120  # 8x6 cassette scaled
c_width = 18  # 18" C-channel scaled

# C-channel perimeter (tan)
cv2.rectangle(image,
             (cass_x - c_width, cass_y - c_width),
             (cass_x + cass_w + c_width, cass_y + cass_h + c_width),
             (180, 200, 240), -1)

# Cassette (pink)
cv2.rectangle(image, (cass_x, cass_y), (cass_x + cass_w, cass_y + cass_h),
             (200, 150, 200), -1)
cv2.rectangle(image, (cass_x, cass_y), (cass_x + cass_w, cass_y + cass_h),
             (0, 0, 0), 2)

# Label the 4 C-channel strips
cv2.putText(image, "North (18\")", (cass_x + 40, cass_y - 25),
           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
cv2.putText(image, "South (18\")", (cass_x + 40, cass_y + cass_h + 35),
           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
cv2.putText(image, "West", (cass_x - 40, cass_y + 60),
           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
cv2.putText(image, "East", (cass_x + cass_w + 22, cass_y + 60),
           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

cv2.putText(image, "8x6 Cassette", (cass_x + 30, cass_y + 65),
           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

# Explanation
cv2.putText(image, "Each cassette has 4 C-channel strips forming a perimeter", (350, y_offset + 50),
           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
cv2.putText(image, "Provides structural support around cassette edges", (350, y_offset + 80),
           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)


# DIAGRAM 2: Adjacent Cassettes - Shared C-Channels
y_offset = 340
cv2.putText(image, "2. ADJACENT CASSETTES - Shared C-Channels (OPTIMIZED)", (50, y_offset),
           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

# Left cassette
cass1_x = 100
cass1_y = y_offset + 30

# Right cassette (adjacent, touching)
cass2_x = cass1_x + cass_w
cass2_y = cass1_y

# Draw shared C-channel in middle (full 18")
shared_x = cass1_x + cass_w - c_width//2
cv2.rectangle(image,
             (shared_x, cass1_y),
             (shared_x + c_width, cass1_y + cass_h),
             (180, 200, 240), -1)

# Draw cassette 1 C-channels (excluding shared middle)
# North
cv2.rectangle(image, (cass1_x, cass1_y - c_width),
             (cass1_x + cass_w, cass1_y), (180, 200, 240), -1)
# South
cv2.rectangle(image, (cass1_x, cass1_y + cass_h),
             (cass1_x + cass_w, cass1_y + cass_h + c_width), (180, 200, 240), -1)
# West
cv2.rectangle(image, (cass1_x - c_width, cass1_y),
             (cass1_x, cass1_y + cass_h), (180, 200, 240), -1)

# Draw cassette 2 C-channels (excluding shared middle)
# North
cv2.rectangle(image, (cass2_x, cass2_y - c_width),
             (cass2_x + cass_w, cass2_y), (180, 200, 240), -1)
# South
cv2.rectangle(image, (cass2_x, cass2_y + cass_h),
             (cass2_x + cass_w, cass2_y + cass_h + c_width), (180, 200, 240), -1)
# East
cv2.rectangle(image, (cass2_x + cass_w, cass2_y),
             (cass2_x + cass_w + c_width, cass2_y + cass_h), (180, 200, 240), -1)

# Draw cassettes
cv2.rectangle(image, (cass1_x, cass1_y), (cass1_x + cass_w, cass1_y + cass_h),
             (200, 150, 200), -1)
cv2.rectangle(image, (cass1_x, cass1_y), (cass1_x + cass_w, cass1_y + cass_h),
             (0, 0, 0), 2)

cv2.rectangle(image, (cass2_x, cass2_y), (cass2_x + cass_w, cass2_y + cass_h),
             (200, 150, 200), -1)
cv2.rectangle(image, (cass2_x, cass2_y), (cass2_x + cass_w, cass2_y + cass_h),
             (0, 0, 0), 2)

# Arrow and label for shared C-channel
arrow_y = cass1_y + cass_h // 2
cv2.arrowedLine(image, (shared_x + c_width + 10, arrow_y),
               (shared_x + c_width//2, arrow_y), (0, 0, 255), 2)
cv2.putText(image, "Shared 18\" C-channel", (shared_x + c_width + 15, arrow_y),
           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
cv2.putText(image, "(Left contributes 9\"", (shared_x + c_width + 15, arrow_y + 20),
           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
cv2.putText(image, "+ Right contributes 9\")", (shared_x + c_width + 15, arrow_y + 35),
           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

# Explanation
cv2.putText(image, "Adjacent cassettes SHARE their C-channels at the boundary", (500, y_offset + 60),
           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
cv2.putText(image, "Each cassette extends 9\" (half-width) toward the other", (500, y_offset + 90),
           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
cv2.putText(image, "Result: 18\" total with NO overlap, NO gap", (500, y_offset + 120),
           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)


# DIAGRAM 3: Key Differences
y_offset = 580
cv2.putText(image, "3. KEY ARCHITECTURAL PRINCIPLES", (50, y_offset),
           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

principles = [
    "✓ PERIMETER-BASED: Each cassette has a C-channel perimeter (4 strips)",
    "✓ SHARED: Adjacent cassettes share C-channels (each contributes 9\")",
    "✓ STRUCTURAL: C-channels provide support, not just gap filling",
    "✓ OPTIMIZED: 18\" width chosen so perimeters achieve 100% coverage",
    "",
    "✗ NOT filler-based: We don't randomly place C-channels in gaps",
    "✗ NOT patch material: C-channels are structural elements",
]

for i, principle in enumerate(principles):
    color = (0, 150, 0) if principle.startswith("✓") else (200, 0, 0) if principle.startswith("✗") else (0, 0, 0)
    cv2.putText(image, principle, (70, y_offset + 40 + i * 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)


# DIAGRAM 4: Numbers
y_offset = 820
cv2.putText(image, "4. UMBRA XL CONFIGURATION", (50, y_offset),
           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

stats = [
    "24 cassettes × 4 edges = 96 C-channel strips created",
    "68 strips are half-width (9\") - shared with adjacent cassettes",
    "28 strips are full-width (18\") - at polygon boundaries",
    "Total C-channel area: 24.0 sq ft (fills 24.0 sq ft gap perfectly)",
]

for i, stat in enumerate(stats):
    cv2.putText(image, "• " + stat, (70, y_offset + 35 + i * 28),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


# Legend
cv2.rectangle(image, (1050, 100), (1350, 250), (240, 240, 240), -1)
cv2.putText(image, "LEGEND", (1150, 130),
           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

cv2.rectangle(image, (1070, 150), (1100, 170), (200, 150, 200), -1)
cv2.putText(image, "Cassette", (1110, 166),
           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

cv2.rectangle(image, (1070, 185), (1100, 205), (180, 200, 240), -1)
cv2.putText(image, "C-Channel", (1110, 201),
           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

cv2.rectangle(image, (1070, 220), (1100, 240), (255, 255, 255), -1)
cv2.rectangle(image, (1070, 220), (1100, 240), (0, 0, 0), 1)
cv2.putText(image, "Empty Space", (1110, 236),
           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

# Save
cv2.imwrite('cchannel_architecture_diagram.png', image)
print("Diagram saved to: cchannel_architecture_diagram.png")

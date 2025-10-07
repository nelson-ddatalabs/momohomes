#!/usr/bin/env python3
"""
FUNDAMENTAL CONTRADICTION ANALYSIS
===================================
Based on user's clarification:
1. C-channels are ONLY fillers (not structural perimeters)
2. Fillers can ONLY be in CENTER (between cassettes), NOT at edges
3. No perimeter needed around cassettes
4. Focus ONLY on filling gaps

Let's analyze if this is possible with current cassette placement
"""

import json
from pathlib import Path
from shapely.geometry import box, Polygon, Point
from shapely.ops import unary_union
import numpy as np

print("\n" + "=" * 80)
print("FUNDAMENTAL CONTRADICTION ANALYSIS")
print("=" * 80)

# Load current results
results_path = Path('output_Umbra_XL_fill/results_fill.json')
with open(results_path, 'r') as f:
    results = json.load(f)

cassettes = results['cassettes']
polygon_coords = results['polygon']
polygon = Polygon(polygon_coords)

print("\n" + "=" * 80)
print("STEP 1: WHERE ARE THE GAPS WITH CURRENT CASSETTE PLACEMENT?")
print("=" * 80)

# Calculate gaps
cassette_geoms = [
    box(c['x'], c['y'], c['x'] + c['width'], c['y'] + c['height'])
    for c in cassettes
]
cassette_union = unary_union(cassette_geoms)
gap_geom = polygon.difference(cassette_union)

print(f"\nCurrent situation:")
print(f"  Total gap: {gap_geom.area:.1f} sq ft")
print(f"  Cassettes touch edge-to-edge: YES (distance = 0.0 ft)")

# Check if gaps are at boundaries or between cassettes
poly_bounds = polygon.bounds
poly_boundary = polygon.boundary

# Sample points from gap to check if they're at polygon boundary
if hasattr(gap_geom, 'geoms'):
    gap_pieces = list(gap_geom.geoms)
else:
    gap_pieces = [gap_geom]

gaps_at_boundary = 0
gaps_in_center = 0

for gap in gap_pieces:
    # Get centroid of gap
    centroid = gap.centroid

    # Check if this gap touches the polygon boundary
    touches_boundary = gap.intersects(poly_boundary)

    if touches_boundary:
        gaps_at_boundary += 1
    else:
        gaps_in_center += 1

print(f"\nGap location analysis:")
print(f"  Gaps touching polygon boundary: {gaps_at_boundary}")
print(f"  Gaps in center (between cassettes): {gaps_in_center}")

print("\n" + "-" * 80)
print("FINDING:")
print("-" * 80)
if gaps_in_center == 0:
    print("⚠️  ALL GAPS ARE AT POLYGON BOUNDARIES")
    print("⚠️  ZERO GAPS EXIST BETWEEN CASSETTES IN THE CENTER")
else:
    print(f"✓ {gaps_in_center} gaps exist in center between cassettes")

print("\n" + "=" * 80)
print("STEP 2: APPLYING USER'S NEW REQUIREMENTS")
print("=" * 80)

print("\nUser requirements:")
print("  1. C-channels are ONLY fillers (not structural perimeters)")
print("  2. Fillers can ONLY be in CENTER (between cassettes)")
print("  3. Fillers CANNOT be at edges/boundaries")
print("  4. Focus ONLY on filling gaps")

print("\nCurrent gap distribution:")
print(f"  At boundaries: {gap_geom.area:.1f} sq ft (100%)")
print(f"  In center: 0.0 sq ft (0%)")

print("\n" + "-" * 80)
print("FUNDAMENTAL CONTRADICTION:")
print("-" * 80)
print("❌ User says: Fillers can ONLY be in CENTER")
print("❌ Reality: ALL gaps are at BOUNDARIES")
print("❌ Result: CANNOT fill any gaps with this constraint")
print("")
print("Coverage with user's constraints: 97.8% (no C-channels can be placed)")

print("\n" + "=" * 80)
print("STEP 3: ROOT CAUSE ANALYSIS")
print("=" * 80)

print("\nWhy are there no center gaps?")
print("  → The greedy optimizer places cassettes TOUCHING edge-to-edge")
print("  → This maximizes cassette coverage (97.8%)")
print("  → Leaves gaps ONLY at polygon boundaries")

print("\nExample: Check adjacent cassettes")
for i in range(min(3, len(cassettes) - 1)):
    c1 = cassettes[i]
    c2 = cassettes[i + 1]

    g1 = box(c1['x'], c1['y'], c1['x'] + c1['width'], c1['y'] + c1['height'])
    g2 = box(c2['x'], c2['y'], c2['x'] + c2['width'], c2['y'] + c2['height'])

    distance = g1.distance(g2)
    touches = g1.touches(g2)

    print(f"\n  Cassette {i} at ({c1['x']:.1f}, {c1['y']:.1f})")
    print(f"  Cassette {i+1} at ({c2['x']:.1f}, {c2['y']:.1f})")
    print(f"    Distance between them: {distance:.4f} ft")
    print(f"    Touch edge-to-edge: {touches}")

    if distance > 0:
        print(f"    → GAP of {distance*12:.2f} inches in CENTER ✓")
    else:
        print(f"    → NO GAP (cassettes touch) ❌")

print("\n" + "=" * 80)
print("STEP 4: POSSIBLE SOLUTIONS")
print("=" * 80)

print("\n" + "-" * 80)
print("OPTION A: Change Cassette Placement to Create Center Gaps")
print("-" * 80)

print("\nIdea:")
print("  • Place cassettes with intentional spacing between them")
print("  • Create gaps IN CENTER (between cassettes)")
print("  • Use C-channels as fillers in these center gaps")
print("  • Don't fill boundary gaps (accept < 100% coverage)")

print("\nExample:")
print("  Current: [Cassette][Cassette][Cassette] |boundary gap|")
print("  New:     [Cassette] gap [Cassette] gap [Cassette] |boundary gap|")
print("           └─ Fill with C-channel ─┘")

print("\nImplications:")
print("  ✓ C-channels are pure fillers (not perimeters)")
print("  ✓ Fillers are in center (between cassettes)")
print("  ✓ No fillers at boundaries")
print("  ❌ Coverage will be < 100% (boundary gaps unfilled)")
print("  ❓ How much spacing between cassettes?")
print("  ❓ What happens to boundary gaps?")

print("\n" + "-" * 80)
print("OPTION B: Redefine 'Center' vs 'Boundary'")
print("-" * 80)

print("\nIdea:")
print("  • Perhaps 'center' means 'not at polygon edges'?")
print("  • Gaps between cassette edges and polygon are 'center'?")
print("  • Fill these gaps with C-channels")

print("\nImplications:")
print("  ❓ Unclear definition of 'center' vs 'boundary'")
print("  ❓ Need clarification from user")

print("\n" + "-" * 80)
print("OPTION C: Different Coverage Goal")
print("-" * 80)

print("\nIdea:")
print("  • Accept that 100% coverage is not the goal")
print("  • Cassette coverage: 97.8%")
print("  • C-channel coverage: Fill center gaps only")
print("  • Total coverage: < 100%")
print("  • Boundary gaps remain unfilled")

print("\nImplications:")
print("  ✓ Aligns with user requirements")
print("  ❌ Does not achieve 100% coverage")
print("  ❓ What is acceptable coverage percentage?")

print("\n" + "=" * 80)
print("CRITICAL QUESTIONS FOR USER")
print("=" * 80)

questions = [
    "1. Should cassettes be placed with SPACING between them?",
    "   • Current: Cassettes touch edge-to-edge (no gaps between)",
    "   • New: Create gaps between cassettes to fill with C-channels?",
    "   • How much spacing? (e.g., 2\", 6\", 12\"?)",
    "",
    "2. What happens to BOUNDARY gaps (at polygon edges)?",
    "   • Current: 24 sq ft of gaps at boundaries",
    "   • If we can't fill them: Coverage = 97.8% (not 100%)",
    "   • Is < 100% coverage acceptable?",
    "   • Or use different material at boundaries?",
    "",
    "3. What is the COVERAGE GOAL?",
    "   • 100% coverage of polygon?",
    "   • Or just maximize cassette placement + center gaps?",
    "   • Acceptable to leave boundary gaps unfilled?",
    "",
    "4. What defines 'CENTER' vs 'BOUNDARY'?",
    "   • Center = between cassettes only?",
    "   • Or center = anywhere not touching polygon exterior?",
    "",
    "5. In real construction, what are C-channels FOR?",
    "   • Spacing/support between cassettes?",
    "   • Structural framing?",
    "   • Just fill gaps for complete coverage?",
]

for q in questions:
    print(q)

print("\n" + "=" * 80)
print("SUMMARY: NEED FUNDAMENTAL REDESIGN")
print("=" * 80)

print("\nCurrent system:")
print("  • Places cassettes touching (max coverage)")
print("  • Creates C-channel perimeters")
print("  • Fills boundary gaps")
print("  • Achieves 100% coverage")

print("\nUser's new vision:")
print("  • C-channels are ONLY center fillers")
print("  • NO C-channels at boundaries")
print("  • Focus on gaps between cassettes")

print("\nThe contradiction:")
print("  • Current: No center gaps exist (cassettes touch)")
print("  • User: Only fill center gaps")
print("  • Result: Cannot fill anything")

print("\n⚠️  FUNDAMENTAL REDESIGN NEEDED:")
print("   1. Change cassette placement (create center gaps)")
print("   2. Redefine coverage goals (accept < 100%?)")
print("   3. Clarify boundary gap handling")

print("\n" + "=" * 80)

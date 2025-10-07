#!/usr/bin/env python3
"""
DEEP INVESTIGATION: What's Really Happening Based on Logs
===========================================================
Re-examining the two questions based on actual log output
"""

import json
from pathlib import Path
from shapely.geometry import box, Polygon
from shapely.ops import unary_union
import numpy as np

print("\n" + "=" * 80)
print("DEEP INVESTIGATION: ANALYZING ACTUAL SYSTEM BEHAVIOR FROM LOGS")
print("=" * 80)

# Load results
results_path = Path('output_Umbra_XL_fill/results_fill.json')
with open(results_path, 'r') as f:
    results = json.load(f)

cassettes = results['cassettes']
polygon_coords = results['polygon']
stats = results['statistics']

print("\n" + "=" * 80)
print("STEP 1: UNDERSTANDING THE GAP")
print("=" * 80)

polygon = Polygon(polygon_coords)
polygon_area = polygon.area

# Calculate cassette coverage
cassette_geoms = [
    box(c['x'], c['y'], c['x'] + c['width'], c['y'] + c['height'])
    for c in cassettes
]
cassette_union = unary_union(cassette_geoms)
cassette_area = cassette_union.area

# Calculate the actual gap geometry
gap_geom = polygon.difference(cassette_union)
gap_area = gap_geom.area

print(f"\nFrom Logs:")
print(f"  Total polygon area: {stats['total_area']:.1f} sq ft")
print(f"  Cassette area: {stats['cassette_area']:.1f} sq ft ({stats['cassette_area']/stats['total_area']*100:.1f}%)")
print(f"  Gap area: {gap_area:.1f} sq ft ({gap_area/stats['total_area']*100:.1f}%)")

print(f"\nVerified Calculation:")
print(f"  Polygon area: {polygon_area:.1f} sq ft")
print(f"  Cassette union area: {cassette_area:.1f} sq ft")
print(f"  Gap (polygon - cassettes): {gap_area:.1f} sq ft")
print(f"  Coverage without C-channels: {cassette_area/polygon_area*100:.1f}%")

print("\n" + "-" * 80)
print("CRITICAL OBSERVATION FROM LOGS:")
print("-" * 80)
print("The system has a GAP of 24.0 sq ft that needs to be filled.")
print("This gap exists BEFORE any C-channels are added.")
print("The C-channels are being sized to fill this gap.")

print("\n" + "=" * 80)
print("STEP 2: WHERE IS THE GAP?")
print("=" * 80)

# Analyze gap geometry
if hasattr(gap_geom, 'geoms'):
    # MultiPolygon
    gap_pieces = list(gap_geom.geoms)
else:
    # Single Polygon
    gap_pieces = [gap_geom]

print(f"\nGap consists of {len(gap_pieces)} region(s)")

# Find gaps between cassettes and at boundaries
between_cassettes_gaps = []
boundary_gaps = []

# Simplified analysis: Check if gap touches polygon boundary
poly_bounds = polygon.bounds
for i, gap_piece in enumerate(gap_pieces):
    gap_bounds = gap_piece.bounds
    # Check if this gap touches the polygon boundary
    touches_boundary = (
        abs(gap_bounds[0] - poly_bounds[0]) < 0.01 or  # Left edge
        abs(gap_bounds[1] - poly_bounds[1]) < 0.01 or  # Bottom edge
        abs(gap_bounds[2] - poly_bounds[2]) < 0.01 or  # Right edge
        abs(gap_bounds[3] - poly_bounds[3]) < 0.01     # Top edge
    )

    if touches_boundary:
        boundary_gaps.append(gap_piece)
    else:
        between_cassettes_gaps.append(gap_piece)

print(f"\nGap locations:")
print(f"  Between cassettes: {len(between_cassettes_gaps)} regions")
print(f"  At boundaries: {len(boundary_gaps)} regions")

# Calculate areas
if between_cassettes_gaps:
    between_area = sum(g.area for g in between_cassettes_gaps)
    print(f"  Area between cassettes: {between_area:.1f} sq ft")
if boundary_gaps:
    boundary_area = sum(g.area for g in boundary_gaps)
    print(f"  Area at boundaries: {boundary_area:.1f} sq ft")

print("\n" + "-" * 80)
print("CRITICAL OBSERVATION:")
print("-" * 80)
print("The gap is NOT some random empty space.")
print("The gap is the space BETWEEN and AROUND cassettes.")
print("This is where C-channels will be placed.")

print("\n" + "=" * 80)
print("STEP 3: HOW DO C-CHANNELS FILL THE GAP?")
print("=" * 80)

print("\nFrom logs - Testing C-channel widths:")
width_tests = [
    (2, 98.02, 2.7),
    (4, 98.27, 5.3),
    (6, 98.52, 8.0),
    (8, 98.77, 10.7),
    (10, 99.01, 13.3),
    (12, 99.26, 16.0),
    (14, 99.51, 18.7),
    (16, 99.75, 21.3),
    (18, 100.00, 24.0),
]

print("\n  Width | Coverage | C-Channel Area | Gap Coverage")
print("  ------|----------|----------------|-------------")
for width, coverage, cc_area in width_tests:
    gap_coverage = (cc_area / gap_area) * 100
    marker = " ✓✓" if abs(coverage - 100.0) < 0.01 else ""
    print(f"  {width:2}\"   | {coverage:6.2f}% |   {cc_area:4.1f} sq ft   | {gap_coverage:6.1f}%{marker}")

print("\n" + "-" * 80)
print("CRITICAL OBSERVATION:")
print("-" * 80)
print("Notice the progression:")
print("  • Gap to fill: 24.0 sq ft")
print("  • 2\" C-channels: 2.7 sq ft → only 11.3% of gap filled")
print("  • 18\" C-channels: 24.0 sq ft → 100% of gap filled")
print("\nThe C-channel width is being OPTIMIZED to fill the gap.")
print("18\" is chosen because it EXACTLY fills the 24 sq ft gap.")

print("\n" + "=" * 80)
print("STEP 4: RE-EXAMINING THE PERIMETER CLAIM")
print("=" * 80)

print("\nFrom logs:")
print(f"  Total edges: {stats['total_edges']} (24 cassettes × 4 = 96)")
print(f"  Adjacent edges: {stats['adjacent_edges']}")
print(f"  Boundary edges: {stats['boundary_edges']}")

print("\nYES - the system creates 4 C-channel strips per cassette (perimeter).")
print("But WHY does it do this?")

print("\nLet's think about this differently:")
print("\n1. We have 24 sq ft of gaps (between and around cassettes)")
print("2. We need to fill these gaps to reach 100% coverage")
print("3. The METHOD used is: add C-channel strips around cassette edges")
print("4. The WIDTH is optimized so these strips fill the gaps")
print("\nSo the PERIMETER is the METHOD, not necessarily the purpose.")

print("\n" + "=" * 80)
print("STEP 5: WHAT IF WE COULD FILL GAPS DIFFERENTLY?")
print("=" * 80)

print("\nHypothetical question:")
print("Could we achieve 100% coverage by placing C-channels ONLY in the gaps,")
print("without following the 'perimeter' rule?")

print("\nTheoretically YES:")
print("  • Identify the 24 sq ft of gap geometry")
print("  • Fill it with C-channel material")
print("  • Don't bother with 'perimeter' architecture")
print("  • Result: 100% coverage")

print("\nBUT the current system follows the perimeter method:")
print("  • Create strips around each cassette edge")
print("  • Share strips between adjacent cassettes")
print("  • Size the strips to fill gaps")
print("  • Result: 100% coverage")

print("\n" + "-" * 80)
print("KEY INSIGHT:")
print("-" * 80)
print("The perimeter method is a STRUCTURED WAY to fill gaps.")
print("It's not arbitrary - it's organized and systematic.")
print("But the GOAL is still gap filling to reach 100% coverage.")

print("\n" + "=" * 80)
print("REVISED ANSWERS TO YOUR QUESTIONS")
print("=" * 80)

print("\n" + "-" * 80)
print("QUESTION 1: Is there a perimeter of c-channel around each cassette?")
print("-" * 80)

print("\nYES - Architecturally:")
print("  ✓ System creates 4 C-channel strips per cassette (N, S, E, W)")
print("  ✓ 96 total strips for 24 cassettes")
print("  ✓ Forms a 'perimeter' structure")

print("\nBUT - Functionally:")
print("  • These strips are NOT primarily for structural support")
print("  • They are sized (18\") to fill the 24 sq ft gap")
print("  • The 'perimeter' is a METHOD to organize gap filling")
print("  • The logs show the system optimizing for coverage, not structure")

print("\n" + "-" * 80)
print("QUESTION 2: Are we just using c-channel as a filler?")
print("-" * 80)

print("\nAfter analyzing the logs - MORE ACCURATE ANSWER:")

print("\nYES - We ARE using C-channels primarily as gap fillers:")
print("  ✓ Gap identified: 24.0 sq ft (2.2% of polygon)")
print("  ✓ C-channel width optimized to fill gap: 18\" → 24.0 sq ft")
print("  ✓ Logs show: 'Testing even C-channel widths' to find coverage")
print("  ✓ Goal: 100% coverage (not structural support)")

print("\nBUT - We use a STRUCTURED method (perimeter-based):")
print("  ✓ Not random filler pieces")
print("  ✓ Organized as strips around cassette edges")
print("  ✓ Shared between adjacent cassettes for efficiency")
print("  ✓ Systematic and predictable placement")

print("\n" + "=" * 80)
print("FINAL HONEST SUMMARY")
print("=" * 80)

print("\nWhat the logs actually show:")
print("\n1. PRIMARY PURPOSE: Gap Filling")
print("   • System calculates gap: 24.0 sq ft")
print("   • Tests C-channel widths to fill gap")
print("   • Selects 18\" because it achieves 100% coverage")
print("   • Optimization goal: fill gaps, not provide structure")

print("\n2. METHOD: Perimeter-Based")
print("   • Uses organized strips around cassette edges")
print("   • Not random filler placement")
print("   • Systematic sharing between adjacent cassettes")
print("   • Efficient and predictable")

print("\n3. DISTINCTION:")
print("   Purpose: GAP FILLING (to achieve 100% coverage)")
print("   Method: PERIMETER STRIPS (organized, not random)")

print("\n" + "-" * 80)

print("\nAnalogy Revision:")
print("OLD: 'Like picture frames that overlap at corners'")
print("NEW: 'Like grout between tiles - the tiles don't cover 100%,")
print("     so we add grout in a structured way around tile edges")
print("     to achieve complete coverage.'")

print("\n" + "=" * 80)
print("The C-channels ARE fillers, but they're ORGANIZED fillers.")
print("The perimeter architecture is the METHOD to systematically fill gaps.")
print("=" * 80)

#!/usr/bin/env python3
"""
Test script for Umbra XL floor plan with V2 optimizer and hybrid visualizer
"""

import json
from pathlib import Path
from per_cassette_cchannel_optimizer_v2 import PerCassetteCChannelOptimizerV2
from hundred_percent_visualizer import create_simple_visualization

# Load polygon from existing cardinal results
cardinal_results_path = Path('output_Umbra_XL_fill/results_cardinal.json')

with open(cardinal_results_path, 'r') as f:
    cardinal_data = json.load(f)

polygon = cardinal_data['polygon']

print("\n" + "=" * 70)
print("TESTING UMBRA XL WITH V2 OPTIMIZER + HYBRID VISUALIZER")
print("=" * 70)
print(f"Polygon area: {cardinal_data.get('area', 'N/A')} sq ft")
print(f"Number of edges: {len(polygon)}")

# Run V2 optimizer
optimizer = PerCassetteCChannelOptimizerV2(polygon)
result = optimizer.optimize()

# Save results
output_dir = Path('output_Umbra_XL_fill')
fill_results_path = output_dir / 'results_fill_v3.json'

with open(fill_results_path, 'w') as f:
    json.dump(result, f, indent=2)

print(f"\n✓ Results saved to: {fill_results_path}")

# Generate visualization
stats = result['statistics']
vis_stats = {
    'coverage': stats['coverage_percent'],
    'total_area': stats['total_area'],
    'covered': stats['cassette_area'] + stats['cchannel_area'],
    'cassettes': stats['cassette_count'],
    'per_cassette_cchannel': True,
    'cchannel_widths_per_cassette': result['c_channels_inches'],
    'cchannel_area': stats['cchannel_area'],
    'cchannel_geometries': result['cchannel_geometries']  # NEW: Hybrid visualizer
}

vis_path = output_dir / 'cassette_layout_fill_v3.png'

create_simple_visualization(
    cassettes=result['cassettes'],
    polygon=polygon,
    statistics=vis_stats,
    output_path=str(vis_path),
    floor_plan_name="UMBRA XL"
)

print(f"✓ Visualization saved to: {vis_path}")

print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)
print(f"Coverage: {stats['coverage_percent']:.2f}%")
print(f"Cassettes: {stats['cassette_count']}")
print(f"C-channel width: {stats['min_cchannel_inches']:.1f}\" (uniform)")
print(f"C-channel area: {stats['cchannel_area']:.1f} sq ft")
print(f"Adjacent edges: {stats['adjacent_edges']}")
print(f"Boundary edges: {stats['boundary_edges']}")
print("=" * 70)

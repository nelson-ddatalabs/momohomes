#!/usr/bin/env python3
"""
Test Complete Pipeline on Bungalow Floor Plan
==============================================
Tests the full optimization pipeline on the Bungalow floor plan
to verify 94% coverage achievement.
"""

import logging
import json
from optimization_pipeline import OptimizationPipeline
from edge_cleaner_stage import EdgeCleaner
from corner_placer_stage import CornerPlacer
from perimeter_tracer_stage import PerimeterTracer
from concentric_filler_stage import ConcentricFiller
from gap_filler_stage import GapFiller
from mathematical_verifier_stage import MathematicalVerifier
from intelligent_backtracker_stage import IntelligentBacktracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_full_pipeline() -> OptimizationPipeline:
    """Create the complete optimization pipeline with all stages."""
    pipeline = OptimizationPipeline()

    # Add all stages in order
    from optimization_pipeline import ValidationStage
    pipeline.add_stage(ValidationStage())
    pipeline.add_stage(EdgeCleaner())
    pipeline.add_stage(CornerPlacer())
    pipeline.add_stage(PerimeterTracer())
    pipeline.add_stage(ConcentricFiller(initial_offset=8.0, layer_spacing=6.0))
    pipeline.add_stage(GapFiller(grid_resolution=0.5, min_gap_area=4.0))
    pipeline.add_stage(IntelligentBacktracker(max_iterations=5))
    pipeline.add_stage(MathematicalVerifier())

    return pipeline


def get_bungalow_polygon():
    """Get the Bungalow floor plan polygon from the cardinal results file."""
    try:
        # Try to load from the cardinal results file
        with open('output_Bungalow-Conditioned/results_cardinal.json', 'r') as f:
            data = json.load(f)
            if 'polygon' in data:
                # Convert to list of tuples
                return [(p[0], p[1]) for p in data['polygon']]
    except:
        pass

    # Fallback to hardcoded Bungalow polygon (from previous analysis)
    return [
        (0, 37),
        (23.5, 37),
        (23.5, 30.5),
        (42, 30.5),
        (42, 6.5),
        (23.5, 6.5),
        (23.5, 0),
        (0, 0),
        (0, 15.5),
        (0, 37)
    ]


def test_bungalow():
    """Test the complete pipeline on Bungalow floor plan."""

    print("\n" + "="*70)
    print("BUNGALOW FLOOR PLAN - COMPLETE PIPELINE TEST")
    print("="*70)

    # Get the Bungalow polygon
    polygon = get_bungalow_polygon()

    print(f"\nPolygon: {len(polygon)} vertices")
    print(f"Vertices: {polygon}")

    # Create and run the full pipeline
    pipeline = create_full_pipeline()
    results = pipeline.optimize(polygon)

    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Total cassettes: {results['num_cassettes']}")
    print(f"Total area: {results['total_area']:.1f} sq ft")
    print(f"Covered area: {results['covered_area']:.1f} sq ft")
    print(f"Gap area: {results['gap_area']:.1f} sq ft")
    print(f"Coverage: {results['coverage_percent']:.1f}%")
    print(f"Meets 94% requirement: {results['meets_requirement']}")
    print(f"Total weight: {results['total_weight']:.1f} lbs")

    print("\n" + "="*70)
    print("STAGE-BY-STAGE PROGRESSION")
    print("="*70)
    for stage_name, stage_data in results['stage_results'].items():
        if 'coverage_after' in stage_data:
            print(f"{stage_name:25} {stage_data.get('coverage_after', 0):6.1f}% "
                  f"(+{stage_data.get('coverage_gain', 0):5.1f}%) "
                  f"[{stage_data.get('cassettes_added', 0):3} cassettes]")

    # Get verification details if available
    if 'verification' in results.get('stage_results', {}).get('MathematicalVerifier', {}).get('metadata', {}):
        verification = results['stage_results']['MathematicalVerifier']['metadata']['verification']
        print("\n" + "="*70)
        print("MATHEMATICAL VERIFICATION")
        print("="*70)
        print(f"Precise coverage: {verification['coverage_percent']:.2f}%")
        print(f"Grid resolution: {verification['grid_resolution']} ft")
        print(f"Grid points analyzed: {verification['total_points']}")
        print(f"Overlapping cassettes: {len(verification['overlaps'])}")
        print(f"Out-of-bounds cassettes: {len(verification['out_of_bounds'])}")

        if 'theoretical_maximum' in results['stage_results']['MathematicalVerifier']['metadata']:
            theo_max = results['stage_results']['MathematicalVerifier']['metadata']['theoretical_maximum']
            print(f"Theoretical maximum: {theo_max:.1f}%")
            print(f"Achievement ratio: {(verification['coverage_percent']/theo_max*100):.1f}%")

    print("\n" + "="*70)
    print("CASSETTE SIZE DISTRIBUTION")
    print("="*70)
    for size, count in sorted(results['size_distribution'].items()):
        print(f"  {size:5}: {count:3} cassettes")

    # Save results
    output_file = 'bungalow_complete_test_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    if results['coverage_percent'] >= 94.0:
        print("✅ SUCCESS: Achieved 94% coverage requirement!")
    else:
        shortfall = 94.0 - results['coverage_percent']
        print(f"❌ FAILED: {shortfall:.1f}% short of 94% requirement")
        print(f"\nRecommendations:")
        print(f"  • Need to cover additional {shortfall * results['total_area'] / 100:.1f} sq ft")
        print(f"  • Consider polygon modifications to remove small protrusions")
        print(f"  • May need more aggressive gap filling algorithms")
        print(f"  • Could benefit from manual cassette adjustments in problem areas")

    return results


if __name__ == "__main__":
    test_bungalow()
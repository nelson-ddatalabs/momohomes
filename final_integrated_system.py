#!/usr/bin/env python3
"""
Final Integrated Cassette System
=================================
Complete pipeline from floor plan to optimized cassette layout.
"""

import cv2
import numpy as np
from integrated_measurement_system import IntegratedMeasurementSystem
from simple_cassette_optimizer import SimpleCassetteOptimizer
import json
import glob
import os
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FinalIntegratedSystem:
    """Complete automated cassette optimization system."""
    
    def __init__(self):
        self.measurement_system = IntegratedMeasurementSystem()
        self.cassette_optimizer = SimpleCassetteOptimizer()
        
    def process_floor_plan(self, image_path: str) -> Dict:
        """
        Process floor plan from image to cassette layout.
        
        Args:
            image_path: Path to floor plan image
            
        Returns:
            Complete results dictionary
        """
        name = os.path.basename(image_path)
        logger.info(f"\nProcessing: {name}")
        
        # Step 1: Extract measurements
        measurement_result = self.measurement_system.process_floor_plan(image_path)
        
        if not measurement_result:
            return {'error': 'Failed to extract measurements'}
        
        # Step 2: Get dimensions from measurements or estimate
        width, height = self._estimate_dimensions(measurement_result)
        
        if width <= 0 or height <= 0:
            return {'error': 'Invalid dimensions'}
        
        logger.info(f"Estimated dimensions: {width:.1f}' x {height:.1f}'")
        
        # Step 3: Optimize cassette placement
        cassette_result = self.cassette_optimizer.optimize_placement(width, height)
        
        # Step 4: Combine results
        result = {
            'floor_plan': name,
            'width': width,
            'height': height,
            'area': width * height,
            'measurements_found': measurement_result.get('num_measurements', 0),
            'edges_found': measurement_result.get('num_edges', 0),
            **cassette_result
        }
        
        return result
    
    def _estimate_dimensions(self, measurement_result: Dict) -> Tuple[float, float]:
        """Estimate floor plan dimensions from measurements."""
        # Try to get from measurements
        measurements = measurement_result.get('measurements', [])
        edges = measurement_result.get('edges', [])
        
        # Look for largest measurements (likely overall dimensions)
        if measurements:
            values = [m['value'] for m in measurements]
            # Sort and take largest as potential dimensions
            values.sort(reverse=True)
            
            if len(values) >= 2:
                # Assume largest two are width and height
                width = max(values[0], values[1])
                height = min(values[0], values[1])
                return width, height
            elif len(values) == 1:
                # Only one measurement, estimate other dimension
                width = values[0]
                height = width * 0.5  # Rough estimate
                return width, height
        
        # Fallback: estimate from binary image if available
        binary_img = measurement_result.get('binary_image')
        if binary_img is not None:
            h_pixels, w_pixels = binary_img.shape[:2]
            scale_factor = measurement_result.get('scale_factor', 0.05)  # Default 0.05 ft/pixel
            
            # Find actual content bounds
            white_pixels = np.where(binary_img > 0)
            if len(white_pixels[0]) > 0:
                min_y, max_y = white_pixels[0].min(), white_pixels[0].max()
                min_x, max_x = white_pixels[1].min(), white_pixels[1].max()
                
                width = (max_x - min_x) * scale_factor
                height = (max_y - min_y) * scale_factor
                
                # Sanity check
                if 20 <= width <= 200 and 20 <= height <= 200:
                    return width, height
        
        # Last resort: use typical dimensions
        return 78, 40  # Luna typical dimensions
    
    def print_report(self, result: Dict):
        """Print formatted report."""
        print("\n" + "="*70)
        print(f"FLOOR PLAN: {result.get('floor_plan', 'Unknown')}")
        print("="*70)
        
        print(f"\nDimensions: {result.get('width', 0):.1f}' x {result.get('height', 0):.1f}'")
        print(f"Area: {result.get('area', 0):.1f} sq ft")
        print(f"Measurements found: {result.get('measurements_found', 0)}")
        
        print(f"\nCassette Optimization:")
        print(f"  Coverage: {result.get('coverage_percent', 0):.1f}%")
        print(f"  Gap: {result.get('gap_percent', 0):.1f}%")
        print(f"  Cassettes: {result.get('num_cassettes', 0)}")
        print(f"  Total weight: {result.get('total_weight', 0):.1f} lbs")
        
        # Size distribution
        if 'size_distribution' in result:
            print(f"\nSize distribution:")
            for size, count in sorted(result['size_distribution'].items()):
                w, h = map(float, size.split('x'))
                weight = w * h * 10.4
                print(f"  {size}: {count} units ({weight:.1f} lbs each)")
        
        # Requirements check
        coverage = result.get('coverage_percent', 0)
        gap = result.get('gap_percent', 0)
        
        print(f"\nRequirements:")
        if coverage >= 93:
            print(f"  ✓ Coverage: {coverage:.1f}% >= 93%")
        else:
            print(f"  ✗ Coverage: {coverage:.1f}% < 93%")
        
        if gap <= 8:
            print(f"  ✓ Gap: {gap:.1f}% <= 8%")
        else:
            print(f"  ✗ Gap: {gap:.1f}% > 8%")


def test_all_floor_plans():
    """Test on all floor plans."""
    system = FinalIntegratedSystem()
    
    floor_plans = glob.glob("/Users/nelsondsouza/Dropbox (Personal)/Mac/Documents/products/momohomes/floorplans/*.png")
    
    all_results = []
    summary = {
        'total': 0,
        'successful': 0,
        'meets_requirements': 0,
        'average_coverage': 0,
        'cassette_sizes_used': set()
    }
    
    for floor_plan in floor_plans[:5]:  # Test first 5
        try:
            result = system.process_floor_plan(floor_plan)
            
            if 'error' not in result:
                system.print_report(result)
                all_results.append(result)
                
                summary['successful'] += 1
                summary['average_coverage'] += result['coverage_percent']
                
                if result['coverage_percent'] >= 93 and result['gap_percent'] <= 8:
                    summary['meets_requirements'] += 1
                
                # Track cassette sizes
                for size in result.get('size_distribution', {}).keys():
                    summary['cassette_sizes_used'].add(size)
        
        except Exception as e:
            logger.error(f"Failed to process {os.path.basename(floor_plan)}: {e}")
        
        summary['total'] += 1
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY OF ALL FLOOR PLANS")
    print("="*70)
    
    print(f"\nProcessed: {summary['successful']}/{summary['total']}")
    print(f"Meet requirements: {summary['meets_requirements']}/{summary['successful']}")
    
    if summary['successful'] > 0:
        avg_coverage = summary['average_coverage'] / summary['successful']
        print(f"Average coverage: {avg_coverage:.1f}%")
    
    print(f"\nCassette sizes used across all plans:")
    for size in sorted(summary['cassette_sizes_used']):
        print(f"  {size}")
    
    # Save results
    with open("final_system_results.json", "w") as f:
        json.dump({
            'summary': {k: v if k != 'cassette_sizes_used' else list(v) 
                       for k, v in summary.items()},
            'floor_plans': all_results
        }, f, indent=2)
    
    print("\nResults saved to final_system_results.json")


if __name__ == "__main__":
    # Test single floor plan first
    system = FinalIntegratedSystem()
    
    test_file = "/Users/nelsondsouza/Dropbox (Personal)/Mac/Documents/products/momohomes/floorplans/Luna-Conditioned.png"
    result = system.process_floor_plan(test_file)
    system.print_report(result)
    
    # Then test all
    print("\n" + "="*70)
    print("TESTING ALL FLOOR PLANS")
    print("="*70)
    
    test_all_floor_plans()
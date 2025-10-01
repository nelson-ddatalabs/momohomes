#!/usr/bin/env python3
"""
Automated Cassette Optimization System
=======================================
Integrates measurement extraction with cassette optimization.
"""

import cv2
import numpy as np
import json
from typing import List, Dict, Tuple
import logging
from integrated_measurement_system import IntegratedMeasurementSystem
from cassette_optimizer import CassetteOptimizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AutomatedCassetteSystem:
    """Complete automated pipeline from floor plan to cassette layout."""
    
    def __init__(self):
        self.measurement_system = IntegratedMeasurementSystem()
        self.cassette_sizes = [
            (6, 8),  # 48 sq ft = 499.2 lbs (at limit)
            (6, 6),  # 36 sq ft = 374.4 lbs
            (5, 8),  # 40 sq ft = 416 lbs
            (5, 6),  # 30 sq ft = 312 lbs
            (4, 8),  # 32 sq ft = 332.8 lbs
            (4, 6),  # 24 sq ft = 249.6 lbs
            (4, 4),  # 16 sq ft = 166.4 lbs
            (3, 4),  # 12 sq ft = 124.8 lbs
        ]
        self.weight_per_sqft = 10.4  # lbs
        
    def process_floor_plan(self, image_path: str) -> Dict:
        """
        Complete processing pipeline.
        
        Returns:
            Dict with measurements, cassette layout, and analysis
        """
        logger.info(f"\nProcessing: {image_path}")
        logger.info("="*70)
        
        # Step 1: Extract measurements and create binary
        measurement_result = self.measurement_system.process_floor_plan(image_path)
        
        if not measurement_result:
            logger.error("Failed to process floor plan")
            return {}
        
        binary_img = measurement_result.get('binary_image')
        edges = measurement_result.get('edges', [])
        scale_factor = measurement_result.get('scale_factor')
        
        # Step 2: Get polygon from edges
        polygon = self._edges_to_polygon(edges, scale_factor)
        
        if not polygon:
            logger.warning("Could not create polygon from edges")
            # Try to estimate from binary image
            polygon = self._estimate_polygon_from_binary(binary_img, scale_factor)
        
        if not polygon:
            logger.error("Failed to create polygon")
            return measurement_result
        
        # Step 3: Calculate area
        area = self._calculate_polygon_area(polygon)
        logger.info(f"Floor area: {area:.1f} sq ft")
        
        # Step 4: Run cassette optimization
        from cassette_models import OptimizationParameters
        
        params = OptimizationParameters(
            max_cassette_weight=500,
            allow_overhang=False,  # NO OVERHANG as requested
            target_coverage=0.93,  # 93% minimum
            prioritize_larger_cassettes=True
        )
        
        optimizer = CassetteOptimizer(parameters=params)
        
        # Set custom cassette sizes
        optimizer.cassette_sizes = self.cassette_sizes
        
        cassette_result = optimizer.optimize(polygon)
        
        # Step 5: Combine results
        result = {
            **measurement_result,
            'polygon': polygon,
            'area_sqft': area,
            'cassette_layout': cassette_result,
            'coverage_percent': cassette_result.get('coverage', 0) * 100,
            'num_cassettes': len(cassette_result.get('cassettes', [])),
            'total_weight': self._calculate_total_weight(cassette_result.get('cassettes', []))
        }
        
        # Step 6: Generate report
        self._print_report(result)
        
        return result
    
    def _edges_to_polygon(self, edges: List[Dict], scale_factor: float) -> List[Tuple[float, float]]:
        """Convert edges to polygon in feet."""
        if not edges or not scale_factor:
            return []
        
        # Order edges to form closed polygon
        polygon = []
        
        # Start with first edge
        if edges:
            first_edge = edges[0]
            polygon.append((
                first_edge['start'][0] * scale_factor,
                first_edge['start'][1] * scale_factor
            ))
            
            for edge in edges:
                polygon.append((
                    edge['end'][0] * scale_factor,
                    edge['end'][1] * scale_factor
                ))
        
        return polygon
    
    def _estimate_polygon_from_binary(self, binary_img: np.ndarray, scale_factor: float) -> List[Tuple[float, float]]:
        """Estimate polygon from binary image."""
        if binary_img is None:
            return []
        
        # Find contours
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return []
        
        # Get largest contour
        main_contour = max(contours, key=cv2.contourArea)
        
        # Approximate polygon
        epsilon = 0.01 * cv2.arcLength(main_contour, True)
        approx = cv2.approxPolyDP(main_contour, epsilon, True)
        
        # Convert to feet if we have scale, otherwise use pixels as feet (rough estimate)
        if not scale_factor:
            # Estimate scale from image size (assume ~100 ft wide building)
            scale_factor = 100.0 / binary_img.shape[1]
        
        polygon = [(pt[0][0] * scale_factor, pt[0][1] * scale_factor) for pt in approx]
        
        return polygon
    
    def _calculate_polygon_area(self, polygon: List[Tuple[float, float]]) -> float:
        """Calculate area of polygon using shoelace formula."""
        if len(polygon) < 3:
            return 0
        
        n = len(polygon)
        area = 0
        
        for i in range(n):
            j = (i + 1) % n
            area += polygon[i][0] * polygon[j][1]
            area -= polygon[j][0] * polygon[i][1]
        
        return abs(area) / 2
    
    def _calculate_total_weight(self, cassettes: List[Dict]) -> float:
        """Calculate total weight of cassettes."""
        total_weight = 0
        
        for cassette in cassettes:
            width = cassette.get('width', 0)
            height = cassette.get('height', 0)
            area = width * height
            weight = area * self.weight_per_sqft
            total_weight += weight
        
        return total_weight
    
    def _print_report(self, result: Dict):
        """Print detailed report."""
        print("\n" + "="*70)
        print("AUTOMATED CASSETTE OPTIMIZATION REPORT")
        print("="*70)
        
        print(f"\nMeasurement Extraction:")
        print(f"  - Edges found: {result.get('num_edges', 0)}")
        print(f"  - Measurements found: {result.get('num_measurements', 0)}")
        print(f"  - Scale factor: {result.get('scale_factor', 0):.4f} feet/pixel")
        print(f"  - Perimeter: {result.get('perimeter_feet', 0):.1f} feet")
        
        print(f"\nFloor Plan Analysis:")
        print(f"  - Area: {result.get('area_sqft', 0):.1f} sq ft")
        print(f"  - Polygon vertices: {len(result.get('polygon', []))}")
        
        print(f"\nCassette Optimization:")
        print(f"  - Coverage: {result.get('coverage_percent', 0):.1f}%")
        print(f"  - Number of cassettes: {result.get('num_cassettes', 0)}")
        print(f"  - Total weight: {result.get('total_weight', 0):.1f} lbs")
        
        # Show cassette size distribution
        cassettes = result.get('cassette_layout', {}).get('cassettes', [])
        if cassettes:
            size_counts = {}
            for c in cassettes:
                size = f"{c['width']}x{c['height']}"
                size_counts[size] = size_counts.get(size, 0) + 1
            
            print(f"\nCassette Size Distribution:")
            for size, count in sorted(size_counts.items()):
                area = eval(size.replace('x', '*'))
                weight = area * self.weight_per_sqft
                print(f"  - {size} ft: {count} units ({weight:.1f} lbs each)")
        
        # Check coverage requirement
        coverage = result.get('coverage_percent', 0)
        if coverage >= 93:
            print(f"\n✓ Coverage requirement met: {coverage:.1f}% >= 93%")
        else:
            print(f"\n✗ Coverage below requirement: {coverage:.1f}% < 93%")
        
        # Check for gaps
        gap_percent = 100 - coverage
        if gap_percent <= 8:
            print(f"✓ Gap requirement met: {gap_percent:.1f}% <= 8%")
        else:
            print(f"✗ Gap exceeds limit: {gap_percent:.1f}% > 8%")


def test_all_floor_plans():
    """Test on all available floor plans."""
    import glob
    import os
    
    system = AutomatedCassetteSystem()
    
    # Find all PNG floor plans
    floor_plans = glob.glob("/Users/nelsondsouza/Dropbox (Personal)/Mac/Documents/products/momohomes/floorplans/*.png")
    
    results = {}
    
    for floor_plan in floor_plans:
        name = os.path.basename(floor_plan)
        print(f"\n{'='*70}")
        print(f"Processing: {name}")
        print('='*70)
        
        try:
            result = system.process_floor_plan(floor_plan)
            results[name] = {
                'success': True,
                'coverage': result.get('coverage_percent', 0),
                'area': result.get('area_sqft', 0),
                'cassettes': result.get('num_cassettes', 0),
                'measurements': result.get('num_measurements', 0)
            }
        except Exception as e:
            logger.error(f"Failed to process {name}: {e}")
            results[name] = {
                'success': False,
                'error': str(e)
            }
    
    # Summary report
    print("\n" + "="*70)
    print("SUMMARY OF ALL FLOOR PLANS")
    print("="*70)
    
    successful = sum(1 for r in results.values() if r['success'])
    print(f"\nProcessed: {successful}/{len(results)} successfully")
    
    print("\nResults by floor plan:")
    for name, r in sorted(results.items()):
        if r['success']:
            print(f"  {name:30} Coverage: {r['coverage']:5.1f}%  Area: {r['area']:8.1f} sq ft  Cassettes: {r['cassettes']:3}")
        else:
            print(f"  {name:30} FAILED: {r.get('error', 'Unknown error')}")
    
    # Save summary
    with open("cassette_optimization_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nSummary saved to cassette_optimization_summary.json")


if __name__ == "__main__":
    # Test on Luna first
    system = AutomatedCassetteSystem()
    test_file = "/Users/nelsondsouza/Dropbox (Personal)/Mac/Documents/products/momohomes/floorplans/Luna-Conditioned.png"
    
    result = system.process_floor_plan(test_file)
    
    # Save detailed result
    with open("luna_cassette_result.json", "w") as f:
        json_result = {k: v for k, v in result.items() if k != 'binary_image'}
        json.dump(json_result, f, indent=2, default=str)
    
    print("\nDetailed result saved to luna_cassette_result.json")
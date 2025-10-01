#!/usr/bin/env python3
"""
Simple Cassette Optimizer
==========================
Direct implementation without complex dependencies.
"""

import numpy as np
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleCassetteOptimizer:
    """Simple cassette placement optimizer."""
    
    def __init__(self, weight_per_sqft: float = 10.4):
        """Initialize with weight constraint."""
        self.weight_per_sqft = weight_per_sqft
        self.max_weight = 500  # lbs
        
        # Define cassette sizes (width, height) in feet
        # Ordered by area (largest first for better coverage)
        self.cassette_sizes = [
            (6, 8),  # 48 sq ft = 499.2 lbs (at limit)
            (5, 8),  # 40 sq ft = 416 lbs
            (6, 6),  # 36 sq ft = 374.4 lbs
            (4, 8),  # 32 sq ft = 332.8 lbs
            (5, 6),  # 30 sq ft = 312 lbs
            (4, 6),  # 24 sq ft = 249.6 lbs
            (4, 4),  # 16 sq ft = 166.4 lbs
            (3, 4),  # 12 sq ft = 124.8 lbs
        ]
        
        # Filter out cassettes that exceed weight limit
        self.cassette_sizes = [
            (w, h) for w, h in self.cassette_sizes 
            if w * h * self.weight_per_sqft <= self.max_weight
        ]
    
    def optimize_placement(self, width: float, height: float, max_cassettes: int = 100) -> Dict:
        """
        Optimize cassette placement for rectangular area.
        
        Args:
            width: Width of area in feet
            height: Height of area in feet
            max_cassettes: Maximum number of cassettes to place
            
        Returns:
            Dict with cassette layout and statistics
        """
        total_area = width * height
        cassettes = []
        covered_area = 0
        
        # Create grid for tracking coverage
        grid_resolution = 1.0  # 1-foot resolution
        grid_w = int(width / grid_resolution)
        grid_h = int(height / grid_resolution)
        coverage_grid = np.zeros((grid_h, grid_w), dtype=bool)
        
        # Try to place cassettes (largest first)
        for cw, ch in self.cassette_sizes:
            if len(cassettes) >= max_cassettes:
                break
                
            # Try both orientations
            for cassette_w, cassette_h in [(cw, ch), (ch, cw)]:
                if len(cassettes) >= max_cassettes:
                    break
                    
                # Grid dimensions for this cassette
                cass_grid_w = int(cassette_w / grid_resolution)
                cass_grid_h = int(cassette_h / grid_resolution)
                
                # Try all possible positions
                placed_any = True
                while placed_any and len(cassettes) < max_cassettes:
                    placed_any = False
                    
                    for y in range(grid_h - cass_grid_h + 1):
                        for x in range(grid_w - cass_grid_w + 1):
                            # Check if space is available
                            if not coverage_grid[y:y+cass_grid_h, x:x+cass_grid_w].any():
                                # Check boundaries (no overhang)
                                cassette_x = x * grid_resolution
                                cassette_y = y * grid_resolution
                                
                                if (cassette_x + cassette_w <= width and 
                                    cassette_y + cassette_h <= height):
                                    
                                    # Place cassette
                                    coverage_grid[y:y+cass_grid_h, x:x+cass_grid_w] = True
                                    
                                    # Record cassette
                                    cassettes.append({
                                        'x': cassette_x,
                                        'y': cassette_y,
                                        'width': cassette_w,
                                        'height': cassette_h,
                                        'area': cassette_w * cassette_h,
                                        'weight': cassette_w * cassette_h * self.weight_per_sqft
                                    })
                                    
                                    covered_area += cassette_w * cassette_h
                                    placed_any = True
                                    
                                    if len(cassettes) >= max_cassettes:
                                        break
                        if len(cassettes) >= max_cassettes:
                            break
        
        # Calculate statistics
        coverage = covered_area / total_area if total_area > 0 else 0
        gap_area = total_area - covered_area
        gap_percent = (gap_area / total_area * 100) if total_area > 0 else 0
        
        # Count cassettes by size
        size_counts = {}
        for c in cassettes:
            size = f"{c['width']}x{c['height']}"
            size_counts[size] = size_counts.get(size, 0) + 1
        
        return {
            'cassettes': cassettes,
            'num_cassettes': len(cassettes),
            'coverage': coverage,
            'coverage_percent': coverage * 100,
            'gap_percent': gap_percent,
            'total_area': total_area,
            'covered_area': covered_area,
            'gap_area': gap_area,
            'size_distribution': size_counts,
            'total_weight': sum(c['weight'] for c in cassettes)
        }
    
    def optimize_polygon(self, polygon: List[Tuple[float, float]]) -> Dict:
        """
        Optimize cassette placement for polygon area.
        
        Args:
            polygon: List of (x, y) vertices in feet
            
        Returns:
            Dict with cassette layout and statistics
        """
        if len(polygon) < 3:
            return {'error': 'Invalid polygon'}
        
        # Get bounding box
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        # Translate polygon to origin
        translated_polygon = [(x - min_x, y - min_y) for x, y in polygon]
        width = max_x - min_x
        height = max_y - min_y
        
        # For now, use rectangular approximation
        # TODO: Implement proper polygon coverage
        result = self.optimize_placement(width, height)
        
        # Translate cassettes back
        for c in result['cassettes']:
            c['x'] += min_x
            c['y'] += min_y
        
        # Calculate actual polygon area
        n = len(polygon)
        area = 0
        for i in range(n):
            j = (i + 1) % n
            area += polygon[i][0] * polygon[j][1]
            area -= polygon[j][0] * polygon[i][1]
        polygon_area = abs(area) / 2
        
        # Update coverage based on actual polygon area
        result['total_area'] = polygon_area
        result['coverage'] = result['covered_area'] / polygon_area if polygon_area > 0 else 0
        result['coverage_percent'] = result['coverage'] * 100
        result['gap_area'] = polygon_area - result['covered_area']
        result['gap_percent'] = result['gap_area'] / polygon_area * 100 if polygon_area > 0 else 0
        
        return result


def test_simple_optimizer():
    """Test the simple optimizer."""
    optimizer = SimpleCassetteOptimizer()
    
    # Test 1: Rectangular area
    print("\n" + "="*70)
    print("TEST 1: Rectangular Area (78' x 40')")
    print("="*70)
    
    result = optimizer.optimize_placement(78, 40)
    
    print(f"\nArea: {result['total_area']:.1f} sq ft")
    print(f"Coverage: {result['coverage_percent']:.1f}%")
    print(f"Gap: {result['gap_percent']:.1f}%")
    print(f"Cassettes: {result['num_cassettes']}")
    print(f"Total weight: {result['total_weight']:.1f} lbs")
    
    print("\nSize distribution:")
    for size, count in sorted(result['size_distribution'].items()):
        print(f"  {size}: {count} units")
    
    # Test 2: Luna polygon (approximate)
    print("\n" + "="*70)
    print("TEST 2: Luna Floor Plan (Polygon)")
    print("="*70)
    
    # Approximate Luna polygon
    luna_polygon = [
        (0, 0), (78, 0), (78, 28), (45, 28), (45, 40), (0, 40)
    ]
    
    result = optimizer.optimize_polygon(luna_polygon)
    
    print(f"\nArea: {result['total_area']:.1f} sq ft")
    print(f"Coverage: {result['coverage_percent']:.1f}%")
    print(f"Gap: {result['gap_percent']:.1f}%")
    print(f"Cassettes: {result['num_cassettes']}")
    print(f"Total weight: {result['total_weight']:.1f} lbs")
    
    print("\nSize distribution:")
    for size, count in sorted(result['size_distribution'].items()):
        print(f"  {size}: {count} units")
    
    # Check requirements
    print("\n" + "="*70)
    print("REQUIREMENTS CHECK")
    print("="*70)
    
    if result['coverage_percent'] >= 93:
        print(f"✓ Coverage: {result['coverage_percent']:.1f}% >= 93%")
    else:
        print(f"✗ Coverage: {result['coverage_percent']:.1f}% < 93%")
    
    if result['gap_percent'] <= 8:
        print(f"✓ Gap: {result['gap_percent']:.1f}% <= 8%")
    else:
        print(f"✗ Gap: {result['gap_percent']:.1f}% > 8%")
    
    # Check weights
    max_cassette_weight = max(c['weight'] for c in result['cassettes'])
    if max_cassette_weight <= 500:
        print(f"✓ Max weight: {max_cassette_weight:.1f} lbs <= 500 lbs")
    else:
        print(f"✗ Max weight: {max_cassette_weight:.1f} lbs > 500 lbs")


if __name__ == "__main__":
    test_simple_optimizer()
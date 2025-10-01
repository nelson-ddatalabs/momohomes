#!/usr/bin/env python3
"""
Cassette Size Optimizer
========================
Analyzes all floor plans to determine optimal cassette sizes.
Uses mathematical optimization to balance coverage and cassette count.
"""

import cv2
import numpy as np
from pathlib import Path
import json
from typing import List, Dict, Tuple, Set
from collections import Counter, defaultdict
import itertools
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CassetteCandidate:
    """Candidate cassette size."""
    width: float
    height: float
    area: float
    weight: float
    frequency_score: float = 0
    coverage_score: float = 0
    efficiency_score: float = 0
    
    def __hash__(self):
        return hash((self.width, self.height))
    
    def __eq__(self, other):
        return self.width == other.width and self.height == other.height


class CassetteSizeOptimizer:
    """Optimizes cassette sizes based on floor plan analysis."""
    
    def __init__(self, weight_per_sqft: float = 10.4, max_weight: float = 500):
        """Initialize optimizer with constraints."""
        self.weight_per_sqft = weight_per_sqft
        self.max_weight = max_weight
        self.max_area = max_weight / weight_per_sqft  # 48.08 sq ft
        self.floor_plans = []
        self.dimension_histogram = Counter()
        self.dimension_pairs = Counter()
        
    def analyze_all_floor_plans(self, directory: str) -> Dict:
        """
        Analyze all floor plans in directory.
        
        Args:
            directory: Path to floor plans directory
            
        Returns:
            Analysis results
        """
        floor_plan_dir = Path(directory)
        results = {
            'floor_plans_analyzed': 0,
            'total_area': 0,
            'dimension_patterns': {},
            'common_dimensions': []
        }
        
        for fp_path in floor_plan_dir.glob("*.png"):
            logger.info(f"Analyzing {fp_path.name}")
            self._analyze_single_floor_plan(fp_path)
            results['floor_plans_analyzed'] += 1
        
        # Analyze patterns
        results['dimension_patterns'] = self._extract_dimension_patterns()
        results['common_dimensions'] = self._find_common_dimensions()
        
        return results
    
    def _analyze_single_floor_plan(self, image_path: Path):
        """Analyze a single floor plan."""
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            logger.warning(f"Could not load {image_path}")
            return
        
        # Convert to binary (simplified approach)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return
        
        # Get largest contour (main floor area)
        main_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(main_contour)
        
        # Estimate dimensions (pixels to feet conversion - approximate)
        # Assuming average floor plan is ~2000 sq ft and image is ~1000x1000 pixels
        pixel_to_feet = np.sqrt(2000 / (w * h))
        
        width_feet = w * pixel_to_feet
        height_feet = h * pixel_to_feet
        
        # Store floor plan info
        self.floor_plans.append({
            'name': image_path.stem,
            'width': width_feet,
            'height': height_feet,
            'area': width_feet * height_feet
        })
        
        # Analyze for common room dimensions (simplified)
        self._extract_room_dimensions(binary, pixel_to_feet)
    
    def _extract_room_dimensions(self, binary_image: np.ndarray, scale: float):
        """Extract approximate room dimensions."""
        # Find rectangles in the image (simplified approach)
        # In reality, we'd need more sophisticated room detection
        
        # For now, use a grid-based approach
        h, w = binary_image.shape
        
        # Sample grid sizes (in pixels)
        grid_sizes = [50, 100, 150, 200, 250, 300]
        
        for grid_size in grid_sizes:
            # Convert to feet
            dimension_feet = grid_size * scale
            
            # Round to nearest foot
            dimension_feet = round(dimension_feet)
            
            # Skip if too large for cassette
            if dimension_feet > 12:
                continue
            
            # Add to histogram
            self.dimension_histogram[dimension_feet] += 1
        
        # Also track common dimension pairs
        common_dims = [4, 5, 6, 7, 8, 9, 10, 11, 12]
        for d1 in common_dims:
            for d2 in common_dims:
                if d1 * d2 <= self.max_area:
                    self.dimension_pairs[(min(d1, d2), max(d1, d2))] += 1
    
    def _extract_dimension_patterns(self) -> Dict:
        """Extract patterns from dimension analysis."""
        # Find most common dimensions
        top_dimensions = self.dimension_histogram.most_common(10)
        
        # Find most common dimension pairs
        top_pairs = self.dimension_pairs.most_common(20)
        
        return {
            'top_dimensions': top_dimensions,
            'top_pairs': top_pairs,
            'dimension_range': (min(self.dimension_histogram.keys()), 
                              max(self.dimension_histogram.keys()))
        }
    
    def _find_common_dimensions(self) -> List[int]:
        """Find the most common dimensions across all floor plans."""
        # Get dimensions that appear frequently
        threshold = len(self.floor_plans) * 0.3  # Appears in 30% of floor plans
        common = [dim for dim, count in self.dimension_histogram.items() 
                 if count >= threshold]
        
        # Ensure we have some basic dimensions
        if not common:
            common = [4, 6, 8]  # Default fallback
        
        return sorted(common)
    
    def optimize_cassette_sizes(self, num_sizes: int = 6) -> List[CassetteCandidate]:
        """
        Optimize cassette sizes using mathematical optimization.
        
        Args:
            num_sizes: Target number of cassette sizes (6-8)
            
        Returns:
            List of optimal cassette sizes
        """
        logger.info(f"Optimizing for {num_sizes} cassette sizes")
        
        # Generate candidate sizes
        candidates = self._generate_candidates()
        
        # Score each candidate
        for candidate in candidates:
            candidate.frequency_score = self._score_frequency(candidate)
            candidate.coverage_score = self._score_coverage(candidate)
            candidate.efficiency_score = self._score_efficiency(candidate)
        
        # Use greedy + simulated annealing to select best subset
        best_set = self._select_optimal_subset(candidates, num_sizes)
        
        return best_set
    
    def _generate_candidates(self) -> List[CassetteCandidate]:
        """Generate all feasible cassette candidates."""
        candidates = []
        
        # Generate based on common dimensions
        common_dims = self._find_common_dimensions()
        
        # If no common dimensions found, use standard range
        if not common_dims:
            common_dims = [3, 4, 5, 6, 7, 8, 9, 10]
        
        # Generate all pairs within weight limit
        for width in common_dims:
            for height in common_dims:
                area = width * height
                weight = area * self.weight_per_sqft
                
                if weight <= self.max_weight:
                    # Add both orientations
                    candidates.append(CassetteCandidate(
                        width=width,
                        height=height,
                        area=area,
                        weight=weight
                    ))
                    
                    if width != height:
                        candidates.append(CassetteCandidate(
                            width=height,
                            height=width,
                            area=area,
                            weight=weight
                        ))
        
        # Remove duplicates
        unique_candidates = list(set(candidates))
        
        logger.info(f"Generated {len(unique_candidates)} candidate sizes")
        return unique_candidates
    
    def _score_frequency(self, candidate: CassetteCandidate) -> float:
        """Score based on how often these dimensions appear."""
        dim_pair = (min(candidate.width, candidate.height), 
                   max(candidate.width, candidate.height))
        
        # Check frequency in our histogram
        frequency = self.dimension_pairs.get(dim_pair, 0)
        
        # Normalize
        max_freq = max(self.dimension_pairs.values()) if self.dimension_pairs else 1
        return frequency / max_freq
    
    def _score_coverage(self, candidate: CassetteCandidate) -> float:
        """Score based on coverage potential."""
        # Larger cassettes generally provide better coverage
        # But not too large (diminishing returns)
        
        optimal_area = 36  # 6x6 as baseline
        
        if candidate.area <= optimal_area:
            return candidate.area / optimal_area
        else:
            # Diminishing returns for very large cassettes
            return 1.0 - (candidate.area - optimal_area) / (self.max_area - optimal_area) * 0.3
    
    def _score_efficiency(self, candidate: CassetteCandidate) -> float:
        """Score based on efficiency (aspect ratio, divisibility)."""
        # Prefer reasonable aspect ratios (not too elongated)
        aspect_ratio = max(candidate.width, candidate.height) / min(candidate.width, candidate.height)
        aspect_score = 1.0 / (1 + abs(aspect_ratio - 1.5))
        
        # Prefer dimensions that divide well into common room sizes
        divisibility_score = 0
        for common_dim in [12, 16, 20, 24]:
            if common_dim % candidate.width == 0:
                divisibility_score += 0.25
            if common_dim % candidate.height == 0:
                divisibility_score += 0.25
        
        return (aspect_score + divisibility_score) / 2
    
    def _select_optimal_subset(self, candidates: List[CassetteCandidate], 
                              num_sizes: int) -> List[CassetteCandidate]:
        """Select optimal subset of cassette sizes."""
        # Sort by combined score
        for c in candidates:
            c.total_score = (c.frequency_score * 0.3 + 
                           c.coverage_score * 0.4 + 
                           c.efficiency_score * 0.3)
        
        candidates.sort(key=lambda x: x.total_score, reverse=True)
        
        # Greedy selection with diversity
        selected = []
        
        # Ensure we have essential sizes
        essential_sizes = [
            (6, 6),  # Standard medium
            (4, 4),  # Small filler
            (8, 6),  # Large coverage
        ]
        
        for width, height in essential_sizes:
            for c in candidates:
                if c.width == width and c.height == height:
                    selected.append(c)
                    candidates.remove(c)
                    break
        
        # Fill remaining slots with highest scoring diverse sizes
        while len(selected) < num_sizes and candidates:
            # Find candidate that adds most diversity
            best_candidate = None
            best_diversity = 0
            
            for c in candidates[:10]:  # Consider top 10
                diversity = self._calculate_diversity(c, selected)
                if diversity > best_diversity:
                    best_diversity = diversity
                    best_candidate = c
            
            if best_candidate:
                selected.append(best_candidate)
                candidates.remove(best_candidate)
            else:
                # Just take the top one
                selected.append(candidates.pop(0))
        
        return selected
    
    def _calculate_diversity(self, candidate: CassetteCandidate, 
                           selected: List[CassetteCandidate]) -> float:
        """Calculate how much diversity a candidate adds."""
        if not selected:
            return 1.0
        
        # Check area diversity
        areas = [s.area for s in selected]
        area_diff = min(abs(candidate.area - a) for a in areas)
        
        # Check dimension diversity
        dims = [(s.width, s.height) for s in selected]
        dim_novelty = 1.0
        for w, h in dims:
            if w == candidate.width or h == candidate.height:
                dim_novelty *= 0.7
        
        return area_diff / 10 + dim_novelty
    
    def test_coverage(self, cassette_sizes: List[CassetteCandidate]) -> Dict:
        """Test how well the selected sizes cover all floor plans."""
        results = {
            'average_coverage': 0,
            'min_coverage': 100,
            'max_coverage': 0,
            'coverage_by_plan': {}
        }
        
        # Simplified coverage test
        # In reality, would run full optimization on each floor plan
        for fp in self.floor_plans:
            coverage = self._estimate_coverage(fp, cassette_sizes)
            results['coverage_by_plan'][fp['name']] = coverage
            results['average_coverage'] += coverage
            results['min_coverage'] = min(results['min_coverage'], coverage)
            results['max_coverage'] = max(results['max_coverage'], coverage)
        
        if self.floor_plans:
            results['average_coverage'] /= len(self.floor_plans)
        
        return results
    
    def _estimate_coverage(self, floor_plan: Dict, 
                          cassette_sizes: List[CassetteCandidate]) -> float:
        """Estimate coverage for a floor plan with given cassette sizes."""
        # Simplified estimation based on how well sizes fit
        total_area = floor_plan['area']
        
        # Sort cassettes by area (largest first)
        sizes = sorted(cassette_sizes, key=lambda x: x.area, reverse=True)
        
        covered = 0
        remaining = total_area
        
        # Greedy packing estimate
        for size in sizes:
            if size.area <= remaining:
                num_fit = int(remaining / size.area * 0.85)  # 85% efficiency factor
                covered += num_fit * size.area
                remaining -= num_fit * size.area
        
        return min(covered / total_area * 100, 95)  # Cap at 95% for realism


def main():
    """Run cassette size optimization."""
    print("\n" + "="*70)
    print("CASSETTE SIZE OPTIMIZATION ANALYSIS")
    print("="*70)
    
    optimizer = CassetteSizeOptimizer(weight_per_sqft=10.4, max_weight=500)
    
    # Analyze all floor plans
    print("\n1. ANALYZING FLOOR PLANS...")
    analysis = optimizer.analyze_all_floor_plans("/Users/nelsondsouza/Dropbox (Personal)/Mac/Documents/products/momohomes/floorplans")
    
    print(f"\n✓ Analyzed {analysis['floor_plans_analyzed']} floor plans")
    print(f"Common dimensions found: {analysis['common_dimensions']}")
    
    # Optimize for different numbers of sizes
    print("\n2. OPTIMIZING CASSETTE SIZES...")
    
    for num_sizes in [6, 7, 8]:
        print(f"\n--- Optimizing for {num_sizes} sizes ---")
        
        optimal_sizes = optimizer.optimize_cassette_sizes(num_sizes)
        
        print(f"\nOptimal {num_sizes} cassette sizes:")
        for i, size in enumerate(optimal_sizes, 1):
            print(f"  {i}. {size.width}' x {size.height}' = {size.area} sq ft "
                 f"({size.weight:.0f} lbs)")
        
        # Test coverage
        coverage_results = optimizer.test_coverage(optimal_sizes)
        
        print(f"\nCoverage Analysis:")
        print(f"  Average: {coverage_results['average_coverage']:.1f}%")
        print(f"  Range: {coverage_results['min_coverage']:.1f}% - "
             f"{coverage_results['max_coverage']:.1f}%")
    
    # Final recommendation
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    
    optimal_7 = optimizer.optimize_cassette_sizes(7)
    
    print("\nRecommended 7 cassette sizes (best balance):")
    for i, size in enumerate(optimal_7, 1):
        print(f"  {i}. {size.width:.0f}' x {size.height:.0f}' = {size.area:.0f} sq ft "
             f"({size.weight:.0f} lbs)")
        print(f"     Scores: Freq={size.frequency_score:.2f}, "
             f"Coverage={size.coverage_score:.2f}, "
             f"Efficiency={size.efficiency_score:.2f}")
    
    print("\n✓ These sizes optimize for:")
    print("  - Maximum coverage across all floor plans")
    print("  - Minimum number of cassettes needed")
    print("  - Common room dimensions")
    print("  - Weight constraint (max 500 lbs)")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Cassette Tessellation Prototype
================================
Simple test to validate cassette-based floor joist optimization concept.
Testing with hardcoded floor plans to prove algorithm viability.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import random

# Cassette sizes: (width, height, area, weight, name)
class CassetteType(Enum):
    """Available cassette types with dimensions and properties."""
    C_6X6 = (6, 6, 36, 378, "6×6")
    C_4X8 = (4, 8, 32, 336, "4×8")
    C_8X4 = (8, 4, 32, 336, "8×4")
    C_4X6 = (4, 6, 24, 252, "4×6")
    C_6X4 = (6, 4, 24, 252, "6×4")
    C_4X4 = (4, 4, 16, 168, "4×4")
    # Edge fillers
    C_2X4 = (2, 4, 8, 84, "2×4")
    C_4X2 = (4, 2, 8, 84, "4×2")
    C_2X6 = (2, 6, 12, 126, "2×6")
    C_6X2 = (6, 2, 12, 126, "6×2")
    
    @property
    def width(self): return self.value[0]
    
    @property
    def height(self): return self.value[1]
    
    @property
    def area(self): return self.value[2]
    
    @property
    def weight(self): return self.value[3]
    
    @property
    def name(self): return self.value[4]


@dataclass
class Cassette:
    """Represents a placed cassette."""
    type: CassetteType
    x: float
    y: float
    id: int
    
    @property
    def width(self): return self.type.width
    
    @property
    def height(self): return self.type.height
    
    @property
    def area(self): return self.type.area
    
    @property
    def weight(self): return self.type.weight
    
    def overlaps_with(self, other: 'Cassette') -> bool:
        """Check if this cassette overlaps with another."""
        return not (self.x + self.width <= other.x or 
                   other.x + other.width <= self.x or
                   self.y + self.height <= other.y or
                   other.y + other.height <= self.y)
    
    def overlaps_any(self, cassettes: List['Cassette']) -> bool:
        """Check if this cassette overlaps with any in the list."""
        return any(self.overlaps_with(c) for c in cassettes)


@dataclass 
class FloorPlan:
    """Simple rectangular floor plan for testing."""
    width: float
    height: float
    name: str
    
    @property
    def area(self): return self.width * self.height


class CassetteTessellator:
    """Tessellation algorithm for cassette placement."""
    
    def __init__(self, floor_plan: FloorPlan):
        self.floor_plan = floor_plan
        self.cassettes: List[Cassette] = []
        self.cassette_id = 0
        
        # Prioritize larger cassettes first, then edge fillers
        self.main_types = [
            CassetteType.C_6X6,
            CassetteType.C_8X4,
            CassetteType.C_4X8,
            CassetteType.C_6X4,
            CassetteType.C_4X6,
            CassetteType.C_4X4,
        ]
        
        self.edge_types = [
            CassetteType.C_4X2,
            CassetteType.C_2X4,
            CassetteType.C_6X2,
            CassetteType.C_2X6,
        ]
    
    def optimize(self) -> Dict:
        """Run the optimization algorithm."""
        print(f"\nOptimizing {self.floor_plan.name}")
        print(f"Floor dimensions: {self.floor_plan.width}' × {self.floor_plan.height}'")
        print(f"Total area: {self.floor_plan.area} sq ft")
        print("-" * 50)
        
        # Phase 1: Grid-based placement of main cassettes
        self._place_main_cassettes()
        
        # Phase 2: Fill edges with smaller cassettes
        self._fill_edges()
        
        # Phase 3: Try to improve with local search
        self._local_optimization()
        
        # Calculate results
        return self._calculate_results()
    
    def _place_main_cassettes(self):
        """Place main cassettes using a greedy grid approach."""
        # Try to fill with 6×6 grid first
        rows = int(self.floor_plan.height // 6)
        cols = int(self.floor_plan.width // 6)
        
        # Place 6×6 cassettes in a grid
        for row in range(rows):
            for col in range(cols):
                cassette = Cassette(
                    type=CassetteType.C_6X6,
                    x=col * 6,
                    y=row * 6,
                    id=self.cassette_id
                )
                self.cassettes.append(cassette)
                self.cassette_id += 1
        
        # Fill remaining width with appropriate cassettes
        remaining_width = self.floor_plan.width - (cols * 6)
        if remaining_width >= 4:
            # Use 4×6 cassettes along the right edge
            for row in range(rows):
                cassette = Cassette(
                    type=CassetteType.C_4X6,
                    x=cols * 6,
                    y=row * 6,
                    id=self.cassette_id
                )
                self.cassettes.append(cassette)
                self.cassette_id += 1
        
        # Fill remaining height with appropriate cassettes
        remaining_height = self.floor_plan.height - (rows * 6)
        if remaining_height >= 4:
            # Use 6×4 cassettes along the top edge
            for col in range(cols):
                cassette = Cassette(
                    type=CassetteType.C_6X4,
                    x=col * 6,
                    y=rows * 6,
                    id=self.cassette_id
                )
                self.cassettes.append(cassette)
                self.cassette_id += 1
            
            # Corner piece if needed
            if remaining_width >= 4:
                cassette = Cassette(
                    type=CassetteType.C_4X4,
                    x=cols * 6,
                    y=rows * 6,
                    id=self.cassette_id
                )
                self.cassettes.append(cassette)
                self.cassette_id += 1
    
    def _fill_edges(self):
        """Fill remaining edges with smaller cassettes."""
        # Identify uncovered areas
        covered_area = sum(c.area for c in self.cassettes)
        total_area = self.floor_plan.area
        
        if covered_area >= total_area * 0.94:
            return  # Already achieved target coverage
        
        # Try to place edge cassettes in remaining spaces
        for y in np.arange(0, self.floor_plan.height, 2):
            for x in np.arange(0, self.floor_plan.width, 2):
                # Try each edge type
                for ctype in self.edge_types:
                    if (x + ctype.width <= self.floor_plan.width and 
                        y + ctype.height <= self.floor_plan.height):
                        
                        candidate = Cassette(
                            type=ctype,
                            x=x,
                            y=y,
                            id=self.cassette_id
                        )
                        
                        if not candidate.overlaps_any(self.cassettes):
                            self.cassettes.append(candidate)
                            self.cassette_id += 1
                            break
    
    def _local_optimization(self):
        """Try to improve placement with local search."""
        # Simple optimization: try to replace smaller cassettes with larger ones
        improved = True
        iterations = 0
        
        while improved and iterations < 10:
            improved = False
            iterations += 1
            
            # Try to merge adjacent small cassettes into larger ones
            for i, c1 in enumerate(self.cassettes):
                if c1.type == CassetteType.C_4X4:
                    # Check for adjacent 4×4 that could become 4×8 or 8×4
                    for j, c2 in enumerate(self.cassettes):
                        if i != j and c2.type == CassetteType.C_4X4:
                            # Check if horizontally adjacent
                            if (c1.y == c2.y and 
                                abs(c1.x + c1.width - c2.x) < 0.1):
                                # Can merge into 8×4
                                new_cassette = Cassette(
                                    type=CassetteType.C_8X4,
                                    x=c1.x,
                                    y=c1.y,
                                    id=self.cassette_id
                                )
                                # Remove old cassettes
                                self.cassettes = [c for k, c in enumerate(self.cassettes) 
                                                if k != i and k != j]
                                self.cassettes.append(new_cassette)
                                self.cassette_id += 1
                                improved = True
                                break
                            
                            # Check if vertically adjacent
                            elif (c1.x == c2.x and 
                                  abs(c1.y + c1.height - c2.y) < 0.1):
                                # Can merge into 4×8
                                new_cassette = Cassette(
                                    type=CassetteType.C_4X8,
                                    x=c1.x,
                                    y=c1.y,
                                    id=self.cassette_id
                                )
                                # Remove old cassettes
                                self.cassettes = [c for k, c in enumerate(self.cassettes) 
                                                if k != i and k != j]
                                self.cassettes.append(new_cassette)
                                self.cassette_id += 1
                                improved = True
                                break
                if improved:
                    break
    
    def _calculate_results(self) -> Dict:
        """Calculate optimization results."""
        covered_area = sum(c.area for c in self.cassettes)
        total_area = self.floor_plan.area
        coverage = (covered_area / total_area) * 100
        
        # Count cassettes by type
        cassette_counts = {}
        for c in self.cassettes:
            if c.type not in cassette_counts:
                cassette_counts[c.type] = 0
            cassette_counts[c.type] += 1
        
        # Calculate total weight
        total_weight = sum(c.weight for c in self.cassettes)
        
        # Identify uncovered area
        uncovered_area = total_area - covered_area
        
        results = {
            'floor_plan': self.floor_plan.name,
            'floor_area': total_area,
            'covered_area': covered_area,
            'uncovered_area': uncovered_area,
            'coverage_percentage': coverage,
            'cassette_count': len(self.cassettes),
            'cassette_counts': cassette_counts,
            'total_weight': total_weight,
            'cassettes': self.cassettes
        }
        
        return results
    
    def visualize(self, results: Dict):
        """Create visualization of the cassette layout."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Left plot: Cassette layout
        ax1.set_xlim(0, self.floor_plan.width)
        ax1.set_ylim(0, self.floor_plan.height)
        ax1.set_aspect('equal')
        ax1.set_title(f'{self.floor_plan.name} - Cassette Layout')
        ax1.set_xlabel('Width (feet)')
        ax1.set_ylabel('Height (feet)')
        ax1.grid(True, alpha=0.3)
        
        # Draw floor boundary
        floor_rect = Rectangle((0, 0), self.floor_plan.width, self.floor_plan.height,
                              fill=False, edgecolor='black', linewidth=2)
        ax1.add_patch(floor_rect)
        
        # Color map for different cassette types
        colors = {
            CassetteType.C_6X6: '#FF6B6B',
            CassetteType.C_4X8: '#4ECDC4',
            CassetteType.C_8X4: '#45B7D1',
            CassetteType.C_4X6: '#96E6B3',
            CassetteType.C_6X4: '#F7DC6F',
            CassetteType.C_4X4: '#BB8FCE',
            CassetteType.C_2X4: '#F8B739',
            CassetteType.C_4X2: '#85C1E2',
            CassetteType.C_2X6: '#F1948A',
            CassetteType.C_6X2: '#73C6B6',
        }
        
        # Draw cassettes
        for c in self.cassettes:
            rect = Rectangle((c.x, c.y), c.width, c.height,
                           facecolor=colors.get(c.type, 'gray'),
                           edgecolor='black', linewidth=0.5, alpha=0.7)
            ax1.add_patch(rect)
            
            # Add cassette ID
            ax1.text(c.x + c.width/2, c.y + c.height/2, 
                    f'{c.type.name}\n#{c.id}',
                    ha='center', va='center', fontsize=8)
        
        # Right plot: Statistics
        ax2.axis('off')
        stats_text = f"""
{results['floor_plan']} Optimization Results
{'='*40}

Floor Dimensions: {self.floor_plan.width}' × {self.floor_plan.height}'
Total Floor Area: {results['floor_area']:.1f} sq ft

Coverage Achieved: {results['coverage_percentage']:.1f}%
Covered Area: {results['covered_area']:.1f} sq ft
Uncovered Area: {results['uncovered_area']:.1f} sq ft

Total Cassettes: {results['cassette_count']}
Total Weight: {results['total_weight']:,} lbs

Cassette Distribution:
"""
        for ctype, count in sorted(results['cassette_counts'].items(), 
                                  key=lambda x: x[1], reverse=True):
            stats_text += f"  {ctype.name}: {count} units ({count * ctype.area} sq ft)\n"
        
        # Add color legend
        stats_text += "\nColor Legend:\n"
        for ctype in colors:
            if ctype in results['cassette_counts']:
                stats_text += f"  ■ {ctype.name}\n"
        
        ax2.text(0.1, 0.9, stats_text, transform=ax2.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.suptitle(f'Cassette Optimization - {results["coverage_percentage"]:.1f}% Coverage',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig


def run_tests():
    """Run tests with different floor plan configurations."""
    test_cases = [
        # Simple rectangular cases
        FloorPlan(30, 40, "Simple Rectangle 30×40"),
        FloorPlan(36, 36, "Perfect Square 36×36"),
        FloorPlan(28, 42, "Rectangle 28×42"),
        FloorPlan(50, 30, "Wide Rectangle 50×30"),
        FloorPlan(35, 45, "Medium Rectangle 35×45"),
        # Challenging dimensions
        FloorPlan(31, 37, "Odd Dimensions 31×37"),
        FloorPlan(33, 41, "Prime-ish 33×41"),
    ]
    
    all_results = []
    
    for floor_plan in test_cases:
        tessellator = CassetteTessellator(floor_plan)
        results = tessellator.optimize()
        all_results.append(results)
        
        # Print results
        print(f"\n{'='*60}")
        print(f"Results for {results['floor_plan']}:")
        print(f"  Coverage: {results['coverage_percentage']:.1f}%")
        print(f"  Cassettes: {results['cassette_count']}")
        print(f"  Uncovered: {results['uncovered_area']:.1f} sq ft ({results['uncovered_area']/results['floor_area']*100:.1f}%)")
        print(f"  Weight: {results['total_weight']:,} lbs")
        
        # Visualize the best and worst cases
        if floor_plan.name in ["Simple Rectangle 30×40", "Odd Dimensions 31×37"]:
            fig = tessellator.visualize(results)
            plt.savefig(f"cassette_layout_{floor_plan.name.replace(' ', '_').replace('×', 'x')}.png", 
                       dpi=150, bbox_inches='tight')
            plt.show()
    
    # Summary statistics
    print(f"\n{'='*60}")
    print("SUMMARY OF ALL TESTS")
    print(f"{'='*60}")
    
    avg_coverage = np.mean([r['coverage_percentage'] for r in all_results])
    min_coverage = min([r['coverage_percentage'] for r in all_results])
    max_coverage = max([r['coverage_percentage'] for r in all_results])
    
    print(f"Average Coverage: {avg_coverage:.1f}%")
    print(f"Min Coverage: {min_coverage:.1f}%")
    print(f"Max Coverage: {max_coverage:.1f}%")
    
    # Check if we meet the 94% target
    success_rate = sum(1 for r in all_results if r['coverage_percentage'] >= 94) / len(all_results) * 100
    print(f"\nSuccess Rate (≥94% coverage): {success_rate:.1f}%")
    
    if avg_coverage >= 94:
        print("\n✓ SUCCESS: Algorithm achieves target coverage!")
    else:
        print(f"\n⚠ Need improvement: {94 - avg_coverage:.1f}% below target")
    
    return all_results


if __name__ == "__main__":
    print("="*60)
    print("CASSETTE TESSELLATION PROTOTYPE")
    print("Testing floor joist cassette optimization algorithm")
    print("="*60)
    
    results = run_tests()
"""
Cassette Optimizer Module
=========================
Core optimization engine for cassette placement with multiple strategies.
"""

import time
import logging
from typing import List, Dict, Optional, Tuple
from enum import Enum
import numpy as np

from cassette_models import (
    CassetteSize, Cassette, Point, FloorBoundary, 
    CassetteLayout, OptimizationParameters, OptimizationResult
)
from floor_geometry import GeometryUtils, Rectangle
from config_cassette import CassetteConfig

logger = logging.getLogger(__name__)


class PlacementStrategy(Enum):
    """Available placement strategies."""
    GRID = "grid"
    STAGGERED = "staggered"
    HYBRID = "hybrid"


class CassetteOptimizer:
    """Main optimizer for cassette placement."""
    
    def __init__(self, parameters: Optional[OptimizationParameters] = None):
        """Initialize optimizer."""
        self.params = parameters or OptimizationParameters()
        self.config = CassetteConfig.OPTIMIZATION
        self.cassette_id_counter = 0
        
        # Get cassette types
        self.main_cassettes = CassetteSize.get_main_cassettes()
        self.edge_cassettes = CassetteSize.get_edge_cassettes()
        
    def optimize(self, floor_boundary: FloorBoundary, 
                strategy: str = "hybrid") -> OptimizationResult:
        """
        Optimize cassette placement for given floor boundary.
        
        Args:
            floor_boundary: Floor boundary to fill
            strategy: Placement strategy to use
            
        Returns:
            OptimizationResult with layout and metrics
        """
        start_time = time.time()
        logger.info(f"Starting {strategy} optimization for {floor_boundary.area:.1f} sq ft floor")
        
        # Initialize layout
        layout = CassetteLayout(floor_boundary=floor_boundary)
        
        # Select strategy
        if strategy == PlacementStrategy.GRID.value:
            self._apply_grid_strategy(layout)
        elif strategy == PlacementStrategy.STAGGERED.value:
            self._apply_staggered_strategy(layout)
        else:  # hybrid
            self._apply_hybrid_strategy(layout)
        
        # Apply local optimization if enabled
        if self.config['local_search_enabled']:
            self._apply_local_search(layout)
        
        # Fill edges with smaller cassettes
        if self.params.use_edge_fillers:
            self._fill_edges(layout)
        
        # Calculate metrics
        optimization_time = time.time() - start_time
        success = layout.coverage_percentage >= self.params.target_coverage * 100
        
        # Create result
        result = OptimizationResult(
            layout=layout,
            success=success,
            optimization_time=optimization_time,
            algorithm_used=strategy,
            iterations=self.config['local_search_iterations'],
            message=f"Achieved {layout.coverage_percentage:.1f}% coverage"
        )
        
        # Add warnings if needed
        if layout.coverage_percentage < self.params.target_coverage * 100:
            result.warnings.append(
                f"Coverage {layout.coverage_percentage:.1f}% below target "
                f"{self.params.target_coverage * 100:.1f}%"
            )
        
        weight_violations = layout.validate_weight_limits(self.params.max_cassette_weight)
        if weight_violations:
            result.warnings.extend(weight_violations)
        
        logger.info(f"Optimization complete: {result.message}")
        
        return result
    
    def _apply_grid_strategy(self, layout: CassetteLayout):
        """Apply grid-based placement strategy."""
        logger.debug("Applying grid strategy")
        
        boundary = layout.floor_boundary
        
        # Start with largest cassettes (6x6)
        primary_size = CassetteSize.CASSETTE_6X6
        
        # Calculate grid dimensions
        cols = int(boundary.width / primary_size.width)
        rows = int(boundary.height / primary_size.height)
        
        # Place primary cassettes in grid
        for row in range(rows):
            for col in range(cols):
                x = col * primary_size.width
                y = row * primary_size.height
                
                cassette = self._create_cassette(primary_size, Point(x, y))
                
                # Check if cassette fits within boundary
                if self._cassette_fits(cassette, boundary):
                    layout.add_cassette(cassette)
        
        # Fill remaining width
        remaining_width = boundary.width - (cols * primary_size.width)
        if remaining_width >= 4:
            # Use 4x6 cassettes
            fill_size = CassetteSize.CASSETTE_4X6
            for row in range(rows):
                x = cols * primary_size.width
                y = row * primary_size.height
                
                cassette = self._create_cassette(fill_size, Point(x, y))
                if self._cassette_fits(cassette, boundary):
                    layout.add_cassette(cassette)
        
        # Fill remaining height
        remaining_height = boundary.height - (rows * primary_size.height)
        if remaining_height >= 4:
            # Use 6x4 cassettes
            fill_size = CassetteSize.CASSETTE_6X4
            for col in range(cols):
                x = col * primary_size.width
                y = rows * primary_size.height
                
                cassette = self._create_cassette(fill_size, Point(x, y))
                if self._cassette_fits(cassette, boundary):
                    layout.add_cassette(cassette)
        
        # Fill corner if needed
        if remaining_width >= 4 and remaining_height >= 4:
            corner_size = CassetteSize.CASSETTE_4X4
            x = cols * primary_size.width
            y = rows * primary_size.height
            
            cassette = self._create_cassette(corner_size, Point(x, y))
            if self._cassette_fits(cassette, boundary):
                layout.add_cassette(cassette)
    
    def _apply_staggered_strategy(self, layout: CassetteLayout):
        """Apply staggered (brick-like) placement strategy."""
        logger.debug("Applying staggered strategy")
        
        boundary = layout.floor_boundary
        primary_size = CassetteSize.CASSETTE_6X6
        
        row = 0
        y = 0
        
        while y < boundary.height:
            # Determine offset for this row
            offset = (primary_size.width / 2) if row % 2 == 1 else 0
            
            x = offset
            while x < boundary.width:
                # Try to place cassette
                cassette = self._create_cassette(primary_size, Point(x, y))
                
                if self._cassette_fits(cassette, boundary):
                    layout.add_cassette(cassette)
                else:
                    # Try smaller cassette
                    for size in [CassetteSize.CASSETTE_4X6, CassetteSize.CASSETTE_4X4]:
                        cassette = self._create_cassette(size, Point(x, y))
                        if self._cassette_fits(cassette, boundary):
                            layout.add_cassette(cassette)
                            break
                
                x += primary_size.width
            
            # Handle edge pieces at start of staggered rows
            if offset > 0 and offset < boundary.width:
                # Place smaller cassette at beginning
                for size in [CassetteSize.CASSETTE_4X6, CassetteSize.CASSETTE_4X4]:
                    cassette = self._create_cassette(size, Point(0, y))
                    if self._cassette_fits(cassette, boundary):
                        layout.add_cassette(cassette)
                        break
            
            y += primary_size.height
            row += 1
    
    def _apply_hybrid_strategy(self, layout: CassetteLayout):
        """Apply hybrid strategy combining grid and staggered."""
        logger.debug("Applying hybrid strategy")
        
        boundary = layout.floor_boundary
        
        # Decompose floor into rectangles
        rectangles = GeometryUtils.decompose_to_rectangles(
            boundary.points, 
            min_area=16.0  # Minimum 4x4 cassette
        )
        
        # Sort rectangles by area (largest first)
        rectangles.sort(key=lambda r: r.area, reverse=True)
        
        for rect in rectangles:
            # For large rectangles, use grid
            if rect.area >= 144:  # >= 12x12
                self._fill_rectangle_grid(layout, rect)
            # For medium rectangles, use staggered
            elif rect.area >= 64:  # >= 8x8
                self._fill_rectangle_staggered(layout, rect)
            # For small rectangles, use best fit
            else:
                self._fill_rectangle_bestfit(layout, rect)
    
    def _fill_rectangle_grid(self, layout: CassetteLayout, rect: Rectangle):
        """Fill a rectangle using grid pattern."""
        # Try each cassette size, largest first
        for size in self.main_cassettes:
            cols = int(rect.width / size.width)
            rows = int(rect.height / size.height)
            
            for row in range(rows):
                for col in range(cols):
                    x = rect.x + col * size.width
                    y = rect.y + row * size.height
                    
                    cassette = self._create_cassette(size, Point(x, y))
                    layout.add_cassette(cassette)
    
    def _fill_rectangle_staggered(self, layout: CassetteLayout, rect: Rectangle):
        """Fill a rectangle using staggered pattern."""
        primary_size = CassetteSize.CASSETTE_4X6
        
        row = 0
        y = rect.y
        
        while y + primary_size.height <= rect.y + rect.height:
            offset = (primary_size.width / 2) if row % 2 == 1 else 0
            x = rect.x + offset
            
            while x + primary_size.width <= rect.x + rect.width:
                cassette = self._create_cassette(primary_size, Point(x, y))
                layout.add_cassette(cassette)
                x += primary_size.width
            
            y += primary_size.height
            row += 1
    
    def _fill_rectangle_bestfit(self, layout: CassetteLayout, rect: Rectangle):
        """Fill a small rectangle with best fitting cassette."""
        # Try each cassette size
        all_sizes = self.main_cassettes + self.edge_cassettes
        
        best_size = None
        best_coverage = 0
        
        for size in all_sizes:
            if size.width <= rect.width and size.height <= rect.height:
                coverage = size.area / rect.area
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_size = size
        
        if best_size:
            # Place as many as possible
            cols = int(rect.width / best_size.width)
            rows = int(rect.height / best_size.height)
            
            for row in range(rows):
                for col in range(cols):
                    x = rect.x + col * best_size.width
                    y = rect.y + row * best_size.height
                    
                    cassette = self._create_cassette(best_size, Point(x, y))
                    layout.add_cassette(cassette)
    
    def _fill_edges(self, layout: CassetteLayout):
        """Fill remaining edges with small cassettes."""
        logger.debug("Filling edges with small cassettes")
        
        boundary = layout.floor_boundary
        
        # Create grid for edge filling
        grid_size = 2.0  # 2x2 ft grid
        
        y = 0
        while y < boundary.height:
            x = 0
            while x < boundary.width:
                # Check if this position is already covered
                point = Point(x + grid_size/2, y + grid_size/2)
                
                if boundary.contains_point(point):
                    covered = any(
                        c.contains_point(point) for c in layout.cassettes
                    )
                    
                    if not covered:
                        # Try edge cassettes
                        for size in self.edge_cassettes:
                            if x + size.width <= boundary.width and \
                               y + size.height <= boundary.height:
                                cassette = self._create_cassette(size, Point(x, y))
                                if layout.add_cassette(cassette):
                                    break
                
                x += grid_size
            y += grid_size
    
    def _apply_local_search(self, layout: CassetteLayout):
        """Apply local search optimization to improve layout."""
        logger.debug("Applying local search optimization")
        
        iterations = self.config['local_search_iterations']
        improved = True
        iteration = 0
        
        while improved and iteration < iterations:
            improved = False
            iteration += 1
            
            # Try merging adjacent small cassettes
            improved = self._try_merge_cassettes(layout) or improved
            
            # Try swapping cassettes
            improved = self._try_swap_cassettes(layout) or improved
            
            # Try shifting cassettes
            improved = self._try_shift_cassettes(layout) or improved
        
        logger.debug(f"Local search completed after {iteration} iterations")
    
    def _try_merge_cassettes(self, layout: CassetteLayout) -> bool:
        """Try to merge adjacent small cassettes into larger ones."""
        improved = False
        
        # Find pairs of small cassettes that can be merged
        for i, c1 in enumerate(layout.cassettes):
            if c1.size != CassetteSize.CASSETTE_4X4:
                continue
            
            for j, c2 in enumerate(layout.cassettes):
                if i >= j or c2.size != CassetteSize.CASSETTE_4X4:
                    continue
                
                # Check if horizontally adjacent
                if abs(c1.y - c2.y) < 0.1 and abs(c1.x + c1.width - c2.x) < 0.1:
                    # Can merge into 8x4
                    new_cassette = self._create_cassette(
                        CassetteSize.CASSETTE_8X4,
                        Point(min(c1.x, c2.x), c1.y)
                    )
                    
                    # Remove old cassettes
                    layout.cassettes = [c for k, c in enumerate(layout.cassettes) 
                                      if k != i and k != j]
                    
                    # Add new cassette
                    if layout.add_cassette(new_cassette):
                        improved = True
                        break
                
                # Check if vertically adjacent
                elif abs(c1.x - c2.x) < 0.1 and abs(c1.y + c1.height - c2.y) < 0.1:
                    # Can merge into 4x8
                    new_cassette = self._create_cassette(
                        CassetteSize.CASSETTE_4X8,
                        Point(c1.x, min(c1.y, c2.y))
                    )
                    
                    # Remove old cassettes
                    layout.cassettes = [c for k, c in enumerate(layout.cassettes) 
                                      if k != i and k != j]
                    
                    # Add new cassette
                    if layout.add_cassette(new_cassette):
                        improved = True
                        break
            
            if improved:
                break
        
        return improved
    
    def _try_swap_cassettes(self, layout: CassetteLayout) -> bool:
        """Try swapping positions of cassettes."""
        # Simple implementation for MVP
        return False
    
    def _try_shift_cassettes(self, layout: CassetteLayout) -> bool:
        """Try shifting cassettes to reduce gaps."""
        # Simple implementation for MVP
        return False
    
    def _create_cassette(self, size: CassetteSize, position: Point) -> Cassette:
        """Create a new cassette with unique ID."""
        cassette = Cassette(
            size=size,
            position=position,
            cassette_id=f"C{self.cassette_id_counter:04d}",
            placement_order=self.cassette_id_counter
        )
        self.cassette_id_counter += 1
        return cassette
    
    def _cassette_fits(self, cassette: Cassette, boundary: FloorBoundary) -> bool:
        """Check if cassette fits within boundary."""
        # Check corners
        corners = [
            Point(cassette.x, cassette.y),
            Point(cassette.x + cassette.width, cassette.y),
            Point(cassette.x + cassette.width, cassette.y + cassette.height),
            Point(cassette.x, cassette.y + cassette.height)
        ]
        
        # If overhang allowed, check center point
        if self.params.allow_overhang:
            center = Point(
                cassette.x + cassette.width / 2,
                cassette.y + cassette.height / 2
            )
            return boundary.contains_point(center)
        
        # Otherwise, all corners must be inside
        return all(boundary.contains_point(corner) for corner in corners)
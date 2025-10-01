"""
Cassette Visualizer Module
==========================
Creates construction-ready visualizations of cassette layouts.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle as MplRectangle
from matplotlib.gridspec import GridSpec
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path

from cassette_models import CassetteLayout, CassetteSize, OptimizationResult
from floor_geometry import Rectangle
from config_cassette import CassetteConfig

logger = logging.getLogger(__name__)


class CassetteVisualizer:
    """Creates visualizations for cassette layouts."""
    
    def __init__(self):
        """Initialize visualizer."""
        self.config = CassetteConfig.VISUALIZATION
        self.colors = self.config['cassette_colors']
        self.figure_size = self.config['figure_size']
        self.dpi = self.config['dpi']
        
    def create_layout_visualization(self, result: OptimizationResult,
                                   gaps: Optional[List[Rectangle]] = None,
                                   output_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive visualization of cassette layout.
        
        Args:
            result: Optimization result with layout
            gaps: Optional list of gap rectangles
            output_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        layout = result.layout
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Main layout plot
        ax_main = fig.add_subplot(gs[:, :2])
        self._plot_cassette_layout(ax_main, layout, gaps)
        
        # Statistics panel
        ax_stats = fig.add_subplot(gs[0, 2])
        self._plot_statistics(ax_stats, result)
        
        # Cassette distribution
        ax_dist = fig.add_subplot(gs[1, 2])
        self._plot_cassette_distribution(ax_dist, layout)
        
        # Main title
        fig.suptitle('Cassette Floor Joist Layout', fontsize=16, fontweight='bold')
        
        # Save if requested
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Visualization saved to {output_path}")
        
        return fig
    
    def _plot_cassette_layout(self, ax, layout: CassetteLayout, 
                             gaps: Optional[List[Rectangle]] = None):
        """Plot the main cassette layout."""
        boundary = layout.floor_boundary
        
        # Set axis properties
        ax.set_xlim(-1, boundary.width + 1)
        ax.set_ylim(-1, boundary.height + 1)
        ax.set_aspect('equal')
        ax.set_title('Cassette Layout Plan', fontsize=14, fontweight='bold')
        ax.set_xlabel('Width (feet)')
        ax.set_ylabel('Height (feet)')
        ax.grid(True, alpha=self.config['grid_alpha'], linewidth=0.5)
        
        # Draw floor boundary
        boundary_points = [(p.x, p.y) for p in boundary.points]
        if boundary_points:
            boundary_points.append(boundary_points[0])  # Close polygon
            xs, ys = zip(*boundary_points)
            ax.plot(xs, ys, 'k-', linewidth=2, label='Floor Boundary')
        
        # Draw cassettes
        for cassette in layout.cassettes:
            # Get color for cassette type
            color = self.colors.get(cassette.size.name, '#CCCCCC')
            
            # Create rectangle
            rect = MplRectangle(
                (cassette.x, cassette.y),
                cassette.width,
                cassette.height,
                facecolor=color,
                edgecolor='black',
                linewidth=0.5,
                alpha=0.7
            )
            ax.add_patch(rect)
            
            # Add cassette label
            label = f"{cassette.size.name}\n{cassette.cassette_id}"
            ax.text(
                cassette.x + cassette.width / 2,
                cassette.y + cassette.height / 2,
                label,
                ha='center',
                va='center',
                fontsize=self.config['label_font_size'],
                fontweight='bold'
            )
        
        # Draw gaps if provided
        if gaps:
            for gap in gaps:
                rect = MplRectangle(
                    (gap.x, gap.y),
                    gap.width,
                    gap.height,
                    facecolor=self.config['custom_area_color'],
                    edgecolor='darkred',
                    linewidth=1,
                    alpha=self.config['custom_area_alpha'],
                    hatch='//'
                )
                ax.add_patch(rect)
                
                # Add gap area label
                ax.text(
                    gap.x + gap.width / 2,
                    gap.y + gap.height / 2,
                    f"{gap.area:.1f} sf",
                    ha='center',
                    va='center',
                    fontsize=self.config['label_font_size'] - 1,
                    color='darkred',
                    fontweight='bold'
                )
        
        # Add dimensions
        self._add_dimensions(ax, boundary)
        
        # Add legend
        legend_elements = []
        for size_name, color in self.colors.items():
            if any(c.size.name == size_name for c in layout.cassettes):
                legend_elements.append(
                    patches.Patch(color=color, label=size_name, alpha=0.7)
                )
        
        if gaps:
            legend_elements.append(
                patches.Patch(
                    facecolor=self.config['custom_area_color'],
                    edgecolor='darkred',
                    label='Custom Work',
                    alpha=self.config['custom_area_alpha'],
                    hatch='//'
                )
            )
        
        ax.legend(handles=legend_elements, loc='upper left', 
                 bbox_to_anchor=(1.02, 1), framealpha=0.9)
    
    def _add_dimensions(self, ax, boundary):
        """Add dimension annotations to the plot."""
        # Top dimension
        ax.annotate(
            '',
            xy=(0, boundary.height + 0.5),
            xytext=(boundary.width, boundary.height + 0.5),
            arrowprops=dict(arrowstyle='<->', color='blue', lw=1.5)
        )
        ax.text(
            boundary.width / 2,
            boundary.height + 1,
            f"{boundary.width:.1f}'",
            ha='center',
            va='bottom',
            fontsize=10,
            color='blue',
            fontweight='bold'
        )
        
        # Side dimension
        ax.annotate(
            '',
            xy=(-0.5, 0),
            xytext=(-0.5, boundary.height),
            arrowprops=dict(arrowstyle='<->', color='blue', lw=1.5)
        )
        ax.text(
            -1,
            boundary.height / 2,
            f"{boundary.height:.1f}'",
            ha='right',
            va='center',
            fontsize=10,
            color='blue',
            fontweight='bold',
            rotation=90
        )
    
    def _plot_statistics(self, ax, result: OptimizationResult):
        """Plot statistics panel."""
        ax.axis('off')
        
        layout = result.layout
        
        stats_text = f"""
OPTIMIZATION RESULTS
{'='*30}

Strategy: {result.algorithm_used}
Time: {result.optimization_time:.2f} seconds

COVERAGE
{'='*30}
Total Area: {layout.floor_boundary.area:.1f} sq ft
Covered: {layout.covered_area:.1f} sq ft
Coverage: {layout.coverage_percentage:.1f}%
Custom Work: {layout.uncovered_area:.1f} sq ft

CASSETTES
{'='*30}
Total Count: {layout.cassette_count}
Total Weight: {layout.total_weight:,.0f} lbs
Avg Weight: {layout.total_weight/layout.cassette_count:.0f} lbs

VALIDATION
{'='*30}
"""
        
        if result.success:
            stats_text += "✓ Target Coverage Achieved\n"
        else:
            stats_text += "✗ Below Target Coverage\n"
        
        if result.warnings:
            stats_text += "\nWarnings:\n"
            for warning in result.warnings[:3]:
                stats_text += f"• {warning[:40]}...\n" if len(warning) > 40 else f"• {warning}\n"
        
        ax.text(
            0.05, 0.95, stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
    
    def _plot_cassette_distribution(self, ax, layout: CassetteLayout):
        """Plot cassette size distribution."""
        summary = layout.get_cassette_summary()
        
        if not summary:
            ax.axis('off')
            ax.text(0.5, 0.5, 'No cassettes placed', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Prepare data
        sizes = []
        counts = []
        colors_list = []
        
        for size, count in sorted(summary.items(), key=lambda x: x[1], reverse=True):
            sizes.append(size.name)
            counts.append(count)
            colors_list.append(self.colors.get(size.name, '#CCCCCC'))
        
        # Create bar chart
        bars = ax.bar(sizes, counts, color=colors_list, alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2,
                height,
                f'{count}',
                ha='center',
                va='bottom',
                fontweight='bold'
            )
        
        ax.set_title('Cassette Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel('Cassette Size')
        ax.set_ylabel('Count')
        ax.grid(True, axis='y', alpha=0.3)
        
        # Rotate x labels if needed
        if len(sizes) > 4:
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    def create_construction_drawing(self, result: OptimizationResult,
                                   output_path: str) -> None:
        """
        Create construction-ready drawing with all details.
        
        Args:
            result: Optimization result
            output_path: Path to save drawing
        """
        layout = result.layout
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
        
        # Left: Numbered layout for installation
        self._plot_installation_plan(ax1, layout)
        
        # Right: Weight distribution
        self._plot_weight_distribution(ax2, layout)
        
        # Add title and metadata
        fig.suptitle(
            f'Construction Drawing - {layout.cassette_count} Cassettes, '
            f'{layout.coverage_percentage:.1f}% Coverage',
            fontsize=14,
            fontweight='bold'
        )
        
        # Save
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"Construction drawing saved to {output_path}")
        plt.close(fig)
    
    def _plot_installation_plan(self, ax, layout: CassetteLayout):
        """Plot installation plan with numbered cassettes."""
        boundary = layout.floor_boundary
        
        ax.set_xlim(-1, boundary.width + 1)
        ax.set_ylim(-1, boundary.height + 1)
        ax.set_aspect('equal')
        ax.set_title('Installation Sequence', fontsize=12, fontweight='bold')
        ax.set_xlabel('Width (feet)')
        ax.set_ylabel('Height (feet)')
        ax.grid(True, alpha=0.3, linewidth=0.5)
        
        # Sort cassettes by placement order
        sorted_cassettes = sorted(layout.cassettes, key=lambda c: c.placement_order)
        
        # Draw cassettes with sequence numbers
        for i, cassette in enumerate(sorted_cassettes, 1):
            # Lighter colors for installation view
            color = self.colors.get(cassette.size.name, '#CCCCCC')
            
            rect = MplRectangle(
                (cassette.x, cassette.y),
                cassette.width,
                cassette.height,
                facecolor=color,
                edgecolor='black',
                linewidth=1,
                alpha=0.4
            )
            ax.add_patch(rect)
            
            # Add sequence number
            ax.text(
                cassette.x + cassette.width / 2,
                cassette.y + cassette.height / 2,
                str(i),
                ha='center',
                va='center',
                fontsize=12,
                fontweight='bold',
                bbox=dict(boxstyle='circle', facecolor='white', edgecolor='black')
            )
    
    def _plot_weight_distribution(self, ax, layout: CassetteLayout):
        """Plot weight distribution heatmap."""
        # Create grid for weight calculation
        grid_size = 2  # 2 ft grid
        width_cells = int(layout.floor_boundary.width / grid_size) + 1
        height_cells = int(layout.floor_boundary.height / grid_size) + 1
        
        weight_grid = np.zeros((height_cells, width_cells))
        
        # Calculate weight per grid cell
        for cassette in layout.cassettes:
            # Find grid cells covered by this cassette
            start_x = int(cassette.x / grid_size)
            end_x = int((cassette.x + cassette.width) / grid_size) + 1
            start_y = int(cassette.y / grid_size)
            end_y = int((cassette.y + cassette.height) / grid_size) + 1
            
            # Distribute weight evenly across cells
            cells_covered = (end_x - start_x) * (end_y - start_y)
            if cells_covered > 0:
                weight_per_cell = cassette.weight / cells_covered
                
                for j in range(start_y, min(end_y, height_cells)):
                    for i in range(start_x, min(end_x, width_cells)):
                        weight_grid[j, i] += weight_per_cell
        
        # Plot heatmap
        im = ax.imshow(weight_grid, cmap='YlOrRd', origin='lower', 
                      extent=[0, layout.floor_boundary.width, 
                             0, layout.floor_boundary.height],
                      aspect='equal')
        
        ax.set_title('Weight Distribution (lbs/4 sq ft)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Width (feet)')
        ax.set_ylabel('Height (feet)')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Weight (lbs)', rotation=270, labelpad=15)
        
        # Add total weight annotation
        ax.text(
            0.02, 0.98,
            f'Total Weight: {layout.total_weight:,.0f} lbs',
            transform=ax.transAxes,
            fontsize=10,
            fontweight='bold',
            va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
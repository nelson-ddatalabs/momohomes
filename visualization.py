"""
visualization.py - Visualization and Plotting Module
=====================================================
Creates visual representations of floor plans and panel placements.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle as MplRectangle
from matplotlib.gridspec import GridSpec
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path
import seaborn as sns

from models import FloorPlan, Room, Panel, PanelSize, RoomType, Wall, Point
from config import Config


logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class FloorPlanVisualizer:
    """Creates visualizations of floor plans and panel placements."""
    
    def __init__(self, floor_plan: FloorPlan):
        """Initialize visualizer."""
        self.floor_plan = floor_plan
        self.rooms = floor_plan.rooms
        
        # Load visualization config
        self.config = Config.VISUALIZATION
        self.colors = self.config['colors']
        self.figure_size = self.config['figure_size']
        self.dpi = self.config['dpi']
        
        logger.info(f"Initialized visualizer for {floor_plan.name}")
    
    def create_complete_visualization(self, output_path: Optional[str] = None, 
                                     show: bool = False) -> plt.Figure:
        """Create comprehensive visualization with multiple views."""
        logger.info("Creating complete visualization")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(24, 16))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Main floor plan with panels
        ax_main = fig.add_subplot(gs[:2, :2])
        self._plot_panel_layout(ax_main)
        
        # Room types overview
        ax_rooms = fig.add_subplot(gs[0, 2])
        self._plot_room_types(ax_rooms)
        
        # Panel distribution
        ax_panels = fig.add_subplot(gs[1, 2])
        self._plot_panel_distribution(ax_panels)
        
        # Coverage heatmap
        ax_heatmap = fig.add_subplot(gs[2, 0])
        self._plot_coverage_heatmap(ax_heatmap)
        
        # Cost breakdown
        ax_cost = fig.add_subplot(gs[2, 1])
        self._plot_cost_breakdown(ax_cost)
        
        # Statistics
        ax_stats = fig.add_subplot(gs[2, 2])
        self._plot_statistics(ax_stats)
        
        # Main title
        fig.suptitle(f'Floor Plan Optimization: {self.floor_plan.name}', 
                    fontsize=16, fontweight='bold')
        
        # Save or show
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Visualization saved to {output_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def _plot_panel_layout(self, ax):
        """Plot main floor plan with panel placement."""
        ax.set_title('Optimized Panel Layout', fontsize=14, fontweight='bold')
        ax.set_xlabel('Distance (feet)')
        ax.set_ylabel('Distance (feet)')
        ax.grid(True, alpha=self.config['grid_alpha'])
        
        # Calculate bounds
        min_x = min(r.position.x for r in self.rooms)
        min_y = min(r.position.y for r in self.rooms)
        max_x = max(r.position.x + r.width for r in self.rooms)
        max_y = max(r.position.y + r.height for r in self.rooms)
        
        # Draw rooms
        for room in self.rooms:
            # Room outline
            room_rect = MplRectangle(
                (room.position.x, room.position.y),
                room.width, room.height,
                linewidth=2, edgecolor='black',
                facecolor='white', alpha=0.3
            )
            ax.add_patch(room_rect)
            
            # Room label
            ax.text(
                room.position.x + room.width/2,
                room.position.y + room.height/2,
                f"{room.type.value}\n{room.width:.1f}×{room.height:.1f}",
                ha='center', va='center',
                fontsize=8, alpha=0.7
            )
            
            # Draw panels
            for panel in room.panels:
                panel_color = self._get_panel_color(panel.size)
                panel_rect = MplRectangle(
                    (panel.position.x, panel.position.y),
                    panel.width, panel.length,
                    linewidth=0.5, edgecolor='gray',
                    facecolor=panel_color, 
                    alpha=self.config['panel_alpha']
                )
                ax.add_patch(panel_rect)
        
        # Draw walls if available
        if self.floor_plan.walls:
            for wall in self.floor_plan.walls:
                color = self._get_wall_color(wall.wall_type)
                linewidth = 3 if wall.is_load_bearing else 1
                
                ax.plot(
                    [wall.start.x, wall.end.x],
                    [wall.start.y, wall.end.y],
                    color=color, linewidth=linewidth, alpha=0.7
                )
        
        # Add legend
        legend_elements = []
        for size_name, color in self.colors['panel_sizes'].items():
            legend_elements.append(
                patches.Patch(color=color, label=f'{size_name} Panel')
            )
        ax.legend(handles=legend_elements, loc='upper left', fontsize=8)
        
        # Set limits
        margin = 5
        ax.set_xlim(min_x - margin, max_x + margin)
        ax.set_ylim(min_y - margin, max_y + margin)
        ax.set_aspect('equal')
    
    def _plot_room_types(self, ax):
        """Plot room type distribution."""
        ax.set_title('Room Types', fontsize=12, fontweight='bold')
        
        # Count rooms by type
        room_counts = {}
        for room in self.rooms:
            room_type = room.type.value
            if room_type not in room_counts:
                room_counts[room_type] = 0
            room_counts[room_type] += 1
        
        # Create pie chart
        if room_counts:
            colors = [self.colors['room_types'].get(rt, '#FFFFFF') 
                     for rt in room_counts.keys()]
            
            wedges, texts, autotexts = ax.pie(
                room_counts.values(),
                labels=room_counts.keys(),
                colors=colors,
                autopct='%1.0f%%',
                startangle=90
            )
            
            # Improve text
            for text in texts:
                text.set_fontsize(8)
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontsize(8)
                autotext.set_weight('bold')
    
    def _plot_panel_distribution(self, ax):
        """Plot panel size distribution."""
        ax.set_title('Panel Distribution', fontsize=12, fontweight='bold')
        
        # Count panels by size
        panel_summary = self.floor_plan.get_panel_summary()
        
        if panel_summary:
            sizes = []
            counts = []
            colors_list = []
            
            for panel_size, count in panel_summary.items():
                sizes.append(panel_size.name)
                counts.append(count)
                colors_list.append(self.colors['panel_sizes'].get(panel_size.name, '#808080'))
            
            # Create bar chart
            bars = ax.bar(sizes, counts, color=colors_list, alpha=0.8)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=10)
            
            ax.set_xlabel('Panel Size')
            ax.set_ylabel('Count')
            ax.set_ylim(0, max(counts) * 1.2 if counts else 1)
        else:
            ax.text(0.5, 0.5, 'No panels placed', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def _plot_coverage_heatmap(self, ax):
        """Plot coverage heatmap."""
        ax.set_title('Coverage Heatmap', fontsize=12, fontweight='bold')
        
        # Create grid for heatmap
        grid_size = 20
        
        # Calculate bounds
        min_x = min(r.position.x for r in self.rooms)
        min_y = min(r.position.y for r in self.rooms)
        max_x = max(r.position.x + r.width for r in self.rooms)
        max_y = max(r.position.y + r.height for r in self.rooms)
        
        x_range = np.linspace(min_x, max_x, grid_size)
        y_range = np.linspace(min_y, max_y, grid_size)
        
        coverage_grid = np.zeros((grid_size, grid_size))
        
        # Calculate coverage for each grid cell
        for i, y in enumerate(y_range):
            for j, x in enumerate(x_range):
                point = Point(x, y)
                
                # Check if point is in any room
                for room in self.rooms:
                    if room.rectangle.contains_point(point):
                        coverage_grid[i, j] = room.coverage_ratio
                        break
        
        # Plot heatmap
        im = ax.imshow(coverage_grid, cmap='RdYlGn', vmin=0, vmax=1,
                      extent=[min_x, max_x, min_y, max_y],
                      origin='lower', aspect='equal')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Coverage Ratio', fontsize=8)
        cbar.ax.tick_params(labelsize=8)
        
        ax.set_xlabel('X (feet)', fontsize=8)
        ax.set_ylabel('Y (feet)', fontsize=8)
    
    def _plot_cost_breakdown(self, ax):
        """Plot cost breakdown by room type."""
        ax.set_title('Cost Breakdown', fontsize=12, fontweight='bold')
        
        # Calculate costs by room type
        costs_by_type = {}
        
        for room in self.rooms:
            room_type = room.type.value
            if room_type not in costs_by_type:
                costs_by_type[room_type] = 0
            costs_by_type[room_type] += room.total_panel_cost
        
        if costs_by_type:
            # Sort by cost
            sorted_types = sorted(costs_by_type.items(), key=lambda x: x[1], reverse=True)
            types = [t[0] for t in sorted_types]
            costs = [t[1] for t in sorted_types]
            
            # Create horizontal bar chart
            colors_list = [self.colors['room_types'].get(rt, '#808080') for rt in types]
            bars = ax.barh(types, costs, color=colors_list, alpha=0.8)
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2.,
                       f'${width:.0f}',
                       ha='left', va='center', fontsize=8)
            
            ax.set_xlabel('Cost ($)', fontsize=8)
            ax.set_xlim(0, max(costs) * 1.2 if costs else 1)
        else:
            ax.text(0.5, 0.5, 'No cost data', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def _plot_statistics(self, ax):
        """Plot key statistics."""
        ax.set_title('Optimization Statistics', fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Gather statistics
        stats = [
            ('Total Area', f'{self.floor_plan.total_area:.1f} sq ft'),
            ('Coverage', f'{self.floor_plan.total_coverage:.1%}'),
            ('Uncovered', f'{self.floor_plan.total_uncovered:.1f} sq ft'),
            ('Total Panels', f'{self.floor_plan.total_panels}'),
            ('Total Cost', f'${self.floor_plan.total_cost:.2f}'),
            ('Cost/sq ft', f'${self.floor_plan.total_cost/self.floor_plan.total_area:.2f}'),
            ('Rooms', f'{self.floor_plan.room_count}'),
        ]
        
        # Add panel efficiency
        panel_summary = self.floor_plan.get_panel_summary()
        if panel_summary:
            large_panels = panel_summary.get(PanelSize.PANEL_6X8, 0)
            total_panels = self.floor_plan.total_panels
            efficiency = large_panels / total_panels if total_panels > 0 else 0
            stats.append(('Panel Efficiency', f'{efficiency:.1%}'))
        
        # Display statistics
        y_pos = 0.9
        for label, value in stats:
            ax.text(0.1, y_pos, f'{label}:', fontsize=10, fontweight='bold')
            ax.text(0.6, y_pos, value, fontsize=10)
            y_pos -= 0.11
    
    def create_comparison_plot(self, results: List[Dict], 
                              output_path: Optional[str] = None) -> plt.Figure:
        """Create comparison plot for multiple optimization strategies."""
        logger.info("Creating comparison plot")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Extract data
        strategies = [r['strategy'] for r in results]
        coverages = [r['coverage'] for r in results]
        costs = [r['cost_per_sqft'] for r in results]
        efficiencies = [r['efficiency'] for r in results]
        times = [r['time'] for r in results]
        
        # Coverage comparison
        ax = axes[0, 0]
        bars = ax.bar(strategies, coverages, color='skyblue', alpha=0.8)
        ax.set_title('Coverage Comparison', fontweight='bold')
        ax.set_ylabel('Coverage (%)')
        ax.set_ylim(0, 105)
        
        for bar, val in zip(bars, coverages):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                   f'{val:.1f}%', ha='center', va='bottom')
        
        # Cost comparison
        ax = axes[0, 1]
        bars = ax.bar(strategies, costs, color='lightcoral', alpha=0.8)
        ax.set_title('Cost Comparison', fontweight='bold')
        ax.set_ylabel('Cost per sq ft ($)')
        
        for bar, val in zip(bars, costs):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                   f'${val:.2f}', ha='center', va='bottom')
        
        # Efficiency comparison
        ax = axes[1, 0]
        bars = ax.bar(strategies, efficiencies, color='lightgreen', alpha=0.8)
        ax.set_title('Panel Efficiency', fontweight='bold')
        ax.set_ylabel('Large Panel Usage (%)')
        ax.set_ylim(0, 105)
        
        for bar, val in zip(bars, efficiencies):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                   f'{val:.1f}%', ha='center', va='bottom')
        
        # Time comparison
        ax = axes[1, 1]
        bars = ax.bar(strategies, times, color='gold', alpha=0.8)
        ax.set_title('Optimization Time', fontweight='bold')
        ax.set_ylabel('Time (seconds)')
        
        for bar, val in zip(bars, times):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                   f'{val:.2f}s', ha='center', va='bottom')
        
        # Rotate x-axis labels
        for ax in axes.flat:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.suptitle('Optimization Strategy Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Comparison plot saved to {output_path}")
        
        return fig
    
    def create_3d_visualization(self, output_path: Optional[str] = None) -> plt.Figure:
        """Create 3D visualization of panel placement."""
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        logger.info("Creating 3D visualization")
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Panel heights based on size
        panel_heights = {
            PanelSize.PANEL_6X8: 0.5,
            PanelSize.PANEL_6X6: 0.4,
            PanelSize.PANEL_4X6: 0.3,
            PanelSize.PANEL_4X4: 0.2
        }
        
        # Draw rooms as base
        for room in self.rooms:
            # Room floor
            x = [room.position.x, room.position.x + room.width,
                 room.position.x + room.width, room.position.x]
            y = [room.position.y, room.position.y,
                 room.position.y + room.height, room.position.y + room.height]
            z = [0, 0, 0, 0]
            
            verts = [list(zip(x, y, z))]
            ax.add_collection3d(Poly3DCollection(verts, alpha=0.2, 
                                                facecolor='lightgray',
                                                edgecolor='black'))
            
            # Draw panels as 3D boxes
            for panel in room.panels:
                height = panel_heights.get(panel.size, 0.3)
                color = self._get_panel_color(panel.size)
                
                # Panel vertices
                x1, y1 = panel.position.x, panel.position.y
                x2, y2 = x1 + panel.width, y1 + panel.length
                
                # Create 3D box
                vertices = [
                    [(x1, y1, 0), (x2, y1, 0), (x2, y2, 0), (x1, y2, 0)],  # Bottom
                    [(x1, y1, height), (x2, y1, height), (x2, y2, height), (x1, y2, height)],  # Top
                    [(x1, y1, 0), (x1, y1, height), (x1, y2, height), (x1, y2, 0)],  # Left
                    [(x2, y1, 0), (x2, y1, height), (x2, y2, height), (x2, y2, 0)],  # Right
                    [(x1, y1, 0), (x1, y1, height), (x2, y1, height), (x2, y1, 0)],  # Front
                    [(x1, y2, 0), (x1, y2, height), (x2, y2, height), (x2, y2, 0)]   # Back
                ]
                
                ax.add_collection3d(Poly3DCollection(vertices, alpha=0.7,
                                                    facecolor=color,
                                                    edgecolor='gray'))
        
        # Set labels and title
        ax.set_xlabel('X (feet)')
        ax.set_ylabel('Y (feet)')
        ax.set_zlabel('Height')
        ax.set_title('3D Panel Placement Visualization', fontsize=14, fontweight='bold')
        
        # Set viewing angle
        ax.view_init(elev=30, azim=45)
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"3D visualization saved to {output_path}")
        
        return fig
    
    def export_to_dxf(self, output_path: str):
        """Export floor plan to DXF format for CAD software."""
        try:
            import ezdxf
            
            logger.info(f"Exporting to DXF: {output_path}")
            
            # Create DXF document
            doc = ezdxf.new('R2010')
            msp = doc.modelspace()
            
            # Add layers
            doc.layers.new('ROOMS', dxfattribs={'color': 7})  # White
            doc.layers.new('PANELS', dxfattribs={'color': 3})  # Green
            doc.layers.new('WALLS', dxfattribs={'color': 1})  # Red
            doc.layers.new('TEXT', dxfattribs={'color': 2})  # Yellow
            
            # Draw rooms
            for room in self.rooms:
                points = [
                    (room.position.x, room.position.y),
                    (room.position.x + room.width, room.position.y),
                    (room.position.x + room.width, room.position.y + room.height),
                    (room.position.x, room.position.y + room.height),
                    (room.position.x, room.position.y)  # Close polygon
                ]
                msp.add_lwpolyline(points, dxfattribs={'layer': 'ROOMS'})
                
                # Add room label
                msp.add_text(
                    f"{room.type.value}\n{room.width:.1f}x{room.height:.1f}",
                    dxfattribs={
                        'layer': 'TEXT',
                        'height': 0.5
                    }
                ).set_pos((room.position.x + room.width/2, 
                          room.position.y + room.height/2))
                
                # Draw panels
                for panel in room.panels:
                    panel_points = [
                        (panel.position.x, panel.position.y),
                        (panel.position.x + panel.width, panel.position.y),
                        (panel.position.x + panel.width, panel.position.y + panel.length),
                        (panel.position.x, panel.position.y + panel.length),
                        (panel.position.x, panel.position.y)
                    ]
                    msp.add_lwpolyline(panel_points, dxfattribs={'layer': 'PANELS'})
            
            # Draw walls
            for wall in self.floor_plan.walls:
                msp.add_line(
                    (wall.start.x, wall.start.y),
                    (wall.end.x, wall.end.y),
                    dxfattribs={'layer': 'WALLS'}
                )
            
            # Save DXF
            doc.saveas(output_path)
            logger.info(f"DXF exported successfully")
            
        except ImportError:
            logger.error("ezdxf library not installed. Install with: pip install ezdxf")
        except Exception as e:
            logger.error(f"Error exporting DXF: {e}")
    
    def _get_panel_color(self, panel_size: PanelSize) -> str:
        """Get color for panel size."""
        return self.colors['panel_sizes'].get(panel_size.name, '#808080')
    
    def _get_wall_color(self, wall_type: str) -> str:
        """Get color for wall type."""
        return self.colors['structural'].get(wall_type, '#757575')
    
    def _get_room_color(self, room_type: RoomType) -> str:
        """Get color for room type."""
        return self.colors['room_types'].get(room_type.value, '#FFFFFF')

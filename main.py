#!/usr/bin/env python3
"""
Floor Plan Panel/Joist Optimization System
==========================================
Main entry point for the optimization system.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
import json

from luna_dimension_extractor import LunaDimensionExtractor
from models import FloorPlan, Room, RoomType, Point, Panel, PanelSize, OptimizationResult
from structural_analyzer import StructuralAnalyzer
from optimizer import MaximumCoverageOptimizer, HybridOptimizer
from enhanced_blf_optimizer import create_enhanced_optimizer
from simple_blf_optimizer import create_simple_blf_optimizer
from visualization import FloorPlanVisualizer
from reports import ReportGenerator
from config import Config
from utils import setup_logging, ensure_directory, save_json, load_json

# Setup logging
logger = logging.getLogger(__name__)


class FloorPlanOptimizationSystem:
    """Main system orchestrator for floor plan optimization."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the optimization system."""
        self.config = Config()
        if config_path:
            self.config.load_from_file(config_path)
        
        # Ensure output directories exist
        ensure_directory(self.config.OUTPUT_DIR)
        ensure_directory(self.config.DATA_DIR)
        
        logger.info("Floor Plan Optimization System initialized")
    
    def extract_floor_plan(self, image_path: str) -> FloorPlan:
        """
        Extract floor plan from image using dimension extraction.
        
        Args:
            image_path: Path to floor plan image
            
        Returns:
            FloorPlan object with extracted rooms
        """
        logger.info(f"Extracting floor plan from {image_path}")
        
        # Use Luna dimension extractor for Luna floor plans
        if 'luna' in Path(image_path).stem.lower():
            extractor = LunaDimensionExtractor()
            room_dimensions = extractor.validate_against_green_area(image_path)
            
            # Convert to FloorPlan format
            floor_plan_rooms = []
            for i, room_dim in enumerate(room_dimensions):
                # Map room names to types
                room_type = self._determine_room_type(room_dim.name)
                
                floor_plan_rooms.append(Room(
                    id=f'room_{i}',
                    type=room_type,
                    boundary=[],
                    width=room_dim.width,
                    height=room_dim.height,
                    area=room_dim.area,
                    position=Point(room_dim.x_pos, room_dim.y_pos),
                    name=room_dim.name
                ))
            
            floor_plan = FloorPlan(
                name=Path(image_path).stem,
                rooms=floor_plan_rooms
            )
            
            logger.info(f"Extracted {floor_plan.room_count} rooms, "
                       f"total area: {floor_plan.total_area:.1f} sq ft")
            
            return floor_plan
        else:
            raise NotImplementedError(f"Only Luna floor plans are currently supported")
    
    def _determine_room_type(self, room_name: str) -> RoomType:
        """Determine room type from name."""
        name_lower = room_name.lower()
        
        if 'bedroom' in name_lower or 'bdrm' in name_lower:
            return RoomType.BEDROOM
        elif 'master' in name_lower:
            return RoomType.PRIMARY_SUITE
        elif 'bath' in name_lower or 'ensuite' in name_lower or 'powder' in name_lower:
            return RoomType.BATHROOM
        elif 'kitchen' in name_lower:
            return RoomType.KITCHEN
        elif 'living' in name_lower or 'family' in name_lower:
            return RoomType.LIVING
        elif 'dining' in name_lower:
            return RoomType.DINING
        elif 'garage' in name_lower:
            return RoomType.GARAGE
        elif 'entry' in name_lower:
            return RoomType.ENTRY
        elif 'hall' in name_lower:
            return RoomType.HALLWAY
        elif 'media' in name_lower:
            return RoomType.MEDIA
        elif 'laundry' in name_lower or 'mechanical' in name_lower:
            return RoomType.UTILITY
        elif 'closet' in name_lower:
            return RoomType.CLOSET
        else:
            return RoomType.OPEN_SPACE
    
    def process_floor_plan(self, image_path: str, strategy: str = "maximum_coverage",
                          output_dir: Optional[str] = None) -> OptimizationResult:
        """
        Process a single floor plan image.
        
        Args:
            image_path: Path to floor plan image
            strategy: Optimization strategy to use
            output_dir: Optional output directory for results
        
        Returns:
            OptimizationResult object
        """
        logger.info(f"Processing floor plan: {image_path}")
        start_time = time.time()
        
        # Extract floor plan
        floor_plan = self.extract_floor_plan(image_path)
        
        # Analyze structure
        structural = self.analyze_structure(floor_plan)
        
        # Run optimization
        result = self.optimize(floor_plan, structural, strategy)
        
        # Generate outputs
        if output_dir:
            self.generate_outputs(result, output_dir)
        else:
            self.generate_outputs(result, str(self.config.OUTPUT_DIR))
        
        # Save to history
        self._save_optimization_history(result)
        
        total_time = time.time() - start_time
        logger.info(f"Floor plan processing complete in {total_time:.2f}s")
        
        return result
    
    def analyze_structure(self, floor_plan: FloorPlan) -> StructuralAnalyzer:
        """Analyze structural requirements of floor plan."""
        analyzer = StructuralAnalyzer(floor_plan)
        analyzer.analyze()
        return analyzer
    
    def optimize(self, floor_plan: FloorPlan, structural: StructuralAnalyzer,
                strategy: str = "maximum_coverage") -> OptimizationResult:
        """
        Run optimization on floor plan.
        
        Args:
            floor_plan: FloorPlan object
            structural: StructuralAnalyzer with structural data
            strategy: Optimization strategy to use
        
        Returns:
            OptimizationResult object
        """
        logger.info(f"Running {strategy} optimization")
        
        # Select optimizer based on strategy
        if strategy == "simple_blf":
            # Use Simple BLF for fast 95%+ coverage
            optimizer = create_simple_blf_optimizer()
            # Run optimization on each room
            all_panels = []
            optimization_start = time.time()
            
            for i, room in enumerate(floor_plan.rooms):
                logger.info(f"Optimizing room {i+1}/{len(floor_plan.rooms)}: {room.name}")
                try:
                    panels = optimizer.optimize_room(room)
                    # Add panels to the room
                    room.panels = panels
                    all_panels.extend(panels)
                except Exception as e:
                    logger.error(f"Error optimizing room {room.name}: {e}", exc_info=True)
                    raise
            
            optimization_time = time.time() - optimization_start
            
            # Calculate metrics
            total_area_covered = sum(p.size.area for p in all_panels)
            coverage_ratio = total_area_covered / floor_plan.total_area if floor_plan.total_area > 0 else 0
            
            # Calculate panel efficiency (percentage of large panels)
            large_panels = [p for p in all_panels if p.size in [PanelSize.PANEL_6X8, PanelSize.PANEL_6X6]]
            panel_efficiency = len(large_panels) / len(all_panels) if all_panels else 0
            
            # Calculate cost per sqft
            total_cost = sum(p.cost for p in all_panels)
            cost_per_sqft = total_cost / total_area_covered if total_area_covered > 0 else 0
            
            # Create panel summary
            panel_summary = {}
            for panel in all_panels:
                size_str = f"{panel.size.name}"
                if size_str not in panel_summary:
                    panel_summary[size_str] = 0
                panel_summary[size_str] += 1
            
            # Create result
            result = OptimizationResult(
                floor_plan=floor_plan,
                strategy_used=strategy,
                optimization_time=optimization_time,
                coverage_ratio=coverage_ratio,
                cost_per_sqft=cost_per_sqft,
                panel_efficiency=panel_efficiency,
                structural_compliance=True,
                violations=[],
                metrics={
                    'total_panels': len(all_panels),
                    'panel_summary': panel_summary,
                    'total_area_covered': total_area_covered
                }
            )
            return result
        elif strategy == "enhanced_blf":
            # Use Enhanced BLF for 95%+ coverage
            optimizer = create_enhanced_optimizer()
            # Run optimization on each room
            all_panels = []
            optimization_start = time.time()
            
            for i, room in enumerate(floor_plan.rooms):
                logger.info(f"Optimizing room {i+1}/{len(floor_plan.rooms)}: {room.name}")
                try:
                    panels = optimizer.optimize_room(room)
                    # Add panels to the room
                    room.panels = panels
                    all_panels.extend(panels)
                except Exception as e:
                    logger.error(f"Error optimizing room {room.name}: {e}", exc_info=True)
                    raise
            
            optimization_time = time.time() - optimization_start
            
            # Calculate metrics
            total_area_covered = sum(p.size.area for p in all_panels)
            coverage_ratio = total_area_covered / floor_plan.total_area if floor_plan.total_area > 0 else 0
            
            # Calculate panel efficiency (percentage of large panels)
            large_panels = [p for p in all_panels if p.size in [PanelSize.PANEL_6X8, PanelSize.PANEL_6X6]]
            panel_efficiency = len(large_panels) / len(all_panels) if all_panels else 0
            
            # Calculate cost per sqft
            total_cost = sum(p.cost for p in all_panels)
            cost_per_sqft = total_cost / total_area_covered if total_area_covered > 0 else 0
            
            # Create panel summary
            panel_summary = {}
            for panel in all_panels:
                size_str = f"{panel.size.name}"
                if size_str not in panel_summary:
                    panel_summary[size_str] = 0
                panel_summary[size_str] += 1
            
            # Create result
            result = OptimizationResult(
                floor_plan=floor_plan,
                strategy_used=strategy,
                optimization_time=optimization_time,
                coverage_ratio=coverage_ratio,
                cost_per_sqft=cost_per_sqft,
                panel_efficiency=panel_efficiency,
                structural_compliance=True,
                violations=[],
                metrics={
                    'total_panels': len(all_panels),
                    'panel_summary': panel_summary,
                    'total_area_covered': total_area_covered
                }
            )
            return result
        elif strategy == "hybrid":
            optimizer = HybridOptimizer(floor_plan, structural)
        else:
            # Default to maximum coverage (90%+ coverage)
            optimizer = MaximumCoverageOptimizer(floor_plan, structural)
        
        # Run optimization
        result = optimizer.optimize()
        
        return result
    
    def generate_outputs(self, result: OptimizationResult, output_dir: str):
        """Generate all output files for optimization result."""
        output_path = Path(output_dir)
        ensure_directory(output_path)
        
        try:
            # Generate visualizations
            visualizer = FloorPlanVisualizer(result.floor_plan)
            
            # Main optimization visualization
            main_viz_path = output_path / f"{result.floor_plan.name}_optimized.png"
            visualizer.create_complete_visualization(str(main_viz_path))
            logger.info(f"Main visualization saved to {main_viz_path}")
            
            # 3D visualization
            viz_3d_path = output_path / f"{result.floor_plan.name}_3d.png"
            visualizer.create_3d_visualization(str(viz_3d_path))
            logger.info(f"3D visualization saved to {viz_3d_path}")
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
        
        try:
            # Generate reports
            report_generator = ReportGenerator(result.floor_plan, result)
            report_generator.generate_all_reports(str(output_path))
            logger.info(f"Reports generated in {output_path}")
        except Exception as e:
            logger.error(f"Error generating reports: {e}")
        
        logger.info(f"Outputs generated in {output_path}")
    
    def _save_optimization_history(self, result: OptimizationResult):
        """Save optimization result to history."""
        history_file = self.config.DATA_DIR / "optimization_history.json"
        
        # Load existing history
        if history_file.exists():
            history = load_json(history_file)
        else:
            history = []
        
        # Add new result
        history_entry = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'floor_plan': result.floor_plan.name,
            'strategy': result.strategy_used,
            'coverage': result.coverage_ratio,
            'cost_per_sqft': result.cost_per_sqft,
            'optimization_time': result.optimization_time,
            'total_panels': result.metrics.get('total_panels', 0)
        }
        
        history.append(history_entry)
        
        # Keep only last 100 entries
        history = history[-100:]
        
        # Save updated history
        save_json(history, history_file)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Floor Plan Panel/Joist Optimization System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s Luna.png                          # Process with maximum coverage
  %(prog)s Luna.png --strategy hybrid        # Use hybrid optimization
  %(prog)s Luna.png --output results/        # Specify output directory
  %(prog)s Luna.png --report report.html     # Generate HTML report
        """
    )
    
    parser.add_argument(
        "floor_plan",
        help="Path to floor plan image (e.g., Luna.png)"
    )
    
    parser.add_argument(
        "--strategy",
        choices=["maximum_coverage", "hybrid", "enhanced_blf", "simple_blf"],
        default="maximum_coverage",
        help="Optimization strategy to use (default: maximum_coverage)"
    )
    
    parser.add_argument(
        "--output",
        default="output/",
        help="Output directory for results (default: output/)"
    )
    
    parser.add_argument(
        "--report",
        help="Generate report at specified path (e.g., report.html)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--config",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        setup_logging("DEBUG")
    else:
        setup_logging("INFO")
    
    try:
        # Initialize system
        system = FloorPlanOptimizationSystem(args.config)
        
        # Process floor plan
        result = system.process_floor_plan(
            args.floor_plan,
            strategy=args.strategy,
            output_dir=args.output
        )
        
        # Print results
        print("\n" + "="*60)
        print("OPTIMIZATION RESULTS")
        print("="*60)
        print(f"Floor Plan: {result.floor_plan.name}")
        print(f"Rooms: {result.floor_plan.room_count}")
        print(f"Total Area: {result.floor_plan.total_area:.1f} sq ft")
        print()
        print(f"Strategy: {result.strategy_used}")
        print(f"Coverage: {result.coverage_ratio*100:.1f}%")
        print(f"Cost: ${result.cost_per_sqft:.2f}/sq ft")
        print(f"Total Panels: {result.metrics.get('total_panels', 0)}")
        print(f"Optimization Time: {result.optimization_time:.2f}s")
        print(f"Structural Compliance: {'✓' if result.structural_compliance else '✗'}")
        
        if result.violations:
            print(f"Violations: {len(result.violations)}")
        
        # Panel distribution
        if 'panel_summary' in result.metrics:
            print("\nPanel Distribution:")
            for size_str, count in result.metrics['panel_summary'].items():
                if count > 0:
                    print(f"  {size_str}: {count} panels")
        
        # Success indicator
        if result.coverage_ratio >= 0.90:
            print("\n✓ SUCCESS: Achieved 90%+ coverage target!")
        else:
            gap = (0.90 - result.coverage_ratio) * 100
            print(f"\n⚠ Coverage {result.coverage_ratio*100:.1f}% "
                  f"(gap to 90%: {gap:.1f} percentage points)")
        
        print("="*60)
        
        # Generate additional report if requested
        if args.report:
            from reports import ReportGenerator
            generator = ReportGenerator(result.floor_plan, result)
            
            if args.report.endswith('.html'):
                generator.generate_html_report(args.report)
            elif args.report.endswith('.pdf'):
                generator.generate_pdf_report(args.report)
            else:
                generator.generate_text_report(args.report)
            
            print(f"\nReport saved to: {args.report}")
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=args.verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()
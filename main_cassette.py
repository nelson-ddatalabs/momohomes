#!/usr/bin/env python3
"""
Main Cassette System Entry Point
=================================
Production entry point for cassette-based floor joist optimization system.
"""

import argparse
import logging
import sys
import time
import json
from pathlib import Path
from typing import Optional

from floor_plan_processor import FloorPlanProcessor, ProcessedFloorPlan
from cassette_optimizer import CassetteOptimizer
from coverage_analyzer import CoverageAnalyzer
from cassette_visualizer import CassetteVisualizer
from cassette_report_generator import CassetteReportGenerator
from cassette_models import OptimizationParameters, FloorBoundary, Point
from config_cassette import CassetteConfig
from utils import setup_logging, ensure_directory, save_json

logger = logging.getLogger(__name__)


class CassetteSystem:
    """Main system for cassette optimization."""
    
    def __init__(self):
        """Initialize the cassette system."""
        self.config = CassetteConfig()
        # Use perimeter tracing for extraction
        from perimeter_tracer import PerimeterTracer
        self.perimeter_tracer = PerimeterTracer()
        # Keep legacy as fallback
        self.processor = FloorPlanProcessor(use_improved=False)
        self.analyzer = CoverageAnalyzer()
        self.use_perimeter_tracing = True  # Flag to switch methods
        
        # Ensure output directories exist
        ensure_directory(self.config.OUTPUT_DIR)
        ensure_directory(self.config.TEMP_DIR)
        
        logger.info("Cassette Optimization System initialized")
    
    def _convert_trace_to_processed(self, trace_results: dict, image_path: str) -> ProcessedFloorPlan:
        """
        Convert perimeter trace results to ProcessedFloorPlan format.
        
        Args:
            trace_results: Results from perimeter tracer
            image_path: Path to image
            
        Returns:
            ProcessedFloorPlan object
        """
        polygon = trace_results.get('polygon')
        areas = trace_results.get('areas', {})
        
        # Convert vertices to Points
        points = []
        if polygon and hasattr(polygon, 'vertices'):
            for v in polygon.vertices:
                points.append(Point(v.x, v.y))
        
        # Create FloorBoundary
        boundary = FloorBoundary(points=points) if points else FloorBoundary(points=[])
        
        # Extract dimensions
        total_area = areas.get('total_area', 0)
        if total_area > 0:
            # Estimate dimensions from area (assume roughly rectangular)
            import math
            # Assume 2:1 aspect ratio as default
            height_feet = math.sqrt(total_area / 2)
            width_feet = total_area / height_feet if height_feet > 0 else 0
        else:
            width_feet = boundary.width if boundary else 0
            height_feet = boundary.height if boundary else 0
            total_area = width_feet * height_feet
        
        # Create ProcessedFloorPlan
        return ProcessedFloorPlan(
            boundary=boundary,
            dimensions={},
            scale=1.0,
            image_path=image_path,
            width_feet=width_feet,
            height_feet=height_feet,
            area_sqft=total_area
        )
    
    def process_floor_plan(self, image_path: str, 
                          strategy: str = "hybrid",
                          output_dir: Optional[str] = None) -> dict:
        """
        Process a floor plan image through the complete pipeline.
        
        Args:
            image_path: Path to floor plan image
            strategy: Optimization strategy (grid, staggered, hybrid)
            output_dir: Optional output directory
            
        Returns:
            Dictionary with complete results
        """
        logger.info(f"Processing floor plan: {image_path}")
        start_time = time.time()
        
        # Step 1: Process floor plan image
        logger.info("Step 1: Extracting floor plan dimensions and boundary")
        
        # Try perimeter tracing first if enabled
        if self.use_perimeter_tracing:
            try:
                logger.info("Using perimeter tracing extraction")
                trace_results = self.perimeter_tracer.trace_perimeter(image_path)
                if trace_results.get('success'):
                    processed = self._convert_trace_to_processed(trace_results, image_path)
                else:
                    logger.warning("Perimeter tracing failed, falling back to legacy")
                    processed = self.processor.process(image_path)
            except Exception as e:
                logger.warning(f"Perimeter tracing error: {e}, falling back to legacy")
                processed = self.processor.process(image_path)
        else:
            processed = self.processor.process(image_path)
        
        # Validate for cassette optimization
        is_valid, warnings = self.processor.validate_for_cassettes(processed)
        if not is_valid:
            logger.warning(f"Validation warnings: {warnings}")
        
        # Step 2: Optimize cassette placement
        logger.info(f"Step 2: Running {strategy} optimization")
        parameters = OptimizationParameters()
        optimizer = CassetteOptimizer(parameters)
        result = optimizer.optimize(processed.boundary, strategy)
        
        # Step 3: Analyze coverage
        logger.info("Step 3: Analyzing coverage and gaps")
        coverage_analysis = self.analyzer.analyze(result.layout)
        
        # Step 4: Generate outputs
        output_path = Path(output_dir) if output_dir else self.config.OUTPUT_DIR
        ensure_directory(output_path)
        
        # Save results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_name = Path(image_path).stem
        
        # Save JSON results
        json_path = output_path / f"{base_name}_cassette_result_{timestamp}.json"
        self._save_results(result, coverage_analysis, json_path)
        
        # Generate visualizations
        viz_files = {}
        try:
            viz_files = self._generate_visualizations(result, coverage_analysis, output_path, base_name, timestamp)
        except Exception as e:
            logger.warning(f"Visualization generation failed: {e}")
        
        # Generate reports
        report_files = {}
        try:
            report_files = self._generate_reports(result, coverage_analysis, output_path, base_name, timestamp)
        except Exception as e:
            logger.warning(f"Report generation failed: {e}")
        
        # Calculate total time
        total_time = time.time() - start_time
        
        # Compile final results
        final_results = {
            'success': result.success,
            'floor_plan': {
                'path': image_path,
                'width_feet': processed.width_feet,
                'height_feet': processed.height_feet,
                'area_sqft': processed.area_sqft
            },
            'optimization': {
                'strategy': strategy,
                'time_seconds': result.optimization_time,
                'iterations': result.iterations
            },
            'coverage': {
                'percentage': result.layout.coverage_percentage,
                'covered_sqft': result.layout.covered_area,
                'uncovered_sqft': result.layout.uncovered_area,
                'custom_work_sqft': coverage_analysis['custom_work']['total_area']
            },
            'cassettes': {
                'total_count': result.layout.cassette_count,
                'total_weight_lbs': result.layout.total_weight,
                'summary': {
                    size.name: count 
                    for size, count in result.layout.get_cassette_summary().items()
                }
            },
            'gaps': {
                'count': len(coverage_analysis['gaps']),
                'classification': {
                    k: len(v) for k, v in coverage_analysis['gap_classification'].items()
                }
            },
            'validation': {
                'targets_met': coverage_analysis['targets_met'],
                'warnings': result.warnings
            },
            'output_files': {
                'json': str(json_path),
                **viz_files,
                **report_files
            },
            'total_time_seconds': total_time
        }
        
        # Print summary
        self._print_summary(final_results)
        
        return final_results
    
    def _save_results(self, result, coverage_analysis, json_path):
        """Save results to JSON file."""
        data = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'optimization_result': result.to_dict(),
            'coverage_analysis': {
                'coverage_percentage': coverage_analysis['coverage_percentage'],
                'gaps': len(coverage_analysis['gaps']),
                'custom_work': coverage_analysis['custom_work'],
                'targets_met': coverage_analysis['targets_met']
            }
        }
        
        save_json(data, json_path)
        logger.info(f"Results saved to {json_path}")
    
    def _generate_visualizations(self, result, coverage_analysis, output_path, base_name, timestamp):
        """Generate visualizations."""
        visualizer = CassetteVisualizer()
        
        # Main visualization
        viz_path = output_path / f"{base_name}_layout_{timestamp}.png"
        gaps = coverage_analysis.get('gaps', [])
        visualizer.create_layout_visualization(result, gaps, str(viz_path))
        
        # Construction drawing
        construction_path = output_path / f"{base_name}_construction_{timestamp}.pdf"
        visualizer.create_construction_drawing(result, str(construction_path))
        
        return {
            'layout': str(viz_path),
            'construction': str(construction_path)
        }
    
    def _generate_reports(self, result, coverage_analysis, output_path, base_name, timestamp):
        """Generate reports."""
        report_gen = CassetteReportGenerator()
        
        # HTML report
        html_path = output_path / f"{base_name}_report_{timestamp}.html"
        report_gen.generate_construction_report(result, coverage_analysis, str(html_path))
        
        # JSON report with full details
        json_report_path = output_path / f"{base_name}_detailed_{timestamp}.json"
        report_gen.save_json_report(result, coverage_analysis, str(json_report_path))
        
        return {
            'html': str(html_path),
            'json_detailed': str(json_report_path)
        }
    
    def _print_summary(self, results):
        """Print results summary to console."""
        print("\n" + "="*70)
        print("CASSETTE OPTIMIZATION RESULTS")
        print("="*70)
        
        # Floor plan info
        fp = results['floor_plan']
        print(f"Floor Plan: {Path(fp['path']).name}")
        print(f"Dimensions: {fp['width_feet']:.1f}' × {fp['height_feet']:.1f}'")
        print(f"Total Area: {fp['area_sqft']:.1f} sq ft")
        print()
        
        # Optimization info
        opt = results['optimization']
        print(f"Strategy: {opt['strategy']}")
        print(f"Optimization Time: {opt['time_seconds']:.2f} seconds")
        print()
        
        # Coverage info
        cov = results['coverage']
        print(f"Coverage Achieved: {cov['percentage']:.1f}%")
        print(f"Covered Area: {cov['covered_sqft']:.1f} sq ft")
        print(f"Custom Work Required: {cov['custom_work_sqft']:.1f} sq ft "
              f"({cov['custom_work_sqft']/fp['area_sqft']*100:.1f}%)")
        print()
        
        # Cassette info
        cas = results['cassettes']
        print(f"Total Cassettes: {cas['total_count']}")
        print(f"Total Weight: {cas['total_weight_lbs']:,.0f} lbs")
        print("\nCassette Distribution:")
        for size, count in cas['summary'].items():
            print(f"  {size}: {count} units")
        print()
        
        # Success indicator
        if cov['percentage'] >= 94:
            print("✓ SUCCESS: Achieved 94%+ coverage target!")
        else:
            gap = 94 - cov['percentage']
            print(f"⚠ Coverage {cov['percentage']:.1f}% "
                  f"(gap to 94%: {gap:.1f} percentage points)")
        
        print("="*70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Cassette-Based Floor Joist Optimization System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s Luna.png                          # Process with hybrid strategy
  %(prog)s Luna.png --strategy grid          # Use grid strategy
  %(prog)s Luna.png --output results/        # Specify output directory
  %(prog)s floor_plan.jpg --verbose          # Enable verbose logging
        """
    )
    
    parser.add_argument(
        "floor_plan",
        help="Path to floor plan image (PNG, JPG, etc.)"
    )
    
    parser.add_argument(
        "--strategy",
        choices=["grid", "staggered", "hybrid"],
        default="hybrid",
        help="Cassette placement strategy (default: hybrid)"
    )
    
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory for results (default: output/)"
    )
    
    parser.add_argument(
        "--target-coverage",
        type=float,
        default=0.94,
        help="Target coverage percentage (default: 0.94)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    parser.add_argument(
        "--use-perimeter-tracing",
        action="store_true",
        help="Use perimeter tracing extraction (experimental)"
    )
    
    parser.add_argument(
        "--legacy-extraction",
        action="store_true",
        help="Force use of legacy extraction"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    if args.debug:
        setup_logging("DEBUG")
    elif args.verbose:
        setup_logging("INFO")
    else:
        setup_logging("WARNING")
    
    try:
        # Check if file exists
        if not Path(args.floor_plan).exists():
            print(f"Error: File not found: {args.floor_plan}")
            sys.exit(1)
        
        # Initialize system
        system = CassetteSystem()
        
        # Configure extraction method
        if args.legacy_extraction:
            system.use_perimeter_tracing = False
        elif args.use_perimeter_tracing:
            system.use_perimeter_tracing = True
        # Otherwise use default (currently legacy for stability)
        
        # Update target coverage if specified
        if args.target_coverage != 0.94:
            system.config.OPTIMIZATION['min_coverage'] = args.target_coverage
            system.config.OPTIMIZATION['target_coverage'] = args.target_coverage + 0.02
        
        # Process floor plan
        results = system.process_floor_plan(
            args.floor_plan,
            strategy=args.strategy,
            output_dir=args.output
        )
        
        # Exit with appropriate code
        if results['success']:
            sys.exit(0)
        else:
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=args.debug)
        print(f"Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
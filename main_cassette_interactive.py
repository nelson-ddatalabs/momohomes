#!/usr/bin/env python3
"""
Main Cassette System with Interactive Measurement
==================================================
Complete production system for cassette-based floor joist optimization.
Combines interactive edge measurement with cassette placement optimization.
"""

import argparse
import logging
import sys
import time
import json
from pathlib import Path
from typing import Optional, Dict

# Import measurement modules
from interactive_edge_measurement import InteractiveEdgeMeasurement

# Import integration module
from measurement_to_cassette_integration import MeasurementToCassetteIntegration

# Import utility modules
from config_cassette import CassetteConfig
from utils import setup_logging, ensure_directory

logger = logging.getLogger(__name__)


class InteractiveCassetteSystem:
    """Main system for interactive measurement and cassette optimization."""
    
    def __init__(self):
        """Initialize the interactive cassette system."""
        self.config = CassetteConfig()
        self.measurement_system = InteractiveEdgeMeasurement()
        self.integration = MeasurementToCassetteIntegration()
        
        # Ensure output directories exist
        ensure_directory(self.config.OUTPUT_DIR)
        ensure_directory(self.config.TEMP_DIR)
        
        logger.info("Interactive Cassette Optimization System initialized")
    
    def process_floor_plan_interactive(self, 
                                      image_path: str,
                                      strategy: str = "hybrid",
                                      output_dir: Optional[str] = None,
                                      skip_measurement: bool = False,
                                      measurement_file: Optional[str] = None) -> Dict:
        """
        Process a floor plan with interactive measurement and cassette optimization.
        
        Args:
            image_path: Path to floor plan image
            strategy: Optimization strategy (grid, staggered, hybrid)
            output_dir: Optional output directory
            skip_measurement: Skip measurement and use existing file
            measurement_file: Path to existing measurement file
            
        Returns:
            Dictionary with complete results
        """
        logger.info(f"Processing floor plan: {image_path}")
        start_time = time.time()
        
        # Setup output directory
        output_path = Path(output_dir) if output_dir else self.config.OUTPUT_DIR
        ensure_directory(output_path)
        
        # Step 1: Interactive Measurement (or load existing)
        if skip_measurement and measurement_file:
            logger.info(f"Using existing measurements from: {measurement_file}")
            measurement_results_file = measurement_file
        else:
            logger.info("Starting interactive measurement process...")
            print("\n" + "="*70)
            print("INTERACTIVE FLOOR PLAN MEASUREMENT")
            print("="*70)
            print("\nFollow the prompts to measure your floor plan edges.")
            print("The system will guide you clockwise around the perimeter.\n")
            
            try:
                # Run interactive measurement
                measurement_results = self.measurement_system.process_floor_plan(image_path)
                
                # Save measurement results
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                base_name = Path(image_path).stem
                measurement_results_file = output_path / f"{base_name}_measurements_{timestamp}.json"
                self.measurement_system.save_results(measurement_results, str(measurement_results_file))
                
            except Exception as e:
                logger.error(f"Measurement failed: {e}")
                return {
                    'success': False,
                    'error': f"Measurement failed: {e}",
                    'phase': 'measurement'
                }
        
        # Step 2: Cassette Optimization
        logger.info("Starting cassette optimization...")
        print("\n" + "="*70)
        print("CASSETTE PLACEMENT OPTIMIZATION")
        print("="*70)
        
        try:
            # Run complete optimization pipeline
            optimization_results = self.integration.run_complete_pipeline(
                measurement_file=str(measurement_results_file),
                strategy=strategy,
                output_dir=str(output_path)
            )
            
            if not optimization_results['success']:
                return optimization_results
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return {
                'success': False,
                'error': f"Optimization failed: {e}",
                'phase': 'optimization'
            }
        
        # Calculate total time
        total_time = time.time() - start_time
        
        # Add timing and file paths to results
        optimization_results['total_time_seconds'] = total_time
        optimization_results['measurement_file'] = str(measurement_results_file)
        optimization_results['floor_plan_image'] = image_path
        
        # Print final summary
        self._print_final_summary(optimization_results)
        
        return optimization_results
    
    def _print_final_summary(self, results: Dict):
        """Print final summary of the complete process."""
        print("\n" + "="*70)
        print("COMPLETE CASSETTE OPTIMIZATION RESULTS")
        print("="*70)
        
        print(f"\nFloor Plan: {Path(results['floor_plan_image']).name}")
        print(f"Processing Time: {results['total_time_seconds']:.1f} seconds")
        
        print(f"\nFloor Dimensions:")
        print(f"  Area: {results['input']['area_sqft']:.1f} sq ft")
        print(f"  Perimeter: {results['input']['perimeter_feet']:.1f} ft")
        
        print(f"\nCassette Placement:")
        print(f"  Total cassettes: {results['optimization']['cassette_count']}")
        print(f"  Coverage achieved: {results['optimization']['coverage_percentage']:.1f}%")
        print(f"  Total weight: {results['optimization']['total_weight']:.0f} lbs")
        
        print(f"\nCassette Distribution:")
        for size_name, info in results['cassettes'].items():
            print(f"  {size_name}: {info['count']} units "
                 f"({info['dimensions']}, {info['weight']} lbs each)")
        
        print(f"\nCustom Work Required:")
        print(f"  Gap area: {results['coverage']['custom_work_sqft']:.1f} sq ft")
        print(f"  Percentage: {results['coverage']['custom_work_sqft']/results['input']['area_sqft']*100:.1f}%")
        
        print(f"\nOutput Files:")
        print(f"  Measurements: {Path(results['measurement_file']).name}")
        print(f"  Visualization: {Path(results['outputs']['visualization']).name}")
        print(f"  Report: {Path(results['outputs']['report']).name}")
        
        # Success indicator
        if results['optimization']['coverage_percentage'] >= 94:
            print(f"\n✓✓✓ SUCCESS: Achieved {results['optimization']['coverage_percentage']:.1f}% coverage!")
            print("The floor plan is ready for cassette installation.")
        else:
            gap = 94 - results['optimization']['coverage_percentage']
            print(f"\n⚠ Coverage {results['optimization']['coverage_percentage']:.1f}% "
                 f"(gap to 94%: {gap:.1f} percentage points)")
        
        print("="*70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Interactive Cassette-Based Floor Joist Optimization System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s Luna.png                          # Interactive measurement + optimization
  %(prog)s Luna.png --strategy grid          # Use grid strategy
  %(prog)s Luna.png --output results/        # Specify output directory
  %(prog)s Luna.png --use-existing measurements.json  # Skip measurement, use existing
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
        "--use-existing",
        metavar="MEASUREMENT_FILE",
        help="Skip interactive measurement and use existing measurement file"
    )
    
    parser.add_argument(
        "--auto-measure",
        action="store_true",
        help="Attempt automatic edge detection without user interaction (experimental)"
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
        system = InteractiveCassetteSystem()
        
        # Update target coverage if specified
        if args.target_coverage != 0.94:
            system.config.OPTIMIZATION['min_coverage'] = args.target_coverage
            system.config.OPTIMIZATION['target_coverage'] = args.target_coverage + 0.02
        
        # Process floor plan
        results = system.process_floor_plan_interactive(
            args.floor_plan,
            strategy=args.strategy,
            output_dir=args.output,
            skip_measurement=bool(args.use_existing),
            measurement_file=args.use_existing
        )
        
        # Exit with appropriate code
        if results['success']:
            print("\n✓ Process completed successfully!")
            sys.exit(0)
        else:
            print(f"\n✗ Process failed: {results.get('error', 'Unknown error')}")
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
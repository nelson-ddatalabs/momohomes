#!/usr/bin/env python3
"""
Measurement to Cassette Integration Module
===========================================
Integrates interactive measurements with cassette optimization.
"""

import json
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
import logging

# Import cassette modules
from cassette_models import (
    CassetteSize, Cassette, CassetteLayout, 
    OptimizationParameters, FloorBoundary, Point
)
from cassette_optimizer import CassetteOptimizer
from cassette_visualizer import CassetteVisualizer
from cassette_report_generator import CassetteReportGenerator
from coverage_analyzer import CoverageAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MeasurementToCassetteIntegration:
    """Integrates measured floor plans with cassette optimization."""
    
    def __init__(self):
        """Initialize integration system."""
        self.measurements = None
        self.floor_boundary = None
        self.optimization_result = None
        self.coverage_analysis = None
        
    def load_measurements(self, measurement_file: str) -> bool:
        """
        Load measurements from JSON file.
        
        Args:
            measurement_file: Path to measurement JSON file
            
        Returns:
            True if loaded successfully
        """
        try:
            with open(measurement_file, 'r') as f:
                self.measurements = json.load(f)
            
            logger.info(f"Loaded measurements: {self.measurements['perimeter_feet']:.1f} ft perimeter, "
                       f"{self.measurements['area_sqft']:.1f} sq ft area")
            
            # Check if polygon closes
            if not self.measurements['is_closed']:
                logger.warning(f"Polygon does not close properly (error: "
                             f"{self.measurements['closure_error_feet']:.2f} ft)")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load measurements: {e}")
            return False
    
    def create_floor_boundary(self) -> FloorBoundary:
        """
        Create FloorBoundary object from measurements.
        
        Returns:
            FloorBoundary object for cassette optimization
        """
        if not self.measurements:
            raise ValueError("No measurements loaded")
        
        # Extract vertices (excluding the duplicate closing vertex)
        vertices = self.measurements['vertices'][:-1]
        
        # Convert to Point objects
        points = []
        for x, y in vertices:
            # Note: y-coordinate might need adjustment based on coordinate system
            points.append(Point(float(x), float(y)))
        
        # Create FloorBoundary
        self.floor_boundary = FloorBoundary(points=points)
        
        # Calculate and log boundary properties
        logger.info(f"Created floor boundary with {len(points)} vertices")
        logger.info(f"Boundary dimensions: {self.floor_boundary.width:.1f}' x "
                   f"{self.floor_boundary.height:.1f}'")
        
        return self.floor_boundary
    
    def optimize_cassette_placement(self, strategy: str = "hybrid") -> Dict:
        """
        Run cassette optimization on measured floor plan.
        
        Args:
            strategy: Optimization strategy (grid, staggered, hybrid)
            
        Returns:
            Optimization results dictionary
        """
        if not self.floor_boundary:
            self.create_floor_boundary()
        
        # Create optimizer with default parameters
        parameters = OptimizationParameters()
        optimizer = CassetteOptimizer(parameters)
        
        logger.info(f"Running {strategy} optimization...")
        
        # Run optimization
        self.optimization_result = optimizer.optimize(self.floor_boundary, strategy)
        
        if self.optimization_result.success:
            logger.info(f"✓ Optimization successful!")
            logger.info(f"  Cassettes placed: {self.optimization_result.layout.cassette_count}")
            logger.info(f"  Coverage: {self.optimization_result.layout.coverage_percentage:.1f}%")
            logger.info(f"  Total weight: {self.optimization_result.layout.total_weight:.0f} lbs")
        else:
            error_msg = getattr(self.optimization_result, 'error_message', 'Optimization failed')
            logger.warning(f"⚠ Optimization failed: {error_msg}")
        
        return {
            'success': self.optimization_result.success,
            'cassette_count': self.optimization_result.layout.cassette_count,
            'coverage_percentage': self.optimization_result.layout.coverage_percentage,
            'covered_area': self.optimization_result.layout.covered_area,
            'uncovered_area': self.optimization_result.layout.uncovered_area,
            'total_weight': self.optimization_result.layout.total_weight,
            'optimization_time': self.optimization_result.optimization_time
        }
    
    def analyze_coverage(self) -> Dict:
        """
        Analyze coverage and gaps.
        
        Returns:
            Coverage analysis results
        """
        if not self.optimization_result:
            raise ValueError("No optimization result available")
        
        analyzer = CoverageAnalyzer()
        self.coverage_analysis = analyzer.analyze(self.optimization_result.layout)
        
        logger.info(f"Coverage Analysis:")
        logger.info(f"  Coverage: {self.coverage_analysis['coverage_percentage']:.1f}%")
        logger.info(f"  Gaps: {len(self.coverage_analysis['gaps'])}")
        logger.info(f"  Custom work area: {self.coverage_analysis['custom_work']['total_area']:.1f} sq ft")
        
        return self.coverage_analysis
    
    def generate_visualization(self, output_path: str = "cassette_layout.png"):
        """
        Generate visualization of cassette layout.
        
        Args:
            output_path: Path for output image
        """
        if not self.optimization_result:
            raise ValueError("No optimization result available")
        
        visualizer = CassetteVisualizer()
        
        # Get gaps from coverage analysis
        gaps = self.coverage_analysis.get('gaps', []) if self.coverage_analysis else []
        
        # Create visualization
        visualizer.create_layout_visualization(
            self.optimization_result, 
            gaps, 
            output_path
        )
        
        logger.info(f"Visualization saved to: {output_path}")
    
    def generate_report(self, output_path: str = "cassette_report.html"):
        """
        Generate detailed HTML report.
        
        Args:
            output_path: Path for output report
        """
        if not self.optimization_result or not self.coverage_analysis:
            raise ValueError("No results available for report")
        
        report_gen = CassetteReportGenerator()
        report_gen.generate_construction_report(
            self.optimization_result,
            self.coverage_analysis,
            output_path
        )
        
        logger.info(f"Report saved to: {output_path}")
    
    def get_cassette_summary(self) -> Dict:
        """
        Get summary of cassette distribution.
        
        Returns:
            Dictionary with cassette counts by size
        """
        if not self.optimization_result:
            return {}
        
        summary = self.optimization_result.layout.get_cassette_summary()
        
        # Convert to readable format
        result = {}
        for size, count in summary.items():
            result[size.name] = {
                'count': count,
                'dimensions': f"{size.value[0]}'x{size.value[1]}'",
                'area': size.value[0] * size.value[1],
                'weight': size.value[2]
            }
        
        return result
    
    def run_complete_pipeline(self, measurement_file: str, 
                            strategy: str = "hybrid",
                            output_dir: str = "output") -> Dict:
        """
        Run complete pipeline from measurements to cassette optimization.
        
        Args:
            measurement_file: Path to measurement JSON file
            strategy: Optimization strategy
            output_dir: Directory for output files
            
        Returns:
            Complete results dictionary
        """
        logger.info("\n" + "="*70)
        logger.info("CASSETTE OPTIMIZATION PIPELINE")
        logger.info("="*70)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Step 1: Load measurements
        if not self.load_measurements(measurement_file):
            return {'success': False, 'error': 'Failed to load measurements'}
        
        # Step 2: Create floor boundary
        self.create_floor_boundary()
        
        # Step 3: Optimize cassette placement
        optimization_results = self.optimize_cassette_placement(strategy)
        
        if not optimization_results['success']:
            return optimization_results
        
        # Step 4: Analyze coverage
        coverage_results = self.analyze_coverage()
        
        # Step 5: Generate outputs
        self.generate_visualization(str(output_path / "cassette_layout.png"))
        self.generate_report(str(output_path / "cassette_report.html"))
        
        # Step 6: Get cassette summary
        cassette_summary = self.get_cassette_summary()
        
        # Compile complete results
        results = {
            'success': True,
            'input': {
                'measurement_file': measurement_file,
                'perimeter_feet': self.measurements['perimeter_feet'],
                'area_sqft': self.measurements['area_sqft']
            },
            'optimization': optimization_results,
            'coverage': {
                'percentage': coverage_results['coverage_percentage'],
                'gaps': len(coverage_results['gaps']),
                'custom_work_sqft': coverage_results['custom_work']['total_area']
            },
            'cassettes': cassette_summary,
            'outputs': {
                'visualization': str(output_path / "cassette_layout.png"),
                'report': str(output_path / "cassette_report.html")
            }
        }
        
        # Save complete results
        results_file = output_path / "complete_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\n✓ Pipeline complete! Results saved to: {results_file}")
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results: Dict):
        """Print results summary."""
        print("\n" + "="*70)
        print("CASSETTE OPTIMIZATION SUMMARY")
        print("="*70)
        
        print(f"\nFloor Plan:")
        print(f"  Area: {results['input']['area_sqft']:.1f} sq ft")
        print(f"  Perimeter: {results['input']['perimeter_feet']:.1f} ft")
        
        print(f"\nOptimization Results:")
        print(f"  Cassettes placed: {results['optimization']['cassette_count']}")
        print(f"  Coverage: {results['optimization']['coverage_percentage']:.1f}%")
        print(f"  Weight: {results['optimization']['total_weight']:.0f} lbs")
        
        print(f"\nCassette Distribution:")
        for size_name, info in results['cassettes'].items():
            print(f"  {size_name}: {info['count']} units "
                 f"({info['dimensions']}, {info['weight']} lbs each)")
        
        print(f"\nCustom Work Required:")
        print(f"  Gap area: {results['coverage']['custom_work_sqft']:.1f} sq ft")
        print(f"  Number of gaps: {results['coverage']['gaps']}")
        
        # Success indicator
        if results['optimization']['coverage_percentage'] >= 94:
            print(f"\n✓ SUCCESS: Achieved {results['optimization']['coverage_percentage']:.1f}% coverage!")
        else:
            print(f"\n⚠ Coverage {results['optimization']['coverage_percentage']:.1f}% "
                 f"(target: 94%+)")


def test_integration():
    """Test the integration with Luna measurements."""
    print("\n" + "="*70)
    print("TESTING MEASUREMENT TO CASSETTE INTEGRATION")
    print("="*70)
    
    integration = MeasurementToCassetteIntegration()
    
    # Check if measurement file exists
    measurement_file = "luna_measured_auto.json"
    if not Path(measurement_file).exists():
        print(f"Error: {measurement_file} not found")
        print("Please run test_interactive_measurement.py first")
        return
    
    # Run complete pipeline
    results = integration.run_complete_pipeline(
        measurement_file=measurement_file,
        strategy="hybrid",
        output_dir="cassette_output"
    )
    
    if results['success']:
        print("\n✓ Integration test successful!")
        print(f"Check 'cassette_output' directory for generated files:")
        print("  - cassette_layout.png: Visual layout")
        print("  - cassette_report.html: Detailed report")
        print("  - complete_results.json: Full results data")
    else:
        print(f"\n✗ Integration test failed: {results.get('error', 'Unknown error')}")


if __name__ == "__main__":
    test_integration()
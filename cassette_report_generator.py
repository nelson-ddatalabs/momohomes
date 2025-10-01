"""
Cassette Report Generator Module
================================
Generates construction reports and bill of materials for cassette layouts.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

from cassette_models import OptimizationResult, CassetteLayout
from config_cassette import CassetteConfig

logger = logging.getLogger(__name__)


class CassetteReportGenerator:
    """Generates reports for cassette layouts."""
    
    def __init__(self):
        """Initialize report generator."""
        self.config = CassetteConfig.REPORTING
        
    def generate_construction_report(self, result: OptimizationResult,
                                    coverage_analysis: Dict,
                                    output_path: str) -> None:
        """
        Generate comprehensive construction report.
        
        Args:
            result: Optimization result
            coverage_analysis: Coverage analysis data
            output_path: Path to save report
        """
        layout = result.layout
        
        # Generate HTML report
        html_content = self._generate_html_report(result, coverage_analysis)
        
        # Save report
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Construction report saved to {output_path}")
    
    def generate_bill_of_materials(self, layout: CassetteLayout) -> Dict:
        """
        Generate bill of materials for cassette layout.
        
        Args:
            layout: Cassette layout
            
        Returns:
            Dictionary with BOM data
        """
        # Group cassettes by size
        cassette_summary = layout.get_cassette_summary()
        
        bom = {
            'timestamp': datetime.now().strftime(self.config['date_format']),
            'summary': {
                'total_cassettes': layout.cassette_count,
                'total_weight_lbs': layout.total_weight,
                'total_area_sqft': layout.covered_area,
                'total_joist_count': layout.total_joist_count
            },
            'cassette_list': [],
            'by_size': {},
            'installation_notes': []
        }
        
        # Add details for each cassette size
        for size, count in cassette_summary.items():
            size_data = {
                'size': size.name,
                'dimensions': f"{size.width}' x {size.height}'",
                'quantity': count,
                'unit_area_sqft': size.area,
                'total_area_sqft': size.area * count,
                'unit_weight_lbs': size.weight,
                'total_weight_lbs': size.weight * count,
                'joist_count_per_unit': size.joist_count,
                'total_joist_count': size.joist_count * count,
                'joist_spacing_inches': size.spacing
            }
            bom['by_size'][size.name] = size_data
        
        # Add individual cassette list
        for cassette in sorted(layout.cassettes, key=lambda c: c.placement_order):
            cassette_data = {
                'id': cassette.cassette_id,
                'size': cassette.size.name,
                'position': f"({cassette.x:.1f}, {cassette.y:.1f})",
                'weight_lbs': cassette.weight,
                'placement_order': cassette.placement_order
            }
            bom['cassette_list'].append(cassette_data)
        
        # Add installation notes
        bom['installation_notes'] = self._generate_installation_notes(layout)
        
        return bom
    
    def generate_installation_sequence(self, layout: CassetteLayout) -> List[Dict]:
        """
        Generate installation sequence for cassettes.
        
        Args:
            layout: Cassette layout
            
        Returns:
            List of installation steps
        """
        sequence = []
        
        # Sort cassettes by placement order
        sorted_cassettes = sorted(layout.cassettes, key=lambda c: c.placement_order)
        
        # Group by rows for easier installation
        rows = {}
        for cassette in sorted_cassettes:
            row_key = int(cassette.y / 6) * 6  # Group by 6 ft increments
            if row_key not in rows:
                rows[row_key] = []
            rows[row_key].append(cassette)
        
        step_num = 1
        for y_pos in sorted(rows.keys()):
            row_cassettes = sorted(rows[y_pos], key=lambda c: c.x)
            
            for cassette in row_cassettes:
                step = {
                    'step': step_num,
                    'cassette_id': cassette.cassette_id,
                    'size': cassette.size.name,
                    'position': f"({cassette.x:.1f}, {cassette.y:.1f})",
                    'weight_lbs': cassette.weight,
                    'crew_required': 2 if cassette.weight > 200 else 1,
                    'notes': self._get_installation_notes_for_cassette(cassette)
                }
                sequence.append(step)
                step_num += 1
        
        return sequence
    
    def _generate_html_report(self, result: OptimizationResult, 
                             coverage_analysis: Dict) -> str:
        """Generate HTML report content."""
        layout = result.layout
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Cassette Layout Construction Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
        .header {{ background-color: #2c3e50; color: white; padding: 20px; border-radius: 5px; }}
        .section {{ background-color: white; margin: 20px 0; padding: 20px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .metric {{ display: inline-block; margin: 10px 20px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
        .metric-label {{ font-size: 12px; color: #7f8c8d; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th {{ background-color: #34495e; color: white; padding: 10px; text-align: left; }}
        td {{ padding: 8px; border-bottom: 1px solid #ecf0f1; }}
        tr:hover {{ background-color: #f8f9fa; }}
        .success {{ color: #27ae60; font-weight: bold; }}
        .warning {{ color: #f39c12; font-weight: bold; }}
        .error {{ color: #e74c3c; font-weight: bold; }}
        .footer {{ text-align: center; margin-top: 40px; padding: 20px; background-color: #ecf0f1; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Cassette Layout Construction Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="section">
        <h2>Project Summary</h2>
        <div class="metric">
            <div class="metric-value">{layout.floor_boundary.area:.1f} sq ft</div>
            <div class="metric-label">Total Floor Area</div>
        </div>
        <div class="metric">
            <div class="metric-value">{layout.coverage_percentage:.1f}%</div>
            <div class="metric-label">Coverage Achieved</div>
        </div>
        <div class="metric">
            <div class="metric-value">{layout.cassette_count}</div>
            <div class="metric-label">Total Cassettes</div>
        </div>
        <div class="metric">
            <div class="metric-value">{layout.total_weight:,.0f} lbs</div>
            <div class="metric-label">Total Weight</div>
        </div>
    </div>
    
    <div class="section">
        <h2>Cassette Distribution</h2>
        <table>
            <tr>
                <th>Size</th>
                <th>Quantity</th>
                <th>Unit Area</th>
                <th>Total Area</th>
                <th>Unit Weight</th>
                <th>Total Weight</th>
            </tr>
"""
        
        # Add cassette summary rows
        for size, count in sorted(layout.get_cassette_summary().items(), 
                                 key=lambda x: x[1], reverse=True):
            html += f"""
            <tr>
                <td>{size.name}</td>
                <td>{count}</td>
                <td>{size.area} sq ft</td>
                <td>{size.area * count} sq ft</td>
                <td>{size.weight} lbs</td>
                <td>{size.weight * count:,} lbs</td>
            </tr>
"""
        
        html += """
        </table>
    </div>
    
    <div class="section">
        <h2>Coverage Analysis</h2>
"""
        
        # Add coverage status
        if layout.coverage_percentage >= 94:
            status_class = "success"
            status_text = "TARGET ACHIEVED"
        else:
            status_class = "warning"
            status_text = "BELOW TARGET"
        
        html += f"""
        <p class="{status_class}">Status: {status_text}</p>
        <p>Covered Area: {layout.covered_area:.1f} sq ft</p>
        <p>Custom Work Required: {layout.uncovered_area:.1f} sq ft ({layout.uncovered_area/layout.total_area*100:.1f}%)</p>
"""
        
        # Add gap analysis if available
        if coverage_analysis and 'gaps' in coverage_analysis:
            gap_count = len(coverage_analysis['gaps'])
            custom_area = coverage_analysis['custom_work']['total_area']
            html += f"""
        <p>Gaps Identified: {gap_count}</p>
        <p>Total Gap Area: {custom_area:.1f} sq ft</p>
"""
        
        html += """
    </div>
    
    <div class="section">
        <h2>Installation Requirements</h2>
        <ul>
            <li>Crew Size: 2-3 workers recommended</li>
            <li>Equipment: Forklift or crane for cassettes over 300 lbs</li>
            <li>Estimated Installation Time: {0} hours</li>
            <li>Safety: Follow OSHA guidelines for material handling</li>
        </ul>
    </div>
""".format(layout.cassette_count * 0.5)  # Rough estimate: 30 min per cassette
        
        # Add optimization details
        html += f"""
    <div class="section">
        <h2>Optimization Details</h2>
        <p>Strategy Used: {result.algorithm_used}</p>
        <p>Optimization Time: {result.optimization_time:.2f} seconds</p>
        <p>Iterations: {result.iterations}</p>
"""
        
        if result.warnings:
            html += "<h3>Warnings:</h3><ul>"
            for warning in result.warnings:
                html += f"<li class='warning'>{warning}</li>"
            html += "</ul>"
        
        html += """
    </div>
    
    <div class="footer">
        <p>This report is for construction planning purposes only.</p>
        <p>Verify all measurements and weights before installation.</p>
    </div>
</body>
</html>
"""
        
        return html
    
    def _generate_installation_notes(self, layout: CassetteLayout) -> List[str]:
        """Generate installation notes."""
        notes = []
        
        # Check for heavy cassettes
        heavy_count = sum(1 for c in layout.cassettes if c.weight > 300)
        if heavy_count > 0:
            notes.append(f"{heavy_count} cassettes exceed 300 lbs - crane required")
        
        # Check cassette variety
        size_count = len(layout.get_cassette_summary())
        if size_count > 4:
            notes.append(f"Multiple cassette sizes ({size_count}) - organize by type before installation")
        
        # Row-by-row installation
        notes.append("Install cassettes row by row, starting from bottom left")
        
        # Edge cassettes
        edge_count = sum(1 for c in layout.cassettes if c.size.area <= 12)
        if edge_count > 0:
            notes.append(f"{edge_count} edge cassettes - install after main cassettes")
        
        return notes
    
    def _get_installation_notes_for_cassette(self, cassette) -> str:
        """Get specific installation notes for a cassette."""
        notes = []
        
        if cassette.weight > 300:
            notes.append("Use crane")
        elif cassette.weight > 200:
            notes.append("2-person lift")
        
        if cassette.size.area <= 12:
            notes.append("Edge filler")
        
        return "; ".join(notes) if notes else "Standard installation"
    
    def save_json_report(self, result: OptimizationResult, 
                        coverage_analysis: Dict,
                        output_path: str) -> None:
        """
        Save report as JSON.
        
        Args:
            result: Optimization result
            coverage_analysis: Coverage analysis
            output_path: Path to save JSON
        """
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'optimization_result': result.to_dict(),
            'coverage_analysis': coverage_analysis,
            'bill_of_materials': self.generate_bill_of_materials(result.layout),
            'installation_sequence': self.generate_installation_sequence(result.layout)
        }
        
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"JSON report saved to {output_path}")
    
    def generate_summary_text(self, result: OptimizationResult) -> str:
        """
        Generate text summary of results.
        
        Args:
            result: Optimization result
            
        Returns:
            Text summary
        """
        layout = result.layout
        
        lines = [
            "CASSETTE LAYOUT SUMMARY",
            "=" * 40,
            f"Total Area: {layout.floor_boundary.area:.1f} sq ft",
            f"Coverage: {layout.coverage_percentage:.1f}%",
            f"Cassettes: {layout.cassette_count}",
            f"Total Weight: {layout.total_weight:,.0f} lbs",
            "",
            "Cassette Breakdown:",
        ]
        
        for size, count in sorted(layout.get_cassette_summary().items(), 
                                 key=lambda x: x[1], reverse=True):
            lines.append(f"  {size.name}: {count} units")
        
        lines.extend([
            "",
            f"Custom Work: {layout.uncovered_area:.1f} sq ft",
            f"Strategy: {result.algorithm_used}",
            f"Time: {result.optimization_time:.2f} seconds",
        ])
        
        return "\n".join(lines)
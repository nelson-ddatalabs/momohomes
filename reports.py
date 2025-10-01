"""
reports.py - Report Generation Module
======================================
Generates detailed reports in various formats.
"""

import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from jinja2 import Template

from models import FloorPlan, OptimizationResult, PanelSize
from config import Config


logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates reports in various formats."""
    
    def __init__(self, floor_plan: FloorPlan, result: OptimizationResult):
        """Initialize report generator."""
        self.floor_plan = floor_plan
        self.result = result
        self.config = Config.REPORTING
        
        # Report data
        self.report_data = self._prepare_report_data()
        
        logger.info(f"Initialized ReportGenerator for {floor_plan.name}")
    
    def _prepare_report_data(self) -> Dict:
        """Prepare data for report generation."""
        panel_summary = self.floor_plan.get_panel_summary()
        
        data = {
            'timestamp': datetime.now().strftime(self.config['date_format']),
            'floor_plan_name': self.floor_plan.name,
            'strategy_used': self.result.strategy_used,
            'optimization_time': self.result.optimization_time,
            
            # Area metrics
            'total_area': round(self.floor_plan.total_area, 2),
            'covered_area': round(self.floor_plan.total_area * self.result.coverage_ratio, 2),
            'uncovered_area': round(self.floor_plan.total_uncovered, 2),
            'coverage_ratio': round(self.result.coverage_ratio * 100, 1),
            
            # Cost metrics
            'total_cost': round(self.floor_plan.total_cost, 2),
            'cost_per_sqft': round(self.result.cost_per_sqft, 2),
            
            # Panel metrics
            'total_panels': self.floor_plan.total_panels,
            'panel_summary': {
                size.name: count for size, count in panel_summary.items()
            },
            'panel_efficiency': round(self.result.panel_efficiency * 100, 1),
            
            # Room details
            'room_count': self.floor_plan.room_count,
            'rooms': [room.to_dict() for room in self.floor_plan.rooms],
            
            # Structural compliance
            'structural_compliance': self.result.structural_compliance,
            'violations': self.result.violations,
            
            # Additional metrics
            'metrics': self.result.metrics
        }
        
        return data
    
    def generate_text_report(self, output_path: Optional[str] = None) -> str:
        """Generate text report."""
        logger.info("Generating text report")
        
        lines = []
        lines.append("=" * 80)
        lines.append("FLOOR PLAN PANEL OPTIMIZATION REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        # Header
        lines.append(f"Report Generated: {self.report_data['timestamp']}")
        lines.append(f"Floor Plan: {self.report_data['floor_plan_name']}")
        lines.append(f"Optimization Strategy: {self.report_data['strategy_used']}")
        lines.append(f"Processing Time: {self.report_data['optimization_time']:.2f} seconds")
        lines.append("")
        
        # Executive Summary
        lines.append("EXECUTIVE SUMMARY")
        lines.append("-" * 40)
        lines.append(f"Total Floor Area: {self.report_data['total_area']} sq ft")
        lines.append(f"Coverage Achieved: {self.report_data['coverage_ratio']}%")
        lines.append(f"Total Cost: ${self.report_data['total_cost']}")
        lines.append(f"Cost per Square Foot: ${self.report_data['cost_per_sqft']}")
        lines.append(f"Structural Compliance: {'✓ PASS' if self.report_data['structural_compliance'] else '✗ FAIL'}")
        lines.append("")
        
        # Panel Requirements
        lines.append("PANEL REQUIREMENTS")
        lines.append("-" * 40)
        total_panels = 0
        for size, count in self.report_data['panel_summary'].items():
            lines.append(f"{size:10} : {count:3} panels")
            total_panels += count
        lines.append(f"{'TOTAL':10} : {total_panels:3} panels")
        lines.append("")
        
        # Cost Breakdown
        lines.append("COST BREAKDOWN")
        lines.append("-" * 40)
        for size, count in self.report_data['panel_summary'].items():
            if size == "6x8":
                unit_cost = 48.00
            elif size == "6x6":
                unit_cost = 41.40
            elif size == "4x6":
                unit_cost = 32.40
            elif size == "4x4":
                unit_cost = 25.60
            else:
                unit_cost = 0
            
            total = count * unit_cost
            lines.append(f"{size:10} : {count:3} × ${unit_cost:6.2f} = ${total:8.2f}")
        
        lines.append(f"{'TOTAL COST':30} = ${self.report_data['total_cost']:8.2f}")
        lines.append("")
        
        # Room-by-Room Breakdown
        lines.append("ROOM-BY-ROOM BREAKDOWN")
        lines.append("-" * 40)
        
        for room in self.report_data['rooms']:
            lines.append(f"\n{room['type'].upper()} (ID: {room['id']})")
            lines.append(f"  Dimensions: {room['dimensions']}")
            lines.append(f"  Area: {room['area']:.2f} sq ft")
            lines.append(f"  Panels Used: {room['panel_count']}")
            lines.append(f"  Coverage: {room['coverage_ratio']*100:.1f}%")
            lines.append(f"  Cost: ${room['total_cost']:.2f}")
            
            if room['uncovered_area'] > 1:
                lines.append(f"  ⚠ Uncovered Area: {room['uncovered_area']:.2f} sq ft")
        
        lines.append("")
        
        # Structural Violations
        if not self.report_data['structural_compliance']:
            lines.append("STRUCTURAL VIOLATIONS")
            lines.append("-" * 40)
            for violation in self.report_data['violations'][:10]:  # Limit to 10
                lines.append(f"  • {violation}")
            if len(self.report_data['violations']) > 10:
                lines.append(f"  ... and {len(self.report_data['violations'])-10} more")
            lines.append("")
        
        # Performance Metrics
        lines.append("PERFORMANCE METRICS")
        lines.append("-" * 40)
        lines.append(f"Coverage Ratio: {self.report_data['coverage_ratio']}%")
        lines.append(f"Panel Efficiency: {self.report_data['panel_efficiency']}%")
        lines.append(f"Cost Efficiency: ${self.report_data['cost_per_sqft']}/sq ft")
        
        if 'fitness' in self.report_data['metrics']:
            lines.append(f"Optimization Score: {self.report_data['metrics']['fitness']:.2f}")
        
        lines.append("")
        
        # Recommendations
        lines.append("RECOMMENDATIONS")
        lines.append("-" * 40)
        
        recommendations = self._generate_recommendations()
        for i, rec in enumerate(recommendations, 1):
            lines.append(f"{i}. {rec}")
        
        lines.append("")
        lines.append("=" * 80)
        lines.append("END OF REPORT")
        lines.append("=" * 80)
        
        report_text = "\n".join(lines)
        
        # Save to file if path provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Text report saved to {output_path}")
        
        return report_text
    
    def generate_json_report(self, output_path: Optional[str] = None) -> Dict:
        """Generate JSON report."""
        logger.info("Generating JSON report")
        
        # Clean data for JSON serialization
        json_data = self._clean_for_json(self.report_data)
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            logger.info(f"JSON report saved to {output_path}")
        
        return json_data
    
    def generate_html_report(self, output_path: Optional[str] = None) -> str:
        """Generate HTML report."""
        logger.info("Generating HTML report")
        
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Floor Plan Optimization Report - {{ floor_plan_name }}</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
                .header { background-color: #2c3e50; color: white; padding: 20px; border-radius: 5px; }
                .section { background-color: white; margin: 20px 0; padding: 20px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .metric { display: inline-block; margin: 10px 20px; }
                .metric-value { font-size: 24px; font-weight: bold; color: #2c3e50; }
                .metric-label { font-size: 12px; color: #7f8c8d; }
                table { width: 100%; border-collapse: collapse; margin: 20px 0; }
                th { background-color: #34495e; color: white; padding: 10px; text-align: left; }
                td { padding: 8px; border-bottom: 1px solid #ecf0f1; }
                tr:hover { background-color: #f8f9fa; }
                .success { color: #27ae60; font-weight: bold; }
                .warning { color: #f39c12; font-weight: bold; }
                .error { color: #e74c3c; font-weight: bold; }
                .progress-bar { width: 100%; height: 30px; background-color: #ecf0f1; border-radius: 15px; overflow: hidden; }
                .progress-fill { height: 100%; background-color: #3498db; text-align: center; line-height: 30px; color: white; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Floor Plan Optimization Report</h1>
                <p>{{ floor_plan_name }} | Generated: {{ timestamp }}</p>
            </div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <div class="metric">
                    <div class="metric-value">{{ total_area }} sq ft</div>
                    <div class="metric-label">Total Area</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{{ coverage_ratio }}%</div>
                    <div class="metric-label">Coverage</div>
                </div>
                <div class="metric">
                    <div class="metric-value">${{ total_cost }}</div>
                    <div class="metric-label">Total Cost</div>
                </div>
                <div class="metric">
                    <div class="metric-value">${{ cost_per_sqft }}/sq ft</div>
                    <div class="metric-label">Cost per sq ft</div>
                </div>
                
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {{ coverage_ratio }}%">
                        Coverage: {{ coverage_ratio }}%
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>Panel Requirements</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Panel Size</th>
                            <th>Quantity</th>
                            <th>Unit Cost</th>
                            <th>Total Cost</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for size, count in panel_summary.items() %}
                        <tr>
                            <td>{{ size }}</td>
                            <td>{{ count }}</td>
                            <td>${{ panel_costs[size] }}</td>
                            <td>${{ panel_costs[size] * count }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
            
            <div class="section">
                <h2>Room Details</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Room</th>
                            <th>Type</th>
                            <th>Dimensions</th>
                            <th>Area</th>
                            <th>Panels</th>
                            <th>Coverage</th>
                            <th>Cost</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for room in rooms %}
                        <tr>
                            <td>{{ room.id }}</td>
                            <td>{{ room.type }}</td>
                            <td>{{ room.dimensions }}</td>
                            <td>{{ room.area }} sq ft</td>
                            <td>{{ room.panel_count }}</td>
                            <td class="{% if room.coverage_ratio > 0.95 %}success{% elif room.coverage_ratio > 0.9 %}warning{% else %}error{% endif %}">
                                {{ (room.coverage_ratio * 100)|round(1) }}%
                            </td>
                            <td>${{ room.total_cost }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
            
            <div class="section">
                <h2>Structural Compliance</h2>
                <p class="{% if structural_compliance %}success{% else %}error{% endif %}">
                    Status: {% if structural_compliance %}✓ COMPLIANT{% else %}✗ NON-COMPLIANT{% endif %}
                </p>
                {% if violations %}
                <h3>Violations:</h3>
                <ul>
                    {% for violation in violations[:5] %}
                    <li>{{ violation }}</li>
                    {% endfor %}
                </ul>
                {% endif %}
            </div>
            
            <div class="section">
                <h2>Optimization Details</h2>
                <p>Strategy: <strong>{{ strategy_used }}</strong></p>
                <p>Processing Time: <strong>{{ optimization_time|round(2) }} seconds</strong></p>
                <p>Panel Efficiency: <strong>{{ panel_efficiency }}%</strong></p>
            </div>
        </body>
        </html>
        """
        
        # Prepare template data
        template_data = self.report_data.copy()
        template_data['panel_costs'] = {
            '6x8': 48.00,
            '6x6': 41.40,
            '4x6': 32.40,
            '4x4': 25.60
        }
        
        # Render template
        template = Template(html_template)
        html_content = template.render(**template_data)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(html_content)
            logger.info(f"HTML report saved to {output_path}")
        
        return html_content
    
    def generate_csv_report(self, output_path: Optional[str] = None) -> List[Dict]:
        """Generate CSV report for room data."""
        logger.info("Generating CSV report")
        
        rows = []
        
        for room in self.report_data['rooms']:
            row = {
                'Room ID': room['id'],
                'Room Type': room['type'],
                'Width (ft)': room['dimensions'].split('x')[0],
                'Height (ft)': room['dimensions'].split('x')[1],
                'Area (sq ft)': room['area'],
                'Panel Count': room['panel_count'],
                'Coverage (%)': round(room['coverage_ratio'] * 100, 1),
                'Uncovered (sq ft)': room['uncovered_area'],
                'Cost ($)': room['total_cost'],
                'Load Bearing': room.get('is_load_bearing', False),
                'Span Direction': room.get('span_direction', 'N/A')
            }
            
            # Add panel counts by size
            for size in ['6x8', '6x6', '4x6', '4x4']:
                count = 0
                for panel in room.get('panels', []):
                    if panel.get('size') == size:
                        count += 1
                row[f'{size} Panels'] = count
            
            rows.append(row)
        
        if output_path:
            with open(output_path, 'w', newline='') as f:
                if rows:
                    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)
            logger.info(f"CSV report saved to {output_path}")
        
        return rows
    
    def generate_summary_report(self) -> str:
        """Generate brief summary report."""
        summary = f"""
OPTIMIZATION SUMMARY - {self.report_data['floor_plan_name']}
{'='*50}
Coverage: {self.report_data['coverage_ratio']}%
Cost: ${self.report_data['total_cost']} (${self.report_data['cost_per_sqft']}/sq ft)
Panels: {self.report_data['total_panels']} total
  - 6x8: {self.report_data['panel_summary'].get('6x8', 0)}
  - 6x6: {self.report_data['panel_summary'].get('6x6', 0)}
  - 4x6: {self.report_data['panel_summary'].get('4x6', 0)}
  - 4x4: {self.report_data['panel_summary'].get('4x4', 0)}
Compliance: {'✓' if self.report_data['structural_compliance'] else '✗'}
Strategy: {self.report_data['strategy_used']}
Time: {self.report_data['optimization_time']:.2f}s
"""
        return summary
    
    def generate_all_reports(self, output_dir: str):
        """Generate all report formats."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        base_name = f"{self.floor_plan.name}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Generate all formats
        self.generate_text_report(output_dir / f"{base_name}.txt")
        self.generate_json_report(output_dir / f"{base_name}.json")
        self.generate_html_report(output_dir / f"{base_name}.html")
        self.generate_csv_report(output_dir / f"{base_name}.csv")
        
        logger.info(f"All reports generated in {output_dir}")
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on results."""
        recommendations = []
        
        # Coverage recommendations
        if self.result.coverage_ratio < 0.95:
            recommendations.append(
                f"Coverage is {self.result.coverage_ratio:.1%}. "
                "Consider manual adjustment of panels in uncovered areas."
            )
        
        # Cost recommendations
        if self.result.cost_per_sqft > 1.50:
            recommendations.append(
                f"Cost per sq ft (${self.result.cost_per_sqft:.2f}) is high. "
                "Review panel size distribution and consider using more large panels."
            )
        
        # Efficiency recommendations
        if self.result.panel_efficiency < 0.65:
            recommendations.append(
                f"Panel efficiency is {self.result.panel_efficiency:.1%}. "
                "Increase usage of 6x8 panels where possible for better cost efficiency."
            )
        
        # Structural recommendations
        if not self.result.structural_compliance:
            recommendations.append(
                "CRITICAL: Structural violations detected. "
                "Review panel placement and ensure compliance with load-bearing requirements."
            )
            
            if len(self.result.violations) > 5:
                recommendations.append(
                    "Multiple structural issues found. "
                    "Consider consulting a structural engineer."
                )
        
        # Room-specific recommendations
        for room in self.floor_plan.rooms:
            if room.coverage_ratio < 0.9:
                recommendations.append(
                    f"{room.type.value} (ID: {room.id}) has low coverage "
                    f"({room.coverage_ratio:.1%}). Review panel placement."
                )
        
        # General recommendations
        if not recommendations:
            recommendations.append("Optimization successful. All metrics within acceptable ranges.")
        
        return recommendations[:10]  # Limit to 10 recommendations
    
    def _clean_for_json(self, data: Any) -> Any:
        """Clean data for JSON serialization."""
        if isinstance(data, dict):
            return {k: self._clean_for_json(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._clean_for_json(v) for v in data]
        elif hasattr(data, '__dict__'):
            return self._clean_for_json(data.__dict__)
        elif isinstance(data, (int, float, str, bool, type(None))):
            return data
        else:
            return str(data)

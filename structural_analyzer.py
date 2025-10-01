"""
structural_analyzer.py - Structural Analysis and Compliance Module
===================================================================
Analyzes structural requirements and ensures panel placement compliance.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
import logging
from dataclasses import dataclass
from enum import Enum

from models import Room, Panel, Wall, Point, FloorPlan, PanelSize, Rectangle
from config import Config


logger = logging.getLogger(__name__)


class LoadPath(Enum):
    """Types of structural load paths."""
    ROOF_TO_FOUNDATION = "roof_to_foundation"
    FLOOR_TO_FLOOR = "floor_to_floor"
    LATERAL = "lateral"
    POINT_LOAD = "point_load"


@dataclass
class StructuralViolation:
    """Represents a structural compliance violation."""
    violation_type: str
    severity: str  # "critical", "major", "minor"
    description: str
    location: Optional[Point] = None
    room_id: Optional[str] = None
    panel_id: Optional[str] = None
    recommendation: str = ""
    
    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.description}"


class StructuralAnalyzer:
    """Analyzes structural requirements for floor plans."""
    
    def __init__(self, floor_plan: FloorPlan):
        """Initialize structural analyzer."""
        self.floor_plan = floor_plan
        self.rooms = floor_plan.rooms
        self.walls = floor_plan.walls
        
        # Structural parameters from config
        self.span_limits = Config.STRUCTURAL['span_limits']
        self.safety_factor = Config.STRUCTURAL['safety_factor']
        
        # Analysis results
        self.load_bearing_walls: List[Wall] = []
        self.support_points: List[Point] = []
        self.load_paths: Dict[str, List[Point]] = {}
        self.span_directions: Dict[str, str] = {}
        self.max_spans: Dict[str, float] = {}
        
        logger.info(f"Initialized StructuralAnalyzer for {floor_plan.name}")
    
    def analyze(self) -> Dict:
        """Perform complete structural analysis."""
        logger.info("Starting structural analysis")
        
        # Step 1: Identify load-bearing elements
        self.identify_load_bearing_walls()
        
        # Step 2: Determine support points
        self.identify_support_points()
        
        # Step 3: Calculate load paths
        self.calculate_load_paths()
        
        # Step 4: Determine span directions
        self.determine_span_directions()
        
        # Step 5: Calculate maximum spans
        self.calculate_max_spans()
        
        # Step 6: Check existing panels if any
        violations = self.check_compliance()
        
        results = {
            'load_bearing_walls': len(self.load_bearing_walls),
            'support_points': len(self.support_points),
            'span_directions': self.span_directions,
            'max_spans': self.max_spans,
            'violations': violations,
            'compliant': len(violations) == 0
        }
        
        logger.info(f"Structural analysis complete: {len(violations)} violations found")
        
        return results
    
    def identify_load_bearing_walls(self):
        """Identify load-bearing walls in the floor plan."""
        logger.info("Identifying load-bearing walls")
        
        self.load_bearing_walls = []
        
        # 1. All exterior walls are load-bearing
        exterior_walls = [w for w in self.walls if w.wall_type == "exterior"]
        self.load_bearing_walls.extend(exterior_walls)
        
        # 2. Walls marked as load-bearing
        marked_walls = [w for w in self.walls if w.wall_type == "load_bearing"]
        self.load_bearing_walls.extend(marked_walls)
        
        # 3. Central walls in large buildings
        if self.floor_plan.building_dimensions:
            building_width = self.floor_plan.building_dimensions.width
            building_height = self.floor_plan.building_dimensions.height
            center_x = self.floor_plan.building_dimensions.center.x
            center_y = self.floor_plan.building_dimensions.center.y
            
            tolerance = Config.STRUCTURAL['load_bearing']['center_tolerance']
            
            for wall in self.walls:
                if wall in self.load_bearing_walls:
                    continue
                
                # Check if wall is near center line
                wall_center = wall.midpoint
                
                if (abs(wall_center.x - center_x) < building_width * tolerance or
                    abs(wall_center.y - center_y) < building_height * tolerance):
                    
                    # Additional check: wall should be long enough
                    if wall.length > min(building_width, building_height) * 0.3:
                        wall.wall_type = "load_bearing"
                        self.load_bearing_walls.append(wall)
        
        # 4. Mark rooms adjacent to load-bearing walls
        for room in self.rooms:
            room.is_load_bearing = self._room_has_load_bearing_wall(room)
        
        logger.info(f"Identified {len(self.load_bearing_walls)} load-bearing walls")
    
    def _room_has_load_bearing_wall(self, room: Room) -> bool:
        """Check if room has load-bearing walls."""
        room_rect = room.rectangle
        
        for wall in self.load_bearing_walls:
            # Check if wall is part of room boundary
            # Simplified: check if wall endpoints are near room perimeter
            for point in [wall.start, wall.end]:
                # Check if point is on room perimeter
                on_left = abs(point.x - room_rect.x) < 1
                on_right = abs(point.x - (room_rect.x + room_rect.width)) < 1
                on_top = abs(point.y - room_rect.y) < 1
                on_bottom = abs(point.y - (room_rect.y + room_rect.height)) < 1
                
                if (on_left or on_right) and room_rect.y <= point.y <= room_rect.y + room_rect.height:
                    return True
                if (on_top or on_bottom) and room_rect.x <= point.x <= room_rect.x + room_rect.width:
                    return True
        
        return False
    
    def identify_support_points(self):
        """Identify key support points in the structure."""
        logger.info("Identifying support points")
        
        self.support_points = []
        
        # 1. Intersection of load-bearing walls
        for i, wall1 in enumerate(self.load_bearing_walls):
            for wall2 in self.load_bearing_walls[i+1:]:
                intersection = self._find_wall_intersection(wall1, wall2)
                if intersection:
                    self.support_points.append(intersection)
        
        # 2. Corners of the building
        if self.floor_plan.building_dimensions:
            corners = self.floor_plan.building_dimensions.corners
            self.support_points.extend(corners)
        
        # 3. Remove duplicates
        self.support_points = self._remove_duplicate_points(self.support_points)
        
        logger.info(f"Identified {len(self.support_points)} support points")
    
    def _find_wall_intersection(self, wall1: Wall, wall2: Wall) -> Optional[Point]:
        """Find intersection point of two walls."""
        # Convert walls to line segments
        p1, p2 = wall1.start, wall1.end
        p3, p4 = wall2.start, wall2.end
        
        # Line intersection algorithm
        x1, y1 = p1.x, p1.y
        x2, y2 = p2.x, p2.y
        x3, y3 = p3.x, p3.y
        x4, y4 = p4.x, p4.y
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        
        if abs(denom) < 0.001:  # Parallel lines
            return None
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        
        # Check if intersection is within both line segments
        if 0 <= t <= 1 and 0 <= u <= 1:
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            return Point(x, y)
        
        return None
    
    def _remove_duplicate_points(self, points: List[Point], 
                                tolerance: float = 1.0) -> List[Point]:
        """Remove duplicate points within tolerance."""
        unique = []
        
        for point in points:
            is_duplicate = False
            
            for existing in unique:
                if point.distance_to(existing) < tolerance:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique.append(point)
        
        return unique
    
    def calculate_load_paths(self):
        """Calculate structural load paths."""
        logger.info("Calculating load paths")
        
        # Simplified load path calculation
        # In reality, this would involve complex structural analysis
        
        self.load_paths = {
            'primary': [],
            'secondary': []
        }
        
        # Primary load paths: from center to corners
        if self.floor_plan.building_dimensions:
            center = self.floor_plan.building_dimensions.center
            corners = self.floor_plan.building_dimensions.corners
            
            for corner in corners:
                self.load_paths['primary'].append(center)
                self.load_paths['primary'].append(corner)
        
        # Secondary load paths: along load-bearing walls
        for wall in self.load_bearing_walls:
            self.load_paths['secondary'].append(wall.start)
            self.load_paths['secondary'].append(wall.end)
    
    def determine_span_directions(self):
        """Determine optimal span direction for each room."""
        logger.info("Determining span directions")
        
        for room in self.rooms:
            # Find nearest support points
            support_distances = {
                'north': float('inf'),
                'south': float('inf'),
                'east': float('inf'),
                'west': float('inf')
            }
            
            room_center = room.center
            
            # Check walls for support
            for wall in self.load_bearing_walls:
                # North wall
                if wall.is_horizontal() and wall.start.y < room_center.y:
                    dist = abs(wall.start.y - room.position.y)
                    support_distances['north'] = min(support_distances['north'], dist)
                
                # South wall
                if wall.is_horizontal() and wall.start.y > room_center.y:
                    dist = abs(wall.start.y - (room.position.y + room.height))
                    support_distances['south'] = min(support_distances['south'], dist)
                
                # East wall
                if wall.is_vertical() and wall.start.x > room_center.x:
                    dist = abs(wall.start.x - (room.position.x + room.width))
                    support_distances['east'] = min(support_distances['east'], dist)
                
                # West wall
                if wall.is_vertical() and wall.start.x < room_center.x:
                    dist = abs(wall.start.x - room.position.x)
                    support_distances['west'] = min(support_distances['west'], dist)
            
            # Determine span direction (shortest span)
            horizontal_span = min(support_distances['east'] + support_distances['west'], room.width)
            vertical_span = min(support_distances['north'] + support_distances['south'], room.height)
            
            if horizontal_span <= vertical_span:
                room.span_direction = "horizontal"
                self.span_directions[room.id] = "horizontal"
            else:
                room.span_direction = "vertical"
                self.span_directions[room.id] = "vertical"
            
            logger.debug(f"Room {room.id}: span direction = {room.span_direction}")
    
    def calculate_max_spans(self):
        """Calculate maximum allowable spans for each room."""
        logger.info("Calculating maximum spans")
        
        for room in self.rooms:
            # Base max span on room type and load conditions
            base_span = 12.0  # Default max span in feet
            
            # Adjust for room type
            if room.type.value in ['bathroom', 'kitchen']:
                # Heavier loads in wet areas
                base_span *= 0.9
            elif room.type.value == 'bedroom':
                # Standard residential load
                base_span *= 1.0
            elif room.type.value in ['living', 'family']:
                # Potentially higher occupancy
                base_span *= 0.95
            elif room.type.value == 'garage':
                # Very heavy loads
                base_span *= 0.7
            
            # Adjust for load-bearing conditions
            if room.is_load_bearing:
                base_span *= 0.9
            
            # Apply safety factor
            max_span = base_span / self.safety_factor
            
            room.max_span = max_span
            self.max_spans[room.id] = max_span
    
    def check_compliance(self, panels: Optional[List[Panel]] = None) -> List[StructuralViolation]:
        """Check structural compliance of panel placement."""
        logger.info("Checking structural compliance")
        
        violations = []
        
        # Get all panels if not provided
        if panels is None:
            panels = []
            for room in self.rooms:
                panels.extend(room.panels)
        
        # Check each panel
        for panel in panels:
            panel_violations = self._check_panel_compliance(panel)
            violations.extend(panel_violations)
        
        # Check room-level compliance
        for room in self.rooms:
            room_violations = self._check_room_compliance(room)
            violations.extend(room_violations)
        
        # Check global compliance
        global_violations = self._check_global_compliance()
        violations.extend(global_violations)
        
        logger.info(f"Found {len(violations)} structural violations")
        
        return violations
    
    def _check_panel_compliance(self, panel: Panel) -> List[StructuralViolation]:
        """Check compliance for individual panel."""
        violations = []
        
        # 1. Check span limits
        panel_span = max(panel.width, panel.length)
        max_allowed_span = self.span_limits.get(panel.size.name, 12)
        
        if panel_span > max_allowed_span:
            violations.append(StructuralViolation(
                violation_type="span_exceeded",
                severity="critical",
                description=f"Panel {panel.panel_id} span ({panel_span:.1f}ft) exceeds limit ({max_allowed_span}ft)",
                location=panel.position,
                panel_id=panel.panel_id,
                recommendation=f"Use smaller panels or add intermediate support"
            ))
        
        # 2. Check orientation vs span direction
        room = self.floor_plan.get_room_by_id(panel.room_id)
        if room and room.span_direction:
            # Panels should generally align with span direction
            if room.span_direction == "horizontal" and panel.orientation == "vertical":
                violations.append(StructuralViolation(
                    violation_type="orientation_mismatch",
                    severity="minor",
                    description=f"Panel orientation conflicts with room span direction",
                    location=panel.position,
                    panel_id=panel.panel_id,
                    room_id=room.id,
                    recommendation="Consider rotating panel to align with span direction"
                ))
        
        # 3. Check support at panel edges
        if not self._panel_has_adequate_support(panel):
            violations.append(StructuralViolation(
                violation_type="inadequate_support",
                severity="major",
                description=f"Panel {panel.panel_id} lacks adequate edge support",
                location=panel.position,
                panel_id=panel.panel_id,
                recommendation="Ensure panel edges align with support walls or beams"
            ))
        
        return violations
    
    def _check_room_compliance(self, room: Room) -> List[StructuralViolation]:
        """Check compliance for room."""
        violations = []
        
        # 1. Check coverage in load-bearing areas
        if room.is_load_bearing and room.coverage_ratio < 0.95:
            violations.append(StructuralViolation(
                violation_type="insufficient_coverage",
                severity="major",
                description=f"Load-bearing room {room.id} has insufficient coverage ({room.coverage_ratio:.1%})",
                room_id=room.id,
                recommendation="Increase panel coverage in load-bearing areas"
            ))
        
        # 2. Check for unsupported spans
        if room.span_direction:
            span = room.width if room.span_direction == "horizontal" else room.height
            
            if span > room.max_span:
                violations.append(StructuralViolation(
                    violation_type="excessive_room_span",
                    severity="critical",
                    description=f"Room {room.id} span ({span:.1f}ft) exceeds maximum ({room.max_span:.1f}ft)",
                    room_id=room.id,
                    recommendation="Add intermediate support or use engineered joists"
                ))
        
        # 3. Check panel distribution
        if room.panels:
            panel_distribution = self._analyze_panel_distribution(room)
            
            if panel_distribution['uniformity'] < 0.7:
                violations.append(StructuralViolation(
                    violation_type="poor_load_distribution",
                    severity="minor",
                    description=f"Uneven panel distribution in room {room.id}",
                    room_id=room.id,
                    recommendation="Distribute panels more evenly for better load distribution"
                ))
        
        return violations
    
    def _check_global_compliance(self) -> List[StructuralViolation]:
        """Check overall structural compliance."""
        violations = []
        
        # 1. Check load path continuity
        if not self._check_load_path_continuity():
            violations.append(StructuralViolation(
                violation_type="load_path_discontinuity",
                severity="critical",
                description="Discontinuous load path detected",
                recommendation="Ensure continuous load transfer from roof to foundation"
            ))
        
        # 2. Check overall coverage
        total_coverage = self.floor_plan.total_coverage
        min_coverage = Config.OPTIMIZATION['targets']['min_coverage_ratio']
        
        if total_coverage < min_coverage:
            violations.append(StructuralViolation(
                violation_type="insufficient_total_coverage",
                severity="major",
                description=f"Total coverage ({total_coverage:.1%}) below minimum ({min_coverage:.1%})",
                recommendation="Add panels to uncovered areas"
            ))
        
        # 3. Check deflection limits
        max_deflection = self._estimate_max_deflection()
        deflection_limit = self._parse_deflection_limit()
        
        if max_deflection > deflection_limit:
            violations.append(StructuralViolation(
                violation_type="excessive_deflection",
                severity="major",
                description=f"Estimated deflection exceeds limit",
                recommendation="Use stiffer panels or reduce spans"
            ))
        
        return violations
    
    def _panel_has_adequate_support(self, panel: Panel) -> bool:
        """Check if panel has adequate support."""
        # Simplified check: panel should be near load-bearing walls
        panel_rect = panel.rectangle
        
        for wall in self.load_bearing_walls:
            # Check if panel edge is near wall
            for corner in panel_rect.corners:
                for wall_point in [wall.start, wall.end]:
                    if corner.distance_to(wall_point) < 2.0:  # Within 2 feet
                        return True
        
        return False
    
    def _analyze_panel_distribution(self, room: Room) -> Dict:
        """Analyze panel distribution in room."""
        if not room.panels:
            return {'uniformity': 0, 'coverage_variance': 1.0}
        
        # Divide room into grid cells
        grid_size = 4  # 4x4 grid
        cell_width = room.width / grid_size
        cell_height = room.height / grid_size
        
        # Count panels in each cell
        grid_counts = np.zeros((grid_size, grid_size))
        
        for panel in room.panels:
            # Find which cell(s) panel occupies
            panel_rect = panel.rectangle
            
            for i in range(grid_size):
                for j in range(grid_size):
                    cell_x = room.position.x + i * cell_width
                    cell_y = room.position.y + j * cell_height
                    cell_rect = Rectangle(cell_x, cell_y, cell_width, cell_height)
                    
                    if panel_rect.intersects(cell_rect):
                        grid_counts[i, j] += 1
        
        # Calculate uniformity (0 to 1, where 1 is perfectly uniform)
        if grid_counts.max() > 0:
            uniformity = grid_counts.min() / grid_counts.max()
        else:
            uniformity = 0
        
        # Calculate coverage variance
        coverage_variance = np.var(grid_counts) / np.mean(grid_counts) if np.mean(grid_counts) > 0 else 1.0
        
        return {
            'uniformity': uniformity,
            'coverage_variance': coverage_variance,
            'grid_counts': grid_counts
        }
    
    def _check_load_path_continuity(self) -> bool:
        """Check if load paths are continuous."""
        # Simplified check: ensure primary load paths exist
        return len(self.load_paths.get('primary', [])) > 0
    
    def _estimate_max_deflection(self) -> float:
        """Estimate maximum deflection in structure."""
        # Simplified deflection calculation
        # In reality, this would involve finite element analysis
        
        max_deflection = 0
        
        for room in self.rooms:
            if room.span_direction:
                span = room.width if room.span_direction == "horizontal" else room.height
                
                # Simple beam deflection formula: δ = (5 * w * L^4) / (384 * E * I)
                # Using simplified assumptions
                deflection = span / 360  # Approximate deflection
                max_deflection = max(max_deflection, deflection)
        
        return max_deflection
    
    def _parse_deflection_limit(self) -> float:
        """Parse deflection limit from config."""
        limit_str = Config.STRUCTURAL.get('deflection_limit', 'L/360')
        
        # Parse L/X format
        if '/' in limit_str:
            divisor = float(limit_str.split('/')[1])
            # Assume maximum span for L
            max_span = max(self.max_spans.values()) if self.max_spans else 12
            return max_span / divisor
        
        return 0.5  # Default to 0.5 feet
    
    def generate_recommendations(self, violations: List[StructuralViolation]) -> List[str]:
        """Generate recommendations based on violations."""
        recommendations = []
        
        # Group violations by type
        violation_types = {}
        for violation in violations:
            if violation.violation_type not in violation_types:
                violation_types[violation.violation_type] = []
            violation_types[violation.violation_type].append(violation)
        
        # Generate recommendations
        if 'span_exceeded' in violation_types:
            recommendations.append(
                "Consider using engineered joists (LVL, I-joists) for longer spans"
            )
        
        if 'inadequate_support' in violation_types:
            recommendations.append(
                "Add blocking or intermediate supports at panel edges"
            )
        
        if 'poor_load_distribution' in violation_types:
            recommendations.append(
                "Redistribute panels for more uniform load distribution"
            )
        
        if 'insufficient_coverage' in violation_types:
            recommendations.append(
                "Increase panel coverage, especially in load-bearing areas"
            )
        
        # Add general recommendations
        if len(violations) > 5:
            recommendations.append(
                "Consider consulting a structural engineer for complex issues"
            )
        
        return recommendations
    
    def export_analysis(self, filepath: str):
        """Export structural analysis to file."""
        import json
        
        analysis_data = {
            'floor_plan': self.floor_plan.name,
            'total_area': self.floor_plan.total_area,
            'load_bearing_walls': len(self.load_bearing_walls),
            'support_points': len(self.support_points),
            'span_directions': self.span_directions,
            'max_spans': self.max_spans,
            'rooms': [
                {
                    'id': room.id,
                    'type': room.type.value,
                    'is_load_bearing': room.is_load_bearing,
                    'span_direction': room.span_direction,
                    'max_span': room.max_span
                }
                for room in self.rooms
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(analysis_data, f, indent=2, default=str)
        
        logger.info(f"Structural analysis exported to {filepath}")

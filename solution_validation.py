from typing import List, Dict, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from datetime import datetime
import logging

from core import PackingState, Panel, Room, Position, PlacedPanel
from spatial_index import SpatialIndex, OccupancyGrid

logger = logging.getLogger(__name__)


class ValidationErrorType(Enum):
    OVERLAP = "overlap"
    BOUNDARY_VIOLATION = "boundary_violation"
    STRUCTURAL_VIOLATION = "structural_violation"
    COVERAGE_MISMATCH = "coverage_mismatch"
    PANEL_DUPLICATE = "panel_duplicate"
    PANEL_MISSING = "panel_missing"
    INVALID_POSITION = "invalid_position"
    INVALID_ROTATION = "invalid_rotation"
    CONSTRAINT_VIOLATION = "constraint_violation"


class ValidationSeverity(Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    error_type: ValidationErrorType
    severity: ValidationSeverity
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    panels_involved: List[int] = field(default_factory=list)
    position: Optional[Position] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ValidationResult:
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)
    
    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == ValidationSeverity.ERROR)
    
    @property
    def warning_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == ValidationSeverity.WARNING)


class OverlapChecker:
    """Checks for overlapping panels in placement"""
    
    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance
        self.overlap_count = 0
        self.total_overlap_area = 0.0
    
    def check(self, state: PackingState) -> List[ValidationIssue]:
        """Check for overlapping panels"""
        issues = []
        placed_panels = state.placed_panels
        
        # Check each pair of panels
        for i in range(len(placed_panels)):
            panel_i = placed_panels[i]
            bounds_i = self._get_panel_bounds(panel_i)
            
            for j in range(i + 1, len(placed_panels)):
                panel_j = placed_panels[j]
                bounds_j = self._get_panel_bounds(panel_j)
                
                # Check for overlap
                overlap = self._calculate_overlap(bounds_i, bounds_j)
                if overlap > self.tolerance:
                    self.overlap_count += 1
                    self.total_overlap_area += overlap
                    
                    issues.append(ValidationIssue(
                        error_type=ValidationErrorType.OVERLAP,
                        severity=ValidationSeverity.ERROR,
                        message=f"Panels {i} and {j} overlap",
                        details={
                            "panel_i_index": i,
                            "panel_j_index": j,
                            "overlap_area": overlap,
                            "panel_i_bounds": bounds_i,
                            "panel_j_bounds": bounds_j
                        },
                        panels_involved=[i, j],
                        position=panel_i.position
                    ))
        
        return issues
    
    def _get_panel_bounds(self, panel: PlacedPanel) -> Tuple[float, float, float, float]:
        """Get bounding box of placed panel"""
        x, y = panel.position.x, panel.position.y
        if panel.rotated:
            width, height = panel.panel.height, panel.panel.width
        else:
            width, height = panel.panel.width, panel.panel.height
        
        return (x, y, x + width, y + height)
    
    def _calculate_overlap(self, bounds1: Tuple[float, float, float, float],
                          bounds2: Tuple[float, float, float, float]) -> float:
        """Calculate overlap area between two rectangles"""
        x1_min, y1_min, x1_max, y1_max = bounds1
        x2_min, y2_min, x2_max, y2_max = bounds2
        
        # Calculate intersection
        x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        
        return x_overlap * y_overlap
    
    def check_with_grid(self, state: PackingState, grid: OccupancyGrid) -> List[ValidationIssue]:
        """Alternative overlap check using occupancy grid"""
        issues = []
        
        for i, panel in enumerate(state.placed_panels):
            x, y = int(panel.position.x), int(panel.position.y)
            if panel.rotated:
                width, height = int(panel.panel.height), int(panel.panel.width)
            else:
                width, height = int(panel.panel.width), int(panel.panel.height)
            
            # Check if region is free before placement
            if not grid.is_region_free(x, y, width, height):
                # Find overlapping cells
                overlapping_cells = []
                for dx in range(width):
                    for dy in range(height):
                        if grid.grid[y + dy, x + dx]:
                            overlapping_cells.append((x + dx, y + dy))
                
                if overlapping_cells:
                    issues.append(ValidationIssue(
                        error_type=ValidationErrorType.OVERLAP,
                        severity=ValidationSeverity.ERROR,
                        message=f"Panel {i} overlaps with occupied cells",
                        details={
                            "panel_index": i,
                            "overlapping_cells": overlapping_cells[:10],  # Limit to first 10
                            "total_overlapping_cells": len(overlapping_cells)
                        },
                        panels_involved=[i],
                        position=panel.position
                    ))
        
        return issues


class BoundaryDetector:
    """Detects boundary violations in panel placement"""
    
    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance
        self.violation_count = 0
        self.max_violation_distance = 0.0
    
    def check(self, state: PackingState, room: Room) -> List[ValidationIssue]:
        """Check for boundary violations"""
        issues = []
        
        for i, panel in enumerate(state.placed_panels):
            bounds = self._get_panel_bounds(panel)
            violations = self._check_room_boundaries(bounds, room)
            
            if violations:
                self.violation_count += 1
                max_dist = max(abs(v) for v in violations.values())
                self.max_violation_distance = max(self.max_violation_distance, max_dist)
                
                issues.append(ValidationIssue(
                    error_type=ValidationErrorType.BOUNDARY_VIOLATION,
                    severity=ValidationSeverity.ERROR,
                    message=f"Panel {i} violates room boundaries",
                    details={
                        "panel_index": i,
                        "violations": violations,
                        "panel_bounds": bounds,
                        "room_bounds": (0, 0, room.width, room.height)
                    },
                    panels_involved=[i],
                    position=panel.position
                ))
        
        return issues
    
    def _get_panel_bounds(self, panel: PlacedPanel) -> Tuple[float, float, float, float]:
        """Get bounding box of placed panel"""
        x, y = panel.position.x, panel.position.y
        if panel.rotated:
            width, height = panel.panel.height, panel.panel.width
        else:
            width, height = panel.panel.width, panel.panel.height
        
        return (x, y, x + width, y + height)
    
    def _check_room_boundaries(self, bounds: Tuple[float, float, float, float],
                              room: Room) -> Dict[str, float]:
        """Check if bounds violate room boundaries"""
        x_min, y_min, x_max, y_max = bounds
        violations = {}
        
        # Check left boundary
        if x_min < -self.tolerance:
            violations["left"] = x_min
        
        # Check right boundary
        if x_max > room.width + self.tolerance:
            violations["right"] = x_max - room.width
        
        # Check top boundary
        if y_min < -self.tolerance:
            violations["top"] = y_min
        
        # Check bottom boundary
        if y_max > room.height + self.tolerance:
            violations["bottom"] = y_max - room.height
        
        return violations
    
    def check_obstacles(self, state: PackingState, room: Room) -> List[ValidationIssue]:
        """Check for collisions with room obstacles"""
        issues = []
        
        if not hasattr(room, 'obstacles') or not room.obstacles:
            return issues
        
        for i, panel in enumerate(state.placed_panels):
            panel_bounds = self._get_panel_bounds(panel)
            
            for j, obstacle in enumerate(room.obstacles):
                obstacle_bounds = (
                    obstacle.x, obstacle.y,
                    obstacle.x + obstacle.width,
                    obstacle.y + obstacle.height
                )
                
                if self._rectangles_intersect(panel_bounds, obstacle_bounds):
                    issues.append(ValidationIssue(
                        error_type=ValidationErrorType.BOUNDARY_VIOLATION,
                        severity=ValidationSeverity.ERROR,
                        message=f"Panel {i} collides with obstacle {j}",
                        details={
                            "panel_index": i,
                            "obstacle_index": j,
                            "panel_bounds": panel_bounds,
                            "obstacle_bounds": obstacle_bounds
                        },
                        panels_involved=[i],
                        position=panel.position
                    ))
        
        return issues
    
    def _rectangles_intersect(self, bounds1: Tuple[float, float, float, float],
                            bounds2: Tuple[float, float, float, float]) -> bool:
        """Check if two rectangles intersect"""
        x1_min, y1_min, x1_max, y1_max = bounds1
        x2_min, y2_min, x2_max, y2_max = bounds2
        
        return not (x1_max <= x2_min or x2_max <= x1_min or
                   y1_max <= y2_min or y2_max <= y1_min)


class StructuralValidator:
    """Validates structural constraints and requirements"""
    
    def __init__(self):
        self.min_support_ratio = 0.3  # Minimum support for stability
        self.max_cantilever = 100.0  # Maximum unsupported overhang
        self.min_edge_distance = 10.0  # Minimum distance from edges
    
    def check(self, state: PackingState, room: Room) -> List[ValidationIssue]:
        """Check structural constraints"""
        issues = []
        
        # Check panel support
        support_issues = self._check_panel_support(state)
        issues.extend(support_issues)
        
        # Check edge distances
        edge_issues = self._check_edge_distances(state, room)
        issues.extend(edge_issues)
        
        # Check load distribution
        load_issues = self._check_load_distribution(state)
        issues.extend(load_issues)
        
        # Check alignment constraints
        alignment_issues = self._check_alignment(state)
        issues.extend(alignment_issues)
        
        return issues
    
    def _check_panel_support(self, state: PackingState) -> List[ValidationIssue]:
        """Check if panels have adequate support"""
        issues = []
        
        # Sort panels by y-position (bottom to top)
        sorted_panels = sorted(enumerate(state.placed_panels),
                             key=lambda x: x[1].position.y)
        
        for i, (idx, panel) in enumerate(sorted_panels):
            if i == 0:
                continue  # Bottom panels don't need support
            
            support_ratio = self._calculate_support_ratio(panel, sorted_panels[:i])
            
            if support_ratio < self.min_support_ratio:
                issues.append(ValidationIssue(
                    error_type=ValidationErrorType.STRUCTURAL_VIOLATION,
                    severity=ValidationSeverity.WARNING,
                    message=f"Panel {idx} has insufficient support",
                    details={
                        "panel_index": idx,
                        "support_ratio": support_ratio,
                        "required_ratio": self.min_support_ratio
                    },
                    panels_involved=[idx],
                    position=panel.position
                ))
        
        return issues
    
    def _calculate_support_ratio(self, panel: PlacedPanel,
                                supporting_panels: List[Tuple[int, PlacedPanel]]) -> float:
        """Calculate how much of panel is supported by panels below"""
        panel_bounds = self._get_panel_bounds(panel)
        panel_width = panel_bounds[2] - panel_bounds[0]
        
        supported_length = 0.0
        for _, support_panel in supporting_panels:
            support_bounds = self._get_panel_bounds(support_panel)
            
            # Check if support panel is directly below
            if abs(support_bounds[3] - panel_bounds[1]) < 1.0:
                # Calculate horizontal overlap
                overlap_start = max(panel_bounds[0], support_bounds[0])
                overlap_end = min(panel_bounds[2], support_bounds[2])
                if overlap_end > overlap_start:
                    supported_length += overlap_end - overlap_start
        
        return supported_length / panel_width if panel_width > 0 else 0.0
    
    def _check_edge_distances(self, state: PackingState, room: Room) -> List[ValidationIssue]:
        """Check minimum distances from room edges"""
        issues = []
        
        for i, panel in enumerate(state.placed_panels):
            bounds = self._get_panel_bounds(panel)
            
            distances = {
                "left": bounds[0],
                "right": room.width - bounds[2],
                "top": bounds[1],
                "bottom": room.height - bounds[3]
            }
            
            violations = {k: v for k, v in distances.items()
                        if v < self.min_edge_distance and v > -1e-6}
            
            if violations:
                issues.append(ValidationIssue(
                    error_type=ValidationErrorType.STRUCTURAL_VIOLATION,
                    severity=ValidationSeverity.INFO,
                    message=f"Panel {i} too close to edges",
                    details={
                        "panel_index": i,
                        "violations": violations,
                        "min_distance": self.min_edge_distance
                    },
                    panels_involved=[i],
                    position=panel.position
                ))
        
        return issues
    
    def _check_load_distribution(self, state: PackingState) -> List[ValidationIssue]:
        """Check if load is evenly distributed"""
        issues = []
        
        if len(state.placed_panels) < 2:
            return issues
        
        # Calculate center of mass
        total_mass = 0.0
        center_x = 0.0
        center_y = 0.0
        
        for panel in state.placed_panels:
            bounds = self._get_panel_bounds(panel)
            area = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
            cx = (bounds[0] + bounds[2]) / 2
            cy = (bounds[1] + bounds[3]) / 2
            
            total_mass += area
            center_x += cx * area
            center_y += cy * area
        
        if total_mass > 0:
            center_x /= total_mass
            center_y /= total_mass
            
            # Check if center of mass is reasonably centered
            room_center_x = state.room.width / 2
            room_center_y = state.room.height / 2
            
            offset_x = abs(center_x - room_center_x) / state.room.width
            offset_y = abs(center_y - room_center_y) / state.room.height
            
            if offset_x > 0.3 or offset_y > 0.3:
                issues.append(ValidationIssue(
                    error_type=ValidationErrorType.STRUCTURAL_VIOLATION,
                    severity=ValidationSeverity.INFO,
                    message="Load distribution is unbalanced",
                    details={
                        "center_of_mass": (center_x, center_y),
                        "room_center": (room_center_x, room_center_y),
                        "offset_ratio": (offset_x, offset_y)
                    }
                ))
        
        return issues
    
    def _check_alignment(self, state: PackingState) -> List[ValidationIssue]:
        """Check panel alignment constraints"""
        issues = []
        alignment_tolerance = 5.0
        
        # Group panels by similar x and y positions
        x_groups = {}
        y_groups = {}
        
        for i, panel in enumerate(state.placed_panels):
            x = panel.position.x
            y = panel.position.y
            
            # Find or create x group
            x_group_found = False
            for group_x in x_groups:
                if abs(x - group_x) < alignment_tolerance:
                    x_groups[group_x].append(i)
                    x_group_found = True
                    break
            if not x_group_found:
                x_groups[x] = [i]
            
            # Find or create y group
            y_group_found = False
            for group_y in y_groups:
                if abs(y - group_y) < alignment_tolerance:
                    y_groups[group_y].append(i)
                    y_group_found = True
                    break
            if not y_group_found:
                y_groups[y] = [i]
        
        # Check for near-misses in alignment
        for group_x, panels in x_groups.items():
            if len(panels) >= 3:
                # Multiple panels almost aligned vertically
                x_positions = [state.placed_panels[i].position.x for i in panels]
                std_dev = np.std(x_positions)
                if std_dev > 0.5 and std_dev < alignment_tolerance:
                    issues.append(ValidationIssue(
                        error_type=ValidationErrorType.STRUCTURAL_VIOLATION,
                        severity=ValidationSeverity.INFO,
                        message="Panels nearly aligned vertically",
                        details={
                            "panels": panels,
                            "x_positions": x_positions,
                            "std_deviation": std_dev
                        },
                        panels_involved=panels
                    ))
        
        return issues
    
    def _get_panel_bounds(self, panel: PlacedPanel) -> Tuple[float, float, float, float]:
        """Get bounding box of placed panel"""
        x, y = panel.position.x, panel.position.y
        if panel.rotated:
            width, height = panel.panel.height, panel.panel.width
        else:
            width, height = panel.panel.width, panel.panel.height
        
        return (x, y, x + width, y + height)


class CoverageVerifier:
    """Verifies coverage calculations and metrics"""
    
    def __init__(self, tolerance: float = 0.001):
        self.tolerance = tolerance
    
    def check(self, state: PackingState, room: Room) -> List[ValidationIssue]:
        """Verify coverage calculations"""
        issues = []
        
        # Calculate coverage independently
        calculated_coverage = self._calculate_coverage(state, room)
        
        # Compare with state's coverage
        if hasattr(state, 'coverage'):
            diff = abs(calculated_coverage - state.coverage)
            if diff > self.tolerance:
                issues.append(ValidationIssue(
                    error_type=ValidationErrorType.COVERAGE_MISMATCH,
                    severity=ValidationSeverity.WARNING,
                    message="Coverage calculation mismatch",
                    details={
                        "state_coverage": state.coverage,
                        "calculated_coverage": calculated_coverage,
                        "difference": diff
                    }
                ))
        
        # Check for duplicate panels
        duplicate_issues = self._check_duplicates(state)
        issues.extend(duplicate_issues)
        
        # Check for missing panels
        missing_issues = self._check_missing_panels(state)
        issues.extend(missing_issues)
        
        # Verify total area calculation
        area_issues = self._verify_area_calculation(state, room)
        issues.extend(area_issues)
        
        return issues
    
    def _calculate_coverage(self, state: PackingState, room: Room) -> float:
        """Calculate coverage from scratch"""
        total_panel_area = 0.0
        
        for panel in state.placed_panels:
            if panel.rotated:
                area = panel.panel.width * panel.panel.height
            else:
                area = panel.panel.width * panel.panel.height
            total_panel_area += area
        
        room_area = room.width * room.height
        return total_panel_area / room_area if room_area > 0 else 0.0
    
    def _check_duplicates(self, state: PackingState) -> List[ValidationIssue]:
        """Check for duplicate panel placements"""
        issues = []
        seen_panels = set()
        
        for i, placed_panel in enumerate(state.placed_panels):
            panel_id = id(placed_panel.panel)
            if panel_id in seen_panels:
                issues.append(ValidationIssue(
                    error_type=ValidationErrorType.PANEL_DUPLICATE,
                    severity=ValidationSeverity.ERROR,
                    message=f"Panel at index {i} is a duplicate",
                    details={
                        "panel_index": i,
                        "panel_id": panel_id
                    },
                    panels_involved=[i]
                ))
            seen_panels.add(panel_id)
        
        return issues
    
    def _check_missing_panels(self, state: PackingState) -> List[ValidationIssue]:
        """Check for panels that should be placed but aren't"""
        issues = []
        
        if hasattr(state, 'available_panels'):
            placed_ids = {id(p.panel) for p in state.placed_panels}
            available_ids = {id(p) for p in state.available_panels}
            
            # Check if high-value panels are missing
            missing_valuable = []
            for panel in state.available_panels:
                if id(panel) not in placed_ids:
                    # Check if this is a valuable panel that should be placed
                    if hasattr(panel, 'value') and panel.value > 100:
                        missing_valuable.append(panel)
            
            if missing_valuable:
                issues.append(ValidationIssue(
                    error_type=ValidationErrorType.PANEL_MISSING,
                    severity=ValidationSeverity.INFO,
                    message=f"{len(missing_valuable)} high-value panels not placed",
                    details={
                        "count": len(missing_valuable),
                        "total_value": sum(p.value for p in missing_valuable if hasattr(p, 'value'))
                    }
                ))
        
        return issues
    
    def _verify_area_calculation(self, state: PackingState, room: Room) -> List[ValidationIssue]:
        """Verify area calculations are correct"""
        issues = []
        
        # Check individual panel areas
        for i, placed_panel in enumerate(state.placed_panels):
            panel = placed_panel.panel
            expected_area = panel.width * panel.height
            
            # Verify dimensions are positive
            if panel.width <= 0 or panel.height <= 0:
                issues.append(ValidationIssue(
                    error_type=ValidationErrorType.INVALID_POSITION,
                    severity=ValidationSeverity.ERROR,
                    message=f"Panel {i} has invalid dimensions",
                    details={
                        "panel_index": i,
                        "width": panel.width,
                        "height": panel.height
                    },
                    panels_involved=[i]
                ))
        
        # Verify room area
        room_area = room.width * room.height
        if room_area <= 0:
            issues.append(ValidationIssue(
                error_type=ValidationErrorType.STRUCTURAL_VIOLATION,
                severity=ValidationSeverity.ERROR,
                message="Room has invalid dimensions",
                details={
                    "width": room.width,
                    "height": room.height,
                    "area": room_area
                }
            ))
        
        return issues


class SolutionValidator:
    """Main validator that coordinates all validation checks"""
    
    def __init__(self):
        self.overlap_checker = OverlapChecker()
        self.boundary_detector = BoundaryDetector()
        self.structural_validator = StructuralValidator()
        self.coverage_verifier = CoverageVerifier()
        self.validation_cache = {}
    
    def validate(self, state: PackingState, room: Room,
                spatial_index: Optional[SpatialIndex] = None,
                occupancy_grid: Optional[OccupancyGrid] = None) -> ValidationResult:
        """Perform comprehensive validation of packing solution"""
        
        # Check cache
        state_hash = hash(state)
        if state_hash in self.validation_cache:
            logger.debug("Using cached validation result")
            return self.validation_cache[state_hash]
        
        issues = []
        metrics = {}
        suggestions = []
        
        # Run overlap checks
        overlap_issues = self.overlap_checker.check(state)
        issues.extend(overlap_issues)
        metrics["overlap_count"] = self.overlap_checker.overlap_count
        metrics["total_overlap_area"] = self.overlap_checker.total_overlap_area
        
        # Additional overlap check with grid if available
        if occupancy_grid:
            grid_overlap_issues = self.overlap_checker.check_with_grid(state, occupancy_grid)
            issues.extend(grid_overlap_issues)
        
        # Run boundary checks
        boundary_issues = self.boundary_detector.check(state, room)
        issues.extend(boundary_issues)
        metrics["boundary_violations"] = self.boundary_detector.violation_count
        metrics["max_violation_distance"] = self.boundary_detector.max_violation_distance
        
        # Check obstacles if present
        obstacle_issues = self.boundary_detector.check_obstacles(state, room)
        issues.extend(obstacle_issues)
        
        # Run structural validation
        structural_issues = self.structural_validator.check(state, room)
        issues.extend(structural_issues)
        
        # Verify coverage
        coverage_issues = self.coverage_verifier.check(state, room)
        issues.extend(coverage_issues)
        
        # Determine overall validity
        has_errors = any(issue.severity == ValidationSeverity.ERROR for issue in issues)
        is_valid = not has_errors
        
        # Generate suggestions based on issues
        suggestions = self._generate_suggestions(issues, state, room)
        
        # Create result
        result = ValidationResult(
            is_valid=is_valid,
            issues=issues,
            metrics=metrics,
            suggestions=suggestions
        )
        
        # Cache result
        self.validation_cache[state_hash] = result
        
        # Log summary
        logger.info(f"Validation complete: valid={is_valid}, "
                   f"errors={result.error_count}, warnings={result.warning_count}")
        
        return result
    
    def _generate_suggestions(self, issues: List[ValidationIssue],
                            state: PackingState, room: Room) -> List[str]:
        """Generate improvement suggestions based on issues"""
        suggestions = []
        
        # Count issue types
        issue_counts = {}
        for issue in issues:
            issue_counts[issue.error_type] = issue_counts.get(issue.error_type, 0) + 1
        
        # Generate suggestions based on common issues
        if issue_counts.get(ValidationErrorType.OVERLAP, 0) > 0:
            suggestions.append("Consider using smaller panels or better placement algorithm to avoid overlaps")
        
        if issue_counts.get(ValidationErrorType.BOUNDARY_VIOLATION, 0) > 0:
            suggestions.append("Ensure panels are placed within room boundaries with proper margin")
        
        if issue_counts.get(ValidationErrorType.STRUCTURAL_VIOLATION, 0) > 0:
            suggestions.append("Improve panel support and alignment for better structural integrity")
        
        if issue_counts.get(ValidationErrorType.COVERAGE_MISMATCH, 0) > 0:
            suggestions.append("Verify coverage calculation methodology")
        
        # Check for optimization opportunities
        coverage = self.coverage_verifier._calculate_coverage(state, room)
        if coverage < 0.9:
            suggestions.append(f"Current coverage is {coverage:.1%}. Consider more aggressive packing strategies")
        
        if len(state.placed_panels) < 10 and coverage < 0.5:
            suggestions.append("Very few panels placed. Check if panel sizes are appropriate for room")
        
        return suggestions
    
    def validate_incremental(self, state: PackingState, new_panel: PlacedPanel,
                           room: Room) -> ValidationResult:
        """Validate just the addition of a new panel (faster than full validation)"""
        issues = []
        
        # Check new panel against existing panels for overlap
        new_bounds = self._get_panel_bounds(new_panel)
        for i, existing_panel in enumerate(state.placed_panels):
            existing_bounds = self._get_panel_bounds(existing_panel)
            overlap = self._calculate_overlap(new_bounds, existing_bounds)
            
            if overlap > 1e-6:
                issues.append(ValidationIssue(
                    error_type=ValidationErrorType.OVERLAP,
                    severity=ValidationSeverity.ERROR,
                    message=f"New panel overlaps with panel {i}",
                    details={
                        "existing_panel_index": i,
                        "overlap_area": overlap
                    },
                    panels_involved=[i],
                    position=new_panel.position
                ))
        
        # Check boundary violations for new panel
        violations = self._check_room_boundaries(new_bounds, room)
        if violations:
            issues.append(ValidationIssue(
                error_type=ValidationErrorType.BOUNDARY_VIOLATION,
                severity=ValidationSeverity.ERROR,
                message="New panel violates room boundaries",
                details={"violations": violations},
                position=new_panel.position
            ))
        
        is_valid = not any(issue.severity == ValidationSeverity.ERROR for issue in issues)
        
        return ValidationResult(
            is_valid=is_valid,
            issues=issues
        )
    
    def _get_panel_bounds(self, panel: PlacedPanel) -> Tuple[float, float, float, float]:
        """Get bounding box of placed panel"""
        x, y = panel.position.x, panel.position.y
        if panel.rotated:
            width, height = panel.panel.height, panel.panel.width
        else:
            width, height = panel.panel.width, panel.panel.height
        
        return (x, y, x + width, y + height)
    
    def _calculate_overlap(self, bounds1: Tuple[float, float, float, float],
                          bounds2: Tuple[float, float, float, float]) -> float:
        """Calculate overlap area between two rectangles"""
        x1_min, y1_min, x1_max, y1_max = bounds1
        x2_min, y2_min, x2_max, y2_max = bounds2
        
        x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        
        return x_overlap * y_overlap
    
    def _check_room_boundaries(self, bounds: Tuple[float, float, float, float],
                              room: Room) -> Dict[str, float]:
        """Check if bounds violate room boundaries"""
        x_min, y_min, x_max, y_max = bounds
        violations = {}
        
        if x_min < 0:
            violations["left"] = x_min
        if x_max > room.width:
            violations["right"] = x_max - room.width
        if y_min < 0:
            violations["top"] = y_min
        if y_max > room.height:
            violations["bottom"] = y_max - room.height
        
        return violations
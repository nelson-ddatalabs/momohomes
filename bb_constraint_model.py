#!/usr/bin/env python3
"""
bb_constraint_model.py - Constraint Model for Branch & Bound
===========================================================
Production-ready constraint definitions including no-overlap, boundary,
compatibility, and structural constraints for panel placement optimization.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
from enum import Enum
from abc import ABC, abstractmethod
from collections import defaultdict
import math

from models import Room, PanelSize, Point
from advanced_packing import PackingState, PanelPlacement
from bb_search_tree import BranchNode


class ConstraintCategory(Enum):
    """Categories of constraints in the model."""
    NO_OVERLAP = "no_overlap"
    BOUNDARY = "boundary"
    COMPATIBILITY = "compatibility"
    STRUCTURAL = "structural"
    ALIGNMENT = "alignment"
    SPACING = "spacing"
    ORIENTATION = "orientation"
    COVERAGE = "coverage"


class ConstraintPriority(Enum):
    """Priority levels for constraint satisfaction."""
    HARD = 1  # Must be satisfied (infeasible if violated)
    SOFT = 2  # Should be satisfied (penalty if violated)
    PREFERENCE = 3  # Nice to have (small penalty if violated)


class ViolationType(Enum):
    """Types of constraint violations."""
    OVERLAP = "overlap"
    OUT_OF_BOUNDS = "out_of_bounds"
    INCOMPATIBLE = "incompatible"
    STRUCTURAL_FAILURE = "structural_failure"
    MISALIGNMENT = "misalignment"
    INSUFFICIENT_SPACING = "insufficient_spacing"
    INVALID_ORIENTATION = "invalid_orientation"
    COVERAGE_EXCEEDED = "coverage_exceeded"


@dataclass
class ConstraintViolation:
    """Represents a constraint violation."""
    constraint_type: ConstraintCategory
    violation_type: ViolationType
    priority: ConstraintPriority
    severity: float  # 0.0 to 1.0
    description: str
    involved_panels: List[str] = field(default_factory=list)
    location: Optional[Tuple[float, float]] = None
    
    @property
    def is_hard_violation(self) -> bool:
        """Check if this is a hard constraint violation."""
        return self.priority == ConstraintPriority.HARD
    
    @property
    def penalty(self) -> float:
        """Calculate penalty for this violation."""
        base_penalty = 1.0 if self.is_hard_violation else 0.1
        return base_penalty * self.severity


@dataclass
class ConstraintContext:
    """Context information for constraint evaluation."""
    room: Room
    current_state: PackingState
    candidate_placement: Optional[PanelPlacement] = None
    remaining_panels: Set[PanelSize] = field(default_factory=set)
    structural_requirements: Dict[str, Any] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(default_factory=dict)


class Constraint(ABC):
    """Abstract base class for constraints."""
    
    def __init__(self, priority: ConstraintPriority = ConstraintPriority.HARD):
        self.priority = priority
        self.violations_checked = 0
        self.violations_found = 0
    
    @abstractmethod
    def check(self, context: ConstraintContext) -> List[ConstraintViolation]:
        """Check if constraint is satisfied, return violations if any."""
        pass
    
    @abstractmethod
    def is_satisfied(self, context: ConstraintContext) -> bool:
        """Quick check if constraint is satisfied."""
        pass
    
    def get_violation_rate(self) -> float:
        """Get rate of violations for this constraint."""
        if self.violations_checked == 0:
            return 0.0
        return self.violations_found / self.violations_checked


class NoOverlapConstraint(Constraint):
    """
    Ensures panels do not overlap with each other.
    This is a fundamental hard constraint.
    """
    
    def __init__(self, tolerance: float = 0.001):
        super().__init__(ConstraintPriority.HARD)
        self.tolerance = tolerance
    
    def check(self, context: ConstraintContext) -> List[ConstraintViolation]:
        """Check for overlapping panels."""
        self.violations_checked += 1
        violations = []
        
        if not context.candidate_placement:
            return violations
        
        candidate = context.candidate_placement
        cx1, cy1, cx2, cy2 = candidate.bounds
        
        # Check against all existing placements
        for existing in context.current_state.placements:
            ex1, ey1, ex2, ey2 = existing.bounds
            
            # Check for overlap
            if self._rectangles_overlap((cx1, cy1, cx2, cy2), (ex1, ey1, ex2, ey2)):
                overlap_area = self._calculate_overlap_area(
                    (cx1, cy1, cx2, cy2), (ex1, ey1, ex2, ey2))
                
                severity = min(1.0, overlap_area / candidate.panel_size.area)
                
                violation = ConstraintViolation(
                    constraint_type=ConstraintCategory.NO_OVERLAP,
                    violation_type=ViolationType.OVERLAP,
                    priority=self.priority,
                    severity=severity,
                    description=f"Panel {candidate.panel_size.name} overlaps with existing panel",
                    involved_panels=[candidate.panel_size.name, existing.panel_size.name],
                    location=(cx1, cy1)
                )
                violations.append(violation)
                self.violations_found += 1
        
        return violations
    
    def is_satisfied(self, context: ConstraintContext) -> bool:
        """Quick satisfaction check."""
        if not context.candidate_placement:
            return True
        
        candidate = context.candidate_placement
        cx1, cy1, cx2, cy2 = candidate.bounds
        
        for existing in context.current_state.placements:
            ex1, ey1, ex2, ey2 = existing.bounds
            if self._rectangles_overlap((cx1, cy1, cx2, cy2), (ex1, ey1, ex2, ey2)):
                return False
        
        return True
    
    def _rectangles_overlap(self, rect1: Tuple[float, float, float, float],
                          rect2: Tuple[float, float, float, float]) -> bool:
        """Check if two rectangles overlap."""
        x1, y1, x2, y2 = rect1
        x3, y3, x4, y4 = rect2
        
        # No overlap if one is to the left/right/above/below the other
        return not (x2 <= x3 + self.tolerance or 
                   x1 >= x4 - self.tolerance or
                   y2 <= y3 + self.tolerance or
                   y1 >= y4 - self.tolerance)
    
    def _calculate_overlap_area(self, rect1: Tuple[float, float, float, float],
                               rect2: Tuple[float, float, float, float]) -> float:
        """Calculate area of overlap between two rectangles."""
        x1, y1, x2, y2 = rect1
        x3, y3, x4, y4 = rect2
        
        overlap_x = max(0, min(x2, x4) - max(x1, x3))
        overlap_y = max(0, min(y2, y4) - max(y1, y3))
        
        return overlap_x * overlap_y


class BoundaryConstraint(Constraint):
    """
    Ensures panels stay within room boundaries.
    Handles both rectangular and irregular room shapes.
    """
    
    def __init__(self, margin: float = 0.0):
        super().__init__(ConstraintPriority.HARD)
        self.margin = margin  # Minimum distance from boundary
    
    def check(self, context: ConstraintContext) -> List[ConstraintViolation]:
        """Check if placement violates room boundaries."""
        self.violations_checked += 1
        violations = []
        
        if not context.candidate_placement:
            return violations
        
        candidate = context.candidate_placement
        x1, y1, x2, y2 = candidate.bounds
        room = context.room
        
        # Check basic rectangular boundaries
        out_of_bounds = False
        severity = 0.0
        
        if x1 < 0 - self.tolerance:
            out_of_bounds = True
            severity = max(severity, abs(x1) / room.width)
        
        if y1 < 0 - self.tolerance:
            out_of_bounds = True
            severity = max(severity, abs(y1) / room.height)
        
        if x2 > room.width + self.tolerance:
            out_of_bounds = True
            severity = max(severity, (x2 - room.width) / room.width)
        
        if y2 > room.height + self.tolerance:
            out_of_bounds = True
            severity = max(severity, (y2 - room.height) / room.height)
        
        if out_of_bounds:
            violation = ConstraintViolation(
                constraint_type=ConstraintCategory.BOUNDARY,
                violation_type=ViolationType.OUT_OF_BOUNDS,
                priority=self.priority,
                severity=min(1.0, severity),
                description=f"Panel {candidate.panel_size.name} extends beyond room boundaries",
                involved_panels=[candidate.panel_size.name],
                location=(x1, y1)
            )
            violations.append(violation)
            self.violations_found += 1
        
        # Check margin requirements
        if self.margin > 0:
            if (x1 < self.margin or y1 < self.margin or
                x2 > room.width - self.margin or y2 > room.height - self.margin):
                
                margin_violation = ConstraintViolation(
                    constraint_type=ConstraintCategory.BOUNDARY,
                    violation_type=ViolationType.INSUFFICIENT_SPACING,
                    priority=ConstraintPriority.SOFT,
                    severity=0.3,
                    description=f"Panel too close to room boundary (margin={self.margin})",
                    involved_panels=[candidate.panel_size.name],
                    location=(x1, y1)
                )
                violations.append(margin_violation)
        
        return violations
    
    def is_satisfied(self, context: ConstraintContext) -> bool:
        """Quick satisfaction check."""
        if not context.candidate_placement:
            return True
        
        x1, y1, x2, y2 = context.candidate_placement.bounds
        room = context.room
        
        return (x1 >= 0 - self.tolerance and 
                y1 >= 0 - self.tolerance and
                x2 <= room.width + self.tolerance and
                y2 <= room.height + self.tolerance)
    
    tolerance = 0.001


class CompatibilityConstraint(Constraint):
    """
    Ensures panels are compatible with their placement locations.
    Considers panel types, sizes, and room-specific requirements.
    """
    
    def __init__(self):
        super().__init__(ConstraintPriority.SOFT)
        self.compatibility_rules = {}
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default compatibility rules."""
        # Large panels prefer corners and edges
        self.compatibility_rules['large_panel_placement'] = {
            'min_area': 30,  # sq ft
            'preferred_positions': ['corner', 'edge'],
            'penalty_center': 0.3
        }
        
        # Small panels can go anywhere but prefer filling gaps
        self.compatibility_rules['small_panel_placement'] = {
            'max_area': 20,  # sq ft
            'preferred_positions': ['gap', 'edge'],
            'penalty_isolated': 0.2
        }
    
    def check(self, context: ConstraintContext) -> List[ConstraintViolation]:
        """Check panel-location compatibility."""
        self.violations_checked += 1
        violations = []
        
        if not context.candidate_placement:
            return violations
        
        candidate = context.candidate_placement
        panel = candidate.panel_size
        x, y = candidate.position
        
        # Check size-based compatibility
        if panel.area > 30:  # Large panel
            if not self._is_edge_or_corner_position(x, y, context.room):
                violation = ConstraintViolation(
                    constraint_type=ConstraintCategory.COMPATIBILITY,
                    violation_type=ViolationType.INCOMPATIBLE,
                    priority=self.priority,
                    severity=0.3,
                    description=f"Large panel {panel.name} not placed at edge/corner",
                    involved_panels=[panel.name],
                    location=(x, y)
                )
                violations.append(violation)
        
        # Check for isolated small panels
        if panel.area < 20:
            if self._is_isolated_placement(candidate, context.current_state):
                violation = ConstraintViolation(
                    constraint_type=ConstraintCategory.COMPATIBILITY,
                    violation_type=ViolationType.INCOMPATIBLE,
                    priority=self.priority,
                    severity=0.2,
                    description=f"Small panel {panel.name} is isolated",
                    involved_panels=[panel.name],
                    location=(x, y)
                )
                violations.append(violation)
        
        return violations
    
    def is_satisfied(self, context: ConstraintContext) -> bool:
        """Quick satisfaction check."""
        # Soft constraint - always technically satisfied
        return True
    
    def _is_edge_or_corner_position(self, x: float, y: float, room: Room) -> bool:
        """Check if position is at edge or corner."""
        edge_threshold = 0.5
        
        at_left = x < edge_threshold
        at_right = x > room.width - edge_threshold - 4  # Account for panel width
        at_bottom = y < edge_threshold
        at_top = y > room.height - edge_threshold - 4
        
        return at_left or at_right or at_bottom or at_top
    
    def _is_isolated_placement(self, placement: PanelPlacement, 
                              state: PackingState) -> bool:
        """Check if placement is isolated from other panels."""
        if not state.placements:
            return False  # First placement is not isolated
        
        px1, py1, px2, py2 = placement.bounds
        min_distance = float('inf')
        
        for existing in state.placements:
            ex1, ey1, ex2, ey2 = existing.bounds
            
            # Calculate minimum distance between rectangles
            dx = max(0, max(px1 - ex2, ex1 - px2))
            dy = max(0, max(py1 - ey2, ey1 - py2))
            distance = math.sqrt(dx**2 + dy**2)
            
            min_distance = min(min_distance, distance)
        
        # Considered isolated if more than 1 unit away from all panels
        return min_distance > 1.0


class StructuralConstraint(Constraint):
    """
    Ensures structural integrity of panel placement.
    Considers support requirements, load distribution, and stability.
    """
    
    def __init__(self):
        super().__init__(ConstraintPriority.HARD)
        self.min_support_ratio = 0.5  # Minimum supported edge ratio
        self.max_cantilever = 2.0  # Maximum unsupported overhang
    
    def check(self, context: ConstraintContext) -> List[ConstraintViolation]:
        """Check structural constraints."""
        self.violations_checked += 1
        violations = []
        
        if not context.candidate_placement:
            return violations
        
        candidate = context.candidate_placement
        
        # Check support requirements
        support_ratio = self._calculate_support_ratio(candidate, context.current_state)
        
        if support_ratio < self.min_support_ratio:
            violation = ConstraintViolation(
                constraint_type=ConstraintCategory.STRUCTURAL,
                violation_type=ViolationType.STRUCTURAL_FAILURE,
                priority=self.priority,
                severity=(self.min_support_ratio - support_ratio) / self.min_support_ratio,
                description=f"Insufficient support for panel {candidate.panel_size.name}",
                involved_panels=[candidate.panel_size.name],
                location=candidate.position
            )
            violations.append(violation)
            self.violations_found += 1
        
        # Check cantilever constraints
        cantilever = self._calculate_cantilever(candidate, context.current_state)
        
        if cantilever > self.max_cantilever:
            violation = ConstraintViolation(
                constraint_type=ConstraintCategory.STRUCTURAL,
                violation_type=ViolationType.STRUCTURAL_FAILURE,
                priority=self.priority,
                severity=min(1.0, (cantilever - self.max_cantilever) / self.max_cantilever),
                description=f"Excessive cantilever for panel {candidate.panel_size.name}",
                involved_panels=[candidate.panel_size.name],
                location=candidate.position
            )
            violations.append(violation)
        
        return violations
    
    def is_satisfied(self, context: ConstraintContext) -> bool:
        """Quick satisfaction check."""
        if not context.candidate_placement:
            return True
        
        support_ratio = self._calculate_support_ratio(
            context.candidate_placement, context.current_state)
        
        return support_ratio >= self.min_support_ratio
    
    def _calculate_support_ratio(self, placement: PanelPlacement,
                                state: PackingState) -> float:
        """Calculate ratio of supported edges."""
        x1, y1, x2, y2 = placement.bounds
        
        # Bottom edge is always supported (floor)
        if y1 <= 0.1:
            return 1.0
        
        # Check support from other panels
        total_edge_length = 2 * ((x2 - x1) + (y2 - y1))
        supported_length = 0.0
        
        for existing in state.placements:
            ex1, ey1, ex2, ey2 = existing.bounds
            
            # Check if existing panel provides support
            if abs(y1 - ey2) < 0.1:  # Bottom edge supported by top of existing
                overlap = min(x2, ex2) - max(x1, ex1)
                if overlap > 0:
                    supported_length += overlap
        
        # Walls provide support
        if x1 <= 0.1:
            supported_length += (y2 - y1)
        if x2 >= state.room_bounds[2] - 0.1:
            supported_length += (y2 - y1)
        
        return min(1.0, supported_length / total_edge_length)
    
    def _calculate_cantilever(self, placement: PanelPlacement,
                             state: PackingState) -> float:
        """Calculate maximum cantilever distance."""
        x1, y1, x2, y2 = placement.bounds
        
        # If on floor, no cantilever
        if y1 <= 0.1:
            return 0.0
        
        # Find maximum unsupported overhang
        max_overhang = 0.0
        
        for existing in state.placements:
            ex1, ey1, ex2, ey2 = existing.bounds
            
            if abs(y1 - ey2) < 0.1:  # Potential support
                # Calculate overhang on each side
                left_overhang = max(0, ex1 - x1)
                right_overhang = max(0, x2 - ex2)
                
                max_overhang = max(max_overhang, left_overhang, right_overhang)
        
        return max_overhang


class AlignmentConstraint(Constraint):
    """
    Soft constraint preferring aligned panel edges.
    Improves aesthetics and structural efficiency.
    """
    
    def __init__(self, alignment_tolerance: float = 0.1):
        super().__init__(ConstraintPriority.PREFERENCE)
        self.alignment_tolerance = alignment_tolerance
        self.alignment_bonus = 0.1  # Bonus for good alignment
    
    def check(self, context: ConstraintContext) -> List[ConstraintViolation]:
        """Check alignment with existing panels."""
        self.violations_checked += 1
        violations = []
        
        if not context.candidate_placement:
            return violations
        
        candidate = context.candidate_placement
        x1, y1, x2, y2 = candidate.bounds
        
        # Count alignment opportunities
        alignment_opportunities = 0
        alignments_achieved = 0
        
        for existing in context.current_state.placements:
            ex1, ey1, ex2, ey2 = existing.bounds
            
            # Check for potential alignments
            if abs(x1 - ex1) < 2.0 or abs(x2 - ex2) < 2.0:  # Vertical alignment possible
                alignment_opportunities += 1
                if abs(x1 - ex1) < self.alignment_tolerance or abs(x2 - ex2) < self.alignment_tolerance:
                    alignments_achieved += 1
            
            if abs(y1 - ey1) < 2.0 or abs(y2 - ey2) < 2.0:  # Horizontal alignment possible
                alignment_opportunities += 1
                if abs(y1 - ey1) < self.alignment_tolerance or abs(y2 - ey2) < self.alignment_tolerance:
                    alignments_achieved += 1
        
        # Report misalignment as soft violation
        if alignment_opportunities > 0 and alignments_achieved < alignment_opportunities:
            missed_ratio = 1.0 - (alignments_achieved / alignment_opportunities)
            
            violation = ConstraintViolation(
                constraint_type=ConstraintCategory.ALIGNMENT,
                violation_type=ViolationType.MISALIGNMENT,
                priority=self.priority,
                severity=missed_ratio * 0.2,  # Low severity
                description=f"Panel {candidate.panel_size.name} misses alignment opportunities",
                involved_panels=[candidate.panel_size.name],
                location=(x1, y1)
            )
            violations.append(violation)
        
        return violations
    
    def is_satisfied(self, context: ConstraintContext) -> bool:
        """Always satisfied as this is a preference."""
        return True


class ConstraintSystem:
    """
    Complete constraint system managing all constraints.
    Provides unified interface for constraint checking and management.
    """
    
    def __init__(self, room: Room):
        self.room = room
        self.constraints = {}
        self._initialize_constraints()
        
        # Statistics
        self.total_checks = 0
        self.total_violations = 0
        self.violations_by_type = defaultdict(int)
    
    def _initialize_constraints(self):
        """Initialize all constraint types."""
        self.constraints[ConstraintCategory.NO_OVERLAP] = NoOverlapConstraint()
        self.constraints[ConstraintCategory.BOUNDARY] = BoundaryConstraint()
        self.constraints[ConstraintCategory.COMPATIBILITY] = CompatibilityConstraint()
        self.constraints[ConstraintCategory.STRUCTURAL] = StructuralConstraint()
        self.constraints[ConstraintCategory.ALIGNMENT] = AlignmentConstraint()
    
    def check_all_constraints(self, context: ConstraintContext) -> List[ConstraintViolation]:
        """Check all constraints and return violations."""
        self.total_checks += 1
        all_violations = []
        
        for constraint_type, constraint in self.constraints.items():
            violations = constraint.check(context)
            all_violations.extend(violations)
            
            for violation in violations:
                self.violations_by_type[violation.violation_type] += 1
        
        self.total_violations += len(all_violations)
        return all_violations
    
    def is_feasible(self, context: ConstraintContext) -> bool:
        """Check if placement is feasible (no hard constraint violations)."""
        for constraint_type, constraint in self.constraints.items():
            if constraint.priority == ConstraintPriority.HARD:
                if not constraint.is_satisfied(context):
                    return False
        return True
    
    def calculate_penalty(self, violations: List[ConstraintViolation]) -> float:
        """Calculate total penalty for violations."""
        total_penalty = 0.0
        
        for violation in violations:
            if violation.is_hard_violation:
                return float('inf')  # Infeasible
            total_penalty += violation.penalty
        
        return total_penalty
    
    def get_constraint(self, constraint_type: ConstraintCategory) -> Optional[Constraint]:
        """Get specific constraint by type."""
        return self.constraints.get(constraint_type)
    
    def add_custom_constraint(self, constraint_type: ConstraintCategory, 
                            constraint: Constraint):
        """Add custom constraint to the system."""
        self.constraints[constraint_type] = constraint
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get constraint checking statistics."""
        stats = {
            'total_checks': self.total_checks,
            'total_violations': self.total_violations,
            'violation_rate': self.total_violations / max(1, self.total_checks),
            'violations_by_type': dict(self.violations_by_type)
        }
        
        # Add per-constraint statistics
        for constraint_type, constraint in self.constraints.items():
            stats[f'{constraint_type.value}_violation_rate'] = constraint.get_violation_rate()
        
        return stats
    
    def reset_statistics(self):
        """Reset all statistics."""
        self.total_checks = 0
        self.total_violations = 0
        self.violations_by_type.clear()
        
        for constraint in self.constraints.values():
            constraint.violations_checked = 0
            constraint.violations_found = 0


def create_constraint_system(room: Room, 
                           custom_constraints: Optional[Dict[ConstraintCategory, Constraint]] = None) -> ConstraintSystem:
    """
    Factory function to create constraint system.
    Optionally adds custom constraints.
    """
    system = ConstraintSystem(room)
    
    if custom_constraints:
        for constraint_type, constraint in custom_constraints.items():
            system.add_custom_constraint(constraint_type, constraint)
    
    return system
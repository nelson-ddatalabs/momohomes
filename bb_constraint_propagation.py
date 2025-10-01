#!/usr/bin/env python3
"""
bb_constraint_propagation.py - Constraint Propagation Engine
===========================================================
Production-ready arc consistency, domain reduction, queue management,
and propagation optimization for efficient constraint solving.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Deque
from collections import deque, defaultdict
from enum import Enum
from abc import ABC, abstractmethod
import math
import heapq

from models import Room, PanelSize, Point
from advanced_packing import PackingState, PanelPlacement
from bb_search_tree import BranchNode
from bb_constraint_model import (
    ConstraintCategory, ConstraintViolation, ConstraintContext,
    ConstraintSystem, Constraint
)


class ConsistencyLevel(Enum):
    """Different levels of consistency enforcement."""
    FORWARD_CHECKING = "forward_checking"  # Check immediate constraints
    AC3 = "ac3"  # Arc consistency
    PATH_CONSISTENCY = "path_consistency"  # Path consistency (stronger)
    FULL = "full"  # Full consistency checking


class DomainValue:
    """Represents a value in a variable's domain."""
    def __init__(self, position: Tuple[float, float], orientation: str):
        self.position = position
        self.orientation = orientation
        self.is_valid = True
        self.support_count = 0
        self.supporting_values = set()
    
    def __hash__(self):
        return hash((self.position, self.orientation))
    
    def __eq__(self, other):
        return self.position == other.position and self.orientation == other.orientation


@dataclass
class Variable:
    """Represents a constraint variable (panel to be placed)."""
    panel_size: PanelSize
    domain: Set[DomainValue]
    domain_size_history: List[int] = field(default_factory=list)
    
    def add_domain_value(self, value: DomainValue):
        """Add a value to the domain."""
        self.domain.add(value)
    
    def remove_domain_value(self, value: DomainValue):
        """Remove a value from the domain."""
        if value in self.domain:
            self.domain.remove(value)
            self.domain_size_history.append(len(self.domain))
    
    def is_assigned(self) -> bool:
        """Check if variable has been assigned (domain size = 1)."""
        return len(self.domain) == 1
    
    def is_unassigned(self) -> bool:
        """Check if variable has empty domain (no valid assignment)."""
        return len(self.domain) == 0
    
    def get_domain_reduction_rate(self) -> float:
        """Get the rate of domain reduction."""
        if not self.domain_size_history:
            return 0.0
        initial = self.domain_size_history[0] if self.domain_size_history else len(self.domain)
        current = len(self.domain)
        if initial == 0:
            return 0.0
        return 1.0 - (current / initial)


@dataclass
class Arc:
    """Represents a constraint arc between two variables."""
    source: Variable
    target: Variable
    constraint_type: ConstraintCategory
    
    def __hash__(self):
        return hash((id(self.source), id(self.target), self.constraint_type))
    
    def __eq__(self, other):
        return (self.source == other.source and 
                self.target == other.target and
                self.constraint_type == other.constraint_type)


class PropagationQueue:
    """
    Manages the queue of arcs to be processed during constraint propagation.
    Supports different queueing strategies for optimization.
    """
    
    def __init__(self, strategy: str = "fifo"):
        self.strategy = strategy
        self.queue = deque() if strategy == "fifo" else []
        self.in_queue = set()
        self.process_count = 0
        self.revision_count = 0
        
    def add_arc(self, arc: Arc, priority: float = 0.0):
        """Add an arc to the queue."""
        if arc not in self.in_queue:
            if self.strategy == "fifo":
                self.queue.append(arc)
            elif self.strategy == "priority":
                heapq.heappush(self.queue, (priority, self.process_count, arc))
                self.process_count += 1
            else:  # lifo
                self.queue.append(arc)
            self.in_queue.add(arc)
    
    def get_next_arc(self) -> Optional[Arc]:
        """Get the next arc to process."""
        if not self.has_arcs():
            return None
        
        if self.strategy == "fifo":
            arc = self.queue.popleft()
        elif self.strategy == "priority":
            _, _, arc = heapq.heappop(self.queue)
        else:  # lifo
            arc = self.queue.pop()
        
        self.in_queue.discard(arc)
        self.revision_count += 1
        return arc
    
    def has_arcs(self) -> bool:
        """Check if queue has arcs to process."""
        return len(self.queue) > 0
    
    def clear(self):
        """Clear the queue."""
        self.queue.clear()
        self.in_queue.clear()
    
    def size(self) -> int:
        """Get current queue size."""
        return len(self.queue)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get queue statistics."""
        return {
            'strategy': self.strategy,
            'current_size': self.size(),
            'total_processed': self.process_count,
            'revisions': self.revision_count
        }


class DomainReducer:
    """
    Implements domain reduction techniques to prune invalid values.
    """
    
    def __init__(self, room: Room, constraint_system: ConstraintSystem):
        self.room = room
        self.constraint_system = constraint_system
        self.reduction_count = 0
        self.total_values_removed = 0
        
    def reduce_domain_bounds(self, variable: Variable) -> int:
        """
        Reduce domain based on boundary constraints.
        Returns number of values removed.
        """
        removed_count = 0
        invalid_values = []
        
        for value in variable.domain:
            x, y = value.position
            width, height = variable.panel_size.get_dimensions()
            
            if value.orientation == "vertical":
                width, height = height, width
            
            # Check if placement would exceed room bounds
            if (x < 0 or y < 0 or 
                x + width > self.room.width or 
                y + height > self.room.height):
                invalid_values.append(value)
        
        for value in invalid_values:
            variable.remove_domain_value(value)
            removed_count += 1
        
        self.total_values_removed += removed_count
        return removed_count
    
    def reduce_domain_overlaps(self, variable: Variable, 
                              placed_panels: List[PanelPlacement]) -> int:
        """
        Reduce domain based on overlap constraints with placed panels.
        Returns number of values removed.
        """
        removed_count = 0
        invalid_values = []
        
        for value in variable.domain:
            x, y = value.position
            width, height = variable.panel_size.get_dimensions()
            
            if value.orientation == "vertical":
                width, height = height, width
            
            # Check overlap with each placed panel
            for placement in placed_panels:
                px1, py1, px2, py2 = placement.bounds
                
                # Check if this value would overlap
                if not (x + width <= px1 or x >= px2 or 
                       y + height <= py1 or y >= py2):
                    invalid_values.append(value)
                    break
        
        for value in invalid_values:
            variable.remove_domain_value(value)
            removed_count += 1
        
        self.total_values_removed += removed_count
        return removed_count
    
    def reduce_domain_structural(self, variable: Variable,
                                support_map: Dict[Tuple[float, float], bool]) -> int:
        """
        Reduce domain based on structural support requirements.
        Returns number of values removed.
        """
        removed_count = 0
        invalid_values = []
        
        min_support_ratio = 0.5  # Minimum support required
        
        for value in variable.domain:
            x, y = value.position
            width, height = variable.panel_size.get_dimensions()
            
            if value.orientation == "vertical":
                width, height = height, width
            
            # Check if position has adequate support
            if y > 0.1:  # Not on floor
                # Calculate support from below
                support_length = 0.0
                for dx in range(int(width * 10)):  # Sample points
                    check_x = x + dx * 0.1
                    check_y = y - 0.1
                    if support_map.get((check_x, check_y), False):
                        support_length += 0.1
                
                support_ratio = support_length / width
                if support_ratio < min_support_ratio:
                    invalid_values.append(value)
        
        for value in invalid_values:
            variable.remove_domain_value(value)
            removed_count += 1
        
        self.total_values_removed += removed_count
        return removed_count
    
    def forward_check(self, assigned_var: Variable, unassigned_vars: List[Variable]) -> bool:
        """
        Forward checking: reduce domains of unassigned variables based on assignment.
        Returns False if any domain becomes empty.
        """
        if not assigned_var.is_assigned():
            return True
        
        assigned_value = list(assigned_var.domain)[0]
        ax, ay = assigned_value.position
        awidth, aheight = assigned_var.panel_size.get_dimensions()
        
        if assigned_value.orientation == "vertical":
            awidth, aheight = aheight, awidth
        
        for var in unassigned_vars:
            invalid_values = []
            
            for value in var.domain:
                x, y = value.position
                width, height = var.panel_size.get_dimensions()
                
                if value.orientation == "vertical":
                    width, height = height, width
                
                # Check if would overlap with assigned
                if not (x + width <= ax or x >= ax + awidth or
                       y + height <= ay or y >= ay + aheight):
                    invalid_values.append(value)
            
            for value in invalid_values:
                var.remove_domain_value(value)
            
            if var.is_unassigned():
                return False  # Domain wipeout
        
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get domain reduction statistics."""
        return {
            'total_reductions': self.reduction_count,
            'values_removed': self.total_values_removed
        }


class ArcConsistency:
    """
    Implements AC-3 algorithm for arc consistency.
    """
    
    def __init__(self, constraint_system: ConstraintSystem):
        self.constraint_system = constraint_system
        self.revision_count = 0
        self.consistency_checks = 0
        
    def revise(self, arc: Arc) -> bool:
        """
        Revise arc by removing inconsistent values from source domain.
        Returns True if domain was revised.
        """
        revised = False
        self.revision_count += 1
        
        invalid_values = []
        
        for source_value in arc.source.domain:
            # Check if source_value has support in target domain
            has_support = False
            
            for target_value in arc.target.domain:
                if self._is_consistent(source_value, target_value, arc):
                    has_support = True
                    break
            
            if not has_support:
                invalid_values.append(source_value)
        
        # Remove invalid values
        for value in invalid_values:
            arc.source.remove_domain_value(value)
            revised = True
        
        return revised
    
    def _is_consistent(self, value1: DomainValue, value2: DomainValue, arc: Arc) -> bool:
        """Check if two values are consistent with respect to the constraint."""
        self.consistency_checks += 1
        
        # Extract positions and dimensions
        x1, y1 = value1.position
        panel1 = arc.source.panel_size
        w1, h1 = panel1.get_dimensions()
        if value1.orientation == "vertical":
            w1, h1 = h1, w1
        
        x2, y2 = value2.position
        panel2 = arc.target.panel_size
        w2, h2 = panel2.get_dimensions()
        if value2.orientation == "vertical":
            w2, h2 = h2, w2
        
        # Check based on constraint type
        if arc.constraint_type == ConstraintCategory.NO_OVERLAP:
            # Values are consistent if they don't overlap
            return (x1 + w1 <= x2 or x1 >= x2 + w2 or
                   y1 + h1 <= y2 or y1 >= y2 + h2)
        
        return True  # Default to consistent
    
    def ac3(self, variables: List[Variable], arcs: List[Arc]) -> bool:
        """
        AC-3 algorithm for establishing arc consistency.
        Returns False if inconsistency detected.
        """
        queue = PropagationQueue(strategy="fifo")
        
        # Initialize queue with all arcs
        for arc in arcs:
            queue.add_arc(arc)
        
        while queue.has_arcs():
            arc = queue.get_next_arc()
            
            if self.revise(arc):
                if arc.source.is_unassigned():
                    return False  # Domain wipeout
                
                # Add all arcs pointing to revised variable
                for other_arc in arcs:
                    if other_arc.target == arc.source and other_arc.source != arc.target:
                        queue.add_arc(other_arc)
        
        return True
    
    def maintain_arc_consistency(self, variable: Variable, variables: List[Variable],
                                arcs: List[Arc]) -> bool:
        """
        Maintain arc consistency after variable assignment.
        Returns False if inconsistency detected.
        """
        # Add arcs from assigned variable to others
        queue = PropagationQueue(strategy="fifo")
        
        for arc in arcs:
            if arc.source == variable:
                queue.add_arc(arc)
        
        while queue.has_arcs():
            arc = queue.get_next_arc()
            
            if self.revise(arc):
                if arc.target.is_unassigned():
                    return False
                
                # Add affected arcs
                for other_arc in arcs:
                    if other_arc.target == arc.target and other_arc.source != variable:
                        queue.add_arc(other_arc)
        
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get arc consistency statistics."""
        return {
            'revisions': self.revision_count,
            'consistency_checks': self.consistency_checks
        }


class PropagationOptimizer:
    """
    Optimizes constraint propagation for efficiency.
    """
    
    def __init__(self):
        self.propagation_history = []
        self.domain_reduction_rates = defaultdict(list)
        self.effective_constraints = defaultdict(int)
        
    def select_propagation_order(self, arcs: List[Arc]) -> List[Arc]:
        """
        Select optimal order for arc processing.
        Orders arcs to maximize early pruning.
        """
        # Score each arc based on expected pruning power
        scored_arcs = []
        
        for arc in arcs:
            score = self._score_arc(arc)
            scored_arcs.append((score, arc))
        
        # Sort by score (higher score = process first)
        scored_arcs.sort(key=lambda x: x[0], reverse=True)
        
        return [arc for _, arc in scored_arcs]
    
    def _score_arc(self, arc: Arc) -> float:
        """Score an arc based on expected pruning effectiveness."""
        score = 0.0
        
        # Prefer arcs with smaller target domains (more constraining)
        target_domain_size = len(arc.target.domain)
        if target_domain_size > 0:
            score += 100.0 / target_domain_size
        
        # Prefer hard constraints
        if arc.constraint_type == ConstraintCategory.NO_OVERLAP:
            score += 50.0
        elif arc.constraint_type == ConstraintCategory.BOUNDARY:
            score += 40.0
        elif arc.constraint_type == ConstraintCategory.STRUCTURAL:
            score += 30.0
        
        # Use historical effectiveness
        if arc.constraint_type in self.effective_constraints:
            score += self.effective_constraints[arc.constraint_type] * 10.0
        
        return score
    
    def update_effectiveness(self, constraint_type: ConstraintCategory, 
                           reduction_rate: float):
        """Update constraint effectiveness based on actual reduction."""
        self.effective_constraints[constraint_type] = (
            0.7 * self.effective_constraints[constraint_type] + 
            0.3 * reduction_rate
        )
    
    def should_propagate(self, variable: Variable, threshold: float = 0.1) -> bool:
        """
        Decide if propagation is worthwhile based on expected benefit.
        """
        # Always propagate if domain is very small
        if len(variable.domain) <= 2:
            return True
        
        # Check historical reduction rates
        reduction_history = variable.domain_size_history
        if len(reduction_history) >= 2:
            recent_reduction = (reduction_history[-2] - reduction_history[-1]) / reduction_history[-2]
            if recent_reduction < threshold:
                return False  # Not much reduction expected
        
        return True
    
    def optimize_queue_strategy(self, statistics: Dict[str, Any]) -> str:
        """
        Select optimal queue strategy based on problem characteristics.
        """
        if statistics.get('variable_count', 0) > 50:
            return "priority"  # Use priority queue for large problems
        elif statistics.get('constraint_density', 0) > 0.5:
            return "lifo"  # LIFO for highly constrained problems
        else:
            return "fifo"  # Default FIFO


class ConstraintPropagationEngine:
    """
    Main constraint propagation engine combining all techniques.
    """
    
    def __init__(self, room: Room, constraint_system: ConstraintSystem):
        self.room = room
        self.constraint_system = constraint_system
        self.domain_reducer = DomainReducer(room, constraint_system)
        self.arc_consistency = ArcConsistency(constraint_system)
        self.optimizer = PropagationOptimizer()
        
        # Statistics
        self.propagation_count = 0
        self.total_time = 0.0
        
    def initialize_domains(self, panels: List[PanelSize], 
                         resolution: float = 0.5) -> Dict[PanelSize, Variable]:
        """
        Initialize variable domains for all panels.
        """
        variables = {}
        
        for panel in panels:
            domain = set()
            width, height = panel.get_dimensions()
            
            # Generate possible positions
            for x in range(int(self.room.width / resolution)):
                for y in range(int(self.room.height / resolution)):
                    pos_x = x * resolution
                    pos_y = y * resolution
                    
                    # Horizontal orientation
                    if pos_x + width <= self.room.width and pos_y + height <= self.room.height:
                        domain.add(DomainValue((pos_x, pos_y), "horizontal"))
                    
                    # Vertical orientation (if different)
                    if width != height:
                        if pos_x + height <= self.room.width and pos_y + width <= self.room.height:
                            domain.add(DomainValue((pos_x, pos_y), "vertical"))
            
            variable = Variable(panel_size=panel, domain=domain)
            variable.domain_size_history.append(len(domain))
            variables[panel] = variable
        
        return variables
    
    def create_constraint_network(self, variables: List[Variable]) -> List[Arc]:
        """
        Create constraint network (arcs) between variables.
        """
        arcs = []
        
        # Create arcs for each pair of variables
        for i, var1 in enumerate(variables):
            for j, var2 in enumerate(variables):
                if i != j:
                    # No-overlap constraint between all pairs
                    arc = Arc(var1, var2, ConstraintCategory.NO_OVERLAP)
                    arcs.append(arc)
        
        return arcs
    
    def propagate(self, variables: Dict[PanelSize, Variable], 
                 placed_panels: List[PanelPlacement] = None) -> bool:
        """
        Perform full constraint propagation.
        Returns False if inconsistency detected.
        """
        import time
        start_time = time.time()
        self.propagation_count += 1
        
        placed_panels = placed_panels or []
        variable_list = list(variables.values())
        
        # Initial domain reduction
        for variable in variable_list:
            # Reduce based on bounds
            self.domain_reducer.reduce_domain_bounds(variable)
            
            # Reduce based on placed panels
            if placed_panels:
                self.domain_reducer.reduce_domain_overlaps(variable, placed_panels)
            
            if variable.is_unassigned():
                return False
        
        # Create constraint network
        arcs = self.create_constraint_network(variable_list)
        
        # Optimize arc order
        arcs = self.optimizer.select_propagation_order(arcs)
        
        # Establish arc consistency
        if not self.arc_consistency.ac3(variable_list, arcs):
            return False
        
        # Update statistics
        for variable in variable_list:
            reduction_rate = variable.get_domain_reduction_rate()
            if reduction_rate > 0:
                self.optimizer.update_effectiveness(ConstraintCategory.NO_OVERLAP, reduction_rate)
        
        self.total_time += time.time() - start_time
        return True
    
    def incremental_propagate(self, assigned_var: Variable, 
                            unassigned_vars: List[Variable]) -> bool:
        """
        Incremental propagation after variable assignment.
        More efficient than full propagation.
        """
        # Forward checking
        if not self.domain_reducer.forward_check(assigned_var, unassigned_vars):
            return False
        
        # Maintain arc consistency
        all_vars = [assigned_var] + unassigned_vars
        arcs = self.create_constraint_network(all_vars)
        
        if not self.arc_consistency.maintain_arc_consistency(assigned_var, all_vars, arcs):
            return False
        
        return True
    
    def get_most_constrained_variable(self, variables: List[Variable]) -> Optional[Variable]:
        """
        Get the most constrained variable (MRV heuristic).
        """
        unassigned = [v for v in variables if not v.is_assigned()]
        if not unassigned:
            return None
        
        return min(unassigned, key=lambda v: len(v.domain))
    
    def get_least_constraining_value(self, variable: Variable, 
                                    other_variables: List[Variable]) -> Optional[DomainValue]:
        """
        Get the least constraining value for a variable.
        """
        if not variable.domain:
            return None
        
        value_scores = []
        
        for value in variable.domain:
            # Count how many values this eliminates from other domains
            eliminated_count = 0
            
            for other_var in other_variables:
                if other_var == variable:
                    continue
                
                for other_value in other_var.domain:
                    if not self._values_consistent(value, other_value, 
                                                  variable.panel_size, other_var.panel_size):
                        eliminated_count += 1
            
            value_scores.append((eliminated_count, value))
        
        # Return value that eliminates fewest options
        value_scores.sort(key=lambda x: x[0])
        return value_scores[0][1] if value_scores else None
    
    def _values_consistent(self, value1: DomainValue, value2: DomainValue,
                          panel1: PanelSize, panel2: PanelSize) -> bool:
        """Check if two values are consistent."""
        x1, y1 = value1.position
        w1, h1 = panel1.get_dimensions()
        if value1.orientation == "vertical":
            w1, h1 = h1, w1
        
        x2, y2 = value2.position
        w2, h2 = panel2.get_dimensions()
        if value2.orientation == "vertical":
            w2, h2 = h2, w2
        
        # Check overlap
        return (x1 + w1 <= x2 or x1 >= x2 + w2 or
               y1 + h1 <= y2 or y1 >= y2 + h2)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get propagation engine statistics."""
        stats = {
            'propagation_count': self.propagation_count,
            'total_time': self.total_time,
            'avg_time': self.total_time / max(1, self.propagation_count)
        }
        
        stats.update(self.domain_reducer.get_statistics())
        stats.update(self.arc_consistency.get_statistics())
        
        return stats


def create_propagation_engine(room: Room, 
                            constraint_system: ConstraintSystem) -> ConstraintPropagationEngine:
    """
    Factory function to create constraint propagation engine.
    """
    return ConstraintPropagationEngine(room, constraint_system)
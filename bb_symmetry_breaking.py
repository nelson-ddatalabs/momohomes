#!/usr/bin/env python3
"""
bb_symmetry_breaking.py - Symmetry Breaking Mechanisms
======================================================
Production-ready lexicographic ordering, rotation elimination,
position detection, and dynamic symmetry breaking for search space reduction.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any, FrozenSet
from collections import defaultdict
from enum import Enum
import math
import hashlib

from models import Room, PanelSize, Point
from advanced_packing import PackingState, PanelPlacement
from bb_search_tree import BranchNode


class SymmetryType(Enum):
    """Types of symmetries in panel placement."""
    ROTATION = "rotation"  # 90, 180, 270 degree rotations
    REFLECTION = "reflection"  # Horizontal/vertical reflection
    TRANSLATION = "translation"  # Position shift
    PERMUTATION = "permutation"  # Panel order permutation
    COMPOSITE = "composite"  # Combination of symmetries


@dataclass
class SymmetryGroup:
    """Represents a group of symmetric solutions."""
    representative: FrozenSet[Tuple[PanelSize, Tuple[float, float], str]]
    members: Set[FrozenSet[Tuple[PanelSize, Tuple[float, float], str]]]
    symmetry_type: SymmetryType
    transformation: str  # Description of transformation
    
    def __hash__(self):
        return hash(self.representative)
    
    def add_member(self, member: FrozenSet):
        """Add a symmetric solution to the group."""
        self.members.add(member)
    
    @property
    def size(self) -> int:
        """Size of the symmetry group."""
        return len(self.members) + 1  # +1 for representative


class LexicographicOrderer:
    """
    Enforces lexicographic ordering to eliminate symmetric solutions.
    Ensures canonical representation of placements.
    """
    
    def __init__(self, priority_rules: Optional[List[str]] = None):
        self.priority_rules = priority_rules or [
            'position_x',  # X coordinate first
            'position_y',  # Then Y coordinate
            'panel_size',  # Then panel size
            'orientation'  # Finally orientation
        ]
    
    def create_canonical_order(self,
                              panels: List[PanelSize]) -> List[PanelSize]:
        """Create canonical ordering of panels."""
        # Sort panels by multiple criteria
        return sorted(panels, key=lambda p: (
            -p.width * p.height,  # Larger panels first
            -p.width,  # Wider panels first
            -p.height,  # Taller panels first
            p.id  # Finally by ID for consistency
        ))
    
    def is_canonical_placement(self,
                              placement: PanelPlacement,
                              previous_placements: List[PanelPlacement]) -> bool:
        """Check if placement maintains canonical ordering."""
        if not previous_placements:
            return True
        
        last_placement = previous_placements[-1]
        
        # Compare based on priority rules
        for rule in self.priority_rules:
            if rule == 'position_x':
                if placement.position[0] < last_placement.position[0]:
                    return False
                elif placement.position[0] > last_placement.position[0]:
                    return True
            
            elif rule == 'position_y':
                if placement.position[0] == last_placement.position[0]:
                    if placement.position[1] < last_placement.position[1]:
                        return False
                    elif placement.position[1] > last_placement.position[1]:
                        return True
            
            elif rule == 'panel_size':
                last_area = last_placement.panel_size.width * last_placement.panel_size.height
                current_area = placement.panel_size.width * placement.panel_size.height
                if current_area > last_area:
                    return False
                elif current_area < last_area:
                    return True
        
        return True
    
    def enforce_ordering_constraint(self,
                                   candidates: List[Tuple[PanelSize, Tuple[float, float], str]],
                                   placed: List[PanelPlacement]) -> List[Tuple[PanelSize, Tuple[float, float], str]]:
        """Filter candidates to maintain lexicographic ordering."""
        if not placed:
            return candidates
        
        filtered = []
        for panel_size, position, orientation in candidates:
            # Create temporary placement
            temp_placement = PanelPlacement(
                panel_size=panel_size,
                position=position,
                placed_index=len(placed)
            )
            
            if self.is_canonical_placement(temp_placement, placed):
                filtered.append((panel_size, position, orientation))
        
        return filtered
    
    def compute_placement_signature(self,
                                   placements: List[PanelPlacement]) -> str:
        """Compute canonical signature for placement set."""
        # Sort placements canonically
        sorted_placements = sorted(placements, key=lambda p: (
            p.position[0],
            p.position[1],
            -p.panel_size.width * p.panel_size.height,
            p.panel_size.id
        ))
        
        # Create signature string
        signature_parts = []
        for p in sorted_placements:
            part = f"{p.panel_size.id}@({p.position[0]},{p.position[1]})"
            signature_parts.append(part)
        
        return "|".join(signature_parts)


class RotationEliminator:
    """
    Eliminates rotational symmetries by fixing orientation choices.
    Reduces search space by avoiding equivalent rotated solutions.
    """
    
    def __init__(self, room: Room):
        self.room = room
        self.rotation_groups = defaultdict(set)
        self.fixed_orientations = {}
    
    def should_fix_orientation(self,
                              panel: PanelSize,
                              position: Tuple[float, float]) -> Optional[str]:
        """Determine if orientation should be fixed for symmetry breaking."""
        # Fix orientation for square panels (no benefit from rotation)
        if abs(panel.width - panel.height) < 0.01:
            return "horizontal"
        
        # Fix orientation for first placement in each corner
        corners = [
            (0, 0),
            (self.room.width - panel.width, 0),
            (0, self.room.height - panel.height),
            (self.room.width - panel.width, self.room.height - panel.height)
        ]
        
        if position in corners and position not in self.fixed_orientations:
            self.fixed_orientations[position] = "horizontal"
            return "horizontal"
        
        # Check if position has fixed orientation
        if position in self.fixed_orientations:
            return self.fixed_orientations[position]
        
        return None
    
    def eliminate_rotation_symmetry(self,
                                   candidates: List[Tuple[PanelSize, Tuple[float, float], str]]) -> List[Tuple[PanelSize, Tuple[float, float], str]]:
        """Remove rotationally symmetric candidates."""
        filtered = []
        seen_rotations = set()
        
        for panel_size, position, orientation in candidates:
            # Check if we should fix orientation
            fixed_orientation = self.should_fix_orientation(panel_size, position)
            if fixed_orientation and orientation != fixed_orientation:
                continue
            
            # Create rotation key
            rotation_key = self._get_rotation_key(panel_size, position, orientation)
            
            # Check if we've seen an equivalent rotation
            if rotation_key not in seen_rotations:
                filtered.append((panel_size, position, orientation))
                seen_rotations.add(rotation_key)
                
                # Add all equivalent rotations to seen set
                for equiv in self._get_equivalent_rotations(panel_size, position, orientation):
                    seen_rotations.add(equiv)
        
        return filtered
    
    def _get_rotation_key(self,
                         panel: PanelSize,
                         position: Tuple[float, float],
                         orientation: str) -> Tuple:
        """Get canonical rotation key."""
        # Normalize based on panel dimensions
        if panel.width == panel.height:
            # Square panel - all rotations equivalent
            return (panel.id, position, "square")
        elif orientation == "horizontal":
            return (panel.id, position, "h")
        else:
            return (panel.id, position, "v")
    
    def _get_equivalent_rotations(self,
                                 panel: PanelSize,
                                 position: Tuple[float, float],
                                 orientation: str) -> List[Tuple]:
        """Get all rotationally equivalent placements."""
        equivalent = []
        
        if panel.width == panel.height:
            # Square panel - both orientations are equivalent
            equivalent.append((panel.id, position, "horizontal"))
            equivalent.append((panel.id, position, "vertical"))
        
        # 180 degree rotation
        rotated_pos = (
            self.room.width - position[0] - panel.width,
            self.room.height - position[1] - panel.height
        )
        if self._is_valid_position(rotated_pos, panel):
            equivalent.append((panel.id, rotated_pos, orientation))
        
        return equivalent
    
    def _is_valid_position(self,
                          position: Tuple[float, float],
                          panel: PanelSize) -> bool:
        """Check if position is valid in room."""
        x, y = position
        return (0 <= x <= self.room.width - panel.width and
                0 <= y <= self.room.height - panel.height)


class PositionDetector:
    """
    Detects and exploits positional symmetries.
    Identifies equivalent positions and reduces redundant exploration.
    """
    
    def __init__(self, room: Room, grid_size: float = 1.0):
        self.room = room
        self.grid_size = grid_size
        self.position_classes = defaultdict(set)
        self.symmetric_positions = defaultdict(set)
        self._initialize_position_classes()
    
    def _initialize_position_classes(self):
        """Initialize position equivalence classes."""
        # Identify symmetric positions
        center_x = self.room.width / 2
        center_y = self.room.height / 2
        
        # Grid positions
        x_positions = [i * self.grid_size for i in range(int(self.room.width / self.grid_size) + 1)]
        y_positions = [i * self.grid_size for i in range(int(self.room.height / self.grid_size) + 1)]
        
        for x in x_positions:
            for y in y_positions:
                # Classify position
                pos_class = self._get_position_class(x, y, center_x, center_y)
                self.position_classes[pos_class].add((x, y))
                
                # Find symmetric positions
                symmetric = self._find_symmetric_positions(x, y, center_x, center_y)
                self.symmetric_positions[(x, y)].update(symmetric)
    
    def _get_position_class(self,
                           x: float,
                           y: float,
                           center_x: float,
                           center_y: float) -> str:
        """Classify position based on symmetry properties."""
        # Determine quadrant
        if x < center_x and y < center_y:
            quadrant = "top_left"
        elif x >= center_x and y < center_y:
            quadrant = "top_right"
        elif x < center_x and y >= center_y:
            quadrant = "bottom_left"
        else:
            quadrant = "bottom_right"
        
        # Check if on axis
        on_x_axis = abs(y - center_y) < self.grid_size / 2
        on_y_axis = abs(x - center_x) < self.grid_size / 2
        
        if on_x_axis and on_y_axis:
            return "center"
        elif on_x_axis:
            return "horizontal_axis"
        elif on_y_axis:
            return "vertical_axis"
        else:
            return quadrant
    
    def _find_symmetric_positions(self,
                                 x: float,
                                 y: float,
                                 center_x: float,
                                 center_y: float) -> Set[Tuple[float, float]]:
        """Find all positions symmetric to given position."""
        symmetric = set()
        
        # Horizontal reflection
        h_reflected = (2 * center_x - x, y)
        if self._is_valid_grid_position(h_reflected):
            symmetric.add(h_reflected)
        
        # Vertical reflection
        v_reflected = (x, 2 * center_y - y)
        if self._is_valid_grid_position(v_reflected):
            symmetric.add(v_reflected)
        
        # Central reflection
        c_reflected = (2 * center_x - x, 2 * center_y - y)
        if self._is_valid_grid_position(c_reflected):
            symmetric.add(c_reflected)
        
        return symmetric
    
    def _is_valid_grid_position(self, position: Tuple[float, float]) -> bool:
        """Check if position is valid on grid."""
        x, y = position
        return (0 <= x <= self.room.width and
                0 <= y <= self.room.height and
                x % self.grid_size == 0 and
                y % self.grid_size == 0)
    
    def detect_equivalent_positions(self,
                                   position: Tuple[float, float]) -> Set[Tuple[float, float]]:
        """Detect positions equivalent by symmetry."""
        return self.symmetric_positions.get(position, set())
    
    def select_canonical_position(self,
                                 positions: List[Tuple[float, float]]) -> Tuple[float, float]:
        """Select canonical representative from equivalent positions."""
        if not positions:
            return None
        
        # Sort lexicographically and take first
        return min(positions, key=lambda p: (p[0], p[1]))
    
    def filter_symmetric_positions(self,
                                  candidates: List[Tuple[PanelSize, Tuple[float, float], str]],
                                  placed: List[PanelPlacement]) -> List[Tuple[PanelSize, Tuple[float, float], str]]:
        """Filter out symmetric position candidates."""
        if not placed:
            return candidates
        
        # Track used position classes
        used_classes = set()
        for p in placed:
            pos_class = self._get_position_class(
                p.position[0],
                p.position[1],
                self.room.width / 2,
                self.room.height / 2
            )
            used_classes.add(pos_class)
        
        filtered = []
        seen_canonical = set()
        
        for panel_size, position, orientation in candidates:
            # Get equivalent positions
            equivalent = self.detect_equivalent_positions(position)
            equivalent.add(position)
            
            # Select canonical
            canonical = self.select_canonical_position(list(equivalent))
            
            # Check if we've seen this canonical position
            key = (panel_size.id, canonical, orientation)
            if key not in seen_canonical:
                filtered.append((panel_size, position, orientation))
                seen_canonical.add(key)
        
        return filtered


class DynamicSymmetryBreaker:
    """
    Dynamically detects and breaks symmetries during search.
    Adapts symmetry breaking based on search progress.
    """
    
    def __init__(self, room: Room, learning_rate: float = 0.1):
        self.room = room
        self.learning_rate = learning_rate
        self.symmetry_groups = {}
        self.detection_history = []
        self.breaking_rules = []
        self.effectiveness_scores = defaultdict(float)
    
    def detect_symmetry(self, state: PackingState) -> Optional[SymmetryGroup]:
        """Detect symmetries in current state."""
        if not state.placed_panels:
            return None
        
        # Create state signature
        state_signature = self._create_state_signature(state)
        
        # Check for known symmetry groups
        for group_key, group in self.symmetry_groups.items():
            if self._is_symmetric_to_group(state_signature, group):
                return group
        
        # Try to detect new symmetries
        new_group = self._detect_new_symmetry(state)
        if new_group:
            self.symmetry_groups[hash(new_group)] = new_group
            return new_group
        
        return None
    
    def _create_state_signature(self, state: PackingState) -> FrozenSet:
        """Create signature for state."""
        return frozenset(
            (p.panel_size, p.position, getattr(p, 'orientation', 'horizontal'))
            for p in state.placed_panels
        )
    
    def _is_symmetric_to_group(self,
                              signature: FrozenSet,
                              group: SymmetryGroup) -> bool:
        """Check if state is symmetric to a group."""
        # Check against representative
        if self._are_symmetric(signature, group.representative):
            return True
        
        # Check against members
        for member in group.members:
            if self._are_symmetric(signature, member):
                return True
        
        return False
    
    def _are_symmetric(self,
                      sig1: FrozenSet,
                      sig2: FrozenSet) -> bool:
        """Check if two signatures are symmetric."""
        if len(sig1) != len(sig2):
            return False
        
        # Check for various symmetry types
        
        # Reflection symmetry
        if self._check_reflection_symmetry(sig1, sig2):
            return True
        
        # Rotation symmetry
        if self._check_rotation_symmetry(sig1, sig2):
            return True
        
        # Translation symmetry
        if self._check_translation_symmetry(sig1, sig2):
            return True
        
        return False
    
    def _check_reflection_symmetry(self,
                                  sig1: FrozenSet,
                                  sig2: FrozenSet) -> bool:
        """Check for reflection symmetry."""
        # Horizontal reflection
        h_reflected = self._reflect_horizontal(sig1)
        if h_reflected == sig2:
            return True
        
        # Vertical reflection
        v_reflected = self._reflect_vertical(sig1)
        if v_reflected == sig2:
            return True
        
        return False
    
    def _reflect_horizontal(self, signature: FrozenSet) -> FrozenSet:
        """Apply horizontal reflection to signature."""
        reflected = set()
        for panel_size, (x, y), orientation in signature:
            new_x = self.room.width - x - panel_size.width
            reflected.add((panel_size, (new_x, y), orientation))
        return frozenset(reflected)
    
    def _reflect_vertical(self, signature: FrozenSet) -> FrozenSet:
        """Apply vertical reflection to signature."""
        reflected = set()
        for panel_size, (x, y), orientation in signature:
            new_y = self.room.height - y - panel_size.height
            reflected.add((panel_size, (x, new_y), orientation))
        return frozenset(reflected)
    
    def _check_rotation_symmetry(self,
                                sig1: FrozenSet,
                                sig2: FrozenSet) -> bool:
        """Check for rotation symmetry."""
        # 180 degree rotation
        rotated = self._rotate_180(sig1)
        return rotated == sig2
    
    def _rotate_180(self, signature: FrozenSet) -> FrozenSet:
        """Apply 180 degree rotation to signature."""
        rotated = set()
        for panel_size, (x, y), orientation in signature:
            new_x = self.room.width - x - panel_size.width
            new_y = self.room.height - y - panel_size.height
            rotated.add((panel_size, (new_x, new_y), orientation))
        return frozenset(rotated)
    
    def _check_translation_symmetry(self,
                                   sig1: FrozenSet,
                                   sig2: FrozenSet) -> bool:
        """Check for translation symmetry."""
        # Find potential translation vector
        if not sig1 or not sig2:
            return False
        
        # Take first element from each
        elem1 = next(iter(sig1))
        elem2 = next(iter(sig2))
        
        if elem1[0] != elem2[0]:  # Different panel sizes
            return False
        
        # Calculate translation
        dx = elem2[1][0] - elem1[1][0]
        dy = elem2[1][1] - elem1[1][1]
        
        # Apply translation to all elements
        translated = self._translate(sig1, dx, dy)
        return translated == sig2
    
    def _translate(self,
                  signature: FrozenSet,
                  dx: float,
                  dy: float) -> FrozenSet:
        """Apply translation to signature."""
        translated = set()
        for panel_size, (x, y), orientation in signature:
            new_x = x + dx
            new_y = y + dy
            # Check if translation is valid
            if (0 <= new_x <= self.room.width - panel_size.width and
                0 <= new_y <= self.room.height - panel_size.height):
                translated.add((panel_size, (new_x, new_y), orientation))
        return frozenset(translated)
    
    def _detect_new_symmetry(self, state: PackingState) -> Optional[SymmetryGroup]:
        """Try to detect new symmetry group."""
        signature = self._create_state_signature(state)
        
        # Try different transformations
        transformations = [
            (self._reflect_horizontal, SymmetryType.REFLECTION, "horizontal"),
            (self._reflect_vertical, SymmetryType.REFLECTION, "vertical"),
            (self._rotate_180, SymmetryType.ROTATION, "180_degree")
        ]
        
        for transform_func, sym_type, description in transformations:
            transformed = transform_func(signature)
            if transformed != signature and self._is_valid_signature(transformed):
                # Found a symmetry
                return SymmetryGroup(
                    representative=signature,
                    members={transformed},
                    symmetry_type=sym_type,
                    transformation=description
                )
        
        return None
    
    def _is_valid_signature(self, signature: FrozenSet) -> bool:
        """Check if signature represents valid placements."""
        for panel_size, (x, y), orientation in signature:
            if not (0 <= x <= self.room.width - panel_size.width and
                   0 <= y <= self.room.height - panel_size.height):
                return False
        return True
    
    def add_breaking_rule(self,
                         rule_type: str,
                         condition: Any,
                         action: Any):
        """Add a dynamic symmetry breaking rule."""
        rule = {
            'type': rule_type,
            'condition': condition,
            'action': action,
            'applications': 0,
            'successes': 0
        }
        self.breaking_rules.append(rule)
    
    def apply_breaking_rules(self,
                           candidates: List[Any],
                           state: PackingState) -> List[Any]:
        """Apply dynamic breaking rules to candidates."""
        filtered = candidates.copy()
        
        for rule in self.breaking_rules:
            if self._evaluate_condition(rule['condition'], state):
                filtered = self._apply_action(rule['action'], filtered, state)
                rule['applications'] += 1
        
        return filtered
    
    def _evaluate_condition(self, condition: Any, state: PackingState) -> bool:
        """Evaluate a breaking rule condition."""
        # Simple condition evaluation
        if callable(condition):
            return condition(state)
        return bool(condition)
    
    def _apply_action(self,
                     action: Any,
                     candidates: List[Any],
                     state: PackingState) -> List[Any]:
        """Apply a breaking rule action."""
        if callable(action):
            return action(candidates, state)
        return candidates
    
    def update_effectiveness(self,
                           symmetry_type: SymmetryType,
                           effective: bool):
        """Update effectiveness scores for symmetry breaking."""
        key = symmetry_type.value
        old_score = self.effectiveness_scores[key]
        
        # Exponential moving average update
        new_score = old_score * (1 - self.learning_rate) + (1.0 if effective else 0.0) * self.learning_rate
        self.effectiveness_scores[key] = new_score


class SymmetryBreakingSystem:
    """Main system coordinating all symmetry breaking components."""
    
    def __init__(self, room: Room, config: Optional[Dict[str, Any]] = None):
        self.room = room
        self.config = config or {}
        
        # Initialize components
        self.lexicographic_orderer = LexicographicOrderer(
            priority_rules=self.config.get('priority_rules')
        )
        
        self.rotation_eliminator = RotationEliminator(room)
        
        self.position_detector = PositionDetector(
            room,
            grid_size=self.config.get('grid_size', 1.0)
        )
        
        self.dynamic_breaker = DynamicSymmetryBreaker(
            room,
            learning_rate=self.config.get('learning_rate', 0.1)
        )
        
        self.breaking_enabled = self.config.get('breaking_enabled', True)
        self.statistics = defaultdict(int)
    
    def break_symmetries(self,
                        candidates: List[Tuple[PanelSize, Tuple[float, float], str]],
                        state: PackingState) -> List[Tuple[PanelSize, Tuple[float, float], str]]:
        """Apply all symmetry breaking techniques."""
        if not self.breaking_enabled:
            return candidates
        
        original_count = len(candidates)
        
        # Apply lexicographic ordering
        candidates = self.lexicographic_orderer.enforce_ordering_constraint(
            candidates,
            state.placed_panels
        )
        self.statistics['lexicographic_filtered'] += original_count - len(candidates)
        
        # Eliminate rotation symmetries
        pre_rotation = len(candidates)
        candidates = self.rotation_eliminator.eliminate_rotation_symmetry(candidates)
        self.statistics['rotation_filtered'] += pre_rotation - len(candidates)
        
        # Filter symmetric positions
        pre_position = len(candidates)
        candidates = self.position_detector.filter_symmetric_positions(
            candidates,
            state.placed_panels
        )
        self.statistics['position_filtered'] += pre_position - len(candidates)
        
        # Apply dynamic breaking rules
        pre_dynamic = len(candidates)
        candidates = self.dynamic_breaker.apply_breaking_rules(candidates, state)
        self.statistics['dynamic_filtered'] += pre_dynamic - len(candidates)
        
        self.statistics['total_filtered'] += original_count - len(candidates)
        self.statistics['total_candidates'] += original_count
        
        return candidates
    
    def detect_state_symmetry(self, state: PackingState) -> Optional[SymmetryGroup]:
        """Detect symmetry in current state."""
        return self.dynamic_breaker.detect_symmetry(state)
    
    def get_canonical_panels(self, panels: List[PanelSize]) -> List[PanelSize]:
        """Get canonical ordering of panels."""
        return self.lexicographic_orderer.create_canonical_order(panels)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get symmetry breaking statistics."""
        stats = dict(self.statistics)
        
        if stats['total_candidates'] > 0:
            stats['reduction_rate'] = stats['total_filtered'] / stats['total_candidates']
        else:
            stats['reduction_rate'] = 0
        
        # Add component-specific stats
        stats['effectiveness_scores'] = dict(self.dynamic_breaker.effectiveness_scores)
        stats['symmetry_groups'] = len(self.dynamic_breaker.symmetry_groups)
        
        return stats
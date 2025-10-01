#!/usr/bin/env python3
"""
bb_constraint_learning.py - Constraint Learning System
======================================================
Production-ready nogood learning, implied constraints, constraint database,
and relevance scoring for intelligent constraint management.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any, FrozenSet
from collections import defaultdict, deque
from enum import Enum
import time
import hashlib
import json

from models import Room, PanelSize, Point
from advanced_packing import PackingState, PanelPlacement
from bb_search_tree import BranchNode
from bb_constraint_model import ConstraintViolation, ConstraintCategory


class NogoodType(Enum):
    """Types of nogood constraints learned from failures."""
    CONFLICT = "conflict"  # Direct conflict between assignments
    INFEASIBILITY = "infeasibility"  # Proven infeasible combination
    DOMINANCE = "dominance"  # Dominated by better solution
    SYMMETRY = "symmetry"  # Symmetric to known nogood
    IMPLIED = "implied"  # Implied by other constraints


@dataclass
class Nogood:
    """Represents a learned nogood constraint."""
    assignments: FrozenSet[Tuple[PanelSize, Tuple[float, float], str]]  # (panel, position, orientation)
    type: NogoodType
    reason: str
    strength: float  # 0-1, higher means stronger evidence
    usage_count: int = 0
    success_count: int = 0
    creation_time: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    
    def __hash__(self):
        return hash(self.assignments)
    
    def __eq__(self, other):
        return self.assignments == other.assignments
    
    def update_usage(self, successful: bool):
        """Update usage statistics."""
        self.usage_count += 1
        if successful:
            self.success_count += 1
        self.last_used = time.time()
    
    @property
    def effectiveness(self) -> float:
        """Calculate effectiveness score."""
        if self.usage_count == 0:
            return self.strength
        success_rate = self.success_count / self.usage_count
        age_factor = min(1.0, (time.time() - self.creation_time) / 3600)  # Decay over 1 hour
        return self.strength * success_rate * (1 - age_factor * 0.5)


@dataclass
class ImpliedConstraint:
    """Represents a constraint implied by problem structure."""
    condition: str  # Condition that triggers the constraint
    implication: str  # What is implied
    confidence: float  # Confidence level (0-1)
    support_count: int = 0  # Number of times validated
    violation_count: int = 0  # Number of violations
    
    def update(self, validated: bool):
        """Update implication statistics."""
        if validated:
            self.support_count += 1
        else:
            self.violation_count += 1
        
        # Update confidence based on evidence
        total = self.support_count + self.violation_count
        if total > 0:
            self.confidence = self.support_count / total
    
    @property
    def reliability(self) -> float:
        """Calculate reliability score."""
        if self.support_count + self.violation_count == 0:
            return self.confidence
        return self.support_count / (self.support_count + self.violation_count)


class NogoodLearner:
    """Learns nogood constraints from search failures."""
    
    def __init__(self, max_nogood_size: int = 5, min_strength: float = 0.3):
        self.max_nogood_size = max_nogood_size
        self.min_strength = min_strength
        self.conflict_graph = defaultdict(set)  # Track conflicts between assignments
        self.failure_patterns = defaultdict(int)  # Count failure patterns
    
    def learn_from_conflict(self, 
                           assignments: List[PanelPlacement],
                           conflict: ConstraintViolation) -> Optional[Nogood]:
        """Learn nogood from a constraint conflict."""
        if not assignments:
            return None
        
        # Extract relevant assignments involved in conflict
        relevant = self._extract_relevant_assignments(assignments, conflict)
        
        if len(relevant) > self.max_nogood_size:
            # Too large, try to minimize
            relevant = self._minimize_nogood(relevant, conflict)
        
        if not relevant:
            return None
        
        # Create frozen set of assignments
        nogood_assignments = frozenset(
            (p.panel_size, p.position, getattr(p, 'orientation', 'horizontal'))
            for p in relevant
        )
        
        # Calculate strength based on conflict severity
        strength = min(1.0, conflict.severity * 1.5)
        
        if strength < self.min_strength:
            return None
        
        # Update conflict graph
        for i, a1 in enumerate(relevant):
            for a2 in relevant[i+1:]:
                key1 = (a1.panel_size, a1.position)
                key2 = (a2.panel_size, a2.position)
                self.conflict_graph[key1].add(key2)
                self.conflict_graph[key2].add(key1)
        
        return Nogood(
            assignments=nogood_assignments,
            type=NogoodType.CONFLICT,
            reason=f"Conflict: {conflict.description}",
            strength=strength
        )
    
    def learn_from_infeasibility(self,
                                assignments: List[PanelPlacement],
                                remaining_panels: List[PanelSize]) -> Optional[Nogood]:
        """Learn nogood from proven infeasibility."""
        if not assignments:
            return None
        
        # Create assignments set
        nogood_assignments = frozenset(
            (p.panel_size, p.position, getattr(p, 'orientation', 'horizontal'))
            for p in assignments[-self.max_nogood_size:]  # Take last few assignments
        )
        
        # High strength for proven infeasibility
        strength = 0.9
        
        return Nogood(
            assignments=nogood_assignments,
            type=NogoodType.INFEASIBILITY,
            reason=f"Infeasible with {len(remaining_panels)} panels remaining",
            strength=strength
        )
    
    def learn_from_dominance(self,
                            dominated: List[PanelPlacement],
                            dominating: List[PanelPlacement]) -> Optional[Nogood]:
        """Learn nogood from dominance relationship."""
        if not dominated or not dominating:
            return None
        
        # Find the difference in assignments
        dom_set = set((p.panel_size, p.position) for p in dominated)
        dominating_set = set((p.panel_size, p.position) for p in dominating)
        
        difference = dom_set - dominating_set
        if not difference or len(difference) > self.max_nogood_size:
            return None
        
        nogood_assignments = frozenset(
            (ps, pos, 'horizontal') for ps, pos in difference
        )
        
        strength = 0.7  # Moderate strength for dominance
        
        return Nogood(
            assignments=nogood_assignments,
            type=NogoodType.DOMINANCE,
            reason="Dominated by better solution",
            strength=strength
        )
    
    def _extract_relevant_assignments(self,
                                     assignments: List[PanelPlacement],
                                     conflict: ConstraintViolation) -> List[PanelPlacement]:
        """Extract assignments relevant to a conflict."""
        relevant = []
        
        # Look for assignments mentioned in conflict description
        for assignment in assignments:
            # Check if this assignment is involved in the conflict
            pos_str = f"({assignment.position[0]}, {assignment.position[1]})"
            if pos_str in conflict.description:
                relevant.append(assignment)
        
        # If no specific assignments found, take recent ones
        if not relevant and assignments:
            relevant = assignments[-min(3, len(assignments)):]
        
        return relevant
    
    def _minimize_nogood(self,
                        assignments: List[PanelPlacement],
                        conflict: ConstraintViolation) -> List[PanelPlacement]:
        """Minimize a nogood to essential assignments."""
        # Simple heuristic: keep assignments with highest conflict involvement
        scored = []
        for assignment in assignments:
            score = 0
            key = (assignment.panel_size, assignment.position)
            
            # Score based on conflict graph
            score += len(self.conflict_graph.get(key, set()))
            
            # Score based on failure patterns
            pattern = (assignment.panel_size.id, assignment.position)
            score += self.failure_patterns.get(pattern, 0)
            
            scored.append((score, assignment))
        
        # Sort by score and take top assignments
        scored.sort(reverse=True, key=lambda x: x[0])
        return [assignment for _, assignment in scored[:self.max_nogood_size]]


class ImpliedConstraintDeriver:
    """Derives implied constraints from problem structure."""
    
    def __init__(self, room: Room):
        self.room = room
        self.implications = []
        self.pattern_counts = defaultdict(int)
    
    def derive_spatial_implications(self,
                                   placed: List[PanelPlacement]) -> List[ImpliedConstraint]:
        """Derive spatial implications from current placements."""
        implications = []
        
        # Check for corner implications
        corner_filled = any(
            p.position == (0, 0) or 
            p.position == (self.room.width - p.panel_size.width, 0) or
            p.position == (0, self.room.height - p.panel_size.height) or
            p.position == (self.room.width - p.panel_size.width, 
                          self.room.height - p.panel_size.height)
            for p in placed
        )
        
        if corner_filled:
            implications.append(ImpliedConstraint(
                condition="All corners filled",
                implication="Must use edge positions for remaining panels",
                confidence=0.8
            ))
        
        # Check for row/column completion
        rows_used = defaultdict(list)
        cols_used = defaultdict(list)
        
        for p in placed:
            rows_used[p.position[1]].append(p)
            cols_used[p.position[0]].append(p)
        
        for row_y, panels in rows_used.items():
            total_width = sum(p.panel_size.width for p in panels)
            if total_width >= self.room.width * 0.8:
                implications.append(ImpliedConstraint(
                    condition=f"Row {row_y} nearly full",
                    implication=f"Cannot place large panels in row {row_y}",
                    confidence=0.9
                ))
        
        return implications
    
    def derive_symmetry_implications(self,
                                    placed: List[PanelPlacement]) -> List[ImpliedConstraint]:
        """Derive symmetry-based implications."""
        implications = []
        
        # Check for symmetric placements
        left_panels = [p for p in placed if p.position[0] < self.room.width / 2]
        right_panels = [p for p in placed if p.position[0] >= self.room.width / 2]
        
        if abs(len(left_panels) - len(right_panels)) > 3:
            implications.append(ImpliedConstraint(
                condition="Asymmetric placement detected",
                implication="Should balance panels on both sides",
                confidence=0.6
            ))
        
        return implications
    
    def derive_efficiency_implications(self,
                                      placed: List[PanelPlacement],
                                      remaining: List[PanelSize]) -> List[ImpliedConstraint]:
        """Derive efficiency-based implications."""
        implications = []
        
        if not remaining:
            return implications
        
        # Calculate remaining area
        placed_area = sum(p.panel_size.width * p.panel_size.height for p in placed)
        remaining_area = sum(ps.width * ps.height for ps in remaining)
        room_area = self.room.width * self.room.height
        
        utilization = placed_area / room_area
        
        if utilization > 0.7 and remaining_area > room_area * 0.2:
            implications.append(ImpliedConstraint(
                condition="High utilization with significant remaining panels",
                implication="Must use tight packing for remaining panels",
                confidence=0.85
            ))
        
        # Check for fragmentation
        if len(placed) > 10:
            # Simple fragmentation check
            implications.append(ImpliedConstraint(
                condition="Many panels placed",
                implication="Avoid creating small gaps",
                confidence=0.7
            ))
        
        return implications


class ConstraintDatabase:
    """Database for storing and managing learned constraints."""
    
    def __init__(self, max_size: int = 1000, eviction_threshold: float = 0.1):
        self.max_size = max_size
        self.eviction_threshold = eviction_threshold
        self.nogoods = {}  # Hash -> Nogood
        self.implications = []
        self.index_by_panel = defaultdict(set)  # Panel -> Set of nogood hashes
        self.index_by_position = defaultdict(set)  # Position -> Set of nogood hashes
        self.statistics = {
            'total_added': 0,
            'total_evicted': 0,
            'total_queries': 0,
            'total_hits': 0
        }
    
    def add_nogood(self, nogood: Nogood) -> bool:
        """Add a nogood to the database."""
        # Check if already exists
        nogood_hash = hash(nogood)
        if nogood_hash in self.nogoods:
            # Update existing
            existing = self.nogoods[nogood_hash]
            existing.strength = max(existing.strength, nogood.strength)
            return False
        
        # Check size limit
        if len(self.nogoods) >= self.max_size:
            self._evict_weak_nogoods()
        
        # Add to database
        self.nogoods[nogood_hash] = nogood
        self.statistics['total_added'] += 1
        
        # Update indices
        for panel_size, position, _ in nogood.assignments:
            self.index_by_panel[panel_size].add(nogood_hash)
            self.index_by_position[position].add(nogood_hash)
        
        return True
    
    def add_implication(self, implication: ImpliedConstraint):
        """Add an implied constraint."""
        # Check for duplicates
        for existing in self.implications:
            if (existing.condition == implication.condition and
                existing.implication == implication.implication):
                # Update existing
                existing.update(implication.reliability > 0.5)
                return
        
        self.implications.append(implication)
    
    def check_nogood(self, assignments: List[PanelPlacement]) -> Optional[Nogood]:
        """Check if assignments violate any nogood."""
        self.statistics['total_queries'] += 1
        
        assignment_set = frozenset(
            (p.panel_size, p.position, getattr(p, 'orientation', 'horizontal'))
            for p in assignments
        )
        
        # Check each relevant nogood
        checked = set()
        for assignment in assignments:
            # Get nogoods involving this panel or position
            panel_nogoods = self.index_by_panel.get(assignment.panel_size, set())
            pos_nogoods = self.index_by_position.get(assignment.position, set())
            
            for nogood_hash in panel_nogoods | pos_nogoods:
                if nogood_hash in checked:
                    continue
                checked.add(nogood_hash)
                
                nogood = self.nogoods[nogood_hash]
                
                # Check if current assignments contain the nogood
                if nogood.assignments.issubset(assignment_set):
                    self.statistics['total_hits'] += 1
                    nogood.update_usage(True)
                    return nogood
        
        return None
    
    def get_relevant_implications(self,
                                 state: PackingState) -> List[ImpliedConstraint]:
        """Get implications relevant to current state."""
        relevant = []
        
        for implication in self.implications:
            # Simple relevance check based on condition keywords
            if "corner" in implication.condition.lower():
                # Check if corners are relevant
                relevant.append(implication)
            elif "row" in implication.condition.lower():
                # Check specific row conditions
                relevant.append(implication)
            elif implication.reliability > 0.7:
                # High reliability implications are always relevant
                relevant.append(implication)
        
        return relevant
    
    def _evict_weak_nogoods(self):
        """Remove weak nogoods to make space."""
        # Calculate scores for all nogoods
        scored = []
        for hash_val, nogood in self.nogoods.items():
            score = nogood.effectiveness
            scored.append((score, hash_val))
        
        # Sort by score
        scored.sort()
        
        # Remove bottom portion
        evict_count = int(len(scored) * 0.2)  # Remove bottom 20%
        for _, hash_val in scored[:evict_count]:
            nogood = self.nogoods[hash_val]
            
            # Remove from indices
            for panel_size, position, _ in nogood.assignments:
                self.index_by_panel[panel_size].discard(hash_val)
                self.index_by_position[position].discard(hash_val)
            
            # Remove from database
            del self.nogoods[hash_val]
            self.statistics['total_evicted'] += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        stats = self.statistics.copy()
        stats['current_nogoods'] = len(self.nogoods)
        stats['current_implications'] = len(self.implications)
        
        if stats['total_queries'] > 0:
            stats['hit_rate'] = stats['total_hits'] / stats['total_queries']
        else:
            stats['hit_rate'] = 0
        
        # Calculate average effectiveness
        if self.nogoods:
            stats['avg_effectiveness'] = sum(
                n.effectiveness for n in self.nogoods.values()
            ) / len(self.nogoods)
        else:
            stats['avg_effectiveness'] = 0
        
        return stats


class RelevanceScorer:
    """Scores relevance of constraints for current search state."""
    
    def __init__(self, decay_factor: float = 0.95):
        self.decay_factor = decay_factor
        self.usage_history = defaultdict(list)  # Constraint -> List of (time, success)
        self.context_scores = defaultdict(float)  # (Constraint, Context) -> Score
    
    def score_nogood(self,
                    nogood: Nogood,
                    current_state: PackingState,
                    search_depth: int) -> float:
        """Score relevance of a nogood for current state."""
        score = 0.0
        
        # Base score from nogood strength
        score += nogood.strength * 0.3
        
        # Recency bonus
        time_since_creation = time.time() - nogood.creation_time
        recency_factor = self.decay_factor ** (time_since_creation / 3600)
        score += recency_factor * 0.2
        
        # Usage effectiveness
        score += nogood.effectiveness * 0.3
        
        # Depth relevance (more relevant at shallow depths)
        depth_factor = 1.0 / (1.0 + search_depth * 0.1)
        score += depth_factor * 0.2
        
        # Context similarity
        context_key = self._get_context_key(current_state)
        if context_key in self.context_scores:
            score += self.context_scores[context_key] * 0.1
        
        return min(1.0, score)
    
    def score_implication(self,
                         implication: ImpliedConstraint,
                         current_state: PackingState) -> float:
        """Score relevance of an implication."""
        score = 0.0
        
        # Base score from confidence
        score += implication.confidence * 0.4
        
        # Reliability bonus
        score += implication.reliability * 0.3
        
        # Support evidence
        if implication.support_count > 0:
            support_factor = min(1.0, implication.support_count / 10)
            score += support_factor * 0.3
        
        return min(1.0, score)
    
    def update_usage(self,
                    constraint: Any,
                    context: PackingState,
                    successful: bool):
        """Update usage history for a constraint."""
        # Record usage
        self.usage_history[id(constraint)].append((time.time(), successful))
        
        # Update context score
        context_key = self._get_context_key(context)
        old_score = self.context_scores.get((id(constraint), context_key), 0.5)
        
        # Exponential moving average update
        alpha = 0.1
        new_score = old_score * (1 - alpha) + (1.0 if successful else 0.0) * alpha
        self.context_scores[(id(constraint), context_key)] = new_score
        
        # Clean old history
        self._clean_old_history()
    
    def _get_context_key(self, state: PackingState) -> str:
        """Generate context key for a state."""
        # Simple context based on coverage and panel count
        coverage_bucket = int(state.coverage_ratio * 10)
        panel_bucket = len(state.placed_panels) // 5
        return f"cov_{coverage_bucket}_panels_{panel_bucket}"
    
    def _clean_old_history(self):
        """Remove old usage history entries."""
        current_time = time.time()
        cutoff_time = current_time - 7200  # 2 hours
        
        for constraint_id in list(self.usage_history.keys()):
            history = self.usage_history[constraint_id]
            # Keep only recent entries
            self.usage_history[constraint_id] = [
                (t, s) for t, s in history if t > cutoff_time
            ]
            
            # Remove if no recent history
            if not self.usage_history[constraint_id]:
                del self.usage_history[constraint_id]


class ConstraintLearningSystem:
    """Main system coordinating all constraint learning components."""
    
    def __init__(self, room: Room, config: Optional[Dict[str, Any]] = None):
        self.room = room
        self.config = config or {}
        
        # Initialize components
        self.nogood_learner = NogoodLearner(
            max_nogood_size=self.config.get('max_nogood_size', 5),
            min_strength=self.config.get('min_strength', 0.3)
        )
        
        self.implication_deriver = ImpliedConstraintDeriver(room)
        
        self.database = ConstraintDatabase(
            max_size=self.config.get('max_database_size', 1000),
            eviction_threshold=self.config.get('eviction_threshold', 0.1)
        )
        
        self.relevance_scorer = RelevanceScorer(
            decay_factor=self.config.get('decay_factor', 0.95)
        )
        
        self.learning_enabled = self.config.get('learning_enabled', True)
    
    def process_conflict(self,
                        state: PackingState,
                        conflict: ConstraintViolation) -> Optional[Nogood]:
        """Process a constraint conflict and learn from it."""
        if not self.learning_enabled:
            return None
        
        # Learn nogood from conflict
        nogood = self.nogood_learner.learn_from_conflict(
            state.placed_panels,
            conflict
        )
        
        if nogood:
            self.database.add_nogood(nogood)
            return nogood
        
        return None
    
    def process_infeasibility(self,
                            state: PackingState,
                            remaining_panels: List[PanelSize]) -> Optional[Nogood]:
        """Process an infeasible state."""
        if not self.learning_enabled:
            return None
        
        nogood = self.nogood_learner.learn_from_infeasibility(
            state.placed_panels,
            remaining_panels
        )
        
        if nogood:
            self.database.add_nogood(nogood)
            return nogood
        
        return None
    
    def derive_implications(self, state: PackingState) -> List[ImpliedConstraint]:
        """Derive implied constraints from current state."""
        if not self.learning_enabled:
            return []
        
        implications = []
        
        # Derive spatial implications
        spatial = self.implication_deriver.derive_spatial_implications(
            state.placed_panels
        )
        implications.extend(spatial)
        
        # Derive symmetry implications
        symmetry = self.implication_deriver.derive_symmetry_implications(
            state.placed_panels
        )
        implications.extend(symmetry)
        
        # Add to database
        for impl in implications:
            self.database.add_implication(impl)
        
        return implications
    
    def check_state(self, state: PackingState) -> Tuple[Optional[Nogood], List[ImpliedConstraint]]:
        """Check state against learned constraints."""
        # Check for nogood violations
        nogood = self.database.check_nogood(state.placed_panels)
        
        # Get relevant implications
        implications = self.database.get_relevant_implications(state)
        
        # Score and filter implications
        scored_implications = []
        for impl in implications:
            score = self.relevance_scorer.score_implication(impl, state)
            if score > 0.5:  # Threshold for relevance
                scored_implications.append(impl)
        
        return nogood, scored_implications
    
    def update_learning(self,
                       constraint: Any,
                       state: PackingState,
                       successful: bool):
        """Update learning based on constraint usage."""
        self.relevance_scorer.update_usage(constraint, state, successful)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        stats = {
            'database': self.database.get_statistics(),
            'total_nogoods': len(self.database.nogoods),
            'total_implications': len(self.database.implications),
            'learning_enabled': self.learning_enabled
        }
        
        # Add effectiveness distribution
        if self.database.nogoods:
            effectiveness_values = [
                n.effectiveness for n in self.database.nogoods.values()
            ]
            stats['effectiveness'] = {
                'min': min(effectiveness_values),
                'max': max(effectiveness_values),
                'avg': sum(effectiveness_values) / len(effectiveness_values)
            }
        
        return stats
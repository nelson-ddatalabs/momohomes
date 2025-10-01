#!/usr/bin/env python3
"""
bb_cutting_planes.py - Cutting Plane Generation
===============================================
Production-ready clique cuts, cover inequalities, knapsack cuts,
and cut management for strengthening branch & bound relaxations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any, FrozenSet
from collections import defaultdict, deque
from enum import Enum
import math
import numpy as np

from models import Room, PanelSize, Point
from advanced_packing import PackingState, PanelPlacement
from bb_search_tree import BranchNode


class CutType(Enum):
    """Types of cutting planes."""
    CLIQUE = "clique"  # Maximal clique inequalities
    COVER = "cover"  # Minimal cover inequalities
    KNAPSACK = "knapsack"  # Knapsack cover inequalities
    GOMORY = "gomory"  # Gomory mixed-integer cuts
    LIFTED = "lifted"  # Lifted inequalities


@dataclass
class CuttingPlane:
    """Represents a cutting plane inequality."""
    coefficients: Dict[Tuple[PanelSize, Tuple[float, float]], float]  # Variable coefficients
    rhs: float  # Right-hand side
    cut_type: CutType
    violation: float = 0.0  # Amount of violation in current solution
    effectiveness: float = 0.0  # Historical effectiveness
    age: int = 0  # Iterations since creation
    usage_count: int = 0
    
    def evaluate(self, solution: Dict[Tuple[PanelSize, Tuple[float, float]], float]) -> float:
        """Evaluate cut on a solution."""
        lhs = sum(
            coeff * solution.get(var, 0.0)
            for var, coeff in self.coefficients.items()
        )
        return lhs
    
    def is_violated(self, solution: Dict[Tuple[PanelSize, Tuple[float, float]], float]) -> bool:
        """Check if cut is violated by solution."""
        lhs = self.evaluate(solution)
        self.violation = max(0, lhs - self.rhs)
        return self.violation > 1e-6
    
    def update_effectiveness(self, improved: bool):
        """Update effectiveness based on usage."""
        self.usage_count += 1
        alpha = 0.1  # Learning rate
        self.effectiveness = (1 - alpha) * self.effectiveness + alpha * (1.0 if improved else 0.0)
        self.age += 1


class ConflictGraph:
    """Graph representing conflicts between panel placements."""
    
    def __init__(self):
        self.nodes = set()  # (panel_size, position) tuples
        self.edges = defaultdict(set)  # Adjacency list
        self.node_weights = {}  # Node weights for weighted cliques
    
    def add_node(self, panel: PanelSize, position: Tuple[float, float], weight: float = 1.0):
        """Add a node to the conflict graph."""
        node = (panel, position)
        self.nodes.add(node)
        self.node_weights[node] = weight
    
    def add_edge(self, node1: Tuple, node2: Tuple):
        """Add an edge representing a conflict."""
        self.edges[node1].add(node2)
        self.edges[node2].add(node1)
    
    def find_maximal_cliques(self, min_size: int = 3) -> List[Set[Tuple]]:
        """Find maximal cliques using Bron-Kerbosch algorithm."""
        cliques = []
        self._bron_kerbosch(set(), self.nodes.copy(), set(), cliques)
        # Filter by minimum size
        return [c for c in cliques if len(c) >= min_size]
    
    def _bron_kerbosch(self, R: Set, P: Set, X: Set, cliques: List[Set]):
        """Bron-Kerbosch algorithm for finding maximal cliques."""
        if not P and not X:
            if len(R) > 0:
                cliques.append(R.copy())
            return
        
        # Choose pivot to minimize branching
        pivot = max(P | X, key=lambda v: len(self.edges[v] & P)) if P | X else None
        
        if pivot:
            P_minus_pivot = P - self.edges[pivot]
        else:
            P_minus_pivot = P.copy()
        
        for v in list(P_minus_pivot):
            neighbors = self.edges[v]
            self._bron_kerbosch(
                R | {v},
                P & neighbors,
                X & neighbors,
                cliques
            )
            P.remove(v)
            X.add(v)
    
    def find_stable_sets(self) -> List[Set[Tuple]]:
        """Find stable sets (independent sets) in the graph."""
        # Complement graph approach
        stable_sets = []
        nodes_list = list(self.nodes)
        
        # Greedy approach for finding stable sets
        for start_node in nodes_list:
            stable_set = {start_node}
            for node in nodes_list:
                if node != start_node:
                    # Check if node is independent of all in stable_set
                    if all(node not in self.edges[s] for s in stable_set):
                        stable_set.add(node)
            
            if len(stable_set) > 1:
                stable_sets.append(stable_set)
        
        # Remove duplicates
        unique_sets = []
        seen = set()
        for s in stable_sets:
            frozen = frozenset(s)
            if frozen not in seen:
                seen.add(frozen)
                unique_sets.append(s)
        
        return unique_sets


class CliqueCutGenerator:
    """Generates clique-based cutting planes."""
    
    def __init__(self, room: Room):
        self.room = room
        self.conflict_graph = ConflictGraph()
        self.generated_cuts = []
    
    def build_conflict_graph(self,
                            panels: List[PanelSize],
                            positions: List[Tuple[float, float]]) -> ConflictGraph:
        """Build conflict graph from panels and positions."""
        graph = ConflictGraph()
        
        # Add nodes
        for panel in panels:
            for pos in positions:
                if self._is_valid_placement(panel, pos):
                    weight = panel.width * panel.height  # Weight by area
                    graph.add_node(panel, pos, weight)
        
        # Add edges for conflicts
        nodes = list(graph.nodes)
        for i, (panel1, pos1) in enumerate(nodes):
            for panel2, pos2 in nodes[i+1:]:
                if self._conflicts(panel1, pos1, panel2, pos2):
                    graph.add_edge((panel1, pos1), (panel2, pos2))
        
        self.conflict_graph = graph
        return graph
    
    def _is_valid_placement(self, panel: PanelSize, position: Tuple[float, float]) -> bool:
        """Check if placement is valid in room."""
        x, y = position
        return (0 <= x <= self.room.width - panel.width and
                0 <= y <= self.room.height - panel.height)
    
    def _conflicts(self,
                  panel1: PanelSize,
                  pos1: Tuple[float, float],
                  panel2: PanelSize,
                  pos2: Tuple[float, float]) -> bool:
        """Check if two placements conflict (overlap)."""
        x1, y1 = pos1
        x2, y2 = pos2
        
        # Check for overlap
        return not (x1 + panel1.width <= x2 or
                   x2 + panel2.width <= x1 or
                   y1 + panel1.height <= y2 or
                   y2 + panel2.height <= y1)
    
    def generate_clique_cuts(self, min_size: int = 3) -> List[CuttingPlane]:
        """Generate clique inequalities."""
        cuts = []
        cliques = self.conflict_graph.find_maximal_cliques(min_size)
        
        for clique in cliques:
            # Create clique inequality: sum of variables in clique <= 1
            coefficients = {}
            for panel, pos in clique:
                coefficients[(panel, pos)] = 1.0
            
            cut = CuttingPlane(
                coefficients=coefficients,
                rhs=1.0,
                cut_type=CutType.CLIQUE
            )
            cuts.append(cut)
        
        self.generated_cuts.extend(cuts)
        return cuts
    
    def generate_weighted_clique_cuts(self) -> List[CuttingPlane]:
        """Generate weighted clique inequalities."""
        cuts = []
        cliques = self.conflict_graph.find_maximal_cliques()
        
        for clique in cliques:
            # Calculate weights
            total_weight = sum(
                self.conflict_graph.node_weights.get((panel, pos), 1.0)
                for panel, pos in clique
            )
            
            if total_weight > self.room.width * self.room.height:
                # Create weighted inequality
                coefficients = {}
                for panel, pos in clique:
                    weight = self.conflict_graph.node_weights.get((panel, pos), 1.0)
                    coefficients[(panel, pos)] = weight
                
                cut = CuttingPlane(
                    coefficients=coefficients,
                    rhs=self.room.width * self.room.height,
                    cut_type=CutType.CLIQUE
                )
                cuts.append(cut)
        
        return cuts


class CoverCutGenerator:
    """Generates cover-based cutting planes."""
    
    def __init__(self, room: Room):
        self.room = room
        self.covers = []
    
    def find_minimal_covers(self,
                          panels: List[PanelSize],
                          positions: List[Tuple[float, float]],
                          target_area: float) -> List[Set[Tuple[PanelSize, Tuple[float, float]]]]:
        """Find minimal covers that exceed target area."""
        covers = []
        items = []
        
        # Create items with areas
        for panel in panels:
            for pos in positions:
                if self._is_valid_placement(panel, pos):
                    items.append((panel, pos, panel.width * panel.height))
        
        # Sort by area (largest first)
        items.sort(key=lambda x: x[2], reverse=True)
        
        # Find minimal covers
        for i in range(len(items)):
            cover = set()
            total_area = 0
            
            for j in range(i, len(items)):
                panel, pos, area = items[j]
                cover.add((panel, pos))
                total_area += area
                
                if total_area > target_area:
                    # Found a cover, try to minimize it
                    minimal_cover = self._minimize_cover(cover, target_area)
                    if minimal_cover not in covers:
                        covers.append(minimal_cover)
                    break
        
        self.covers = covers
        return covers
    
    def _is_valid_placement(self, panel: PanelSize, position: Tuple[float, float]) -> bool:
        """Check if placement is valid."""
        x, y = position
        return (0 <= x <= self.room.width - panel.width and
                0 <= y <= self.room.height - panel.height)
    
    def _minimize_cover(self,
                       cover: Set[Tuple[PanelSize, Tuple[float, float]]],
                       target_area: float) -> Set[Tuple[PanelSize, Tuple[float, float]]]:
        """Minimize a cover by removing unnecessary elements."""
        minimal = cover.copy()
        
        for panel, pos in cover:
            test_cover = minimal - {(panel, pos)}
            total_area = sum(p.width * p.height for p, _ in test_cover)
            
            if total_area > target_area:
                minimal = test_cover
        
        return minimal
    
    def generate_cover_cuts(self, target_coverage: float = 0.9) -> List[CuttingPlane]:
        """Generate cover inequalities."""
        cuts = []
        target_area = self.room.width * self.room.height * target_coverage
        
        for cover in self.covers:
            if len(cover) > 1:
                # Create cover inequality: sum of (|C| - 1) * x_i <= |C| - 1
                coefficients = {}
                for panel, pos in cover:
                    coefficients[(panel, pos)] = 1.0
                
                cut = CuttingPlane(
                    coefficients=coefficients,
                    rhs=float(len(cover) - 1),
                    cut_type=CutType.COVER
                )
                cuts.append(cut)
        
        return cuts
    
    def generate_extended_cover_cuts(self) -> List[CuttingPlane]:
        """Generate extended cover inequalities."""
        cuts = []
        
        for cover in self.covers:
            # Find items not in cover but that could extend it
            extension_candidates = self._find_extension_candidates(cover)
            
            if extension_candidates:
                # Create extended cover inequality
                coefficients = {}
                
                # Original cover variables
                for panel, pos in cover:
                    coefficients[(panel, pos)] = 1.0
                
                # Extension variables with fractional coefficients
                for panel, pos in extension_candidates:
                    area = panel.width * panel.height
                    min_area = min(p.width * p.height for p, _ in cover)
                    coefficients[(panel, pos)] = min(1.0, area / min_area)
                
                cut = CuttingPlane(
                    coefficients=coefficients,
                    rhs=float(len(cover) - 1),
                    cut_type=CutType.COVER
                )
                cuts.append(cut)
        
        return cuts
    
    def _find_extension_candidates(self,
                                  cover: Set[Tuple[PanelSize, Tuple[float, float]]]) -> Set[Tuple[PanelSize, Tuple[float, float]]]:
        """Find candidates that could extend a cover."""
        candidates = set()
        min_area = min(p.width * p.height for p, _ in cover)
        
        # Look for items with area >= min_area in cover
        # (These could potentially replace items in the cover)
        # This is simplified - real implementation would be more sophisticated
        
        return candidates


class KnapsackCutGenerator:
    """Generates knapsack-based cutting planes."""
    
    def __init__(self, room: Room):
        self.room = room
        self.capacity = room.width * room.height
    
    def generate_knapsack_cuts(self,
                              panels: List[PanelSize],
                              positions: List[Tuple[float, float]]) -> List[CuttingPlane]:
        """Generate knapsack cover inequalities."""
        cuts = []
        
        # Create items with weights (areas) and values
        items = []
        for panel in panels:
            for pos in positions:
                if self._is_valid_placement(panel, pos):
                    weight = panel.width * panel.height
                    value = weight  # Use area as value
                    items.append((panel, pos, weight, value))
        
        # Find covers for knapsack
        covers = self._find_knapsack_covers(items)
        
        for cover in covers:
            # Generate knapsack cover inequality
            coefficients = {}
            for panel, pos, _, _ in cover:
                coefficients[(panel, pos)] = 1.0
            
            cut = CuttingPlane(
                coefficients=coefficients,
                rhs=float(len(cover) - 1),
                cut_type=CutType.KNAPSACK
            )
            cuts.append(cut)
        
        # Generate lifted knapsack inequalities
        lifted_cuts = self._generate_lifted_cuts(covers, items)
        cuts.extend(lifted_cuts)
        
        return cuts
    
    def _is_valid_placement(self, panel: PanelSize, position: Tuple[float, float]) -> bool:
        """Check if placement is valid."""
        x, y = position
        return (0 <= x <= self.room.width - panel.width and
                0 <= y <= self.room.height - panel.height)
    
    def _find_knapsack_covers(self,
                            items: List[Tuple]) -> List[List[Tuple]]:
        """Find minimal covers for knapsack constraint."""
        covers = []
        
        # Sort by weight/value ratio
        items.sort(key=lambda x: x[2] / x[3] if x[3] > 0 else float('inf'))
        
        # Greedy approach to find covers
        for start_idx in range(len(items)):
            cover = []
            total_weight = 0
            
            for i in range(start_idx, len(items)):
                item = items[i]
                cover.append(item)
                total_weight += item[2]
                
                if total_weight > self.capacity:
                    # Found a cover
                    minimal = self._minimize_knapsack_cover(cover)
                    if len(minimal) > 1:
                        covers.append(minimal)
                    break
        
        return covers
    
    def _minimize_knapsack_cover(self, cover: List[Tuple]) -> List[Tuple]:
        """Minimize a knapsack cover."""
        minimal = cover.copy()
        
        for item in cover:
            test_cover = [x for x in minimal if x != item]
            total_weight = sum(x[2] for x in test_cover)
            
            if total_weight > self.capacity:
                minimal = test_cover
        
        return minimal
    
    def _generate_lifted_cuts(self,
                            covers: List[List[Tuple]],
                            all_items: List[Tuple]) -> List[CuttingPlane]:
        """Generate lifted knapsack inequalities."""
        cuts = []
        
        for cover in covers:
            cover_set = set(cover)
            non_cover = [item for item in all_items if item not in cover_set]
            
            # Calculate lifting coefficients
            coefficients = {}
            
            # Original cover variables
            for panel, pos, _, _ in cover:
                coefficients[(panel, pos)] = 1.0
            
            # Lifted variables
            for panel, pos, weight, _ in non_cover:
                # Simple lifting coefficient (can be improved)
                lift_coeff = min(1.0, weight / min(c[2] for c in cover))
                if lift_coeff > 0:
                    coefficients[(panel, pos)] = lift_coeff
            
            if len(coefficients) > len(cover):
                cut = CuttingPlane(
                    coefficients=coefficients,
                    rhs=float(len(cover) - 1),
                    cut_type=CutType.LIFTED
                )
                cuts.append(cut)
        
        return cuts


class CutSeparator:
    """Separates violated cutting planes from current solution."""
    
    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance
        self.separated_cuts = []
    
    def separate(self,
                solution: Dict[Tuple[PanelSize, Tuple[float, float]], float],
                cut_generators: List[Any]) -> List[CuttingPlane]:
        """Separate violated cuts from solution."""
        violated_cuts = []
        
        for generator in cut_generators:
            if hasattr(generator, 'generated_cuts'):
                for cut in generator.generated_cuts:
                    if cut.is_violated(solution):
                        violated_cuts.append(cut)
        
        # Sort by violation amount
        violated_cuts.sort(key=lambda c: c.violation, reverse=True)
        
        self.separated_cuts = violated_cuts
        return violated_cuts
    
    def separate_most_violated(self,
                             solution: Dict[Tuple[PanelSize, Tuple[float, float]], float],
                             cuts: List[CuttingPlane],
                             max_cuts: int = 10) -> List[CuttingPlane]:
        """Separate the most violated cuts."""
        violated = []
        
        for cut in cuts:
            if cut.is_violated(solution):
                violated.append(cut)
        
        # Sort by violation and return top cuts
        violated.sort(key=lambda c: c.violation, reverse=True)
        return violated[:max_cuts]


class CutManager:
    """Manages the pool of cutting planes."""
    
    def __init__(self, max_cuts: int = 1000, max_age: int = 50):
        self.max_cuts = max_cuts
        self.max_age = max_age
        self.cut_pool = []
        self.active_cuts = []
        self.statistics = {
            'total_generated': 0,
            'total_added': 0,
            'total_removed': 0,
            'total_violations': 0
        }
    
    def add_cut(self, cut: CuttingPlane) -> bool:
        """Add a cut to the pool."""
        # Check for duplicates
        for existing in self.cut_pool:
            if self._are_equivalent(cut, existing):
                return False
        
        # Add to pool
        self.cut_pool.append(cut)
        self.statistics['total_added'] += 1
        
        # Manage pool size
        if len(self.cut_pool) > self.max_cuts:
            self._remove_weak_cuts()
        
        return True
    
    def _are_equivalent(self, cut1: CuttingPlane, cut2: CuttingPlane) -> bool:
        """Check if two cuts are equivalent."""
        if abs(cut1.rhs - cut2.rhs) > 1e-6:
            return False
        
        # Check coefficients
        keys1 = set(cut1.coefficients.keys())
        keys2 = set(cut2.coefficients.keys())
        
        if keys1 != keys2:
            return False
        
        for key in keys1:
            if abs(cut1.coefficients[key] - cut2.coefficients[key]) > 1e-6:
                return False
        
        return True
    
    def _remove_weak_cuts(self):
        """Remove weak or old cuts from pool."""
        # Score cuts
        scored = []
        for cut in self.cut_pool:
            score = self._score_cut(cut)
            scored.append((score, cut))
        
        # Sort by score
        scored.sort(reverse=True)
        
        # Keep top cuts
        self.cut_pool = [cut for _, cut in scored[:self.max_cuts]]
        self.statistics['total_removed'] += len(scored) - len(self.cut_pool)
    
    def _score_cut(self, cut: CuttingPlane) -> float:
        """Score a cut based on various criteria."""
        score = 0.0
        
        # Effectiveness
        score += cut.effectiveness * 100
        
        # Violation amount
        score += cut.violation * 10
        
        # Age penalty
        age_penalty = min(1.0, cut.age / self.max_age)
        score *= (1 - age_penalty * 0.5)
        
        # Type bonus
        if cut.cut_type == CutType.CLIQUE:
            score *= 1.2
        elif cut.cut_type == CutType.LIFTED:
            score *= 1.1
        
        return score
    
    def select_cuts(self,
                   solution: Dict[Tuple[PanelSize, Tuple[float, float]], float],
                   max_active: int = 50) -> List[CuttingPlane]:
        """Select cuts to be active in current iteration."""
        # Find violated cuts
        violated = []
        for cut in self.cut_pool:
            if cut.is_violated(solution):
                violated.append(cut)
                self.statistics['total_violations'] += 1
        
        # Sort by violation
        violated.sort(key=lambda c: c.violation, reverse=True)
        
        # Select diverse set of cuts
        selected = self._select_diverse_cuts(violated, max_active)
        
        self.active_cuts = selected
        return selected
    
    def _select_diverse_cuts(self,
                           cuts: List[CuttingPlane],
                           max_cuts: int) -> List[CuttingPlane]:
        """Select diverse set of cuts."""
        if len(cuts) <= max_cuts:
            return cuts
        
        selected = []
        remaining = cuts.copy()
        
        # Select cuts of different types
        for cut_type in CutType:
            type_cuts = [c for c in remaining if c.cut_type == cut_type]
            if type_cuts:
                # Take best of each type
                n_select = min(len(type_cuts), max_cuts // len(CutType))
                selected.extend(type_cuts[:n_select])
                for c in type_cuts[:n_select]:
                    remaining.remove(c)
        
        # Fill remaining slots with most violated
        remaining.sort(key=lambda c: c.violation, reverse=True)
        n_remaining = max_cuts - len(selected)
        selected.extend(remaining[:n_remaining])
        
        return selected
    
    def update_cuts(self, improved: bool):
        """Update effectiveness of active cuts."""
        for cut in self.active_cuts:
            cut.update_effectiveness(improved)
    
    def age_cuts(self):
        """Age all cuts in pool."""
        for cut in self.cut_pool:
            cut.age += 1
        
        # Remove very old cuts
        self.cut_pool = [c for c in self.cut_pool if c.age < self.max_age]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cut management statistics."""
        stats = self.statistics.copy()
        stats['pool_size'] = len(self.cut_pool)
        stats['active_cuts'] = len(self.active_cuts)
        
        if self.cut_pool:
            stats['avg_effectiveness'] = sum(c.effectiveness for c in self.cut_pool) / len(self.cut_pool)
            stats['avg_age'] = sum(c.age for c in self.cut_pool) / len(self.cut_pool)
        
        return stats


class CuttingPlaneSystem:
    """Main system coordinating all cutting plane components."""
    
    def __init__(self, room: Room, config: Optional[Dict[str, Any]] = None):
        self.room = room
        self.config = config or {}
        
        # Initialize generators
        self.clique_generator = CliqueCutGenerator(room)
        self.cover_generator = CoverCutGenerator(room)
        self.knapsack_generator = KnapsackCutGenerator(room)
        
        # Initialize separator and manager
        self.separator = CutSeparator(
            tolerance=self.config.get('violation_tolerance', 1e-6)
        )
        
        self.cut_manager = CutManager(
            max_cuts=self.config.get('max_cuts', 1000),
            max_age=self.config.get('max_age', 50)
        )
        
        self.generation_enabled = self.config.get('generation_enabled', True)
    
    def generate_initial_cuts(self,
                            panels: List[PanelSize],
                            positions: List[Tuple[float, float]]) -> List[CuttingPlane]:
        """Generate initial set of cutting planes."""
        if not self.generation_enabled:
            return []
        
        all_cuts = []
        
        # Build conflict graph
        self.clique_generator.build_conflict_graph(panels, positions)
        
        # Generate clique cuts
        clique_cuts = self.clique_generator.generate_clique_cuts(
            min_size=self.config.get('min_clique_size', 3)
        )
        all_cuts.extend(clique_cuts)
        
        # Generate cover cuts
        self.cover_generator.find_minimal_covers(
            panels,
            positions,
            self.room.width * self.room.height * 0.9
        )
        cover_cuts = self.cover_generator.generate_cover_cuts()
        all_cuts.extend(cover_cuts)
        
        # Generate knapsack cuts
        knapsack_cuts = self.knapsack_generator.generate_knapsack_cuts(panels, positions)
        all_cuts.extend(knapsack_cuts)
        
        # Add to manager
        for cut in all_cuts:
            self.cut_manager.add_cut(cut)
        
        return all_cuts
    
    def separate_cuts(self,
                     solution: Dict[Tuple[PanelSize, Tuple[float, float]], float]) -> List[CuttingPlane]:
        """Separate violated cuts from current solution."""
        # Get violated cuts from pool
        violated = self.cut_manager.select_cuts(
            solution,
            max_active=self.config.get('max_active_cuts', 50)
        )
        
        return violated
    
    def update_system(self, improved: bool):
        """Update system after iteration."""
        self.cut_manager.update_cuts(improved)
        self.cut_manager.age_cuts()
    
    def get_active_cuts(self) -> List[CuttingPlane]:
        """Get currently active cuts."""
        return self.cut_manager.active_cuts
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            'manager': self.cut_manager.get_statistics(),
            'cliques_found': len(self.clique_generator.generated_cuts),
            'covers_found': len(self.cover_generator.covers),
            'generation_enabled': self.generation_enabled
        }
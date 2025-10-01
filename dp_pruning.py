#!/usr/bin/env python3
"""
dp_pruning.py - Advanced DP Pruning Strategies
==============================================
Production-ready pruning strategies for DP optimization including dominated state
elimination, bound pruning, symmetry elimination, and early termination.
"""

import time
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, FrozenSet
from collections import defaultdict
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np

from models import Room, PanelSize
from dp_state import DPState, SkylineProfile
from advanced_packing import PanelPlacement
from dp_grid import GridResolution


class PruningStrategy(Enum):
    """Types of pruning strategies available."""
    DOMINATED_STATES = "dominated"
    BOUND_PRUNING = "bounds"
    SYMMETRY_ELIMINATION = "symmetry"
    EARLY_TERMINATION = "early_term"


@dataclass(frozen=True)
class StateFingerprint:
    """
    Compact fingerprint for state comparison and symmetry detection.
    Captures essential state characteristics for pruning decisions.
    """
    coverage_bucket: int  # Coverage rounded to nearest 5%
    panel_counts: Tuple[int, ...]  # Count of each panel type remaining
    placement_count: int  # Total panels placed
    area_utilization: int  # Utilized area bucket
    skyline_signature: Optional[str] = None  # Compact skyline representation
    
    def __hash__(self) -> int:
        return hash((self.coverage_bucket, self.panel_counts, 
                    self.placement_count, self.area_utilization, 
                    self.skyline_signature))


class DominatedStateEliminator:
    """
    Eliminates states that are dominated by others (strictly worse in all aspects).
    Uses efficient comparison strategies to avoid exploring inferior states.
    """
    
    def __init__(self, room: Room, panel_sizes: List[PanelSize]):
        self.room = room
        self.panel_sizes = panel_sizes
        self.room_area = room.width * room.height
        self.dominated_count = 0
        self.comparison_count = 0
        
        # State buckets for efficient comparison
        self.state_buckets: Dict[StateFingerprint, List[DPState]] = defaultdict(list)
        self.best_by_bucket: Dict[StateFingerprint, DPState] = {}
        
    def is_dominated(self, state: DPState) -> bool:
        """
        Check if state is dominated by any existing state.
        Returns True if state should be pruned.
        """
        fingerprint = self._create_fingerprint(state)
        
        # Check against best state in same bucket
        if fingerprint in self.best_by_bucket:
            best_state = self.best_by_bucket[fingerprint]
            if self._dominates(best_state, state):
                self.dominated_count += 1
                return True
        
        # Check against similar states in nearby buckets
        for similar_fingerprint in self._get_similar_fingerprints(fingerprint):
            if similar_fingerprint in self.best_by_bucket:
                candidate = self.best_by_bucket[similar_fingerprint]
                self.comparison_count += 1
                
                if self._dominates(candidate, state):
                    self.dominated_count += 1
                    return True
        
        return False
    
    def register_state(self, state: DPState):
        """Register state for future domination checking."""
        fingerprint = self._create_fingerprint(state)
        
        # Update best state in bucket if this is better
        if (fingerprint not in self.best_by_bucket or 
            self._dominates(state, self.best_by_bucket[fingerprint])):
            self.best_by_bucket[fingerprint] = state
        
        # Add to bucket for comparison
        self.state_buckets[fingerprint].append(state)
    
    def _dominates(self, state1: DPState, state2: DPState) -> bool:
        """Check if state1 dominates state2 (is better in all relevant aspects)."""
        # Coverage comparison (primary criterion)
        if state1.coverage <= state2.coverage:
            return False
        
        # Remaining panels comparison (fewer remaining is better)
        remaining1 = len(state1.remaining_panels)
        remaining2 = len(state2.remaining_panels)
        
        if remaining1 > remaining2:
            return False
        
        # If coverage is significantly higher and similar remaining panels, dominate
        coverage_advantage = state1.coverage - state2.coverage
        if coverage_advantage > 0.1 and abs(remaining1 - remaining2) <= 1:
            return True
        
        # Detailed comparison for close states
        if coverage_advantage > 0.05:
            # Check remaining panel value
            value1 = sum(p.area for p in state1.remaining_panels)
            value2 = sum(p.area for p in state2.remaining_panels)
            
            if value1 <= value2:  # Less valuable panels remaining
                return True
        
        return False
    
    def _create_fingerprint(self, state: DPState) -> StateFingerprint:
        """Create fingerprint for state bucketing."""
        # Coverage bucket (5% intervals)
        coverage_bucket = int(state.coverage * 20)  # 0-20 buckets
        
        # Panel count buckets
        panel_counts = []
        for panel_type in self.panel_sizes:
            count = sum(1 for p in state.remaining_panels if p == panel_type)
            panel_counts.append(count)
        
        # Area utilization bucket
        used_area = state.coverage * self.room_area
        area_bucket = int(used_area / 10)  # 10 sq ft buckets
        
        # Skyline signature (if available)
        skyline_sig = None
        if state.skyline:
            skyline_sig = self._create_skyline_signature(state.skyline)
        
        return StateFingerprint(
            coverage_bucket=coverage_bucket,
            panel_counts=tuple(panel_counts),
            placement_count=len(state.placements),
            area_utilization=area_bucket,
            skyline_signature=skyline_sig
        )
    
    def _create_skyline_signature(self, skyline: SkylineProfile) -> str:
        """Create compact signature for skyline shape."""
        if not skyline.y_coords:
            return "empty"
        
        # Discretize skyline heights into buckets
        max_height = skyline.room_height
        buckets = []
        
        for y in skyline.y_coords:
            bucket = min(9, int(y / max_height * 10))  # 0-9 buckets
            buckets.append(str(bucket))
        
        return ''.join(buckets[:8])  # Limit to 8 positions
    
    def _get_similar_fingerprints(self, fingerprint: StateFingerprint) -> List[StateFingerprint]:
        """Get fingerprints of similar states for comparison."""
        similar = []
        
        # Check nearby coverage buckets
        for coverage_offset in [-1, 0, 1]:
            new_coverage = fingerprint.coverage_bucket + coverage_offset
            if 0 <= new_coverage <= 20:
                # Check nearby placement counts
                for count_offset in [-1, 0, 1]:
                    new_count = fingerprint.placement_count + count_offset
                    if new_count >= 0:
                        similar_fp = StateFingerprint(
                            coverage_bucket=new_coverage,
                            panel_counts=fingerprint.panel_counts,
                            placement_count=new_count,
                            area_utilization=fingerprint.area_utilization,
                            skyline_signature=fingerprint.skyline_signature
                        )
                        similar.append(similar_fp)
        
        return similar
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get domination elimination statistics."""
        return {
            'dominated_states': self.dominated_count,
            'comparisons_made': self.comparison_count,
            'unique_buckets': len(self.state_buckets),
            'total_states_seen': sum(len(states) for states in self.state_buckets.values()),
            'elimination_rate': self.dominated_count / max(1, self.comparison_count)
        }


class BoundPruner:
    """
    Prunes branches that cannot possibly lead to better solutions than current best.
    Uses multiple bounding strategies for tight bounds.
    """
    
    def __init__(self, room: Room, panel_sizes: List[PanelSize]):
        self.room = room
        self.panel_sizes = panel_sizes
        self.room_area = room.width * room.height
        self.pruned_count = 0
        self.bound_checks = 0
        
        # Precompute panel characteristics
        self.panel_areas = [p.area for p in panel_sizes]
        self.max_panel_area = max(self.panel_areas) if self.panel_areas else 0
        self.total_panel_area = sum(self.panel_areas)
        
    def should_prune(self, state: DPState, current_best: float, 
                    remaining_time: float = float('inf')) -> bool:
        """
        Determine if state should be pruned based on bounds.
        Returns True if branch cannot lead to better solution.
        """
        self.bound_checks += 1
        
        # Quick coverage bound
        if self._coverage_bound_prune(state, current_best):
            self.pruned_count += 1
            return True
        
        # Area-based bound
        if self._area_bound_prune(state, current_best):
            self.pruned_count += 1
            return True
        
        # Skyline-based bound (if available)
        if state.skyline and self._skyline_bound_prune(state, current_best):
            self.pruned_count += 1
            return True
        
        # Time-based pruning for real-time scenarios
        if remaining_time < 1.0 and self._time_bound_prune(state, current_best):
            self.pruned_count += 1
            return True
        
        return False
    
    def _coverage_bound_prune(self, state: DPState, current_best: float) -> bool:
        """Simple coverage-based bound."""
        if not state.remaining_panels:
            return state.coverage <= current_best
        
        # Optimistic bound: all remaining panels fit perfectly
        remaining_area = sum(p.area for p in state.remaining_panels)
        max_additional_coverage = remaining_area / self.room_area
        upper_bound = state.coverage + max_additional_coverage
        
        # Add small epsilon to account for numerical errors
        return upper_bound <= current_best + 0.001
    
    def _area_bound_prune(self, state: DPState, current_best: float) -> bool:
        """Area-based bound considering space constraints."""
        if not state.remaining_panels:
            return state.coverage <= current_best
        
        # More realistic bound considering packing efficiency
        remaining_area = sum(p.area for p in state.remaining_panels)
        
        # Assume packing efficiency drops with more panels (realistic)
        efficiency_factor = max(0.7, 1.0 - len(state.remaining_panels) * 0.05)
        realistic_additional = (remaining_area * efficiency_factor) / self.room_area
        
        upper_bound = state.coverage + realistic_additional
        return upper_bound <= current_best + 0.001
    
    def _skyline_bound_prune(self, state: DPState, current_best: float) -> bool:
        """Skyline-based bound using available space analysis."""
        if not state.skyline or not state.remaining_panels:
            return False
        
        # Estimate available space from skyline
        available_space = self._estimate_available_space(state.skyline)
        
        # Check if remaining panels can fit in available space
        remaining_area = sum(p.area for p in state.remaining_panels)
        
        if remaining_area > available_space:
            # Use available space as bound
            max_additional = available_space / self.room_area
            upper_bound = state.coverage + max_additional
            return upper_bound <= current_best + 0.001
        
        return False
    
    def _estimate_available_space(self, skyline: SkylineProfile) -> float:
        """Estimate available space from skyline profile."""
        if not skyline.x_coords or not skyline.y_coords:
            return self.room_area
        
        total_available = 0.0
        
        for i, x in enumerate(skyline.x_coords):
            if i + 1 < len(skyline.x_coords):
                width = skyline.x_coords[i + 1] - x
                available_height = skyline.room_height - skyline.y_coords[i]
                total_available += width * available_height
        
        return max(0.0, total_available)
    
    def _time_bound_prune(self, state: DPState, current_best: float) -> bool:
        """Time-based aggressive pruning for real-time scenarios."""
        # Only explore very promising states when time is limited
        if state.coverage <= current_best * 0.8:  # Must be at least 80% of current best
            return True
        
        # Prune if too many panels remaining (likely won't finish in time)
        if len(state.remaining_panels) > 3:
            return True
        
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get bound pruning statistics."""
        return {
            'pruned_states': self.pruned_count,
            'bound_checks': self.bound_checks,
            'prune_rate': self.pruned_count / max(1, self.bound_checks)
        }


class SymmetryEliminator:
    """
    Eliminates symmetric states to avoid redundant exploration.
    Handles rotational and translational symmetries in panel placement.
    """
    
    def __init__(self, room: Room, panel_sizes: List[PanelSize]):
        self.room = room
        self.panel_sizes = panel_sizes
        self.seen_signatures: Set[str] = set()
        self.eliminated_count = 0
        
        # Check room symmetries
        self.is_square_room = abs(room.width - room.height) < 0.1
        self.has_rotational_symmetry = self.is_square_room
        
    def is_symmetric_duplicate(self, state: DPState) -> bool:
        """
        Check if state is symmetric duplicate of previously seen state.
        Returns True if state should be eliminated.
        """
        signatures = self._generate_symmetry_signatures(state)
        
        for signature in signatures:
            if signature in self.seen_signatures:
                self.eliminated_count += 1
                return True
        
        # Register all signatures to prevent future duplicates
        self.seen_signatures.update(signatures)
        return False
    
    def _generate_symmetry_signatures(self, state: DPState) -> List[str]:
        """Generate all symmetric signatures for state."""
        signatures = []
        
        # Base signature
        base_sig = self._create_base_signature(state)
        signatures.append(base_sig)
        
        # Rotational symmetries (for square rooms)
        if self.has_rotational_symmetry:
            signatures.extend(self._create_rotational_signatures(state))
        
        # Panel type permutation symmetries
        if self._has_identical_panels():
            signatures.extend(self._create_permutation_signatures(state))
        
        return signatures
    
    def _create_base_signature(self, state: DPState) -> str:
        """Create base canonical signature for state."""
        components = [
            f"cov:{int(state.coverage * 1000)}",
            f"placed:{len(state.placements)}",
            f"remaining:{len(state.remaining_panels)}"
        ]
        
        # Add panel type distribution
        panel_dist = defaultdict(int)
        for panel in state.remaining_panels:
            panel_dist[panel.name] += 1
        
        panel_sig = ','.join(f"{k}:{v}" for k, v in sorted(panel_dist.items()))
        components.append(f"panels:{panel_sig}")
        
        # Add spatial signature if available
        if state.skyline:
            spatial_sig = self._create_spatial_signature(state.skyline)
            components.append(f"spatial:{spatial_sig}")
        
        return '|'.join(components)
    
    def _create_rotational_signatures(self, state: DPState) -> List[str]:
        """Create signatures for rotational symmetries."""
        signatures = []
        
        if not self.is_square_room or not state.skyline:
            return signatures
        
        # For square rooms, generate 90°, 180°, 270° rotation signatures
        for rotation in [90, 180, 270]:
            rotated_sig = self._rotate_signature(state.skyline, rotation)
            if rotated_sig:
                base_parts = self._create_base_signature(state).split('|')
                # Replace spatial component with rotated version
                rotated_components = []
                for part in base_parts:
                    if part.startswith('spatial:'):
                        rotated_components.append(f"spatial:{rotated_sig}")
                    else:
                        rotated_components.append(part)
                signatures.append('|'.join(rotated_components))
        
        return signatures
    
    def _rotate_signature(self, skyline: SkylineProfile, angle: int) -> Optional[str]:
        """Create signature for rotated skyline."""
        if angle == 180:
            # 180° rotation: reverse and invert heights
            if not skyline.y_coords:
                return None
            
            max_height = skyline.room_height
            rotated_heights = [max_height - y for y in reversed(skyline.y_coords)]
            return self._heights_to_signature(rotated_heights)
        
        # For 90° and 270°, more complex transformation needed
        # Simplified version for demonstration
        return f"rot{angle}:{hash(tuple(skyline.y_coords)) % 10000}"
    
    def _create_permutation_signatures(self, state: DPState) -> List[str]:
        """Create signatures for panel type permutation symmetries."""
        signatures = []
        
        # If multiple identical panel types exist, create permuted signatures
        identical_groups = self._find_identical_panel_groups()
        
        if len(identical_groups) > 1:
            # Create signature with normalized panel ordering
            normalized_sig = self._normalize_panel_signature(state)
            if normalized_sig != self._create_base_signature(state):
                signatures.append(normalized_sig)
        
        return signatures
    
    def _has_identical_panels(self) -> bool:
        """Check if there are identical panel types that can be permuted."""
        # Check for panels with same dimensions but different names
        dimensions = set()
        for panel in self.panel_sizes:
            dim_pair = (panel.width, panel.length)
            if dim_pair in dimensions:
                return True
            dimensions.add(dim_pair)
        
        return False
    
    def _find_identical_panel_groups(self) -> List[List[PanelSize]]:
        """Find groups of panels with identical properties."""
        dimension_groups = defaultdict(list)
        
        for panel in self.panel_sizes:
            key = (panel.width, panel.length, panel.area)
            dimension_groups[key].append(panel)
        
        return [group for group in dimension_groups.values() if len(group) > 1]
    
    def _normalize_panel_signature(self, state: DPState) -> str:
        """Create normalized signature with canonical panel ordering."""
        # Sort panels by canonical order (by name) to eliminate permutation differences
        remaining_sorted = sorted(state.remaining_panels, key=lambda p: p.name)
        
        # Create signature with sorted panels
        panel_dist = defaultdict(int)
        for panel in remaining_sorted:
            panel_dist[panel.name] += 1
        
        components = [
            f"cov:{int(state.coverage * 1000)}",
            f"placed:{len(state.placements)}",
            f"remaining:{len(remaining_sorted)}"
        ]
        
        panel_sig = ','.join(f"{k}:{v}" for k, v in sorted(panel_dist.items()))
        components.append(f"panels_norm:{panel_sig}")
        
        return '|'.join(components)
    
    def _create_spatial_signature(self, skyline: SkylineProfile) -> str:
        """Create spatial signature from skyline."""
        if not skyline.y_coords:
            return "empty"
        
        # Normalize heights to create signature
        normalized_heights = []
        max_height = skyline.room_height
        
        for height in skyline.y_coords:
            # Discretize to reduce signature space
            bucket = min(9, int(height / max_height * 10))
            normalized_heights.append(str(bucket))
        
        return ''.join(normalized_heights)
    
    def _heights_to_signature(self, heights: List[float]) -> str:
        """Convert height list to signature string."""
        buckets = []
        max_height = self.room.height
        
        for height in heights:
            bucket = min(9, int(height / max_height * 10))
            buckets.append(str(bucket))
        
        return ''.join(buckets)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get symmetry elimination statistics."""
        return {
            'eliminated_states': self.eliminated_count,
            'unique_signatures': len(self.seen_signatures),
            'has_rotational_symmetry': self.has_rotational_symmetry,
            'has_identical_panels': self._has_identical_panels()
        }


class EarlyTerminationManager:
    """
    Manages early termination conditions to stop search when optimal or good-enough
    solutions are found, or when further search is unlikely to improve results.
    """
    
    def __init__(self, target_coverage: float = 0.95, 
                 convergence_threshold: int = 1000,
                 improvement_threshold: float = 0.01):
        self.target_coverage = target_coverage
        self.convergence_threshold = convergence_threshold
        self.improvement_threshold = improvement_threshold
        
        # State tracking
        self.best_coverage = 0.0
        self.states_since_improvement = 0
        self.termination_reason: Optional[str] = None
        self.start_time = time.time()
        
    def should_terminate(self, current_best: float, states_processed: int,
                        elapsed_time: float, max_time: float) -> bool:
        """
        Check if search should terminate early.
        Returns True if termination conditions are met.
        """
        # Update state tracking
        if current_best > self.best_coverage + self.improvement_threshold:
            self.best_coverage = current_best
            self.states_since_improvement = 0
        else:
            self.states_since_improvement += 1
        
        # Target coverage reached
        if current_best >= self.target_coverage:
            self.termination_reason = f"Target coverage {self.target_coverage:.1%} reached"
            return True
        
        # Perfect coverage achieved
        if current_best >= 0.999:
            self.termination_reason = "Perfect coverage achieved"
            return True
        
        # Time limit reached
        if elapsed_time >= max_time:
            self.termination_reason = f"Time limit {max_time:.1f}s reached"
            return True
        
        # Convergence detection (no improvement for many states)
        if self.states_since_improvement >= self.convergence_threshold:
            self.termination_reason = f"Convergence: no improvement for {self.convergence_threshold} states"
            return True
        
        # Diminishing returns (very slow improvement rate)
        if (states_processed > 5000 and 
            self.states_since_improvement >= self.convergence_threshold // 2 and
            current_best < 0.5):
            self.termination_reason = "Diminishing returns: very slow improvement"
            return True
        
        return False
    
    def get_termination_reason(self) -> str:
        """Get reason for termination."""
        return self.termination_reason or "Search ongoing"
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get early termination statistics."""
        return {
            'best_coverage': self.best_coverage,
            'states_since_improvement': self.states_since_improvement,
            'termination_reason': self.termination_reason,
            'target_coverage': self.target_coverage,
            'convergence_threshold': self.convergence_threshold
        }


class ComprehensivePruningEngine:
    """
    Integrates all pruning strategies for maximum DP solver efficiency.
    Coordinates dominated state elimination, bound pruning, symmetry elimination,
    and early termination for optimal performance.
    """
    
    def __init__(self, room: Room, panel_sizes: List[PanelSize],
                 target_coverage: float = 0.95,
                 enabled_strategies: Optional[Set[PruningStrategy]] = None):
        
        self.room = room
        self.panel_sizes = panel_sizes
        self.target_coverage = target_coverage
        
        # Enable all strategies by default
        self.enabled_strategies = enabled_strategies or set(PruningStrategy)
        
        # Initialize pruning components
        self.dominated_eliminator = None
        self.bound_pruner = None
        self.symmetry_eliminator = None
        self.termination_manager = None
        
        if PruningStrategy.DOMINATED_STATES in self.enabled_strategies:
            self.dominated_eliminator = DominatedStateEliminator(room, panel_sizes)
        
        if PruningStrategy.BOUND_PRUNING in self.enabled_strategies:
            self.bound_pruner = BoundPruner(room, panel_sizes)
        
        if PruningStrategy.SYMMETRY_ELIMINATION in self.enabled_strategies:
            self.symmetry_eliminator = SymmetryEliminator(room, panel_sizes)
        
        if PruningStrategy.EARLY_TERMINATION in self.enabled_strategies:
            self.termination_manager = EarlyTerminationManager(target_coverage)
        
        # Statistics tracking
        self.total_states_considered = 0
        self.total_states_pruned = 0
        self.start_time = time.time()
        
    def should_prune_state(self, state: DPState, current_best: float,
                          remaining_time: float = float('inf')) -> bool:
        """
        Comprehensive pruning check using all enabled strategies.
        Returns True if state should be pruned.
        """
        self.total_states_considered += 1
        
        # Dominated state elimination
        if (self.dominated_eliminator and 
            self.dominated_eliminator.is_dominated(state)):
            self.total_states_pruned += 1
            return True
        
        # Bound pruning
        if (self.bound_pruner and 
            self.bound_pruner.should_prune(state, current_best, remaining_time)):
            self.total_states_pruned += 1
            return True
        
        # Symmetry elimination
        if (self.symmetry_eliminator and 
            self.symmetry_eliminator.is_symmetric_duplicate(state)):
            self.total_states_pruned += 1
            return True
        
        # Register state for future domination checking
        if self.dominated_eliminator:
            self.dominated_eliminator.register_state(state)
        
        return False
    
    def should_terminate_early(self, current_best: float, states_processed: int,
                              max_time: float) -> bool:
        """Check if search should terminate early."""
        if not self.termination_manager:
            return False
        
        elapsed_time = time.time() - self.start_time
        return self.termination_manager.should_terminate(
            current_best, states_processed, elapsed_time, max_time
        )
    
    def get_termination_reason(self) -> str:
        """Get reason for early termination."""
        if self.termination_manager:
            return self.termination_manager.get_termination_reason()
        return "No early termination enabled"
    
    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Get statistics from all pruning components."""
        stats = {
            'enabled_strategies': [s.value for s in self.enabled_strategies],
            'total_states_considered': self.total_states_considered,
            'total_states_pruned': self.total_states_pruned,
            'overall_prune_rate': self.total_states_pruned / max(1, self.total_states_considered),
            'processing_time': time.time() - self.start_time
        }
        
        # Add component-specific statistics
        if self.dominated_eliminator:
            stats['dominated_elimination'] = self.dominated_eliminator.get_statistics()
        
        if self.bound_pruner:
            stats['bound_pruning'] = self.bound_pruner.get_statistics()
        
        if self.symmetry_eliminator:
            stats['symmetry_elimination'] = self.symmetry_eliminator.get_statistics()
        
        if self.termination_manager:
            stats['early_termination'] = self.termination_manager.get_statistics()
        
        return stats


def create_pruning_engine(room: Room, 
                         panel_sizes: List[PanelSize],
                         target_coverage: float = 0.95,
                         strategies: Optional[Set[PruningStrategy]] = None) -> ComprehensivePruningEngine:
    """
    Factory function to create comprehensive pruning engine.
    Returns configured engine with all specified pruning strategies.
    """
    return ComprehensivePruningEngine(
        room=room,
        panel_sizes=panel_sizes,
        target_coverage=target_coverage,
        enabled_strategies=strategies
    )
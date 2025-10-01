#!/usr/bin/env python3
"""
blf_backtracking.py - Backtracking Components for BLF Algorithm
================================================================
Implements backtracking mechanisms for the Enhanced Bottom-Left-Fill algorithm.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Set, Dict, Any, Tuple
from collections import deque
import copy

from models import Panel, PanelSize, Point, Room
from advanced_packing import PackingState, PanelPlacement, StateTransition


# Step 2.2.1: Backtrack State Management
# ======================================

@dataclass
class StateSnapshot:
    """
    Immutable snapshot of optimization state for backtracking.
    """
    state: PackingState
    remaining_panels: List[PanelSize]
    decision_point: int
    coverage: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self) -> int:
        """Hash for use in sets and memoization."""
        return hash((self.state.canonical_hash, self.decision_point))
    
    @classmethod
    def create(cls, state: PackingState, remaining: List[PanelSize], 
               decision: int, timestamp: float) -> 'StateSnapshot':
        """Create a snapshot of current state."""
        return cls(
            state=state,  # Already immutable
            remaining_panels=list(remaining),  # Copy list
            decision_point=decision,
            coverage=state.coverage,
            timestamp=timestamp
        )


class PlacementHistoryStack:
    """
    Stack-based history management for panel placements.
    Supports efficient undo/redo operations.
    """
    
    def __init__(self, max_history: int = 1000):
        """Initialize history stack."""
        self.max_history = max_history
        self.history: deque = deque(maxlen=max_history)
        self.redo_stack: deque = deque(maxlen=max_history)
        self.current_state: Optional[PackingState] = None
    
    def push(self, placement: PanelPlacement, state: PackingState):
        """
        Push a new placement onto the history stack.
        """
        # Store the placement and resulting state
        self.history.append({
            'placement': placement,
            'state_before': self.current_state,
            'state_after': state
        })
        
        # Update current state
        self.current_state = state
        
        # Clear redo stack when new action is performed
        self.redo_stack.clear()
    
    def undo(self) -> Optional[PackingState]:
        """
        Undo the last placement.
        Returns the state before the last placement.
        """
        if not self.history:
            return None
        
        # Pop from history
        last_action = self.history.pop()
        
        # Push to redo stack
        self.redo_stack.append(last_action)
        
        # Restore previous state
        self.current_state = last_action['state_before']
        
        return self.current_state
    
    def redo(self) -> Optional[PackingState]:
        """
        Redo the previously undone placement.
        Returns the state after redoing.
        """
        if not self.redo_stack:
            return None
        
        # Pop from redo stack
        action = self.redo_stack.pop()
        
        # Push back to history
        self.history.append(action)
        
        # Restore state
        self.current_state = action['state_after']
        
        return self.current_state
    
    def get_history_depth(self) -> int:
        """Get current history depth."""
        return len(self.history)
    
    def can_undo(self) -> bool:
        """Check if undo is possible."""
        return len(self.history) > 0
    
    def can_redo(self) -> bool:
        """Check if redo is possible."""
        return len(self.redo_stack) > 0
    
    def clear(self):
        """Clear all history."""
        self.history.clear()
        self.redo_stack.clear()
        self.current_state = None
    
    def get_placement_sequence(self) -> List[PanelPlacement]:
        """Get sequence of all placements in history."""
        return [action['placement'] for action in self.history]


class StateSnapshotManager:
    """
    Manages state snapshots for efficient backtracking.
    """
    
    def __init__(self, max_snapshots: int = 100):
        """Initialize snapshot manager."""
        self.max_snapshots = max_snapshots
        self.snapshots: deque = deque(maxlen=max_snapshots)
        self.snapshot_index: Dict[int, StateSnapshot] = {}  # Decision point to snapshot
        self.current_decision_point = 0
    
    def create_snapshot(self, state: PackingState, remaining_panels: List[PanelSize],
                       timestamp: float) -> StateSnapshot:
        """
        Create and store a new snapshot.
        """
        snapshot = StateSnapshot.create(
            state=state,
            remaining=remaining_panels,
            decision=self.current_decision_point,
            timestamp=timestamp
        )
        
        # Store snapshot
        self.snapshots.append(snapshot)
        self.snapshot_index[self.current_decision_point] = snapshot
        
        # Increment decision point
        self.current_decision_point += 1
        
        return snapshot
    
    def restore_snapshot(self, decision_point: int) -> Optional[StateSnapshot]:
        """
        Restore state from a specific decision point.
        """
        if decision_point in self.snapshot_index:
            snapshot = self.snapshot_index[decision_point]
            
            # Update current decision point
            self.current_decision_point = decision_point + 1
            
            # Remove snapshots after this point
            self._prune_after_decision(decision_point)
            
            return snapshot
        
        return None
    
    def get_latest_snapshot(self) -> Optional[StateSnapshot]:
        """Get the most recent snapshot."""
        return self.snapshots[-1] if self.snapshots else None
    
    def find_snapshot_by_coverage(self, min_coverage: float) -> Optional[StateSnapshot]:
        """
        Find the most recent snapshot with at least the specified coverage.
        """
        for snapshot in reversed(self.snapshots):
            if snapshot.coverage >= min_coverage:
                return snapshot
        return None
    
    def _prune_after_decision(self, decision_point: int):
        """Remove all snapshots after a decision point."""
        to_remove = []
        for dp in self.snapshot_index:
            if dp > decision_point:
                to_remove.append(dp)
        
        for dp in to_remove:
            del self.snapshot_index[dp]
        
        # Also prune from deque
        while self.snapshots and self.snapshots[-1].decision_point > decision_point:
            self.snapshots.pop()
    
    def get_branch_points(self) -> List[int]:
        """Get all decision points where branches exist."""
        return sorted(self.snapshot_index.keys())
    
    def clear(self):
        """Clear all snapshots."""
        self.snapshots.clear()
        self.snapshot_index.clear()
        self.current_decision_point = 0


class EfficientStateStorage:
    """
    Efficient storage mechanism for large numbers of states.
    Uses compression and deduplication.
    """
    
    def __init__(self):
        """Initialize efficient storage."""
        self.placement_pool: Dict[str, PanelPlacement] = {}  # Deduplicate placements
        self.state_pool: Dict[str, PackingState] = {}  # Deduplicate states
        self.compressed_states: Dict[str, bytes] = {}  # Compressed state data
    
    def store_state(self, state: PackingState) -> str:
        """
        Store state efficiently, returning a key for retrieval.
        """
        state_hash = state.canonical_hash
        
        if state_hash not in self.state_pool:
            # Deduplicate placements
            deduplicated_placements = set()
            for placement in state.placements:
                placement_key = self._get_placement_key(placement)
                if placement_key not in self.placement_pool:
                    self.placement_pool[placement_key] = placement
                deduplicated_placements.add(placement_key)
            
            # Store deduplicated state
            self.state_pool[state_hash] = state
            
            # Optionally compress for long-term storage
            if len(self.state_pool) > 1000:
                self._compress_old_states()
        
        return state_hash
    
    def retrieve_state(self, state_key: str) -> Optional[PackingState]:
        """
        Retrieve state by key.
        """
        # Check uncompressed pool first
        if state_key in self.state_pool:
            return self.state_pool[state_key]
        
        # Check compressed storage
        if state_key in self.compressed_states:
            return self._decompress_state(state_key)
        
        return None
    
    def _get_placement_key(self, placement: PanelPlacement) -> str:
        """Get unique key for placement."""
        return f"{placement.panel_size}_{placement.position}_{placement.orientation}"
    
    def _compress_old_states(self):
        """Compress old states to save memory."""
        # Keep most recent 100 states uncompressed
        if len(self.state_pool) <= 100:
            return
        
        # Get oldest states
        states_to_compress = list(self.state_pool.keys())[:-100]
        
        for state_key in states_to_compress:
            # Compress state (simplified - would use actual compression)
            state = self.state_pool[state_key]
            # In production, would use pickle + gzip
            self.compressed_states[state_key] = str(state).encode()
            
            # Remove from uncompressed pool
            del self.state_pool[state_key]
    
    def _decompress_state(self, state_key: str) -> Optional[PackingState]:
        """Decompress a state."""
        # In production, would properly decompress
        # For now, return None as placeholder
        return None
    
    def get_storage_stats(self) -> Dict[str, int]:
        """Get storage statistics."""
        return {
            'placements_deduplicated': len(self.placement_pool),
            'states_uncompressed': len(self.state_pool),
            'states_compressed': len(self.compressed_states),
            'total_states': len(self.state_pool) + len(self.compressed_states)
        }
    
    def clear(self):
        """Clear all storage."""
        self.placement_pool.clear()
        self.state_pool.clear()
        self.compressed_states.clear()


class BacktrackStateManager:
    """
    Central manager for all backtracking state operations.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize backtrack state manager."""
        config = config or {}
        
        # Initialize components
        self.history_stack = PlacementHistoryStack(
            max_history=config.get('max_history', 1000)
        )
        self.snapshot_manager = StateSnapshotManager(
            max_snapshots=config.get('max_snapshots', 100)
        )
        self.storage = EfficientStateStorage()
        
        # Track backtrack statistics
        self.backtrack_count = 0
        self.max_backtrack_depth = 0
        self.current_depth = 0
    
    def save_state(self, state: PackingState, remaining_panels: List[PanelSize],
                  timestamp: float) -> StateSnapshot:
        """
        Save current state for potential backtracking.
        """
        # Create snapshot
        snapshot = self.snapshot_manager.create_snapshot(
            state, remaining_panels, timestamp
        )
        
        # Store state efficiently
        state_key = self.storage.store_state(state)
        snapshot.metadata['storage_key'] = state_key
        
        return snapshot
    
    def backtrack_to(self, decision_point: int) -> Optional[StateSnapshot]:
        """
        Backtrack to a specific decision point.
        """
        snapshot = self.snapshot_manager.restore_snapshot(decision_point)
        
        if snapshot:
            self.backtrack_count += 1
            
            # Calculate backtrack depth
            depth = self.current_depth - decision_point
            self.max_backtrack_depth = max(self.max_backtrack_depth, depth)
            
            # Update current depth
            self.current_depth = decision_point
            
            # Clear history after this point
            while (self.history_stack.get_history_depth() > decision_point and
                   self.history_stack.can_undo()):
                self.history_stack.undo()
        
        return snapshot
    
    def record_placement(self, placement: PanelPlacement, state: PackingState):
        """
        Record a placement in history.
        """
        self.history_stack.push(placement, state)
        self.current_depth += 1
    
    def undo_last_placement(self) -> Optional[PackingState]:
        """
        Undo the last placement.
        """
        state = self.history_stack.undo()
        if state:
            self.current_depth -= 1
        return state
    
    def get_alternative_branch(self, min_coverage: float) -> Optional[StateSnapshot]:
        """
        Find an alternative branch point with reasonable coverage.
        """
        return self.snapshot_manager.find_snapshot_by_coverage(min_coverage)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get backtracking statistics.
        """
        return {
            'backtrack_count': self.backtrack_count,
            'max_backtrack_depth': self.max_backtrack_depth,
            'current_depth': self.current_depth,
            'history_depth': self.history_stack.get_history_depth(),
            'snapshot_count': len(self.snapshot_manager.snapshots),
            'storage_stats': self.storage.get_storage_stats()
        }
    
    def clear(self):
        """
        Clear all backtracking state.
        """
        self.history_stack.clear()
        self.snapshot_manager.clear()
        self.storage.clear()
        self.backtrack_count = 0
        self.max_backtrack_depth = 0
        self.current_depth = 0


# Step 2.2.2: Backtrack Triggers
# ===============================

import time
from enum import Enum


class BacktrackReason(Enum):
    """Reasons for triggering backtracking."""
    COVERAGE_PLATEAU = "coverage_plateau"
    WASTE_THRESHOLD = "waste_threshold"
    IMPOSSIBLE_FIT = "impossible_fit"
    TIME_LIMIT = "time_limit"
    NO_VALID_POSITIONS = "no_valid_positions"
    DEAD_END = "dead_end"
    MANUAL = "manual"


class CoveragePlateauDetector:
    """
    Detects when coverage improvement has plateaued.
    """
    
    def __init__(self, plateau_threshold: int = 5, min_improvement: float = 0.01):
        """
        Initialize plateau detector.
        
        Args:
            plateau_threshold: Number of placements without improvement to trigger
            min_improvement: Minimum coverage improvement to reset counter
        """
        self.plateau_threshold = plateau_threshold
        self.min_improvement = min_improvement
        
        self.no_improvement_count = 0
        self.last_coverage = 0.0
        self.coverage_history: List[float] = []
    
    def update(self, current_coverage: float) -> bool:
        """
        Update with current coverage and check for plateau.
        Returns True if plateau detected.
        """
        self.coverage_history.append(current_coverage)
        
        # Check improvement
        improvement = current_coverage - self.last_coverage
        
        if improvement < self.min_improvement:
            self.no_improvement_count += 1
        else:
            self.no_improvement_count = 0
        
        self.last_coverage = current_coverage
        
        # Check if plateau threshold reached
        return self.no_improvement_count >= self.plateau_threshold
    
    def get_plateau_duration(self) -> int:
        """Get number of steps in current plateau."""
        return self.no_improvement_count
    
    def reset(self):
        """Reset plateau detection."""
        self.no_improvement_count = 0
        self.last_coverage = 0.0
        self.coverage_history.clear()
    
    def get_trend(self, window: int = 10) -> float:
        """
        Calculate coverage trend over recent window.
        Returns slope of trend line (positive = improving).
        """
        if len(self.coverage_history) < 2:
            return 0.0
        
        recent = self.coverage_history[-window:]
        if len(recent) < 2:
            return 0.0
        
        # Simple linear regression
        n = len(recent)
        x_mean = (n - 1) / 2
        y_mean = sum(recent) / n
        
        numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(recent))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        return numerator / denominator if denominator > 0 else 0.0


class WasteThresholdMonitor:
    """
    Monitors waste levels and triggers backtracking when threshold exceeded.
    """
    
    def __init__(self, max_waste_ratio: float = 0.3, 
                 critical_waste_ratio: float = 0.4):
        """
        Initialize waste monitor.
        
        Args:
            max_waste_ratio: Soft limit for waste (warning)
            critical_waste_ratio: Hard limit for waste (trigger backtrack)
        """
        self.max_waste_ratio = max_waste_ratio
        self.critical_waste_ratio = critical_waste_ratio
        
        self.waste_history: List[float] = []
        self.consecutive_high_waste = 0
    
    def check_waste(self, placed_area: float, room_area: float) -> Tuple[bool, float]:
        """
        Check current waste level.
        Returns (should_backtrack, waste_ratio).
        """
        waste_ratio = 1.0 - (placed_area / room_area)
        self.waste_history.append(waste_ratio)
        
        # Check critical threshold
        if waste_ratio > self.critical_waste_ratio:
            return True, waste_ratio
        
        # Check sustained high waste
        if waste_ratio > self.max_waste_ratio:
            self.consecutive_high_waste += 1
            if self.consecutive_high_waste >= 3:
                return True, waste_ratio
        else:
            self.consecutive_high_waste = 0
        
        return False, waste_ratio
    
    def get_average_waste(self, window: int = 5) -> float:
        """Get average waste over recent window."""
        if not self.waste_history:
            return 0.0
        
        recent = self.waste_history[-window:]
        return sum(recent) / len(recent)
    
    def suggest_panel_size(self, remaining_area: float) -> Optional[PanelSize]:
        """
        Suggest panel size to reduce waste based on remaining area.
        """
        # Panel sizes and their areas
        panel_options = [
            (PanelSize.PANEL_6X8, 48),
            (PanelSize.PANEL_6X6, 36),
            (PanelSize.PANEL_4X6, 24),
            (PanelSize.PANEL_4X4, 16)
        ]
        
        # Find best fit for remaining area
        for panel_size, area in panel_options:
            if area <= remaining_area:
                waste = (remaining_area - area) / remaining_area
                if waste < self.max_waste_ratio:
                    return panel_size
        
        return None
    
    def reset(self):
        """Reset waste monitoring."""
        self.waste_history.clear()
        self.consecutive_high_waste = 0


class ImpossibleFitDetector:
    """
    Detects when remaining panels cannot possibly fit.
    """
    
    def __init__(self):
        """Initialize fit detector."""
        self.impossible_configurations: Set[str] = set()
    
    def check_fit_possibility(self, remaining_panels: List[PanelSize],
                             available_space: float,
                             max_dimension: Tuple[float, float]) -> bool:
        """
        Check if remaining panels can possibly fit.
        Returns True if impossible to fit.
        """
        # Quick check: total area
        total_panel_area = sum(p.area for p in remaining_panels)
        if total_panel_area > available_space * 1.1:  # Allow 10% margin
            return True
        
        # Check dimensional constraints
        max_width, max_height = max_dimension
        
        for panel in remaining_panels:
            # Check both orientations
            h_width, h_height = panel.get_dimensions("horizontal")
            v_width, v_height = panel.get_dimensions("vertical")
            
            # Panel can't fit in any orientation
            if ((h_width > max_width or h_height > max_height) and
                (v_width > max_width or v_height > max_height)):
                return True
        
        # Check if configuration is known to be impossible
        config_key = self._get_configuration_key(remaining_panels, max_dimension)
        if config_key in self.impossible_configurations:
            return True
        
        return False
    
    def mark_impossible(self, panels: List[PanelSize], 
                       max_dimension: Tuple[float, float]):
        """Mark a configuration as impossible."""
        config_key = self._get_configuration_key(panels, max_dimension)
        self.impossible_configurations.add(config_key)
    
    def _get_configuration_key(self, panels: List[PanelSize],
                              max_dimension: Tuple[float, float]) -> str:
        """Generate unique key for panel configuration."""
        panel_counts = {}
        for panel in panels:
            panel_counts[panel] = panel_counts.get(panel, 0) + 1
        
        sorted_panels = sorted(panel_counts.items(), key=lambda x: (x[0].value, x[1]))
        return f"{sorted_panels}_{max_dimension}"
    
    def estimate_minimum_area_needed(self, panels: List[PanelSize]) -> float:
        """Estimate minimum area needed for panels."""
        # Theoretical minimum (perfect packing)
        total_area = sum(p.area for p in panels)
        
        # Add realistic packing inefficiency (10-20%)
        return total_area * 1.15
    
    def reset(self):
        """Reset impossible configurations."""
        self.impossible_configurations.clear()


class TimeBasedTrigger:
    """
    Triggers backtracking based on time constraints.
    """
    
    def __init__(self, max_time_per_decision: float = 0.5,
                 total_time_limit: float = 5.0):
        """
        Initialize time-based trigger.
        
        Args:
            max_time_per_decision: Max time for single decision (seconds)
            total_time_limit: Total optimization time limit (seconds)
        """
        self.max_time_per_decision = max_time_per_decision
        self.total_time_limit = total_time_limit
        
        self.start_time = time.time()
        self.decision_start_time = time.time()
        self.time_per_decision: List[float] = []
    
    def start_decision(self):
        """Mark start of new decision."""
        self.decision_start_time = time.time()
    
    def check_decision_timeout(self) -> bool:
        """Check if current decision has timed out."""
        elapsed = time.time() - self.decision_start_time
        return elapsed > self.max_time_per_decision
    
    def check_total_timeout(self) -> bool:
        """Check if total time limit exceeded."""
        elapsed = time.time() - self.start_time
        return elapsed > self.total_time_limit
    
    def end_decision(self):
        """Mark end of current decision."""
        elapsed = time.time() - self.decision_start_time
        self.time_per_decision.append(elapsed)
    
    def get_remaining_time(self) -> float:
        """Get remaining time budget."""
        elapsed = time.time() - self.start_time
        return max(0, self.total_time_limit - elapsed)
    
    def get_average_decision_time(self) -> float:
        """Get average time per decision."""
        if not self.time_per_decision:
            return 0.0
        return sum(self.time_per_decision) / len(self.time_per_decision)
    
    def adjust_time_limits(self, remaining_decisions: int):
        """Adjust time limits based on progress."""
        remaining_time = self.get_remaining_time()
        
        if remaining_decisions > 0 and remaining_time > 0:
            # Distribute remaining time
            self.max_time_per_decision = min(
                0.5,  # Cap at original limit
                remaining_time / remaining_decisions * 1.5  # Allow some buffer
            )
    
    def reset(self):
        """Reset time tracking."""
        self.start_time = time.time()
        self.decision_start_time = time.time()
        self.time_per_decision.clear()


class BacktrackTriggerManager:
    """
    Manages all backtrack triggers and decides when to backtrack.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize trigger manager."""
        config = config or {}
        
        # Initialize triggers
        self.plateau_detector = CoveragePlateauDetector(
            plateau_threshold=config.get('plateau_threshold', 5),
            min_improvement=config.get('min_improvement', 0.01)
        )
        
        self.waste_monitor = WasteThresholdMonitor(
            max_waste_ratio=config.get('max_waste_ratio', 0.3),
            critical_waste_ratio=config.get('critical_waste_ratio', 0.4)
        )
        
        self.fit_detector = ImpossibleFitDetector()
        
        self.time_trigger = TimeBasedTrigger(
            max_time_per_decision=config.get('max_time_per_decision', 0.5),
            total_time_limit=config.get('total_time_limit', 5.0)
        )
        
        # Track trigger history
        self.trigger_history: List[Tuple[BacktrackReason, float]] = []
    
    def check_triggers(self, state: PackingState, 
                      remaining_panels: List[PanelSize],
                      room_area: float) -> Tuple[bool, Optional[BacktrackReason]]:
        """
        Check all triggers and return whether to backtrack.
        Returns (should_backtrack, reason).
        """
        # Don't trigger backtracking if no panels placed yet
        if not state.placements:
            return False, None
        
        # Check time limit first (highest priority)
        if self.time_trigger.check_total_timeout():
            self._record_trigger(BacktrackReason.TIME_LIMIT)
            return True, BacktrackReason.TIME_LIMIT
        
        # Check impossible fit
        available_space = room_area - sum(p.panel_size.area for p in state.placements)
        
        # Get max available dimensions (simplified)
        max_dim = (room_area ** 0.5, room_area ** 0.5)  # Rough estimate
        
        if self.fit_detector.check_fit_possibility(
            remaining_panels, available_space, max_dim
        ):
            self._record_trigger(BacktrackReason.IMPOSSIBLE_FIT)
            return True, BacktrackReason.IMPOSSIBLE_FIT
        
        # Check waste threshold - only if panels have been placed
        placed_area = sum(p.panel_size.area for p in state.placements)
        if placed_area > 0:  # Only check waste when panels are placed
            should_backtrack, waste_ratio = self.waste_monitor.check_waste(
                placed_area, room_area
            )
            
            if should_backtrack:
                self._record_trigger(BacktrackReason.WASTE_THRESHOLD)
                return True, BacktrackReason.WASTE_THRESHOLD
        
        # Check coverage plateau
        if self.plateau_detector.update(state.coverage):
            self._record_trigger(BacktrackReason.COVERAGE_PLATEAU)
            return True, BacktrackReason.COVERAGE_PLATEAU
        
        # Check if stuck (no valid positions)
        if not remaining_panels:
            return False, None
        
        return False, None
    
    def _record_trigger(self, reason: BacktrackReason):
        """Record trigger in history."""
        timestamp = time.time()
        self.trigger_history.append((reason, timestamp))
    
    def get_trigger_statistics(self) -> Dict[str, int]:
        """Get statistics on trigger frequencies."""
        stats = {}
        for reason in BacktrackReason:
            count = sum(1 for r, _ in self.trigger_history if r == reason)
            stats[reason.value] = count
        return stats
    
    def suggest_action(self, reason: BacktrackReason, 
                       current_depth: int) -> str:
        """
        Suggest action based on trigger reason.
        """
        if reason == BacktrackReason.COVERAGE_PLATEAU:
            return f"Backtrack {min(3, current_depth)} steps and try different panel order"
        
        elif reason == BacktrackReason.WASTE_THRESHOLD:
            return "Backtrack to last high-coverage state and use smaller panels"
        
        elif reason == BacktrackReason.IMPOSSIBLE_FIT:
            return "Backtrack to last branch point with alternative options"
        
        elif reason == BacktrackReason.TIME_LIMIT:
            return "Return best solution found so far"
        
        else:
            return "Backtrack 1 step and try alternative"
    
    def reset(self):
        """Reset all triggers."""
        self.plateau_detector.reset()
        self.waste_monitor.reset()
        self.fit_detector.reset()
        self.time_trigger.reset()
        self.trigger_history.clear()


# Step 2.2.3: Backtrack Strategy
# ===============================

class BacktrackStrategy:
    """
    Sophisticated backtracking strategy with depth limits and intelligent selection.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize backtrack strategy."""
        config = config or {}
        
        self.max_backtrack_depth = config.get('max_backtrack_depth', 10)
        self.min_backtrack_depth = config.get('min_backtrack_depth', 1)
        self.adaptive_depth = config.get('adaptive_depth', True)
        self.alternative_path_threshold = config.get('alternative_path_threshold', 0.85)
        
        # Track backtrack effectiveness
        self.backtrack_history: List[Dict[str, Any]] = []
        self.successful_backtracks = 0
        self.failed_backtracks = 0
    
    def determine_backtrack_depth(self, reason: BacktrackReason, 
                                 current_state: PackingState,
                                 history_depth: int) -> int:
        """
        Determine optimal backtrack depth based on reason and state.
        """
        # Base depth based on reason
        if reason == BacktrackReason.COVERAGE_PLATEAU:
            base_depth = min(5, history_depth)
        elif reason == BacktrackReason.WASTE_THRESHOLD:
            base_depth = min(7, history_depth)
        elif reason == BacktrackReason.IMPOSSIBLE_FIT:
            base_depth = min(3, history_depth)
        elif reason == BacktrackReason.NO_VALID_POSITIONS:
            base_depth = min(4, history_depth)
        else:
            base_depth = self.min_backtrack_depth
        
        # Adapt based on history if enabled
        if self.adaptive_depth:
            base_depth = self._adapt_depth(base_depth, reason)
        
        # Apply limits
        return max(self.min_backtrack_depth, 
                  min(base_depth, self.max_backtrack_depth, history_depth))
    
    def _adapt_depth(self, base_depth: int, reason: BacktrackReason) -> int:
        """
        Adapt depth based on historical effectiveness.
        """
        # Get recent backtracks with same reason
        recent = [h for h in self.backtrack_history[-10:] 
                 if h['reason'] == reason]
        
        if not recent:
            return base_depth
        
        # Calculate average effectiveness
        avg_improvement = sum(h['improvement'] for h in recent) / len(recent)
        
        # Adjust depth based on effectiveness
        if avg_improvement > 0.05:  # Good improvement
            return base_depth
        elif avg_improvement > 0.02:  # Moderate improvement
            return base_depth + 1
        else:  # Poor improvement
            return base_depth + 2
    
    def select_backtrack_point(self, snapshots: List[StateSnapshot],
                              current_coverage: float,
                              reason: BacktrackReason) -> Optional[StateSnapshot]:
        """
        Intelligently select best backtrack point from available snapshots.
        """
        if not snapshots:
            return None
        
        # Filter viable snapshots
        viable = self._filter_viable_snapshots(snapshots, current_coverage, reason)
        
        if not viable:
            return snapshots[-1]  # Fallback to most recent
        
        # Score and select best snapshot
        scored = [(self._score_snapshot(s, current_coverage, reason), s) 
                 for s in viable]
        scored.sort(key=lambda x: x[0], reverse=True)
        
        return scored[0][1]
    
    def _filter_viable_snapshots(self, snapshots: List[StateSnapshot],
                                current_coverage: float,
                                reason: BacktrackReason) -> List[StateSnapshot]:
        """
        Filter snapshots to only viable backtrack points.
        """
        viable = []
        
        for snapshot in snapshots:
            # Skip if coverage too low
            if snapshot.coverage < current_coverage * 0.8:
                continue
            
            # Skip if too recent for plateau
            if (reason == BacktrackReason.COVERAGE_PLATEAU and 
                snapshot.decision_point > len(snapshots) - 3):
                continue
            
            viable.append(snapshot)
        
        return viable
    
    def _score_snapshot(self, snapshot: StateSnapshot,
                       current_coverage: float,
                       reason: BacktrackReason) -> float:
        """
        Score a snapshot for backtracking potential.
        """
        score = 0.0
        
        # Coverage score (prefer higher coverage)
        score += snapshot.coverage * 10
        
        # Recency score (prefer moderate recency)
        age_factor = 1.0 / (1 + snapshot.decision_point * 0.1)
        score += age_factor * 5
        
        # Reason-specific scoring
        if reason == BacktrackReason.WASTE_THRESHOLD:
            # Prefer points with low waste
            waste = snapshot.metadata.get('waste_ratio', 0)
            score += (1 - waste) * 8
        elif reason == BacktrackReason.COVERAGE_PLATEAU:
            # Prefer points with different panel arrangements
            score += snapshot.metadata.get('arrangement_diversity', 0) * 6
        
        return score
    
    def find_alternative_paths(self, current_snapshot: StateSnapshot,
                              all_snapshots: List[StateSnapshot],
                              num_alternatives: int = 3) -> List[StateSnapshot]:
        """
        Find alternative paths from different branch points.
        """
        alternatives = []
        
        # Find snapshots with similar but different states
        for snapshot in all_snapshots:
            if snapshot == current_snapshot:
                continue
            
            # Check if coverage is promising
            if snapshot.coverage < self.alternative_path_threshold:
                continue
            
            # Check if sufficiently different
            if self._path_similarity(snapshot, current_snapshot) < 0.7:
                alternatives.append(snapshot)
            
            if len(alternatives) >= num_alternatives:
                break
        
        return alternatives
    
    def _path_similarity(self, snap1: StateSnapshot, snap2: StateSnapshot) -> float:
        """
        Calculate similarity between two paths (0 to 1).
        """
        # Compare placement patterns
        placements1 = snap1.state.placements
        placements2 = snap2.state.placements
        
        if not placements1 or not placements2:
            return 0.0
        
        # Calculate Jaccard similarity
        common = len(placements1.intersection(placements2))
        total = len(placements1.union(placements2))
        
        return common / total if total > 0 else 0.0
    
    def record_backtrack_result(self, reason: BacktrackReason,
                               depth: int,
                               improvement: float):
        """
        Record backtrack result for learning.
        """
        self.backtrack_history.append({
            'reason': reason,
            'depth': depth,
            'improvement': improvement,
            'timestamp': time.time()
        })
        
        if improvement > 0:
            self.successful_backtracks += 1
        else:
            self.failed_backtracks += 1
        
        # Keep history bounded
        if len(self.backtrack_history) > 100:
            self.backtrack_history = self.backtrack_history[-100:]
    
    def get_effectiveness_stats(self) -> Dict[str, Any]:
        """
        Get backtracking effectiveness statistics.
        """
        total = self.successful_backtracks + self.failed_backtracks
        
        return {
            'total_backtracks': total,
            'successful': self.successful_backtracks,
            'failed': self.failed_backtracks,
            'success_rate': self.successful_backtracks / total if total > 0 else 0,
            'avg_depth': sum(h['depth'] for h in self.backtrack_history) / len(self.backtrack_history)
                        if self.backtrack_history else 0
        }
#!/usr/bin/env python3
"""
dp_state_optimization.py - Advanced DP State Space Optimization
=============================================================
Production-ready state space optimization including state aggregation,
equivalent state merging, approximation strategies, and adaptive refinement.
"""

import time
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, FrozenSet, Union
from collections import defaultdict, Counter
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np
from functools import lru_cache

from models import Room, PanelSize
from dp_state import DPState, SkylineProfile
from advanced_packing import PanelPlacement
from dp_grid import GridResolution


class AggregationStrategy(Enum):
    """Types of state aggregation strategies."""
    COVERAGE_BASED = "coverage"
    SPATIAL_BASED = "spatial"
    PANEL_COUNT_BASED = "panel_count"
    HYBRID = "hybrid"


class ApproximationLevel(Enum):
    """Levels of approximation for state representation."""
    EXACT = "exact"
    HIGH_PRECISION = "high"
    MEDIUM_PRECISION = "medium"
    LOW_PRECISION = "low"
    COARSE = "coarse"


@dataclass(frozen=True)
class StateCluster:
    """
    Represents a cluster of aggregated states for efficient processing.
    Contains representative state and metadata about the cluster.
    """
    cluster_id: str
    representative_state: DPState
    member_count: int
    coverage_range: Tuple[float, float]  # Min, max coverage in cluster
    aggregation_level: ApproximationLevel
    cluster_quality: float  # Quality metric for adaptive refinement
    
    def __hash__(self) -> int:
        return hash((self.cluster_id, self.aggregation_level))


class StateAggregator:
    """
    Aggregates similar states into clusters to reduce DP state space size.
    Uses multiple clustering strategies for optimal grouping.
    """
    
    def __init__(self, room: Room, panel_sizes: List[PanelSize], 
                 strategy: AggregationStrategy = AggregationStrategy.HYBRID):
        self.room = room
        self.panel_sizes = panel_sizes
        self.strategy = strategy
        self.room_area = room.width * room.height
        
        # Clustering data structures
        self.clusters: Dict[str, StateCluster] = {}
        self.state_to_cluster: Dict[str, str] = {}
        self.aggregation_count = 0
        
        # Clustering parameters
        self.coverage_tolerance = 0.05  # 5% coverage tolerance for clustering
        self.spatial_tolerance = 0.1    # Spatial clustering tolerance
        self.max_cluster_size = 20      # Maximum states per cluster
        
    def aggregate_states(self, states: List[DPState], 
                        approximation_level: ApproximationLevel = ApproximationLevel.MEDIUM_PRECISION) -> List[StateCluster]:
        """
        Aggregate list of states into clusters based on similarity.
        Returns list of state clusters for efficient processing.
        """
        if not states:
            return []
        
        clusters = []
        
        if self.strategy == AggregationStrategy.COVERAGE_BASED:
            clusters = self._aggregate_by_coverage(states, approximation_level)
        elif self.strategy == AggregationStrategy.SPATIAL_BASED:
            clusters = self._aggregate_by_spatial_features(states, approximation_level)
        elif self.strategy == AggregationStrategy.PANEL_COUNT_BASED:
            clusters = self._aggregate_by_panel_count(states, approximation_level)
        else:  # HYBRID
            clusters = self._hybrid_aggregation(states, approximation_level)
        
        # Update internal tracking
        for cluster in clusters:
            self.clusters[cluster.cluster_id] = cluster
            self.aggregation_count += cluster.member_count
        
        return clusters
    
    def _aggregate_by_coverage(self, states: List[DPState], 
                              level: ApproximationLevel) -> List[StateCluster]:
        """Aggregate states based on coverage similarity."""
        # Group states by coverage buckets
        coverage_buckets = defaultdict(list)
        bucket_size = self._get_coverage_bucket_size(level)
        
        for state in states:
            bucket = int(state.coverage / bucket_size)
            coverage_buckets[bucket].append(state)
        
        clusters = []
        for bucket, bucket_states in coverage_buckets.items():
            if len(bucket_states) <= 1:
                continue  # Don't cluster single states
            
            # Create representative state (highest coverage in bucket)
            representative = max(bucket_states, key=lambda s: s.coverage)
            
            coverage_values = [s.coverage for s in bucket_states]
            cluster = StateCluster(
                cluster_id=f"cov_bucket_{bucket}",
                representative_state=representative,
                member_count=len(bucket_states),
                coverage_range=(min(coverage_values), max(coverage_values)),
                aggregation_level=level,
                cluster_quality=self._calculate_cluster_quality(bucket_states)
            )
            clusters.append(cluster)
        
        return clusters
    
    def _aggregate_by_spatial_features(self, states: List[DPState], 
                                     level: ApproximationLevel) -> List[StateCluster]:
        """Aggregate states based on spatial layout similarity."""
        if not any(s.skyline for s in states):
            # Fallback to coverage-based if no spatial info
            return self._aggregate_by_coverage(states, level)
        
        spatial_buckets = defaultdict(list)
        
        for state in states:
            if state.skyline:
                spatial_signature = self._create_spatial_signature(state.skyline, level)
                spatial_buckets[spatial_signature].append(state)
        
        clusters = []
        for signature, bucket_states in spatial_buckets.items():
            if len(bucket_states) <= 1:
                continue
            
            # Representative: state with best coverage in spatial group
            representative = max(bucket_states, key=lambda s: s.coverage)
            
            coverage_values = [s.coverage for s in bucket_states]
            cluster = StateCluster(
                cluster_id=f"spatial_{signature}",
                representative_state=representative,
                member_count=len(bucket_states),
                coverage_range=(min(coverage_values), max(coverage_values)),
                aggregation_level=level,
                cluster_quality=self._calculate_spatial_cluster_quality(bucket_states)
            )
            clusters.append(cluster)
        
        return clusters
    
    def _aggregate_by_panel_count(self, states: List[DPState], 
                                 level: ApproximationLevel) -> List[StateCluster]:
        """Aggregate states based on remaining panel counts."""
        panel_buckets = defaultdict(list)
        
        for state in states:
            # Create signature based on remaining panel counts
            panel_signature = self._create_panel_count_signature(state, level)
            panel_buckets[panel_signature].append(state)
        
        clusters = []
        for signature, bucket_states in panel_buckets.items():
            if len(bucket_states) <= 1:
                continue
            
            representative = max(bucket_states, key=lambda s: s.coverage)
            
            coverage_values = [s.coverage for s in bucket_states]
            cluster = StateCluster(
                cluster_id=f"panels_{signature}",
                representative_state=representative,
                member_count=len(bucket_states),
                coverage_range=(min(coverage_values), max(coverage_values)),
                aggregation_level=level,
                cluster_quality=self._calculate_panel_cluster_quality(bucket_states)
            )
            clusters.append(cluster)
        
        return clusters
    
    def _hybrid_aggregation(self, states: List[DPState], 
                           level: ApproximationLevel) -> List[StateCluster]:
        """Multi-criteria aggregation combining coverage, spatial, and panel features."""
        hybrid_buckets = defaultdict(list)
        
        for state in states:
            # Combine multiple features for hybrid signature
            coverage_bucket = int(state.coverage / self._get_coverage_bucket_size(level))
            spatial_sig = self._create_spatial_signature(state.skyline, level) if state.skyline else "no_spatial"
            panel_sig = self._create_panel_count_signature(state, level)
            
            hybrid_signature = f"{coverage_bucket}|{spatial_sig}|{panel_sig}"
            hybrid_buckets[hybrid_signature].append(state)
        
        clusters = []
        for signature, bucket_states in hybrid_buckets.items():
            if len(bucket_states) <= 1:
                continue
            
            representative = max(bucket_states, key=lambda s: s.coverage)
            
            coverage_values = [s.coverage for s in bucket_states]
            cluster = StateCluster(
                cluster_id=f"hybrid_{hash(signature) % 100000}",
                representative_state=representative,
                member_count=len(bucket_states),
                coverage_range=(min(coverage_values), max(coverage_values)),
                aggregation_level=level,
                cluster_quality=self._calculate_hybrid_cluster_quality(bucket_states)
            )
            clusters.append(cluster)
        
        return clusters
    
    def _get_coverage_bucket_size(self, level: ApproximationLevel) -> float:
        """Get coverage bucket size based on approximation level."""
        bucket_sizes = {
            ApproximationLevel.EXACT: 0.001,
            ApproximationLevel.HIGH_PRECISION: 0.01,
            ApproximationLevel.MEDIUM_PRECISION: 0.05,
            ApproximationLevel.LOW_PRECISION: 0.1,
            ApproximationLevel.COARSE: 0.2
        }
        return bucket_sizes.get(level, 0.05)
    
    def _create_spatial_signature(self, skyline: Optional[SkylineProfile], 
                                 level: ApproximationLevel) -> str:
        """Create spatial signature for skyline clustering."""
        if not skyline or not skyline.y_coords:
            return "empty"
        
        # Discretization level based on approximation
        discretization_levels = {
            ApproximationLevel.EXACT: 100,
            ApproximationLevel.HIGH_PRECISION: 50,
            ApproximationLevel.MEDIUM_PRECISION: 20,
            ApproximationLevel.LOW_PRECISION: 10,
            ApproximationLevel.COARSE: 5
        }
        
        buckets = discretization_levels.get(level, 20)
        max_height = skyline.room_height
        
        signature_parts = []
        for y in skyline.y_coords[:8]:  # Limit to first 8 positions
            bucket = min(buckets - 1, int(y / max_height * buckets))
            signature_parts.append(str(bucket))
        
        return ''.join(signature_parts)
    
    def _create_panel_count_signature(self, state: DPState, 
                                     level: ApproximationLevel) -> str:
        """Create signature based on remaining panel counts."""
        # Count remaining panels by type
        panel_counts = Counter()
        for panel in state.remaining_panels:
            panel_counts[panel.name] += 1
        
        # Create signature with appropriate granularity
        if level in [ApproximationLevel.EXACT, ApproximationLevel.HIGH_PRECISION]:
            # Exact counts
            signature_parts = [f"{k}:{v}" for k, v in sorted(panel_counts.items())]
        else:
            # Bucketed counts for coarser approximation
            bucket_size = 2 if level == ApproximationLevel.MEDIUM_PRECISION else 5
            signature_parts = []
            for k, v in sorted(panel_counts.items()):
                bucketed_count = (v // bucket_size) * bucket_size
                signature_parts.append(f"{k}:{bucketed_count}")
        
        return '|'.join(signature_parts)
    
    def _calculate_cluster_quality(self, states: List[DPState]) -> float:
        """Calculate quality metric for coverage-based cluster."""
        if len(states) <= 1:
            return 1.0
        
        coverages = [s.coverage for s in states]
        coverage_variance = np.var(coverages)
        
        # Lower variance = higher quality
        return max(0.0, 1.0 - coverage_variance * 10)
    
    def _calculate_spatial_cluster_quality(self, states: List[DPState]) -> float:
        """Calculate quality metric for spatial cluster."""
        if len(states) <= 1:
            return 1.0
        
        # Use coverage variance as spatial quality proxy
        coverages = [s.coverage for s in states]
        return max(0.0, 1.0 - np.var(coverages) * 5)
    
    def _calculate_panel_cluster_quality(self, states: List[DPState]) -> float:
        """Calculate quality metric for panel count cluster."""
        if len(states) <= 1:
            return 1.0
        
        # Quality based on consistency of remaining panels
        panel_counts = [len(s.remaining_panels) for s in states]
        count_variance = np.var(panel_counts)
        
        return max(0.0, 1.0 - count_variance * 0.1)
    
    def _calculate_hybrid_cluster_quality(self, states: List[DPState]) -> float:
        """Calculate quality metric for hybrid cluster."""
        if len(states) <= 1:
            return 1.0
        
        # Combine multiple quality factors
        coverage_quality = self._calculate_cluster_quality(states)
        spatial_quality = self._calculate_spatial_cluster_quality(states)
        panel_quality = self._calculate_panel_cluster_quality(states)
        
        # Weighted average
        return (coverage_quality * 0.5 + spatial_quality * 0.3 + panel_quality * 0.2)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregation statistics."""
        if not self.clusters:
            return {'total_clusters': 0, 'aggregated_states': 0}
        
        cluster_sizes = [c.member_count for c in self.clusters.values()]
        cluster_qualities = [c.cluster_quality for c in self.clusters.values()]
        
        return {
            'total_clusters': len(self.clusters),
            'aggregated_states': self.aggregation_count,
            'avg_cluster_size': np.mean(cluster_sizes) if cluster_sizes else 0,
            'max_cluster_size': max(cluster_sizes) if cluster_sizes else 0,
            'avg_cluster_quality': np.mean(cluster_qualities) if cluster_qualities else 0,
            'strategy_used': self.strategy.value
        }


class EquivalentStateMerger:
    """
    Merges equivalent states that lead to identical outcomes.
    Uses advanced state comparison to identify truly equivalent states.
    """
    
    def __init__(self, room: Room, panel_sizes: List[PanelSize]):
        self.room = room
        self.panel_sizes = panel_sizes
        self.equivalence_classes: Dict[str, List[DPState]] = defaultdict(list)
        self.merged_count = 0
        self.comparison_count = 0
        
    def merge_equivalent_states(self, states: List[DPState]) -> List[DPState]:
        """
        Merge equivalent states and return list of unique representatives.
        Maintains best state from each equivalence class.
        """
        if len(states) <= 1:
            return states
        
        # Group states by equivalence signature
        equivalence_groups = defaultdict(list)
        
        for state in states:
            signature = self._create_equivalence_signature(state)
            equivalence_groups[signature].append(state)
        
        merged_states = []
        
        for signature, group_states in equivalence_groups.items():
            if len(group_states) == 1:
                merged_states.append(group_states[0])
            else:
                # Detailed equivalence checking within group
                representative = self._find_best_equivalent_state(group_states)
                merged_states.append(representative)
                self.merged_count += len(group_states) - 1
        
        return merged_states
    
    def _create_equivalence_signature(self, state: DPState) -> str:
        """Create signature for equivalence class identification."""
        components = []
        
        # Coverage component (fine-grained)
        coverage_bucket = int(state.coverage * 1000)  # 0.1% precision
        components.append(f"cov:{coverage_bucket}")
        
        # Remaining panels signature
        panel_signature = self._create_panel_equivalence_signature(state)
        components.append(f"panels:{panel_signature}")
        
        # Spatial signature (if available)
        if state.skyline:
            spatial_sig = self._create_spatial_equivalence_signature(state.skyline)
            components.append(f"spatial:{spatial_sig}")
        
        # Placement count
        components.append(f"placed:{len(state.placements)}")
        
        return '|'.join(components)
    
    def _create_panel_equivalence_signature(self, state: DPState) -> str:
        """Create panel signature for equivalence checking."""
        # Group panels by dimensions to identify equivalent types
        panel_groups = defaultdict(int)
        
        for panel in state.remaining_panels:
            # Key by dimensions (width, length) to group equivalent panels
            key = f"{panel.width}x{panel.length}"
            panel_groups[key] += 1
        
        # Sort by key for consistent signature
        signature_parts = [f"{k}:{v}" for k, v in sorted(panel_groups.items())]
        return '|'.join(signature_parts)
    
    def _create_spatial_equivalence_signature(self, skyline: SkylineProfile) -> str:
        """Create spatial signature for equivalence checking."""
        if not skyline.y_coords:
            return "empty"
        
        # High-precision discretization for equivalence
        max_height = skyline.room_height
        buckets = []
        
        for y in skyline.y_coords:
            # Use high precision for equivalence (50 buckets)
            bucket = min(49, int(y / max_height * 50))
            buckets.append(str(bucket))
        
        return ''.join(buckets[:10])  # Limit to first 10 positions
    
    def _find_best_equivalent_state(self, states: List[DPState]) -> DPState:
        """Find best representative from group of equivalent states."""
        # Detailed equivalence verification
        truly_equivalent = []
        
        for state in states:
            if not truly_equivalent or self._are_truly_equivalent(state, truly_equivalent[0]):
                truly_equivalent.append(state)
                self.comparison_count += 1
        
        # Return state with highest coverage among truly equivalent states
        return max(truly_equivalent, key=lambda s: s.coverage)
    
    def _are_truly_equivalent(self, state1: DPState, state2: DPState) -> bool:
        """Detailed equivalence check between two states."""
        # Coverage must be very close
        if abs(state1.coverage - state2.coverage) > 0.001:
            return False
        
        # Same remaining panel distribution
        if Counter(p.name for p in state1.remaining_panels) != \
           Counter(p.name for p in state2.remaining_panels):
            return False
        
        # Similar spatial structure (if both have skylines)
        if state1.skyline and state2.skyline:
            if not self._skylines_equivalent(state1.skyline, state2.skyline):
                return False
        
        return True
    
    def _skylines_equivalent(self, skyline1: SkylineProfile, skyline2: SkylineProfile) -> bool:
        """Check if two skylines are equivalent."""
        if len(skyline1.y_coords) != len(skyline2.y_coords):
            return False
        
        # Check if height differences are within tolerance
        tolerance = 0.1
        for y1, y2 in zip(skyline1.y_coords, skyline2.y_coords):
            if abs(y1 - y2) > tolerance:
                return False
        
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get merging statistics."""
        return {
            'equivalence_classes': len(self.equivalence_classes),
            'states_merged': self.merged_count,
            'comparisons_made': self.comparison_count,
            'merge_rate': self.merged_count / max(1, self.comparison_count + self.merged_count)
        }


class AdaptiveRefinementManager:
    """
    Manages adaptive refinement of approximation strategies based on solution quality
    and computational efficiency. Dynamically adjusts precision levels.
    """
    
    def __init__(self, initial_level: ApproximationLevel = ApproximationLevel.MEDIUM_PRECISION,
                 target_coverage: float = 0.95):
        self.current_level = initial_level
        self.target_coverage = target_coverage
        self.refinement_history: List[ApproximationLevel] = [initial_level]
        
        # Performance tracking
        self.level_performance: Dict[ApproximationLevel, List[float]] = defaultdict(list)
        self.refinement_count = 0
        self.last_refinement_time = time.time()
        
        # Refinement thresholds
        self.quality_improvement_threshold = 0.02  # 2% coverage improvement
        self.stagnation_threshold = 100  # Iterations without improvement
        self.refinement_cooldown = 5.0  # Seconds between refinements
        
    def should_refine(self, current_best: float, iterations_since_improvement: int,
                     current_time: float) -> bool:
        """
        Determine if approximation level should be refined (increased precision).
        """
        # Cooldown period check
        if current_time - self.last_refinement_time < self.refinement_cooldown:
            return False
        
        # Already at highest precision
        if self.current_level == ApproximationLevel.EXACT:
            return False
        
        # Stagnation detected
        if iterations_since_improvement >= self.stagnation_threshold:
            return True
        
        # Close to target but need more precision
        if (current_best >= self.target_coverage * 0.9 and 
            self.current_level not in [ApproximationLevel.EXACT, ApproximationLevel.HIGH_PRECISION]):
            return True
        
        # Performance-based refinement
        if self._should_refine_based_on_performance(current_best):
            return True
        
        return False
    
    def should_coarsen(self, processing_time: float, memory_usage: float,
                      solution_quality: float) -> bool:
        """
        Determine if approximation should be coarsened (decreased precision) for efficiency.
        """
        # Never coarsen if we're close to target
        if solution_quality >= self.target_coverage * 0.95:
            return False
        
        # Already at lowest precision
        if self.current_level == ApproximationLevel.COARSE:
            return False
        
        # Resource pressure indicators
        high_processing_time = processing_time > 30.0  # 30 seconds
        high_memory_usage = memory_usage > 0.8  # 80% memory usage
        
        # Coarsen if under resource pressure and solution quality is still reasonable
        if (high_processing_time or high_memory_usage) and solution_quality > 0.3:
            return True
        
        return False
    
    def refine_approximation(self) -> ApproximationLevel:
        """
        Increase approximation precision level.
        Returns new approximation level.
        """
        level_order = [
            ApproximationLevel.COARSE,
            ApproximationLevel.LOW_PRECISION,
            ApproximationLevel.MEDIUM_PRECISION,
            ApproximationLevel.HIGH_PRECISION,
            ApproximationLevel.EXACT
        ]
        
        current_index = level_order.index(self.current_level)
        if current_index < len(level_order) - 1:
            self.current_level = level_order[current_index + 1]
            self.refinement_count += 1
            self.last_refinement_time = time.time()
            self.refinement_history.append(self.current_level)
        
        return self.current_level
    
    def coarsen_approximation(self) -> ApproximationLevel:
        """
        Decrease approximation precision level for efficiency.
        Returns new approximation level.
        """
        level_order = [
            ApproximationLevel.EXACT,
            ApproximationLevel.HIGH_PRECISION,
            ApproximationLevel.MEDIUM_PRECISION,
            ApproximationLevel.LOW_PRECISION,
            ApproximationLevel.COARSE
        ]
        
        current_index = level_order.index(self.current_level)
        if current_index < len(level_order) - 1:
            self.current_level = level_order[current_index + 1]
            self.last_refinement_time = time.time()
            self.refinement_history.append(self.current_level)
        
        return self.current_level
    
    def record_performance(self, coverage_achieved: float):
        """Record performance for current approximation level."""
        self.level_performance[self.current_level].append(coverage_achieved)
    
    def _should_refine_based_on_performance(self, current_best: float) -> bool:
        """Check if refinement is warranted based on performance history."""
        current_performance = self.level_performance.get(self.current_level, [])
        
        if len(current_performance) < 3:  # Need sufficient data
            return False
        
        # Check if current level is underperforming
        recent_avg = np.mean(current_performance[-3:])
        
        # Refine if recent performance suggests we're stuck
        return (current_best > 0.5 and recent_avg < current_best * 0.95)
    
    def get_current_level(self) -> ApproximationLevel:
        """Get current approximation level."""
        return self.current_level
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get adaptive refinement statistics."""
        level_avgs = {}
        for level, performances in self.level_performance.items():
            if performances:
                level_avgs[level.value] = np.mean(performances)
        
        return {
            'current_level': self.current_level.value,
            'refinement_count': self.refinement_count,
            'refinement_history': [level.value for level in self.refinement_history],
            'level_performance_averages': level_avgs,
            'time_since_last_refinement': time.time() - self.last_refinement_time
        }


class ComprehensiveStateOptimizer:
    """
    Integrates all state space optimization strategies for maximum DP efficiency.
    Coordinates aggregation, merging, approximation, and adaptive refinement.
    """
    
    def __init__(self, room: Room, panel_sizes: List[PanelSize],
                 target_coverage: float = 0.95,
                 initial_approximation: ApproximationLevel = ApproximationLevel.MEDIUM_PRECISION):
        
        self.room = room
        self.panel_sizes = panel_sizes
        self.target_coverage = target_coverage
        
        # Initialize optimization components
        self.aggregator = StateAggregator(room, panel_sizes, AggregationStrategy.HYBRID)
        self.merger = EquivalentStateMerger(room, panel_sizes)
        self.refinement_manager = AdaptiveRefinementManager(initial_approximation, target_coverage)
        
        # Statistics tracking
        self.optimization_count = 0
        self.total_states_processed = 0
        self.total_states_after_optimization = 0
        self.start_time = time.time()
        
    def optimize_state_space(self, states: List[DPState], current_best: float,
                           iterations_since_improvement: int = 0,
                           processing_time: float = 0.0,
                           memory_usage: float = 0.0) -> Tuple[List[DPState], bool]:
        """
        Comprehensive state space optimization using all strategies.
        Returns optimized states and whether refinement occurred.
        """
        if not states:
            return states, False
        
        self.optimization_count += 1
        self.total_states_processed += len(states)
        original_count = len(states)
        
        current_time = time.time()
        current_level = self.refinement_manager.get_current_level()
        
        # Step 1: Equivalent state merging
        merged_states = self.merger.merge_equivalent_states(states)
        
        # Step 2: Adaptive refinement decision
        refinement_occurred = False
        
        # Check for refinement
        if self.refinement_manager.should_refine(current_best, iterations_since_improvement, current_time):
            new_level = self.refinement_manager.refine_approximation()
            refinement_occurred = True
            print(f"Refined approximation to {new_level.value}")
        
        # Check for coarsening
        elif self.refinement_manager.should_coarsen(processing_time, memory_usage, current_best):
            new_level = self.refinement_manager.coarsen_approximation()
            refinement_occurred = True
            print(f"Coarsened approximation to {new_level.value}")
        
        # Step 3: State aggregation with current approximation level
        current_level = self.refinement_manager.get_current_level()
        clusters = self.aggregator.aggregate_states(merged_states, current_level)
        
        # Step 4: Extract representative states from clusters
        optimized_states = []
        
        # Add clustered representatives
        for cluster in clusters:
            optimized_states.append(cluster.representative_state)
        
        # Add unclustered states (single states that weren't clustered)
        clustered_state_hashes = set()
        for cluster in clusters:
            clustered_state_hashes.add(cluster.representative_state.canonical_hash)
        
        for state in merged_states:
            if state.canonical_hash not in clustered_state_hashes:
                optimized_states.append(state)
        
        # Record performance
        self.refinement_manager.record_performance(current_best)
        
        # Update statistics
        self.total_states_after_optimization += len(optimized_states)
        
        return optimized_states, refinement_occurred
    
    def get_optimization_ratio(self) -> float:
        """Get overall optimization ratio (reduction in state count)."""
        if self.total_states_processed == 0:
            return 0.0
        
        return 1.0 - (self.total_states_after_optimization / self.total_states_processed)
    
    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Get statistics from all optimization components."""
        stats = {
            'optimization_calls': self.optimization_count,
            'total_states_processed': self.total_states_processed,
            'total_states_after_optimization': self.total_states_after_optimization,
            'optimization_ratio': self.get_optimization_ratio(),
            'processing_time': time.time() - self.start_time
        }
        
        # Add component statistics
        stats['aggregation'] = self.aggregator.get_statistics()
        stats['merging'] = self.merger.get_statistics()
        stats['adaptive_refinement'] = self.refinement_manager.get_statistics()
        
        return stats


def create_state_optimizer(room: Room, 
                          panel_sizes: List[PanelSize],
                          target_coverage: float = 0.95,
                          initial_approximation: ApproximationLevel = ApproximationLevel.MEDIUM_PRECISION) -> ComprehensiveStateOptimizer:
    """
    Factory function to create comprehensive state space optimizer.
    Returns configured optimizer with all optimization strategies.
    """
    return ComprehensiveStateOptimizer(
        room=room,
        panel_sizes=panel_sizes,
        target_coverage=target_coverage,
        initial_approximation=initial_approximation
    )
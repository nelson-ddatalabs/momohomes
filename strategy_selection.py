#!/usr/bin/env python3
"""
strategy_selection.py - Strategy Selection System
=================================================
Production-ready room classifier, strategy mapping, fallback chain,
and performance tracking for optimal algorithm selection.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from collections import defaultdict
from enum import Enum
import math
import time
import json

from models import Room, PanelSize
from advanced_packing import PackingState


class RoomComplexity(Enum):
    """Room complexity levels."""
    SIMPLE = "simple"  # Rectangular, no obstacles
    MODERATE = "moderate"  # Some irregularities
    COMPLEX = "complex"  # Highly irregular, many obstacles
    EXTREME = "extreme"  # Very challenging geometry


class RoomSize(Enum):
    """Room size categories."""
    TINY = "tiny"  # < 100 sq ft
    SMALL = "small"  # 100-200 sq ft
    MEDIUM = "medium"  # 200-400 sq ft
    LARGE = "large"  # 400-800 sq ft
    HUGE = "huge"  # > 800 sq ft


class StrategyType(Enum):
    """Available optimization strategies."""
    SIMPLE_BLF = "simple_blf"
    ENHANCED_BLF = "enhanced_blf"
    DYNAMIC_PROGRAMMING = "dynamic_programming"
    BRANCH_AND_BOUND = "branch_and_bound"
    HYBRID = "hybrid"
    GREEDY = "greedy"
    EXHAUSTIVE = "exhaustive"


@dataclass
class RoomClassification:
    """Complete room classification."""
    size: RoomSize
    complexity: RoomComplexity
    aspect_ratio: float
    area: float
    perimeter: float
    num_vertices: int
    has_obstacles: bool
    is_convex: bool
    regularity_score: float  # 0-1, higher is more regular
    features: Dict[str, Any] = field(default_factory=dict)
    
    def to_key(self) -> str:
        """Generate unique key for this classification."""
        return f"{self.size.value}_{self.complexity.value}_{int(self.aspect_ratio*10)}"


@dataclass
class StrategyMapping:
    """Maps room characteristics to strategy."""
    classification: RoomClassification
    primary_strategy: StrategyType
    secondary_strategies: List[StrategyType]
    confidence: float  # 0-1, confidence in this mapping
    reasoning: str
    
    def get_strategies(self) -> List[StrategyType]:
        """Get ordered list of strategies to try."""
        return [self.primary_strategy] + self.secondary_strategies


@dataclass
class StrategyPerformance:
    """Track performance of a strategy."""
    strategy: StrategyType
    room_key: str
    coverage_achieved: float
    time_taken: float
    iterations: int
    success: bool
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class RoomClassifier:
    """Classifies rooms based on geometric properties."""
    
    def __init__(self):
        self.classification_cache = {}
        self.feature_extractors = [
            self._extract_size_features,
            self._extract_shape_features,
            self._extract_complexity_features
        ]
    
    def classify(self, room: Room) -> RoomClassification:
        """Classify a room based on its properties."""
        # Check cache
        room_hash = self._hash_room(room)
        if room_hash in self.classification_cache:
            return self.classification_cache[room_hash]
        
        # Extract features
        features = {}
        for extractor in self.feature_extractors:
            features.update(extractor(room))
        
        # Determine classifications
        size = self._classify_size(room)
        complexity = self._classify_complexity(room, features)
        aspect_ratio = self._calculate_aspect_ratio(room)
        
        # Calculate area and perimeter
        area = room.width * room.height
        perimeter = 2 * (room.width + room.height)
        
        # Check convexity and obstacles
        is_convex = self._is_convex(room)
        has_obstacles = len(getattr(room, 'obstacles', [])) > 0
        
        # Calculate regularity score
        regularity = self._calculate_regularity(room, features)
        
        classification = RoomClassification(
            size=size,
            complexity=complexity,
            aspect_ratio=aspect_ratio,
            area=area,
            perimeter=perimeter,
            num_vertices=len(getattr(room, 'vertices', [])) if hasattr(room, 'vertices') else 4,
            has_obstacles=has_obstacles,
            is_convex=is_convex,
            regularity_score=regularity,
            features=features
        )
        
        # Cache result
        self.classification_cache[room_hash] = classification
        
        return classification
    
    def _hash_room(self, room: Room) -> str:
        """Generate hash for room."""
        # Simple hash based on dimensions
        return f"{room.width}x{room.height}_{id(room)}"
    
    def _extract_size_features(self, room: Room) -> Dict[str, Any]:
        """Extract size-related features."""
        area = room.width * room.height
        return {
            'area': area,
            'width': room.width,
            'height': room.height,
            'perimeter': 2 * (room.width + room.height),
            'diagonal': math.sqrt(room.width**2 + room.height**2)
        }
    
    def _extract_shape_features(self, room: Room) -> Dict[str, Any]:
        """Extract shape-related features."""
        aspect_ratio = room.width / room.height if room.height > 0 else 1.0
        
        return {
            'aspect_ratio': aspect_ratio,
            'is_square': abs(aspect_ratio - 1.0) < 0.1,
            'is_elongated': aspect_ratio > 2.0 or aspect_ratio < 0.5,
            'compactness': min(room.width, room.height) / max(room.width, room.height)
        }
    
    def _extract_complexity_features(self, room: Room) -> Dict[str, Any]:
        """Extract complexity-related features."""
        features = {}
        
        # Count vertices if available
        if hasattr(room, 'vertices'):
            features['num_vertices'] = len(room.vertices)
            features['is_rectangular'] = len(room.vertices) == 4
        else:
            features['num_vertices'] = 4
            features['is_rectangular'] = True
        
        # Check for obstacles
        if hasattr(room, 'obstacles'):
            features['num_obstacles'] = len(room.obstacles)
            features['has_obstacles'] = len(room.obstacles) > 0
        else:
            features['num_obstacles'] = 0
            features['has_obstacles'] = False
        
        return features
    
    def _classify_size(self, room: Room) -> RoomSize:
        """Classify room size."""
        area = room.width * room.height
        
        if area < 100:
            return RoomSize.TINY
        elif area < 200:
            return RoomSize.SMALL
        elif area < 400:
            return RoomSize.MEDIUM
        elif area < 800:
            return RoomSize.LARGE
        else:
            return RoomSize.HUGE
    
    def _classify_complexity(self, room: Room, features: Dict[str, Any]) -> RoomComplexity:
        """Classify room complexity."""
        complexity_score = 0
        
        # Check vertices
        num_vertices = features.get('num_vertices', 4)
        if num_vertices > 8:
            complexity_score += 3
        elif num_vertices > 4:
            complexity_score += 1
        
        # Check obstacles
        num_obstacles = features.get('num_obstacles', 0)
        if num_obstacles > 5:
            complexity_score += 3
        elif num_obstacles > 0:
            complexity_score += 2
        
        # Check aspect ratio
        aspect_ratio = features.get('aspect_ratio', 1.0)
        if aspect_ratio > 3.0 or aspect_ratio < 0.33:
            complexity_score += 1
        
        # Determine complexity level
        if complexity_score >= 6:
            return RoomComplexity.EXTREME
        elif complexity_score >= 4:
            return RoomComplexity.COMPLEX
        elif complexity_score >= 2:
            return RoomComplexity.MODERATE
        else:
            return RoomComplexity.SIMPLE
    
    def _calculate_aspect_ratio(self, room: Room) -> float:
        """Calculate aspect ratio."""
        if room.height == 0:
            return 1.0
        return room.width / room.height
    
    def _is_convex(self, room: Room) -> bool:
        """Check if room is convex."""
        if not hasattr(room, 'vertices'):
            return True  # Assume rectangular rooms are convex
        
        vertices = room.vertices
        if len(vertices) < 3:
            return True
        
        # Check cross products for convexity
        n = len(vertices)
        sign = None
        
        for i in range(n):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % n]
            p3 = vertices[(i + 2) % n]
            
            # Cross product
            cross = (p2[0] - p1[0]) * (p3[1] - p2[1]) - (p2[1] - p1[1]) * (p3[0] - p2[0])
            
            if cross != 0:
                if sign is None:
                    sign = cross > 0
                elif (cross > 0) != sign:
                    return False
        
        return True
    
    def _calculate_regularity(self, room: Room, features: Dict[str, Any]) -> float:
        """Calculate regularity score (0-1, higher is more regular)."""
        score = 1.0
        
        # Penalize non-rectangular shapes
        if not features.get('is_rectangular', True):
            score -= 0.3
        
        # Penalize obstacles
        num_obstacles = features.get('num_obstacles', 0)
        score -= min(0.3, num_obstacles * 0.05)
        
        # Penalize extreme aspect ratios
        aspect_ratio = features.get('aspect_ratio', 1.0)
        if aspect_ratio > 2.5 or aspect_ratio < 0.4:
            score -= 0.2
        
        # Penalize many vertices
        num_vertices = features.get('num_vertices', 4)
        if num_vertices > 4:
            score -= min(0.2, (num_vertices - 4) * 0.05)
        
        return max(0.0, score)


class StrategyMapper:
    """Maps room classifications to optimization strategies."""
    
    def __init__(self):
        self.mapping_rules = self._initialize_rules()
        self.custom_mappings = {}
    
    def _initialize_rules(self) -> List[Dict[str, Any]]:
        """Initialize strategy mapping rules."""
        return [
            # Simple rooms - use fast strategies
            {
                'condition': lambda c: c.complexity == RoomComplexity.SIMPLE and c.size in [RoomSize.TINY, RoomSize.SMALL],
                'primary': StrategyType.SIMPLE_BLF,
                'secondary': [StrategyType.GREEDY],
                'confidence': 0.95,
                'reasoning': "Small simple rooms work well with basic BLF"
            },
            
            # Medium simple rooms
            {
                'condition': lambda c: c.complexity == RoomComplexity.SIMPLE and c.size == RoomSize.MEDIUM,
                'primary': StrategyType.ENHANCED_BLF,
                'secondary': [StrategyType.SIMPLE_BLF, StrategyType.DYNAMIC_PROGRAMMING],
                'confidence': 0.9,
                'reasoning': "Medium simple rooms benefit from enhanced BLF"
            },
            
            # Large simple rooms
            {
                'condition': lambda c: c.complexity == RoomComplexity.SIMPLE and c.size in [RoomSize.LARGE, RoomSize.HUGE],
                'primary': StrategyType.DYNAMIC_PROGRAMMING,
                'secondary': [StrategyType.ENHANCED_BLF, StrategyType.BRANCH_AND_BOUND],
                'confidence': 0.85,
                'reasoning': "Large simple rooms need efficient algorithms"
            },
            
            # Moderate complexity
            {
                'condition': lambda c: c.complexity == RoomComplexity.MODERATE,
                'primary': StrategyType.ENHANCED_BLF,
                'secondary': [StrategyType.BRANCH_AND_BOUND, StrategyType.HYBRID],
                'confidence': 0.8,
                'reasoning': "Moderate complexity requires adaptive strategies"
            },
            
            # Complex rooms
            {
                'condition': lambda c: c.complexity == RoomComplexity.COMPLEX,
                'primary': StrategyType.BRANCH_AND_BOUND,
                'secondary': [StrategyType.HYBRID, StrategyType.ENHANCED_BLF],
                'confidence': 0.75,
                'reasoning': "Complex rooms need sophisticated search"
            },
            
            # Extreme complexity
            {
                'condition': lambda c: c.complexity == RoomComplexity.EXTREME,
                'primary': StrategyType.HYBRID,
                'secondary': [StrategyType.BRANCH_AND_BOUND],
                'confidence': 0.7,
                'reasoning': "Extreme complexity requires hybrid approach"
            },
            
            # Elongated rooms
            {
                'condition': lambda c: c.aspect_ratio > 3.0 or c.aspect_ratio < 0.33,
                'primary': StrategyType.DYNAMIC_PROGRAMMING,
                'secondary': [StrategyType.ENHANCED_BLF],
                'confidence': 0.8,
                'reasoning': "Elongated rooms benefit from DP decomposition"
            },
            
            # Default fallback
            {
                'condition': lambda c: True,
                'primary': StrategyType.ENHANCED_BLF,
                'secondary': [StrategyType.SIMPLE_BLF, StrategyType.GREEDY],
                'confidence': 0.6,
                'reasoning': "Default strategy selection"
            }
        ]
    
    def map_strategy(self, classification: RoomClassification) -> StrategyMapping:
        """Map room classification to strategy."""
        # Check custom mappings first
        room_key = classification.to_key()
        if room_key in self.custom_mappings:
            return self.custom_mappings[room_key]
        
        # Apply rules in order
        for rule in self.mapping_rules:
            if rule['condition'](classification):
                return StrategyMapping(
                    classification=classification,
                    primary_strategy=rule['primary'],
                    secondary_strategies=rule['secondary'],
                    confidence=rule['confidence'],
                    reasoning=rule['reasoning']
                )
        
        # Should never reach here due to default rule
        return StrategyMapping(
            classification=classification,
            primary_strategy=StrategyType.SIMPLE_BLF,
            secondary_strategies=[],
            confidence=0.5,
            reasoning="Fallback mapping"
        )
    
    def add_custom_mapping(self, classification: RoomClassification, mapping: StrategyMapping):
        """Add custom mapping for specific room type."""
        self.custom_mappings[classification.to_key()] = mapping
    
    def adjust_mapping_confidence(self, classification: RoomClassification, performance: StrategyPerformance):
        """Adjust mapping confidence based on performance."""
        room_key = classification.to_key()
        
        if room_key in self.custom_mappings:
            mapping = self.custom_mappings[room_key]
        else:
            mapping = self.map_strategy(classification)
        
        # Adjust confidence based on performance
        if performance.success:
            if performance.coverage_achieved > 0.95:
                mapping.confidence = min(1.0, mapping.confidence * 1.1)
            elif performance.coverage_achieved > 0.9:
                mapping.confidence = min(1.0, mapping.confidence * 1.05)
        else:
            mapping.confidence *= 0.9
        
        # Store adjusted mapping
        self.custom_mappings[room_key] = mapping


class FallbackChain:
    """Manages fallback strategy chain."""
    
    def __init__(self, strategies: List[StrategyType], max_attempts: int = 3):
        self.strategies = strategies
        self.max_attempts = max_attempts
        self.current_index = 0
        self.attempt_count = defaultdict(int)
        self.performance_history = []
    
    def get_next_strategy(self) -> Optional[StrategyType]:
        """Get next strategy to try."""
        if self.current_index >= len(self.strategies):
            return None
        
        strategy = self.strategies[self.current_index]
        
        # Check attempt limit
        if self.attempt_count[strategy] >= self.max_attempts:
            self.current_index += 1
            return self.get_next_strategy()
        
        self.attempt_count[strategy] += 1
        return strategy
    
    def report_failure(self, strategy: StrategyType, reason: str = ""):
        """Report strategy failure."""
        self.performance_history.append({
            'strategy': strategy,
            'success': False,
            'reason': reason,
            'timestamp': time.time()
        })
        
        # Move to next strategy if all attempts exhausted
        if self.attempt_count[strategy] >= self.max_attempts:
            self.current_index += 1
    
    def report_success(self, strategy: StrategyType, coverage: float, time_taken: float):
        """Report strategy success."""
        self.performance_history.append({
            'strategy': strategy,
            'success': True,
            'coverage': coverage,
            'time_taken': time_taken,
            'timestamp': time.time()
        })
    
    def reset(self):
        """Reset fallback chain."""
        self.current_index = 0
        self.attempt_count.clear()
        self.performance_history.clear()
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get performance history."""
        return self.performance_history.copy()


class StrategyTracker:
    """Tracks strategy performance across runs."""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.performance_records = deque(maxlen=history_size)
        self.strategy_stats = defaultdict(lambda: {
            'total_runs': 0,
            'successful_runs': 0,
            'total_coverage': 0.0,
            'total_time': 0.0,
            'best_coverage': 0.0,
            'worst_coverage': 1.0
        })
        self.room_stats = defaultdict(lambda: defaultdict(lambda: {
            'runs': 0,
            'avg_coverage': 0.0,
            'avg_time': 0.0
        }))
    
    def record_performance(self, performance: StrategyPerformance):
        """Record strategy performance."""
        self.performance_records.append(performance)
        
        # Update strategy statistics
        stats = self.strategy_stats[performance.strategy]
        stats['total_runs'] += 1
        if performance.success:
            stats['successful_runs'] += 1
        stats['total_coverage'] += performance.coverage_achieved
        stats['total_time'] += performance.time_taken
        stats['best_coverage'] = max(stats['best_coverage'], performance.coverage_achieved)
        stats['worst_coverage'] = min(stats['worst_coverage'], performance.coverage_achieved)
        
        # Update room-specific statistics
        room_strategy_stats = self.room_stats[performance.room_key][performance.strategy]
        room_strategy_stats['runs'] += 1
        # Running average
        alpha = 0.1
        room_strategy_stats['avg_coverage'] = (
            (1 - alpha) * room_strategy_stats['avg_coverage'] + 
            alpha * performance.coverage_achieved
        )
        room_strategy_stats['avg_time'] = (
            (1 - alpha) * room_strategy_stats['avg_time'] + 
            alpha * performance.time_taken
        )
    
    def get_strategy_stats(self, strategy: StrategyType) -> Dict[str, Any]:
        """Get statistics for a strategy."""
        stats = self.strategy_stats[strategy].copy()
        
        if stats['total_runs'] > 0:
            stats['success_rate'] = stats['successful_runs'] / stats['total_runs']
            stats['avg_coverage'] = stats['total_coverage'] / stats['total_runs']
            stats['avg_time'] = stats['total_time'] / stats['total_runs']
        else:
            stats['success_rate'] = 0.0
            stats['avg_coverage'] = 0.0
            stats['avg_time'] = 0.0
        
        return stats
    
    def get_best_strategy_for_room(self, room_key: str) -> Optional[StrategyType]:
        """Get best performing strategy for a room type."""
        if room_key not in self.room_stats:
            return None
        
        room_strategies = self.room_stats[room_key]
        if not room_strategies:
            return None
        
        # Find strategy with best average coverage
        best_strategy = None
        best_coverage = 0.0
        
        for strategy, stats in room_strategies.items():
            if stats['avg_coverage'] > best_coverage:
                best_coverage = stats['avg_coverage']
                best_strategy = strategy
        
        return best_strategy
    
    def get_recommendation(self, classification: RoomClassification) -> Optional[StrategyType]:
        """Get strategy recommendation based on historical performance."""
        room_key = classification.to_key()
        
        # Check if we have history for this exact room type
        best_strategy = self.get_best_strategy_for_room(room_key)
        if best_strategy:
            return best_strategy
        
        # Otherwise, find similar room types
        similar_keys = self._find_similar_room_keys(classification)
        
        # Aggregate performance across similar rooms
        strategy_scores = defaultdict(float)
        total_weight = 0.0
        
        for key, weight in similar_keys:
            if key in self.room_stats:
                for strategy, stats in self.room_stats[key].items():
                    strategy_scores[strategy] += stats['avg_coverage'] * weight
                total_weight += weight
        
        if total_weight > 0:
            # Normalize scores and find best
            best_strategy = None
            best_score = 0.0
            
            for strategy, score in strategy_scores.items():
                normalized_score = score / total_weight
                if normalized_score > best_score:
                    best_score = normalized_score
                    best_strategy = strategy
            
            return best_strategy
        
        return None
    
    def _find_similar_room_keys(self, classification: RoomClassification) -> List[Tuple[str, float]]:
        """Find similar room keys with similarity weights."""
        similar = []
        
        for room_key in self.room_stats.keys():
            # Parse room key
            parts = room_key.split('_')
            if len(parts) >= 3:
                size = parts[0]
                complexity = parts[1]
                
                # Calculate similarity
                similarity = 0.0
                
                if size == classification.size.value:
                    similarity += 0.5
                if complexity == classification.complexity.value:
                    similarity += 0.5
                
                if similarity > 0:
                    similar.append((room_key, similarity))
        
        return similar
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics."""
        summary = {
            'total_runs': len(self.performance_records),
            'strategies': {}
        }
        
        for strategy in StrategyType:
            summary['strategies'][strategy.value] = self.get_strategy_stats(strategy)
        
        # Add room type statistics
        summary['room_types'] = len(self.room_stats)
        summary['avg_coverage_overall'] = (
            sum(p.coverage_achieved for p in self.performance_records) / 
            len(self.performance_records)
        ) if self.performance_records else 0.0
        
        return summary


class StrategySelectionSystem:
    """Main system coordinating strategy selection."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize components
        self.classifier = RoomClassifier()
        self.mapper = StrategyMapper()
        self.tracker = StrategyTracker(
            history_size=self.config.get('history_size', 1000)
        )
        
        self.current_fallback_chain = None
        self.current_classification = None
        self.current_mapping = None
    
    def select_strategy(self, room: Room, panels: List[PanelSize]) -> StrategyType:
        """Select optimal strategy for room and panels."""
        # Classify room
        classification = self.classifier.classify(room)
        self.current_classification = classification
        
        # Check tracker for historical best
        recommended = self.tracker.get_recommendation(classification)
        
        if recommended:
            # Use historical best with high confidence
            self.current_mapping = StrategyMapping(
                classification=classification,
                primary_strategy=recommended,
                secondary_strategies=[],
                confidence=0.9,
                reasoning="Based on historical performance"
            )
        else:
            # Use rule-based mapping
            self.current_mapping = self.mapper.map_strategy(classification)
        
        # Create fallback chain
        strategies = self.current_mapping.get_strategies()
        self.current_fallback_chain = FallbackChain(
            strategies,
            max_attempts=self.config.get('max_attempts', 3)
        )
        
        # Return primary strategy
        return self.current_mapping.primary_strategy
    
    def get_next_strategy(self) -> Optional[StrategyType]:
        """Get next strategy from fallback chain."""
        if not self.current_fallback_chain:
            return None
        
        return self.current_fallback_chain.get_next_strategy()
    
    def report_result(self, 
                     strategy: StrategyType,
                     coverage: float,
                     time_taken: float,
                     success: bool,
                     metadata: Optional[Dict[str, Any]] = None):
        """Report strategy execution result."""
        if not self.current_classification:
            return
        
        # Create performance record
        performance = StrategyPerformance(
            strategy=strategy,
            room_key=self.current_classification.to_key(),
            coverage_achieved=coverage,
            time_taken=time_taken,
            iterations=metadata.get('iterations', 0) if metadata else 0,
            success=success,
            metadata=metadata or {}
        )
        
        # Record in tracker
        self.tracker.record_performance(performance)
        
        # Update fallback chain
        if self.current_fallback_chain:
            if success:
                self.current_fallback_chain.report_success(strategy, coverage, time_taken)
            else:
                self.current_fallback_chain.report_failure(
                    strategy,
                    metadata.get('failure_reason', '') if metadata else ''
                )
        
        # Adjust mapping confidence
        self.mapper.adjust_mapping_confidence(self.current_classification, performance)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            'tracker': self.tracker.get_summary_statistics(),
            'current_classification': (
                self.current_classification.__dict__ 
                if self.current_classification else None
            ),
            'current_mapping': (
                {
                    'primary': self.current_mapping.primary_strategy.value,
                    'secondary': [s.value for s in self.current_mapping.secondary_strategies],
                    'confidence': self.current_mapping.confidence,
                    'reasoning': self.current_mapping.reasoning
                }
                if self.current_mapping else None
            )
        }
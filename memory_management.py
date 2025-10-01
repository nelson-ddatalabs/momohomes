#!/usr/bin/env python3
"""
memory_management.py - Memory Management System
===============================================
Production-ready memory monitor, cache optimizer, pressure handler,
and garbage collection coordinator for resource management.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Set
from collections import defaultdict, OrderedDict, deque
from enum import Enum
import sys
import gc
import psutil
import weakref
import threading
import time
import tracemalloc


class MemoryPressureLevel(Enum):
    """Memory pressure levels."""
    NORMAL = "normal"  # < 60% usage
    MODERATE = "moderate"  # 60-75% usage
    HIGH = "high"  # 75-85% usage
    CRITICAL = "critical"  # 85-95% usage
    EMERGENCY = "emergency"  # > 95% usage


class CacheStrategy(Enum):
    """Cache optimization strategies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    ADAPTIVE = "adaptive"  # Adaptive strategy
    SIZE_AWARE = "size_aware"  # Consider object size


@dataclass
class MemorySnapshot:
    """Snapshot of memory state."""
    timestamp: float
    total_memory: int
    available_memory: int
    used_memory: int
    percent_used: float
    process_memory: int
    cache_size: int
    pressure_level: MemoryPressureLevel
    gc_stats: Dict[str, Any]
    top_allocations: List[Tuple[str, int]]


@dataclass
class CacheEntry:
    """Entry in cache."""
    key: Any
    value: Any
    size: int
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    creation_time: float = field(default_factory=time.time)
    
    def update_access(self):
        """Update access statistics."""
        self.access_count += 1
        self.last_access = time.time()


class MemoryMonitor:
    """Monitors memory usage and pressure."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.threshold_normal = self.config.get('threshold_normal', 0.6)
        self.threshold_moderate = self.config.get('threshold_moderate', 0.75)
        self.threshold_high = self.config.get('threshold_high', 0.85)
        self.threshold_critical = self.config.get('threshold_critical', 0.95)
        
        self.monitoring = False
        self.monitor_thread = None
        self.snapshots = deque(maxlen=100)
        self.listeners = []
        
        # Start tracemalloc if requested
        if self.config.get('enable_tracemalloc', False):
            tracemalloc.start()
    
    def start_monitoring(self, interval: float = 1.0):
        """Start monitoring memory."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring memory."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
    
    def _monitor_loop(self, interval: float):
        """Main monitoring loop."""
        while self.monitoring:
            snapshot = self.take_snapshot()
            self.snapshots.append(snapshot)
            
            # Notify listeners if pressure changed
            if len(self.snapshots) >= 2:
                prev_level = self.snapshots[-2].pressure_level
                curr_level = snapshot.pressure_level
                
                if prev_level != curr_level:
                    self._notify_pressure_change(prev_level, curr_level)
            
            time.sleep(interval)
    
    def take_snapshot(self) -> MemorySnapshot:
        """Take memory snapshot."""
        # System memory
        memory = psutil.virtual_memory()
        
        # Process memory
        process = psutil.Process()
        process_info = process.memory_info()
        
        # Calculate pressure level
        percent_used = memory.percent / 100
        pressure_level = self._calculate_pressure_level(percent_used)
        
        # GC statistics
        gc_stats = {
            'collections': gc.get_count(),
            'collected': gc.collect(0),
            'uncollectable': len(gc.garbage)
        }
        
        # Top allocations (if tracemalloc is enabled)
        top_allocations = []
        if tracemalloc.is_tracing():
            snapshot_trace = tracemalloc.take_snapshot()
            top_stats = snapshot_trace.statistics('lineno')[:10]
            top_allocations = [(str(stat.traceback), stat.size) for stat in top_stats]
        
        return MemorySnapshot(
            timestamp=time.time(),
            total_memory=memory.total,
            available_memory=memory.available,
            used_memory=memory.used,
            percent_used=percent_used,
            process_memory=process_info.rss,
            cache_size=self._estimate_cache_size(),
            pressure_level=pressure_level,
            gc_stats=gc_stats,
            top_allocations=top_allocations
        )
    
    def _calculate_pressure_level(self, percent_used: float) -> MemoryPressureLevel:
        """Calculate memory pressure level."""
        if percent_used < self.threshold_normal:
            return MemoryPressureLevel.NORMAL
        elif percent_used < self.threshold_moderate:
            return MemoryPressureLevel.MODERATE
        elif percent_used < self.threshold_high:
            return MemoryPressureLevel.HIGH
        elif percent_used < self.threshold_critical:
            return MemoryPressureLevel.CRITICAL
        else:
            return MemoryPressureLevel.EMERGENCY
    
    def _estimate_cache_size(self) -> int:
        """Estimate cache size in bytes."""
        # This is a placeholder - would need actual cache references
        return 0
    
    def _notify_pressure_change(self,
                              old_level: MemoryPressureLevel,
                              new_level: MemoryPressureLevel):
        """Notify listeners of pressure change."""
        for listener in self.listeners:
            try:
                listener(old_level, new_level)
            except Exception as e:
                print(f"Listener error: {e}")
    
    def add_listener(self, listener: Callable):
        """Add pressure change listener."""
        self.listeners.append(listener)
    
    def get_current_pressure(self) -> MemoryPressureLevel:
        """Get current memory pressure level."""
        if self.snapshots:
            return self.snapshots[-1].pressure_level
        
        # Take snapshot if no history
        snapshot = self.take_snapshot()
        return snapshot.pressure_level
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics."""
        snapshot = self.take_snapshot()
        
        return {
            'total_mb': snapshot.total_memory / (1024 * 1024),
            'used_mb': snapshot.used_memory / (1024 * 1024),
            'available_mb': snapshot.available_memory / (1024 * 1024),
            'percent_used': snapshot.percent_used * 100,
            'process_mb': snapshot.process_memory / (1024 * 1024),
            'pressure_level': snapshot.pressure_level.value
        }
    
    def get_trend(self, window: int = 10) -> str:
        """Get memory usage trend."""
        if len(self.snapshots) < 2:
            return "stable"
        
        recent = list(self.snapshots)[-window:]
        if len(recent) < 2:
            return "stable"
        
        # Calculate trend
        first_usage = recent[0].percent_used
        last_usage = recent[-1].percent_used
        
        change = last_usage - first_usage
        
        if change > 0.1:
            return "increasing"
        elif change < -0.1:
            return "decreasing"
        else:
            return "stable"


class CacheOptimizer:
    """Optimizes cache memory usage."""
    
    def __init__(self, 
                max_size: int = 100 * 1024 * 1024,  # 100MB default
                strategy: CacheStrategy = CacheStrategy.LRU):
        self.max_size = max_size
        self.strategy = strategy
        self.cache = OrderedDict()
        self.cache_stats = defaultdict(lambda: {'hits': 0, 'misses': 0})
        self.current_size = 0
        self.eviction_count = 0
        self.lock = threading.Lock()
    
    def get(self, key: Any) -> Optional[Any]:
        """Get item from cache."""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                entry.update_access()
                
                # Move to end for LRU
                if self.strategy == CacheStrategy.LRU:
                    self.cache.move_to_end(key)
                
                self.cache_stats[key]['hits'] += 1
                return entry.value
            else:
                self.cache_stats[key]['misses'] += 1
                return None
    
    def put(self, key: Any, value: Any, size: Optional[int] = None):
        """Put item in cache."""
        if size is None:
            size = sys.getsizeof(value)
        
        with self.lock:
            # Remove if already exists
            if key in self.cache:
                self.current_size -= self.cache[key].size
                del self.cache[key]
            
            # Check if item fits
            if size > self.max_size:
                return  # Item too large
            
            # Evict items if necessary
            while self.current_size + size > self.max_size:
                self._evict()
            
            # Add new entry
            entry = CacheEntry(key=key, value=value, size=size)
            self.cache[key] = entry
            self.current_size += size
    
    def _evict(self):
        """Evict item based on strategy."""
        if not self.cache:
            return
        
        if self.strategy == CacheStrategy.LRU:
            # Remove least recently used
            key, entry = self.cache.popitem(last=False)
        elif self.strategy == CacheStrategy.LFU:
            # Remove least frequently used
            key = min(self.cache.keys(), 
                     key=lambda k: self.cache[k].access_count)
            entry = self.cache.pop(key)
        elif self.strategy == CacheStrategy.FIFO:
            # Remove oldest
            key, entry = self.cache.popitem(last=False)
        elif self.strategy == CacheStrategy.SIZE_AWARE:
            # Remove largest item with low access count
            key = max(self.cache.keys(),
                     key=lambda k: self.cache[k].size / (self.cache[k].access_count + 1))
            entry = self.cache.pop(key)
        else:  # ADAPTIVE
            # Adaptive strategy based on access patterns
            key = self._adaptive_evict()
            entry = self.cache.pop(key)
        
        self.current_size -= entry.size
        self.eviction_count += 1
    
    def _adaptive_evict(self) -> Any:
        """Adaptive eviction strategy."""
        # Calculate scores for each entry
        scores = {}
        current_time = time.time()
        
        for key, entry in self.cache.items():
            # Combine multiple factors
            recency_score = 1.0 / (current_time - entry.last_access + 1)
            frequency_score = entry.access_count / (current_time - entry.creation_time + 1)
            size_penalty = entry.size / self.max_size
            
            scores[key] = recency_score * frequency_score / (size_penalty + 0.1)
        
        # Evict item with lowest score
        return min(scores.keys(), key=lambda k: scores[k])
    
    def clear(self):
        """Clear cache."""
        with self.lock:
            self.cache.clear()
            self.current_size = 0
    
    def resize(self, new_size: int):
        """Resize cache."""
        with self.lock:
            self.max_size = new_size
            
            # Evict if necessary
            while self.current_size > self.max_size:
                self._evict()
    
    def optimize(self, pressure_level: MemoryPressureLevel):
        """Optimize cache based on memory pressure."""
        if pressure_level == MemoryPressureLevel.NORMAL:
            # Can expand cache if beneficial
            if self.get_hit_rate() > 0.8:
                self.resize(min(self.max_size * 1.2, 500 * 1024 * 1024))
        elif pressure_level == MemoryPressureLevel.MODERATE:
            # Maintain current size
            pass
        elif pressure_level == MemoryPressureLevel.HIGH:
            # Reduce cache size
            self.resize(int(self.max_size * 0.8))
        elif pressure_level == MemoryPressureLevel.CRITICAL:
            # Significantly reduce cache
            self.resize(int(self.max_size * 0.5))
        else:  # EMERGENCY
            # Clear most of cache
            self.resize(int(self.max_size * 0.1))
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate."""
        total_hits = sum(stats['hits'] for stats in self.cache_stats.values())
        total_misses = sum(stats['misses'] for stats in self.cache_stats.values())
        
        total = total_hits + total_misses
        if total == 0:
            return 0.0
        
        return total_hits / total
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'size_mb': self.current_size / (1024 * 1024),
            'max_size_mb': self.max_size / (1024 * 1024),
            'entries': len(self.cache),
            'hit_rate': self.get_hit_rate(),
            'eviction_count': self.eviction_count,
            'strategy': self.strategy.value
        }


class MemoryPressureHandler:
    """Handles memory pressure situations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.handlers = {
            MemoryPressureLevel.NORMAL: self._handle_normal,
            MemoryPressureLevel.MODERATE: self._handle_moderate,
            MemoryPressureLevel.HIGH: self._handle_high,
            MemoryPressureLevel.CRITICAL: self._handle_critical,
            MemoryPressureLevel.EMERGENCY: self._handle_emergency
        }
        self.actions_taken = []
        self.recovery_strategies = []
    
    def handle_pressure(self, 
                       pressure_level: MemoryPressureLevel,
                       cache_optimizer: Optional[CacheOptimizer] = None) -> List[str]:
        """Handle memory pressure."""
        actions = []
        
        # Call appropriate handler
        if pressure_level in self.handlers:
            handler_actions = self.handlers[pressure_level](cache_optimizer)
            actions.extend(handler_actions)
        
        # Record actions
        self.actions_taken.append({
            'timestamp': time.time(),
            'pressure_level': pressure_level,
            'actions': actions
        })
        
        return actions
    
    def _handle_normal(self, cache_optimizer: Optional[CacheOptimizer]) -> List[str]:
        """Handle normal memory pressure."""
        actions = []
        
        # Normal operations, maybe expand cache
        if cache_optimizer and cache_optimizer.get_hit_rate() > 0.8:
            cache_optimizer.optimize(MemoryPressureLevel.NORMAL)
            actions.append("expanded_cache")
        
        return actions
    
    def _handle_moderate(self, cache_optimizer: Optional[CacheOptimizer]) -> List[str]:
        """Handle moderate memory pressure."""
        actions = []
        
        # Gentle cleanup
        gc.collect(0)
        actions.append("gc_generation_0")
        
        if cache_optimizer:
            cache_optimizer.optimize(MemoryPressureLevel.MODERATE)
            actions.append("optimized_cache")
        
        return actions
    
    def _handle_high(self, cache_optimizer: Optional[CacheOptimizer]) -> List[str]:
        """Handle high memory pressure."""
        actions = []
        
        # More aggressive cleanup
        gc.collect(1)
        actions.append("gc_generation_1")
        
        if cache_optimizer:
            cache_optimizer.optimize(MemoryPressureLevel.HIGH)
            actions.append("reduced_cache")
        
        # Clear weak references
        gc.collect()
        actions.append("cleared_weak_refs")
        
        return actions
    
    def _handle_critical(self, cache_optimizer: Optional[CacheOptimizer]) -> List[str]:
        """Handle critical memory pressure."""
        actions = []
        
        # Aggressive cleanup
        gc.collect(2)
        actions.append("gc_full")
        
        if cache_optimizer:
            cache_optimizer.optimize(MemoryPressureLevel.CRITICAL)
            actions.append("minimized_cache")
        
        # Clear all weak references
        gc.collect()
        gc.collect()
        actions.append("aggressive_gc")
        
        # Trigger emergency recovery strategies
        for strategy in self.recovery_strategies:
            try:
                strategy()
                actions.append(f"recovery_{strategy.__name__}")
            except Exception as e:
                print(f"Recovery strategy failed: {e}")
        
        return actions
    
    def _handle_emergency(self, cache_optimizer: Optional[CacheOptimizer]) -> List[str]:
        """Handle emergency memory pressure."""
        actions = []
        
        # Emergency measures
        if cache_optimizer:
            cache_optimizer.clear()
            actions.append("cleared_cache")
        
        # Force full GC
        for _ in range(3):
            gc.collect()
        actions.append("emergency_gc")
        
        # Clear all module-level caches
        for module in sys.modules.values():
            if hasattr(module, '__cache__'):
                module.__cache__.clear()
                actions.append(f"cleared_{module.__name__}_cache")
        
        return actions
    
    def add_recovery_strategy(self, strategy: Callable):
        """Add custom recovery strategy."""
        self.recovery_strategies.append(strategy)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pressure handling statistics."""
        stats = {
            'total_actions': len(self.actions_taken),
            'actions_by_level': defaultdict(int)
        }
        
        for action in self.actions_taken:
            level = action['pressure_level'].value
            stats['actions_by_level'][level] += 1
        
        return stats


class GCCoordinator:
    """Coordinates garbage collection."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.auto_tune = self.config.get('auto_tune', True)
        self.gc_threshold = self.config.get('gc_threshold', (700, 10, 10))
        self.gc_history = deque(maxlen=100)
        
        # Set initial threshold
        gc.set_threshold(*self.gc_threshold)
    
    def collect(self, generation: Optional[int] = None) -> int:
        """Perform garbage collection."""
        start_time = time.time()
        
        if generation is None:
            # Automatic generation selection based on memory pressure
            pressure = MemoryMonitor().get_current_pressure()
            if pressure == MemoryPressureLevel.NORMAL:
                generation = 0
            elif pressure == MemoryPressureLevel.MODERATE:
                generation = 1
            else:
                generation = 2
        
        # Perform collection
        collected = gc.collect(generation)
        
        # Record history
        self.gc_history.append({
            'timestamp': time.time(),
            'generation': generation,
            'collected': collected,
            'duration': time.time() - start_time
        })
        
        # Auto-tune if enabled
        if self.auto_tune:
            self._auto_tune_threshold()
        
        return collected
    
    def _auto_tune_threshold(self):
        """Auto-tune GC threshold based on history."""
        if len(self.gc_history) < 10:
            return
        
        # Analyze recent collections
        recent = list(self.gc_history)[-10:]
        avg_collected = sum(h['collected'] for h in recent) / len(recent)
        avg_duration = sum(h['duration'] for h in recent) / len(recent)
        
        # Adjust threshold
        current = gc.get_threshold()
        new_threshold = list(current)
        
        if avg_collected < 10 and avg_duration < 0.001:
            # Too frequent, increase threshold
            new_threshold[0] = min(current[0] * 1.1, 10000)
        elif avg_collected > 1000 or avg_duration > 0.1:
            # Too infrequent, decrease threshold
            new_threshold[0] = max(current[0] * 0.9, 100)
        
        # Apply new threshold
        if new_threshold != list(current):
            gc.set_threshold(*new_threshold)
            self.gc_threshold = tuple(new_threshold)
    
    def schedule_collection(self, delay: float, generation: int = 0) -> threading.Timer:
        """Schedule future garbage collection."""
        timer = threading.Timer(delay, lambda: self.collect(generation))
        timer.start()
        return timer
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get GC statistics."""
        stats = gc.get_stats()
        
        return {
            'threshold': gc.get_threshold(),
            'counts': gc.get_count(),
            'enabled': gc.isenabled(),
            'stats': stats,
            'history_size': len(self.gc_history),
            'recent_collections': list(self.gc_history)[-5:] if self.gc_history else []
        }
    
    def optimize_for_phase(self, phase: str):
        """Optimize GC for specific phase."""
        if phase == "initialization":
            # Disable during initialization
            gc.disable()
        elif phase == "execution":
            # Normal operation
            gc.enable()
            gc.set_threshold(*self.gc_threshold)
        elif phase == "intensive":
            # Less frequent during intensive computation
            gc.set_threshold(self.gc_threshold[0] * 2, 
                           self.gc_threshold[1], 
                           self.gc_threshold[2])
        elif phase == "cleanup":
            # Aggressive cleanup
            gc.enable()
            gc.set_threshold(100, 10, 10)
            gc.collect()


class ObjectPool:
    """Object pool for memory efficiency."""
    
    def __init__(self, 
                factory: Callable,
                max_size: int = 100,
                pre_allocate: int = 0):
        self.factory = factory
        self.max_size = max_size
        self.available = deque()
        self.in_use = weakref.WeakSet()
        
        # Pre-allocate objects
        for _ in range(pre_allocate):
            obj = factory()
            self.available.append(obj)
    
    def acquire(self) -> Any:
        """Acquire object from pool."""
        if self.available:
            obj = self.available.popleft()
        else:
            obj = self.factory()
        
        self.in_use.add(obj)
        return obj
    
    def release(self, obj: Any):
        """Release object back to pool."""
        if obj in self.in_use:
            self.in_use.discard(obj)
            
            if len(self.available) < self.max_size:
                # Reset object if it has reset method
                if hasattr(obj, 'reset'):
                    obj.reset()
                
                self.available.append(obj)
    
    def clear(self):
        """Clear pool."""
        self.available.clear()
        # in_use will be cleared by GC
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            'available': len(self.available),
            'in_use': len(self.in_use),
            'max_size': self.max_size
        }


class MemoryManagementSystem:
    """Main memory management system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize components
        self.monitor = MemoryMonitor(config)
        self.cache_optimizer = CacheOptimizer(
            max_size=self.config.get('cache_size', 100 * 1024 * 1024),
            strategy=CacheStrategy(self.config.get('cache_strategy', 'lru'))
        )
        self.pressure_handler = MemoryPressureHandler(config)
        self.gc_coordinator = GCCoordinator(config)
        
        # Object pools
        self.object_pools = {}
        
        # Setup monitoring
        self.monitor.add_listener(self._on_pressure_change)
        
        # Start monitoring if requested
        if self.config.get('auto_monitor', True):
            self.monitor.start_monitoring()
    
    def _on_pressure_change(self,
                           old_level: MemoryPressureLevel,
                           new_level: MemoryPressureLevel):
        """Handle memory pressure change."""
        # Optimize cache
        self.cache_optimizer.optimize(new_level)
        
        # Handle pressure
        self.pressure_handler.handle_pressure(new_level, self.cache_optimizer)
        
        # Adjust GC
        if new_level.value in ['critical', 'emergency']:
            self.gc_coordinator.collect(2)
        elif new_level == MemoryPressureLevel.HIGH:
            self.gc_coordinator.collect(1)
    
    def create_object_pool(self,
                          name: str,
                          factory: Callable,
                          max_size: int = 100) -> ObjectPool:
        """Create object pool."""
        pool = ObjectPool(factory, max_size)
        self.object_pools[name] = pool
        return pool
    
    def get_object_pool(self, name: str) -> Optional[ObjectPool]:
        """Get object pool by name."""
        return self.object_pools.get(name)
    
    def optimize(self):
        """Perform memory optimization."""
        # Get current pressure
        pressure = self.monitor.get_current_pressure()
        
        # Optimize cache
        self.cache_optimizer.optimize(pressure)
        
        # Run GC if needed
        if pressure.value in ['high', 'critical', 'emergency']:
            self.gc_coordinator.collect()
        
        # Clear unused pools
        for pool in self.object_pools.values():
            if len(pool.in_use) == 0:
                pool.clear()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            'memory': self.monitor.get_memory_usage(),
            'cache': self.cache_optimizer.get_statistics(),
            'gc': self.gc_coordinator.get_statistics(),
            'pressure_handler': self.pressure_handler.get_statistics(),
            'object_pools': {
                name: pool.get_statistics()
                for name, pool in self.object_pools.items()
            }
        }
    
    def shutdown(self):
        """Shutdown memory management."""
        # Stop monitoring
        self.monitor.stop_monitoring()
        
        # Clear cache
        self.cache_optimizer.clear()
        
        # Clear pools
        for pool in self.object_pools.values():
            pool.clear()
        
        # Final GC
        gc.collect()
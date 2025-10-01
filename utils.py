"""
utils.py - Utility Functions and Helpers
=========================================
Common utility functions used throughout the system.
"""

import os
import json
import pickle
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import numpy as np
from functools import wraps
import time


logger = logging.getLogger(__name__)


# ============================================================================
# FILE OPERATIONS
# ============================================================================

def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if necessary."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Dict, filepath: Union[str, Path], indent: int = 2):
    """Save data to JSON file."""
    filepath = Path(filepath)
    ensure_directory(filepath.parent)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent, default=str)
    
    logger.debug(f"Saved JSON to {filepath}")


def load_json(filepath: Union[str, Path]) -> Dict:
    """Load data from JSON file."""
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"JSON file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    logger.debug(f"Loaded JSON from {filepath}")
    return data


def save_pickle(obj: Any, filepath: Union[str, Path]):
    """Save object to pickle file."""
    filepath = Path(filepath)
    ensure_directory(filepath.parent)
    
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    
    logger.debug(f"Saved pickle to {filepath}")


def load_pickle(filepath: Union[str, Path]) -> Any:
    """Load object from pickle file."""
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Pickle file not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
    
    logger.debug(f"Loaded pickle from {filepath}")
    return obj


def get_file_hash(filepath: Union[str, Path]) -> str:
    """Calculate SHA256 hash of file."""
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    sha256_hash = hashlib.sha256()
    
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    
    return sha256_hash.hexdigest()


def find_files(directory: Union[str, Path], pattern: str = "*", 
              recursive: bool = True) -> List[Path]:
    """Find files matching pattern in directory."""
    directory = Path(directory)
    
    if recursive:
        files = list(directory.rglob(pattern))
    else:
        files = list(directory.glob(pattern))
    
    return sorted(files)


# ============================================================================
# GEOMETRIC UTILITIES
# ============================================================================

def calculate_distance(point1: Tuple[float, float], 
                      point2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points."""
    x1, y1 = point1
    x2, y2 = point2
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def calculate_area(width: float, height: float) -> float:
    """Calculate rectangular area."""
    return width * height


def calculate_perimeter(width: float, height: float) -> float:
    """Calculate rectangular perimeter."""
    return 2 * (width + height)


def calculate_aspect_ratio(width: float, height: float) -> float:
    """Calculate aspect ratio."""
    if min(width, height) == 0:
        return float('inf')
    return max(width, height) / min(width, height)


def rectangles_overlap(rect1: Tuple[float, float, float, float],
                       rect2: Tuple[float, float, float, float]) -> bool:
    """Check if two rectangles overlap.
    
    Args:
        rect1: (x, y, width, height)
        rect2: (x, y, width, height)
    """
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    
    return not (x1 + w1 < x2 or x2 + w2 < x1 or
                y1 + h1 < y2 or y2 + h2 < y1)


def point_in_rectangle(point: Tuple[float, float],
                       rect: Tuple[float, float, float, float]) -> bool:
    """Check if point is inside rectangle.
    
    Args:
        point: (x, y)
        rect: (x, y, width, height)
    """
    px, py = point
    rx, ry, rw, rh = rect
    
    return rx <= px <= rx + rw and ry <= py <= ry + rh


def calculate_coverage_grid(rooms: List, panels: List, 
                           grid_size: int = 50) -> np.ndarray:
    """Calculate coverage grid for visualization."""
    # Find bounds
    min_x = min(r.position.x for r in rooms)
    min_y = min(r.position.y for r in rooms)
    max_x = max(r.position.x + r.width for r in rooms)
    max_y = max(r.position.y + r.height for r in rooms)
    
    # Create grid
    x_range = np.linspace(min_x, max_x, grid_size)
    y_range = np.linspace(min_y, max_y, grid_size)
    coverage_grid = np.zeros((grid_size, grid_size))
    
    # Calculate coverage
    for i, y in enumerate(y_range):
        for j, x in enumerate(x_range):
            # Check if point is covered by any panel
            for panel in panels:
                if point_in_rectangle(
                    (x, y),
                    (panel.position.x, panel.position.y, panel.width, panel.length)
                ):
                    coverage_grid[i, j] = 1
                    break
    
    return coverage_grid


# ============================================================================
# VALIDATION UTILITIES
# ============================================================================

def validate_dimensions(width: float, height: float, 
                       min_size: float = 1.0, max_size: float = 1000.0) -> bool:
    """Validate room dimensions."""
    return (min_size <= width <= max_size and 
            min_size <= height <= max_size)


def validate_panel_size(panel_size: str) -> bool:
    """Validate panel size string."""
    valid_sizes = ['6x8', '6x6', '4x6', '4x4']
    return panel_size in valid_sizes


def validate_room_type(room_type: str) -> bool:
    """Validate room type string."""
    valid_types = [
        'bedroom', 'bathroom', 'kitchen', 'living', 'dining',
        'hallway', 'closet', 'entry', 'office', 'garage',
        'utility', 'open_space', 'unknown'
    ]
    return room_type.lower() in valid_types


def validate_file_format(filepath: Union[str, Path], 
                        allowed_formats: List[str]) -> bool:
    """Validate file format."""
    filepath = Path(filepath)
    return filepath.suffix.lower() in allowed_formats


# ============================================================================
# CONVERSION UTILITIES
# ============================================================================

def feet_to_meters(feet: float) -> float:
    """Convert feet to meters."""
    return feet * 0.3048


def meters_to_feet(meters: float) -> float:
    """Convert meters to feet."""
    return meters / 0.3048


def sqft_to_sqm(sqft: float) -> float:
    """Convert square feet to square meters."""
    return sqft * 0.092903


def sqm_to_sqft(sqm: float) -> float:
    """Convert square meters to square feet."""
    return sqm / 0.092903


def parse_dimension_string(dim_str: str) -> Tuple[float, float]:
    """Parse dimension string like '10x12' or '10 x 12'."""
    import re
    
    # Remove spaces and convert to lowercase
    dim_str = dim_str.replace(' ', '').lower()
    
    # Try different patterns
    patterns = [
        r'(\d+\.?\d*)x(\d+\.?\d*)',  # 10x12 or 10.5x12.5
        r'(\d+\.?\d*)[*](\d+\.?\d*)',  # 10*12
        r'(\d+\.?\d*)by(\d+\.?\d*)',  # 10by12
    ]
    
    for pattern in patterns:
        match = re.match(pattern, dim_str)
        if match:
            return float(match.group(1)), float(match.group(2))
    
    raise ValueError(f"Could not parse dimension string: {dim_str}")


# ============================================================================
# PERFORMANCE UTILITIES
# ============================================================================

def timer(func):
    """Decorator to time function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.debug(f"{func.__name__} took {end - start:.2f} seconds")
        return result
    return wrapper


def memoize(func):
    """Decorator to memoize function results."""
    cache = {}
    
    @wraps(func)
    def wrapper(*args):
        if args in cache:
            return cache[args]
        result = func(*args)
        cache[args] = result
        return result
    
    return wrapper


class ProgressTracker:
    """Track and display progress for long operations."""
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()
    
    def update(self, increment: int = 1):
        """Update progress."""
        self.current += increment
        self._display()
    
    def _display(self):
        """Display progress bar."""
        percent = (self.current / self.total) * 100 if self.total > 0 else 0
        elapsed = time.time() - self.start_time
        
        if self.current > 0:
            eta = (elapsed / self.current) * (self.total - self.current)
            eta_str = str(timedelta(seconds=int(eta)))
        else:
            eta_str = "N/A"
        
        bar_length = 40
        filled = int(bar_length * self.current / self.total)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        print(f"\r{self.description}: [{bar}] {percent:.1f}% - ETA: {eta_str}", 
              end='', flush=True)
        
        if self.current >= self.total:
            print()  # New line when complete


# ============================================================================
# LOGGING UTILITIES
# ============================================================================

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration."""
    from config import Config
    
    log_config = Config.LOGGING
    
    # Set log level
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        log_config['format'],
        datefmt=log_config['date_format']
    )
    
    # Setup handlers
    handlers = []
    
    # Console handler
    if log_config.get('console_output', True):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)
    
    # File handler
    if log_file or log_config.get('file'):
        file_path = log_file or log_config['file']
        ensure_directory(Path(file_path).parent)
        
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            file_path,
            maxBytes=log_config.get('max_bytes', 10*1024*1024),
            backupCount=log_config.get('backup_count', 5)
        )
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        handlers=handlers
    )
    
    logger.info(f"Logging configured: level={log_level}")


# ============================================================================
# STATISTICS UTILITIES
# ============================================================================

def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """Calculate basic statistics for a list of values."""
    if not values:
        return {
            'count': 0,
            'min': 0,
            'max': 0,
            'mean': 0,
            'median': 0,
            'std': 0,
            'sum': 0
        }
    
    return {
        'count': len(values),
        'min': min(values),
        'max': max(values),
        'mean': np.mean(values),
        'median': np.median(values),
        'std': np.std(values),
        'sum': sum(values)
    }


def calculate_percentiles(values: List[float], 
                         percentiles: List[int] = [25, 50, 75, 90, 95]) -> Dict[int, float]:
    """Calculate percentiles for a list of values."""
    if not values:
        return {p: 0 for p in percentiles}
    
    return {
        p: np.percentile(values, p) for p in percentiles
    }


# ============================================================================
# OPTIMIZATION UTILITIES
# ============================================================================

def calculate_fitness_score(coverage: float, cost: float, efficiency: float,
                           weights: Optional[Dict[str, float]] = None) -> float:
    """Calculate weighted fitness score."""
    if weights is None:
        weights = {
            'coverage': 0.4,
            'cost': 0.3,
            'efficiency': 0.3
        }
    
    # Normalize cost (lower is better)
    normalized_cost = 1.0 / (1.0 + cost)
    
    score = (
        coverage * weights.get('coverage', 0.4) +
        normalized_cost * weights.get('cost', 0.3) +
        efficiency * weights.get('efficiency', 0.3)
    )
    
    return score * 100


def compare_results(results: List[Dict]) -> Dict:
    """Compare multiple optimization results."""
    comparison = {
        'best_coverage': None,
        'best_cost': None,
        'best_efficiency': None,
        'best_overall': None,
        'rankings': {}
    }
    
    if not results:
        return comparison
    
    # Find best in each category
    comparison['best_coverage'] = max(results, key=lambda r: r.get('coverage', 0))
    comparison['best_cost'] = min(results, key=lambda r: r.get('cost_per_sqft', float('inf')))
    comparison['best_efficiency'] = max(results, key=lambda r: r.get('efficiency', 0))
    
    # Calculate overall scores
    for result in results:
        score = calculate_fitness_score(
            result.get('coverage', 0),
            result.get('cost_per_sqft', 0),
            result.get('efficiency', 0)
        )
        result['overall_score'] = score
    
    comparison['best_overall'] = max(results, key=lambda r: r['overall_score'])
    
    # Create rankings
    for metric in ['coverage', 'cost_per_sqft', 'efficiency', 'overall_score']:
        sorted_results = sorted(
            results,
            key=lambda r: r.get(metric, 0),
            reverse=(metric != 'cost_per_sqft')
        )
        comparison['rankings'][metric] = [
            r.get('strategy', 'unknown') for r in sorted_results
        ]
    
    return comparison


# ============================================================================
# VALIDATION AND ERROR HANDLING
# ============================================================================

class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


def validate_floor_plan(floor_plan) -> List[str]:
    """Validate floor plan for common issues."""
    issues = []
    
    # Check for rooms
    if not floor_plan.rooms:
        issues.append("No rooms detected in floor plan")
    
    # Check room dimensions
    for room in floor_plan.rooms:
        if room.width <= 0 or room.height <= 0:
            issues.append(f"Room {room.id} has invalid dimensions")
        
        if room.area <= 0:
            issues.append(f"Room {room.id} has zero area")
        
        if room.width > 100 or room.height > 100:
            issues.append(f"Room {room.id} has unusually large dimensions")
    
    # Check for overlapping rooms
    for i, room1 in enumerate(floor_plan.rooms):
        for room2 in floor_plan.rooms[i+1:]:
            if room1.rectangle.intersects(room2.rectangle):
                overlap = room1.rectangle.intersection_area(room2.rectangle)
                if overlap > min(room1.area, room2.area) * 0.1:
                    issues.append(f"Rooms {room1.id} and {room2.id} overlap significantly")
    
    return issues


def safe_divide(numerator: float, denominator: float, 
                default: float = 0) -> float:
    """Safe division with default value for division by zero."""
    if denominator == 0:
        return default
    return numerator / denominator

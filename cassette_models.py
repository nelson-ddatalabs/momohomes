"""
Cassette Models for Floor Joist Optimization System
===================================================
Core data structures for cassette-based floor joist placement.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum
import numpy as np
from datetime import datetime


class CassetteSize(Enum):
    """Standard cassette sizes with specifications."""
    # Format: (width, height, area, weight, joist_count, spacing_inches, name)
    CASSETTE_6X6 = (6, 6, 36, 378, 5, 16, "6x6")
    CASSETTE_4X8 = (4, 8, 32, 336, 3, 16, "4x8")
    CASSETTE_8X4 = (8, 4, 32, 336, 5, 16, "8x4")
    CASSETTE_4X6 = (4, 6, 24, 252, 3, 16, "4x6")
    CASSETTE_6X4 = (6, 4, 24, 252, 4, 16, "6x4")
    CASSETTE_4X4 = (4, 4, 16, 168, 3, 16, "4x4")
    # Edge fillers for achieving high coverage
    CASSETTE_2X4 = (2, 4, 8, 84, 2, 16, "2x4")
    CASSETTE_4X2 = (4, 2, 8, 84, 3, 16, "4x2")
    CASSETTE_2X6 = (2, 6, 12, 126, 2, 16, "2x6")
    CASSETTE_6X2 = (6, 2, 12, 126, 4, 16, "6x2")
    
    @property
    def width(self) -> float:
        """Width in feet."""
        return self.value[0]
    
    @property
    def height(self) -> float:
        """Height in feet."""
        return self.value[1]
    
    @property
    def area(self) -> float:
        """Area in square feet."""
        return self.value[2]
    
    @property
    def weight(self) -> float:
        """Weight in pounds."""
        return self.value[3]
    
    @property
    def joist_count(self) -> int:
        """Number of joists in the cassette."""
        return self.value[4]
    
    @property
    def spacing(self) -> int:
        """Joist spacing in inches (on center)."""
        return self.value[5]
    
    @property
    def name(self) -> str:
        """Human-readable name."""
        return self.value[6]
    
    def fits_in_space(self, width: float, height: float) -> bool:
        """Check if cassette fits in given space."""
        return self.width <= width and self.height <= height
    
    def get_dimensions(self) -> Tuple[float, float]:
        """Get dimensions as tuple."""
        return (self.width, self.height)
    
    @classmethod
    def get_main_cassettes(cls) -> List['CassetteSize']:
        """Get primary cassette sizes for main coverage."""
        return [
            cls.CASSETTE_6X6,
            cls.CASSETTE_8X4,
            cls.CASSETTE_4X8,
            cls.CASSETTE_6X4,
            cls.CASSETTE_4X6,
            cls.CASSETTE_4X4,
        ]
    
    @classmethod
    def get_edge_cassettes(cls) -> List['CassetteSize']:
        """Get edge filler cassette sizes."""
        return [
            cls.CASSETTE_4X2,
            cls.CASSETTE_2X4,
            cls.CASSETTE_6X2,
            cls.CASSETTE_2X6,
        ]


@dataclass
class Point:
    """2D point in floor plan coordinate system."""
    x: float
    y: float
    
    def __hash__(self):
        return hash((self.x, self.y))
    
    def __eq__(self, other):
        if not isinstance(other, Point):
            return False
        return abs(self.x - other.x) < 0.01 and abs(self.y - other.y) < 0.01
    
    def distance_to(self, other: 'Point') -> float:
        """Calculate Euclidean distance to another point."""
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def to_tuple(self) -> Tuple[float, float]:
        """Convert to tuple representation."""
        return (self.x, self.y)


@dataclass
class Cassette:
    """Represents a placed cassette in the floor plan."""
    size: CassetteSize
    position: Point
    cassette_id: str
    placement_order: int = 0
    placement_time: Optional[datetime] = None
    
    def __post_init__(self):
        """Initialize computed properties."""
        if self.placement_time is None:
            self.placement_time = datetime.now()
    
    @property
    def x(self) -> float:
        """X coordinate."""
        return self.position.x
    
    @property
    def y(self) -> float:
        """Y coordinate."""
        return self.position.y
    
    @property
    def width(self) -> float:
        """Cassette width."""
        return self.size.width
    
    @property
    def height(self) -> float:
        """Cassette height."""
        return self.size.height
    
    @property
    def area(self) -> float:
        """Cassette area."""
        return self.size.area
    
    @property
    def weight(self) -> float:
        """Cassette weight."""
        return self.size.weight
    
    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Get bounding box (x1, y1, x2, y2)."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)
    
    def overlaps_with(self, other: 'Cassette', tolerance: float = 0.01) -> bool:
        """Check if this cassette overlaps with another."""
        x1, y1, x2, y2 = self.bounds
        ox1, oy1, ox2, oy2 = other.bounds
        
        return not (x2 <= ox1 + tolerance or 
                   ox2 <= x1 + tolerance or
                   y2 <= oy1 + tolerance or
                   oy2 <= y1 + tolerance)
    
    def contains_point(self, point: Point) -> bool:
        """Check if point is inside cassette."""
        return (self.x <= point.x <= self.x + self.width and 
                self.y <= point.y <= self.y + self.height)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'cassette_id': self.cassette_id,
            'size': self.size.name,
            'position': {'x': self.position.x, 'y': self.position.y},
            'dimensions': {'width': self.width, 'height': self.height},
            'area': self.area,
            'weight': self.weight,
            'joist_count': self.size.joist_count,
            'spacing': self.size.spacing,
            'placement_order': self.placement_order
        }


@dataclass
class FloorBoundary:
    """Represents the boundary of the indoor living space."""
    points: List[Point]
    bounding_box: Optional[Tuple[float, float, float, float]] = None
    
    def __post_init__(self):
        """Calculate bounding box."""
        if self.points and not self.bounding_box:
            xs = [p.x for p in self.points]
            ys = [p.y for p in self.points]
            self.bounding_box = (min(xs), min(ys), max(xs), max(ys))
    
    @property
    def width(self) -> float:
        """Width of bounding box."""
        if self.bounding_box:
            return self.bounding_box[2] - self.bounding_box[0]
        return 0
    
    @property
    def height(self) -> float:
        """Height of bounding box."""
        if self.bounding_box:
            return self.bounding_box[3] - self.bounding_box[1]
        return 0
    
    @property
    def area(self) -> float:
        """Approximate area using bounding box."""
        return self.width * self.height
    
    def contains_point(self, point: Point) -> bool:
        """Check if point is inside boundary (simple bounding box check)."""
        if not self.bounding_box:
            return False
        x1, y1, x2, y2 = self.bounding_box
        return x1 <= point.x <= x2 and y1 <= point.y <= y2


@dataclass
class CassetteLayout:
    """Complete cassette layout for a floor plan."""
    floor_boundary: FloorBoundary
    cassettes: List[Cassette] = field(default_factory=list)
    custom_areas: List[Tuple[Point, Point]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_area(self) -> float:
        """Total floor area."""
        return self.floor_boundary.area
    
    @property
    def covered_area(self) -> float:
        """Total area covered by cassettes."""
        return sum(c.area for c in self.cassettes)
    
    @property
    def coverage_percentage(self) -> float:
        """Coverage as percentage."""
        if self.total_area > 0:
            return (self.covered_area / self.total_area) * 100
        return 0
    
    @property
    def uncovered_area(self) -> float:
        """Area not covered by cassettes."""
        return self.total_area - self.covered_area
    
    @property
    def cassette_count(self) -> int:
        """Total number of cassettes."""
        return len(self.cassettes)
    
    @property
    def total_weight(self) -> float:
        """Total weight of all cassettes."""
        return sum(c.weight for c in self.cassettes)
    
    @property
    def total_joist_count(self) -> int:
        """Total number of joists across all cassettes."""
        return sum(c.size.joist_count for c in self.cassettes)
    
    def get_cassette_summary(self) -> Dict[CassetteSize, int]:
        """Get count of each cassette type."""
        summary = {}
        for cassette in self.cassettes:
            if cassette.size not in summary:
                summary[cassette.size] = 0
            summary[cassette.size] += 1
        return summary
    
    def add_cassette(self, cassette: Cassette) -> bool:
        """Add cassette if it doesn't overlap with existing ones."""
        for existing in self.cassettes:
            if cassette.overlaps_with(existing):
                return False
        self.cassettes.append(cassette)
        return True
    
    def validate_weight_limits(self, max_weight_per_cassette: float = 500) -> List[str]:
        """Validate that no cassette exceeds weight limit."""
        violations = []
        for cassette in self.cassettes:
            if cassette.weight > max_weight_per_cassette:
                violations.append(
                    f"Cassette {cassette.cassette_id} ({cassette.size.name}) "
                    f"exceeds weight limit: {cassette.weight} lbs > {max_weight_per_cassette} lbs"
                )
        return violations
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'floor_boundary': {
                'width': self.floor_boundary.width,
                'height': self.floor_boundary.height,
                'area': self.floor_boundary.area
            },
            'total_area': self.total_area,
            'covered_area': self.covered_area,
            'uncovered_area': self.uncovered_area,
            'coverage_percentage': self.coverage_percentage,
            'cassette_count': self.cassette_count,
            'total_weight': self.total_weight,
            'total_joist_count': self.total_joist_count,
            'cassette_summary': {
                size.name: count 
                for size, count in self.get_cassette_summary().items()
            },
            'cassettes': [c.to_dict() for c in self.cassettes],
            'custom_areas': [
                {'start': {'x': p1.x, 'y': p1.y}, 'end': {'x': p2.x, 'y': p2.y}}
                for p1, p2 in self.custom_areas
            ],
            'metadata': self.metadata
        }


@dataclass
class OptimizationParameters:
    """Parameters for cassette optimization."""
    target_coverage: float = 0.94  # 94% minimum coverage
    max_cassette_weight: float = 500  # Maximum weight per cassette in lbs
    allow_overhang: bool = True  # Allow cassettes to extend beyond boundary
    max_overhang: float = 0.5  # Maximum overhang in feet
    prioritize_larger_cassettes: bool = True  # Use larger cassettes first
    use_edge_fillers: bool = True  # Use small cassettes for edges
    optimization_time_limit: float = 10.0  # Maximum optimization time in seconds
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'target_coverage': self.target_coverage,
            'max_cassette_weight': self.max_cassette_weight,
            'allow_overhang': self.allow_overhang,
            'max_overhang': self.max_overhang,
            'prioritize_larger_cassettes': self.prioritize_larger_cassettes,
            'use_edge_fillers': self.use_edge_fillers,
            'optimization_time_limit': self.optimization_time_limit
        }


@dataclass
class OptimizationResult:
    """Results from cassette optimization."""
    layout: CassetteLayout
    success: bool
    optimization_time: float
    algorithm_used: str
    iterations: int = 0
    message: str = ""
    warnings: List[str] = field(default_factory=list)
    
    def get_summary(self) -> str:
        """Generate text summary of results."""
        lines = [
            f"Optimization Algorithm: {self.algorithm_used}",
            f"Success: {'Yes' if self.success else 'No'}",
            f"Time: {self.optimization_time:.2f} seconds",
            f"Iterations: {self.iterations}",
            f"Coverage: {self.layout.coverage_percentage:.1f}%",
            f"Cassettes: {self.layout.cassette_count}",
            f"Total Weight: {self.layout.total_weight:,.0f} lbs",
            f"Custom Area: {self.layout.uncovered_area:.1f} sq ft",
        ]
        
        if self.warnings:
            lines.append("Warnings:")
            for warning in self.warnings[:3]:
                lines.append(f"  - {warning}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'success': self.success,
            'optimization_time': self.optimization_time,
            'algorithm_used': self.algorithm_used,
            'iterations': self.iterations,
            'message': self.message,
            'warnings': self.warnings,
            'layout': self.layout.to_dict(),
            'summary': self.get_summary()
        }
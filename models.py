"""
models.py - Core Data Models for Floor Plan Optimization System
================================================================
Defines all data structures used throughout the system.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Any
from enum import Enum
import numpy as np
from datetime import datetime


class RoomType(Enum):
    """Enumeration of all possible room types."""
    BEDROOM = "bedroom"
    BATHROOM = "bathroom"
    KITCHEN = "kitchen"
    LIVING = "living"
    DINING = "dining"
    FAMILY = "family"
    HALLWAY = "hallway"
    CLOSET = "closet"
    ENTRY = "entry"
    FOYER = "foyer"
    MEDIA = "media"
    OFFICE = "office"
    LAUNDRY = "laundry"
    PANTRY = "pantry"
    GARAGE = "garage"
    PRIMARY_SUITE = "primary_suite"
    UTILITY = "utility"
    OPEN_SPACE = "open_space"
    UNKNOWN = "unknown"


class PanelSize(Enum):
    """Available panel/joist sizes with specifications."""
    # Format: (width, length, area, cost_factor, name)
    PANEL_6X8 = (6, 8, 48, 1.00, "6x8")
    PANEL_6X6 = (6, 6, 36, 1.15, "6x6")
    PANEL_4X6 = (4, 6, 24, 1.35, "4x6")
    PANEL_4X4 = (4, 4, 16, 1.60, "4x4")
    
    @property
    def width(self) -> float:
        """Panel width in feet."""
        return self.value[0]
    
    @property
    def length(self) -> float:
        """Panel length in feet."""
        return self.value[1]
    
    @property
    def area(self) -> float:
        """Panel area in square feet."""
        return self.value[2]
    
    @property
    def cost_factor(self) -> float:
        """Cost multiplier relative to base price."""
        return self.value[3]
    
    @property
    def name(self) -> str:
        """Human-readable panel name."""
        return self.value[4]
    
    def get_dimensions(self, orientation: str = "horizontal") -> Tuple[float, float]:
        """Get width and height based on orientation."""
        if orientation == "horizontal":
            return self.width, self.length
        else:
            return self.length, self.width
    
    def fits_in_space(self, width: float, height: float, orientation: str = "horizontal") -> bool:
        """Check if panel fits in given space."""
        panel_w, panel_h = self.get_dimensions(orientation)
        return panel_w <= width and panel_h <= height


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
    
    def move(self, dx: float, dy: float) -> 'Point':
        """Return new point moved by dx, dy."""
        return Point(self.x + dx, self.y + dy)
    
    def to_tuple(self) -> Tuple[float, float]:
        """Convert to tuple representation."""
        return (self.x, self.y)


@dataclass
class Rectangle:
    """Rectangle representation with position and dimensions."""
    x: float
    y: float
    width: float
    height: float
    
    @property
    def area(self) -> float:
        """Calculate rectangle area."""
        return self.width * self.height
    
    @property
    def perimeter(self) -> float:
        """Calculate rectangle perimeter."""
        return 2 * (self.width + self.height)
    
    @property
    def center(self) -> Point:
        """Get center point of rectangle."""
        return Point(self.x + self.width/2, self.y + self.height/2)
    
    @property
    def corners(self) -> List[Point]:
        """Get all four corner points."""
        return [
            Point(self.x, self.y),
            Point(self.x + self.width, self.y),
            Point(self.x + self.width, self.y + self.height),
            Point(self.x, self.y + self.height)
        ]
    
    @property
    def aspect_ratio(self) -> float:
        """Calculate aspect ratio (width/height)."""
        return self.width / self.height if self.height > 0 else float('inf')
    
    def contains_point(self, point: Point) -> bool:
        """Check if point is inside rectangle."""
        return (self.x <= point.x <= self.x + self.width and 
                self.y <= point.y <= self.y + self.height)
    
    def intersects(self, other: 'Rectangle') -> bool:
        """Check if this rectangle intersects with another."""
        return not (self.x + self.width < other.x or 
                   other.x + other.width < self.x or
                   self.y + self.height < other.y or
                   other.y + other.height < self.y)
    
    def intersection_area(self, other: 'Rectangle') -> float:
        """Calculate intersection area with another rectangle."""
        if not self.intersects(other):
            return 0.0
        
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x + self.width, other.x + other.width)
        y2 = min(self.y + self.height, other.y + other.height)
        
        return max(0, x2 - x1) * max(0, y2 - y1)
    
    def union_area(self, other: 'Rectangle') -> float:
        """Calculate union area with another rectangle."""
        intersection = self.intersection_area(other)
        return self.area + other.area - intersection


@dataclass
class Panel:
    """Represents a placed panel/joist in the floor plan."""
    size: PanelSize
    position: Point
    orientation: str = "horizontal"  # "horizontal" or "vertical"
    room_id: Optional[str] = None
    panel_id: Optional[str] = None
    placement_time: Optional[datetime] = None
    
    def __post_init__(self):
        """Initialize computed properties."""
        if self.panel_id is None:
            import uuid
            self.panel_id = str(uuid.uuid4())[:8]
        if self.placement_time is None:
            self.placement_time = datetime.now()
    
    @property
    def width(self) -> float:
        """Get panel width based on orientation."""
        return self.size.width if self.orientation == "horizontal" else self.size.length
    
    @property
    def length(self) -> float:
        """Get panel length based on orientation."""
        return self.size.length if self.orientation == "horizontal" else self.size.width
    
    @property
    def rectangle(self) -> Rectangle:
        """Get rectangle representation of panel."""
        return Rectangle(self.position.x, self.position.y, self.width, self.length)
    
    @property
    def area(self) -> float:
        """Get panel area."""
        return self.size.area
    
    @property
    def cost(self) -> float:
        """Calculate panel cost."""
        base_price_per_sqft = 1.0  # Base price
        return self.size.area * self.size.cost_factor * base_price_per_sqft
    
    @property
    def corners(self) -> List[Point]:
        """Get corner points of panel."""
        return self.rectangle.corners
    
    def overlaps_with(self, other: 'Panel') -> bool:
        """Check if this panel overlaps with another."""
        return self.rectangle.intersects(other.rectangle)
    
    def to_dict(self) -> Dict:
        """Convert panel to dictionary representation."""
        return {
            'panel_id': self.panel_id,
            'size': self.size.name,
            'position': {'x': self.position.x, 'y': self.position.y},
            'orientation': self.orientation,
            'room_id': self.room_id,
            'width': self.width,
            'length': self.length,
            'area': self.area,
            'cost': self.cost
        }


@dataclass
class Wall:
    """Represents a wall in the floor plan."""
    start: Point
    end: Point
    wall_type: str = "interior"  # "interior", "exterior", "load_bearing"
    thickness: float = 0.5  # Wall thickness in feet
    
    @property
    def length(self) -> float:
        """Calculate wall length."""
        return self.start.distance_to(self.end)
    
    @property
    def midpoint(self) -> Point:
        """Get midpoint of wall."""
        return Point(
            (self.start.x + self.end.x) / 2,
            (self.start.y + self.end.y) / 2
        )
    
    @property
    def is_load_bearing(self) -> bool:
        """Check if wall is load-bearing."""
        return self.wall_type in ["load_bearing", "exterior"]
    
    def is_horizontal(self, tolerance: float = 0.1) -> bool:
        """Check if wall is horizontal."""
        return abs(self.start.y - self.end.y) < tolerance
    
    def is_vertical(self, tolerance: float = 0.1) -> bool:
        """Check if wall is vertical."""
        return abs(self.start.x - self.end.x) < tolerance


@dataclass
class Room:
    """Represents a room in the floor plan."""
    id: str
    type: RoomType
    boundary: List[Point]
    width: float
    height: float
    area: float
    position: Point  # Top-left corner
    name: str = ""
    floor_level: int = 1
    panels: List[Panel] = field(default_factory=list)
    walls: List[Wall] = field(default_factory=list)
    adjacent_rooms: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Structural properties
    is_load_bearing: bool = False
    span_direction: Optional[str] = None  # "horizontal" or "vertical"
    max_span: float = 12.0  # Maximum span in feet
    
    # Coverage tracking
    uncovered_area: float = 0
    coverage_ratio: float = 0
    
    def __post_init__(self):
        """Initialize computed properties."""
        if self.area == 0:
            self.area = self.width * self.height
        self.uncovered_area = self.area
        self.update_coverage()
    
    def update_coverage(self):
        """Update coverage metrics based on placed panels."""
        covered = sum(p.area for p in self.panels)
        self.coverage_ratio = min(1.0, covered / self.area) if self.area > 0 else 0
        self.uncovered_area = max(0, self.area - covered)
    
    @property
    def rectangle(self) -> Rectangle:
        """Get rectangle representation of room."""
        return Rectangle(self.position.x, self.position.y, self.width, self.height)
    
    @property
    def center(self) -> Point:
        """Get center point of room."""
        return self.rectangle.center
    
    @property
    def aspect_ratio(self) -> float:
        """Calculate room aspect ratio."""
        return max(self.width, self.height) / min(self.width, self.height) if min(self.width, self.height) > 0 else float('inf')
    
    @property
    def is_rectangular(self) -> bool:
        """Check if room is rectangular (4 corners)."""
        return len(self.boundary) == 4
    
    @property
    def is_narrow(self) -> bool:
        """Check if room is narrow (hallway-like)."""
        return self.aspect_ratio > 3.0 or min(self.width, self.height) < 5
    
    @property
    def panel_count(self) -> int:
        """Get number of panels in room."""
        return len(self.panels)
    
    @property
    def total_panel_cost(self) -> float:
        """Calculate total cost of panels in room."""
        return sum(p.cost for p in self.panels)
    
    def add_panel(self, panel: Panel) -> bool:
        """Add panel to room if it fits."""
        # Check if panel fits within room boundaries
        panel_rect = panel.rectangle
        room_rect = self.rectangle
        
        if not (room_rect.contains_point(Point(panel_rect.x, panel_rect.y)) and
                room_rect.contains_point(Point(panel_rect.x + panel_rect.width, 
                                              panel_rect.y + panel_rect.height))):
            return False
        
        # Check for overlaps with existing panels
        for existing in self.panels:
            if panel.overlaps_with(existing):
                return False
        
        self.panels.append(panel)
        panel.room_id = self.id
        self.update_coverage()
        return True
    
    def remove_panel(self, panel_id: str) -> bool:
        """Remove panel from room by ID."""
        for i, panel in enumerate(self.panels):
            if panel.panel_id == panel_id:
                del self.panels[i]
                self.update_coverage()
                return True
        return False
    
    def clear_panels(self):
        """Remove all panels from room."""
        self.panels.clear()
        self.update_coverage()
    
    def get_panel_summary(self) -> Dict[PanelSize, int]:
        """Get count of each panel size in room."""
        summary = {}
        for panel in self.panels:
            if panel.size not in summary:
                summary[panel.size] = 0
            summary[panel.size] += 1
        return summary
    
    def to_dict(self) -> Dict:
        """Convert room to dictionary representation."""
        return {
            'id': self.id,
            'type': self.type.value,
            'name': self.name,
            'dimensions': f"{self.width:.1f}x{self.height:.1f}",
            'area': self.area,
            'position': {'x': self.position.x, 'y': self.position.y},
            'panel_count': self.panel_count,
            'coverage_ratio': self.coverage_ratio,
            'uncovered_area': self.uncovered_area,
            'total_cost': self.total_panel_cost,
            'is_load_bearing': self.is_load_bearing,
            'span_direction': self.span_direction,
            'panels': [p.to_dict() for p in self.panels]
        }


@dataclass
class FloorPlan:
    """Represents complete floor plan with all rooms."""
    name: str
    rooms: List[Room]
    walls: List[Wall] = field(default_factory=list)
    total_area: float = 0
    building_dimensions: Optional[Rectangle] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize computed properties."""
        if self.total_area == 0:
            self.total_area = sum(r.area for r in self.rooms)
        
        if self.building_dimensions is None:
            self._calculate_building_dimensions()
    
    def _calculate_building_dimensions(self):
        """Calculate overall building dimensions from rooms."""
        if not self.rooms:
            return
        
        min_x = min(r.position.x for r in self.rooms)
        min_y = min(r.position.y for r in self.rooms)
        max_x = max(r.position.x + r.width for r in self.rooms)
        max_y = max(r.position.y + r.height for r in self.rooms)
        
        self.building_dimensions = Rectangle(
            min_x, min_y, max_x - min_x, max_y - min_y
        )
    
    @property
    def room_count(self) -> int:
        """Get total number of rooms."""
        return len(self.rooms)
    
    @property
    def total_panels(self) -> int:
        """Get total number of panels across all rooms."""
        return sum(r.panel_count for r in self.rooms)
    
    @property
    def total_coverage(self) -> float:
        """Calculate total coverage ratio."""
        covered = sum(r.area * r.coverage_ratio for r in self.rooms)
        return covered / self.total_area if self.total_area > 0 else 0
    
    @property
    def total_uncovered(self) -> float:
        """Calculate total uncovered area."""
        return sum(r.uncovered_area for r in self.rooms)
    
    @property
    def total_cost(self) -> float:
        """Calculate total cost of all panels."""
        return sum(r.total_panel_cost for r in self.rooms)
    
    def get_room_by_id(self, room_id: str) -> Optional[Room]:
        """Find room by ID."""
        for room in self.rooms:
            if room.id == room_id:
                return room
        return None
    
    def get_rooms_by_type(self, room_type: RoomType) -> List[Room]:
        """Get all rooms of specific type."""
        return [r for r in self.rooms if r.type == room_type]
    
    def get_panel_summary(self) -> Dict[PanelSize, int]:
        """Get summary of all panels used."""
        summary = {}
        for room in self.rooms:
            room_summary = room.get_panel_summary()
            for size, count in room_summary.items():
                if size not in summary:
                    summary[size] = 0
                summary[size] += count
        return summary
    
    def to_dict(self) -> Dict:
        """Convert floor plan to dictionary representation."""
        return {
            'name': self.name,
            'total_area': self.total_area,
            'room_count': self.room_count,
            'total_panels': self.total_panels,
            'total_coverage': self.total_coverage,
            'total_uncovered': self.total_uncovered,
            'total_cost': self.total_cost,
            'building_dimensions': {
                'width': self.building_dimensions.width if self.building_dimensions else 0,
                'height': self.building_dimensions.height if self.building_dimensions else 0
            },
            'panel_summary': {
                size.name: count for size, count in self.get_panel_summary().items()
            },
            'rooms': [r.to_dict() for r in self.rooms],
            'metadata': self.metadata
        }


@dataclass
class OptimizationResult:
    """Results from optimization process."""
    floor_plan: FloorPlan
    strategy_used: str
    optimization_time: float
    coverage_ratio: float
    cost_per_sqft: float
    panel_efficiency: float  # Percentage of large panels
    structural_compliance: bool
    violations: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    
    def get_summary(self) -> str:
        """Generate text summary of results."""
        lines = [
            f"Optimization Strategy: {self.strategy_used}",
            f"Time: {self.optimization_time:.2f} seconds",
            f"Coverage: {self.coverage_ratio:.1%}",
            f"Cost: ${self.cost_per_sqft:.2f}/sq ft",
            f"Panel Efficiency: {self.panel_efficiency:.1%}",
            f"Structural Compliance: {'✓' if self.structural_compliance else '✗'}"
        ]
        
        if self.violations:
            lines.append("Violations:")
            for v in self.violations[:3]:
                lines.append(f"  - {v}")
        
        return "\n".join(lines)

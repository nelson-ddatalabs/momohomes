#!/usr/bin/env python3
"""
Manual Input System for Floor Plan Dimensions
==============================================
Allow users to manually input corner and perimeter dimensions.
"""

import json
import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ManualDimension:
    """A manually entered dimension."""
    length_feet: float
    direction: str  # 'E', 'S', 'W', 'N' for East, South, West, North
    description: Optional[str] = None  # e.g., "Living room wall"


@dataclass
class FloorPlanVertex:
    """A vertex in the floor plan polygon."""
    x: float
    y: float
    
    
class ManualInputSystem:
    """System for manual input of floor plan dimensions."""
    
    def __init__(self):
        """Initialize the manual input system."""
        self.dimensions = []
        self.vertices = []
        self.starting_corner = None
        self.area_sqft = 0
        self.perimeter_ft = 0
        
    def get_floor_plan_from_user(self) -> Dict:
        """
        Interactive method to get floor plan dimensions from user.
        
        Returns:
            Dictionary with floor plan data
        """
        print("\n" + "="*70)
        print("FLOOR PLAN MANUAL INPUT SYSTEM")
        print("="*70)
        print("\nThis system will help you input your floor plan dimensions.")
        print("We'll start at one corner and go clockwise around the perimeter.")
        
        # Step 1: Get starting corner
        self.starting_corner = self._get_starting_corner()
        
        # Step 2: Get dimensions
        self.dimensions = self._get_dimensions()
        
        # Step 3: Build polygon
        self.vertices = self._build_polygon()
        
        # Step 4: Validate
        is_valid, error_ft = self._validate_polygon()
        
        # Step 5: Calculate metrics
        self.area_sqft = self._calculate_area()
        self.perimeter_ft = sum(d.length_feet for d in self.dimensions)
        
        # Display results
        self._display_results(is_valid, error_ft)
        
        return {
            'starting_corner': self.starting_corner,
            'dimensions': [(d.length_feet, d.direction, d.description) for d in self.dimensions],
            'vertices': [(v.x, v.y) for v in self.vertices],
            'area_sqft': self.area_sqft,
            'perimeter_ft': self.perimeter_ft,
            'is_closed': is_valid,
            'closure_error_ft': error_ft
        }
    
    def load_preset_luna(self) -> Dict:
        """
        Load preset dimensions for Luna floor plan.
        Based on the floor plan image analysis.
        
        Returns:
            Dictionary with floor plan data
        """
        print("\n" + "="*70)
        print("LOADING LUNA FLOOR PLAN PRESET")
        print("="*70)
        
        self.starting_corner = 'NW'
        
        # Based on Luna.png analysis - approximate dimensions
        # Going clockwise from NW corner
        self.dimensions = [
            # Top edge (going East)
            ManualDimension(7.5, 'E', "Entry"),
            ManualDimension(7.5, 'E', "PWDR"),
            ManualDimension(32.5, 'E', "Living-Dining"),
            ManualDimension(32, 'E', "Patio 2"),
            
            # Right edge (going South)
            ManualDimension(40, 'S', "East wall"),
            
            # Bottom edge (going West)
            ManualDimension(41, 'W', "Media Room"),
            ManualDimension(41, 'W', "Bedroom 4"),
            ManualDimension(40.5, 'W', "Bedroom 3"),
            ManualDimension(41, 'W', "Bedroom 2"),
            ManualDimension(44.5, 'W', "Master Suite"),
            
            # Left edge (going North)
            ManualDimension(45.5, 'N', "West wall including garage"),
        ]
        
        # Build polygon
        self.vertices = self._build_polygon()
        
        # Validate
        is_valid, error_ft = self._validate_polygon()
        
        # Calculate metrics
        self.area_sqft = self._calculate_area()
        self.perimeter_ft = sum(d.length_feet for d in self.dimensions)
        
        print(f"\nLoaded {len(self.dimensions)} dimensions")
        print(f"Total perimeter: {self.perimeter_ft:.1f} ft")
        print(f"Area: {self.area_sqft:.1f} sq ft")
        print(f"Polygon closed: {is_valid} (error: {error_ft:.2f} ft)")
        
        return {
            'starting_corner': self.starting_corner,
            'dimensions': [(d.length_feet, d.direction, d.description) for d in self.dimensions],
            'vertices': [(v.x, v.y) for v in self.vertices],
            'area_sqft': self.area_sqft,
            'perimeter_ft': self.perimeter_ft,
            'is_closed': is_valid,
            'closure_error_ft': error_ft
        }
    
    def _get_starting_corner(self) -> str:
        """Get starting corner from user."""
        print("\n1. SELECT STARTING CORNER")
        print("-" * 40)
        print("Choose your starting corner:")
        print("  NW - Northwest (top-left)")
        print("  NE - Northeast (top-right)")
        print("  SE - Southeast (bottom-right)")
        print("  SW - Southwest (bottom-left)")
        
        while True:
            corner = input("\nEnter corner (NW/NE/SE/SW): ").strip().upper()
            if corner in ['NW', 'NE', 'SE', 'SW']:
                return corner
            print("Invalid input. Please enter NW, NE, SE, or SW.")
    
    def _get_dimensions(self) -> List[ManualDimension]:
        """Get dimensions from user."""
        print("\n2. ENTER DIMENSIONS")
        print("-" * 40)
        print("Enter dimensions going CLOCKWISE from your starting corner.")
        print("For each dimension, specify:")
        print("  - Length in feet (e.g., 32.5)")
        print("  - Direction (E=East/Right, S=South/Down, W=West/Left, N=North/Up)")
        print("  - Optional description (e.g., 'Living room wall')")
        print("\nType 'done' when finished entering all dimensions.")
        
        dimensions = []
        segment_num = 1
        
        while True:
            print(f"\nSegment {segment_num}:")
            
            # Get length
            length_input = input("  Length (feet) or 'done': ").strip()
            if length_input.lower() == 'done':
                if len(dimensions) < 3:
                    print("Need at least 3 dimensions to form a polygon.")
                    continue
                break
            
            try:
                length = float(length_input)
                if length <= 0:
                    print("Length must be positive.")
                    continue
            except ValueError:
                print("Invalid number. Please enter a valid length.")
                continue
            
            # Get direction
            direction = input("  Direction (E/S/W/N): ").strip().upper()
            if direction not in ['E', 'S', 'W', 'N']:
                print("Invalid direction. Use E, S, W, or N.")
                continue
            
            # Get optional description
            description = input("  Description (optional): ").strip()
            if not description:
                description = f"Segment {segment_num}"
            
            dimensions.append(ManualDimension(length, direction, description))
            print(f"  Added: {length} ft going {direction} ({description})")
            segment_num += 1
        
        return dimensions
    
    def _build_polygon(self) -> List[FloorPlanVertex]:
        """Build polygon vertices from dimensions."""
        vertices = []
        
        # Starting position (origin)
        x, y = 0.0, 0.0
        vertices.append(FloorPlanVertex(x, y))
        
        # Direction vectors
        directions = {
            'E': (1, 0),   # East (right)
            'S': (0, -1),  # South (down) - negative Y
            'W': (-1, 0),  # West (left)
            'N': (0, 1)    # North (up) - positive Y
        }
        
        # Build vertices
        for dim in self.dimensions:
            dx, dy = directions[dim.direction]
            x += dx * dim.length_feet
            y += dy * dim.length_feet
            vertices.append(FloorPlanVertex(x, y))
        
        return vertices
    
    def _validate_polygon(self) -> Tuple[bool, float]:
        """
        Validate that polygon closes properly.
        
        Returns:
            Tuple of (is_closed, error_in_feet)
        """
        if len(self.vertices) < 2:
            return False, float('inf')
        
        # Check closure error
        first = self.vertices[0]
        last = self.vertices[-1]
        error = math.sqrt((last.x - first.x)**2 + (last.y - first.y)**2)
        
        # Consider closed if error is less than 1 foot
        is_closed = error < 1.0
        
        return is_closed, error
    
    def _calculate_area(self) -> float:
        """Calculate polygon area using shoelace formula."""
        if len(self.vertices) < 3:
            return 0.0
        
        # Shoelace formula
        area = 0.0
        n = len(self.vertices)
        
        for i in range(n - 1):
            area += self.vertices[i].x * self.vertices[i + 1].y
            area -= self.vertices[i + 1].x * self.vertices[i].y
        
        # Don't close if not already closed
        if self.vertices[-1].x != self.vertices[0].x or self.vertices[-1].y != self.vertices[0].y:
            area += self.vertices[-1].x * self.vertices[0].y
            area -= self.vertices[0].x * self.vertices[-1].y
        
        return abs(area) / 2.0
    
    def _display_results(self, is_valid: bool, error_ft: float):
        """Display input results."""
        print("\n" + "="*70)
        print("FLOOR PLAN SUMMARY")
        print("="*70)
        
        print(f"\nStarting corner: {self.starting_corner}")
        print(f"Number of segments: {len(self.dimensions)}")
        print(f"Total perimeter: {self.perimeter_ft:.1f} ft")
        print(f"Floor area: {self.area_sqft:.1f} sq ft")
        
        if is_valid:
            print(f"✓ Polygon closes properly (error: {error_ft:.3f} ft)")
        else:
            print(f"✗ Polygon does not close (error: {error_ft:.1f} ft)")
            print("  Check your dimensions and directions.")
        
        print("\nDimensions entered:")
        for i, dim in enumerate(self.dimensions, 1):
            print(f"  {i}. {dim.length_feet:.1f} ft {dim.direction} - {dim.description}")
    
    def save_to_file(self, filepath: str):
        """Save floor plan data to JSON file."""
        data = {
            'starting_corner': self.starting_corner,
            'dimensions': [
                {
                    'length_feet': d.length_feet,
                    'direction': d.direction,
                    'description': d.description
                }
                for d in self.dimensions
            ],
            'vertices': [(v.x, v.y) for v in self.vertices],
            'area_sqft': self.area_sqft,
            'perimeter_ft': self.perimeter_ft
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\nFloor plan saved to: {filepath}")
    
    def load_from_file(self, filepath: str) -> Dict:
        """Load floor plan data from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.starting_corner = data['starting_corner']
        self.dimensions = [
            ManualDimension(
                d['length_feet'],
                d['direction'],
                d.get('description', '')
            )
            for d in data['dimensions']
        ]
        self.vertices = [
            FloorPlanVertex(v[0], v[1])
            for v in data['vertices']
        ]
        self.area_sqft = data['area_sqft']
        self.perimeter_ft = data['perimeter_ft']
        
        print(f"Floor plan loaded from: {filepath}")
        return data


def main():
    """Main function for testing."""
    system = ManualInputSystem()
    
    print("\nChoose an option:")
    print("1. Enter floor plan manually")
    print("2. Load Luna preset")
    print("3. Load from file")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == '1':
        floor_plan = system.get_floor_plan_from_user()
        
        # Ask if user wants to save
        save = input("\nSave floor plan to file? (y/n): ").strip().lower()
        if save == 'y':
            filename = input("Enter filename (without extension): ").strip()
            system.save_to_file(f"{filename}.json")
    
    elif choice == '2':
        floor_plan = system.load_preset_luna()
        system.save_to_file("luna_manual.json")
    
    elif choice == '3':
        filename = input("Enter filename to load: ").strip()
        floor_plan = system.load_from_file(filename)
        system._display_results(
            floor_plan['is_closed'],
            floor_plan['closure_error_ft']
        )
    
    else:
        print("Invalid choice.")
        return
    
    print("\n" + "="*70)
    print("Ready for cassette optimization!")
    print("Floor plan data available for processing.")


if __name__ == "__main__":
    main()
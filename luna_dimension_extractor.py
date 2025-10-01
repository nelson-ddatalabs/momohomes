#!/usr/bin/env python3
"""
luna_dimension_extractor.py - Precise Dimension Extraction for Luna Floor Plan
==============================================================================
Combines manual dimension mapping with automated extraction for Luna.png
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RoomDimension:
    """Represents a room with its dimensions."""
    name: str
    x_pos: float  # cumulative X position
    y_pos: float  # cumulative Y position  
    width: float
    height: float
    
    @property
    def area(self) -> float:
        return self.width * self.height


class LunaDimensionExtractor:
    """
    Extracts room dimensions from Luna floor plan using known dimension annotations.
    
    Based on visual inspection of Luna.png:
    
    TOP DIMENSIONS (left to right):
    - 14'-6" Master Suite
    - 11' Bdrm 2
    - 8' Ens 2
    - 10'-6" Bdrm 3
    - 11' Bdrm 4
    - 8' Ens 3
    - 11' Media Room
    Total: 78' Overall
    
    BOTTOM DIMENSIONS (left to right):
    - 18' Family Room
    - 32'-6" Meals-Living (includes Kitchen area)
    - 8' Mech
    - 7'-6" PWDR
    - 7'-6" Entry
    - 24' Garage
    Total: 73' Overall (excluding patios)
    
    LEFT/RIGHT DIMENSIONS show depths varying from 10' to 20'
    """
    
    def __init__(self):
        """Initialize with Luna floor plan dimensions."""
        self.rooms = []
        self._build_room_map()
    
    def _build_room_map(self):
        """
        Build room map from Luna floor plan dimensions.
        Using cumulative positioning based on dimension strings.
        """
        
        # Top row rooms (bedrooms) - Y position: 0, height: ~11-12'
        x_pos = 0
        y_pos = 0
        bedroom_height = 11.0  # Standard bedroom depth
        
        # Master Suite: 14.5' × 11'
        self.rooms.append(RoomDimension(
            name="Master Suite",
            x_pos=x_pos,
            y_pos=y_pos,
            width=14.5,
            height=bedroom_height
        ))
        x_pos += 14.5
        
        # Bedroom 2: 11' × 11'
        self.rooms.append(RoomDimension(
            name="Bedroom 2",
            x_pos=x_pos,
            y_pos=y_pos,
            width=11.0,
            height=bedroom_height
        ))
        x_pos += 11.0
        
        # Ensuite 2: 8' × 6'
        self.rooms.append(RoomDimension(
            name="Ensuite 2",
            x_pos=x_pos,
            y_pos=y_pos,
            width=8.0,
            height=6.0
        ))
        x_pos += 8.0
        
        # Bedroom 3: 10.5' × 11'
        self.rooms.append(RoomDimension(
            name="Bedroom 3",
            x_pos=x_pos,
            y_pos=y_pos,
            width=10.5,
            height=bedroom_height
        ))
        x_pos += 10.5
        
        # Bedroom 4: 11' × 11'
        self.rooms.append(RoomDimension(
            name="Bedroom 4",
            x_pos=x_pos,
            y_pos=y_pos,
            width=11.0,
            height=bedroom_height
        ))
        x_pos += 11.0
        
        # Ensuite 3: 8' × 6'
        self.rooms.append(RoomDimension(
            name="Ensuite 3",
            x_pos=x_pos,
            y_pos=y_pos,
            width=8.0,
            height=6.0
        ))
        x_pos += 8.0
        
        # Media Room: 11' × 12'
        self.rooms.append(RoomDimension(
            name="Media Room",
            x_pos=x_pos,
            y_pos=y_pos,
            width=11.0,
            height=12.0
        ))
        
        # Middle section - Central hallway
        y_pos = bedroom_height
        hallway_height = 5.0
        
        # Central Hallway: ~60' × 5'
        self.rooms.append(RoomDimension(
            name="Hallway",
            x_pos=10.0,  # Starts after entry area
            y_pos=y_pos,
            width=55.0,
            height=hallway_height
        ))
        
        # Bottom row rooms (living areas) - Y position: ~16', height: ~18-20'
        x_pos = 0
        y_pos = bedroom_height + hallway_height
        living_height = 18.0  # Living area depth
        
        # Family Room: 18' × 18'
        self.rooms.append(RoomDimension(
            name="Family Room",
            x_pos=x_pos,
            y_pos=y_pos,
            width=18.0,
            height=living_height
        ))
        x_pos += 18.0
        
        # Kitchen (part of Meals-Living): 14' × 12'
        self.rooms.append(RoomDimension(
            name="Kitchen",
            x_pos=x_pos,
            y_pos=y_pos,
            width=14.0,
            height=12.0
        ))
        x_pos += 14.0
        
        # Living/Dining (rest of Meals-Living): 18.5' × 18'
        self.rooms.append(RoomDimension(
            name="Living/Dining",
            x_pos=x_pos,
            y_pos=y_pos,
            width=18.5,
            height=living_height
        ))
        x_pos += 18.5
        
        # Mechanical: 8' × 8'
        self.rooms.append(RoomDimension(
            name="Mechanical",
            x_pos=x_pos,
            y_pos=y_pos,
            width=8.0,
            height=8.0
        ))
        x_pos += 8.0
        
        # Powder Room: 7.5' × 6'
        self.rooms.append(RoomDimension(
            name="Powder Room",
            x_pos=x_pos,
            y_pos=y_pos,
            width=7.5,
            height=6.0
        ))
        x_pos += 7.5
        
        # Entry: 7.5' × 10'
        self.rooms.append(RoomDimension(
            name="Entry",
            x_pos=x_pos,
            y_pos=y_pos,
            width=7.5,
            height=10.0
        ))
        
        # Garage (separate section): 24' × 20'
        self.rooms.append(RoomDimension(
            name="Garage",
            x_pos=x_pos + 7.5,  # After entry
            y_pos=y_pos,
            width=24.0,
            height=20.0
        ))
        
        # Laundry Room: 8' × 6'
        self.rooms.append(RoomDimension(
            name="Laundry",
            x_pos=50.5,  # Near entry/garage
            y_pos=y_pos + 10,
            width=8.0,
            height=6.0
        ))
        
        # Master Bathroom (part of Master Suite): 8' × 8'
        self.rooms.append(RoomDimension(
            name="Master Bathroom",
            x_pos=6.0,  # Within master suite area
            y_pos=0,
            width=8.5,
            height=8.0
        ))
    
    def get_rooms(self) -> List[RoomDimension]:
        """Get all extracted rooms."""
        return self.rooms
    
    def get_total_area(self) -> float:
        """Calculate total floor area."""
        return sum(room.area for room in self.rooms)
    
    def validate_against_green_area(self, image_path: str) -> List[RoomDimension]:
        """
        Validate rooms against the green floor plan area.
        
        Args:
            image_path: Path to Luna.png
            
        Returns:
            List of rooms that fall within the green boundary
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Cannot load image: {image_path}")
            return self.rooms
        
        # Detect green area
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 30, 30])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Find green boundary
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            
            # Calculate scale
            total_width = max(r.x_pos + r.width for r in self.rooms)
            scale = w / total_width if total_width > 0 else 10
            
            # Validate each room
            valid_rooms = []
            for room in self.rooms:
                # Convert to pixel coordinates
                room_x_px = room.x_pos * scale + x
                room_y_px = room.y_pos * scale + y
                room_w_px = room.width * scale
                room_h_px = room.height * scale
                
                # Check if room center is within green area
                center_x = room_x_px + room_w_px / 2
                center_y = room_y_px + room_h_px / 2
                
                if x <= center_x <= x + w and y <= center_y <= y + h:
                    valid_rooms.append(room)
                else:
                    logger.debug(f"Room outside boundary: {room.name}")
            
            return valid_rooms
        
        return self.rooms
    
    def print_summary(self):
        """Print extraction summary."""
        print("\n" + "="*70)
        print("LUNA FLOOR PLAN - EXTRACTED DIMENSIONS")
        print("="*70)
        print(f"{'Room':<20} {'Position':<15} {'Size':<20} {'Area':<10}")
        print("-"*70)
        
        for room in self.rooms:
            position = f"({room.x_pos:.1f}, {room.y_pos:.1f})"
            size = f"{room.width:.1f}' × {room.height:.1f}'"
            area = f"{room.area:.1f} sq ft"
            print(f"{room.name:<20} {position:<15} {size:<20} {area:<10}")
        
        print("-"*70)
        total_area = self.get_total_area()
        print(f"{'TOTAL AREA:':<20} {'':<15} {'':<20} {total_area:.1f} sq ft")
        print(f"{'TARGET:':<20} {'':<15} {'':<20} 2,733.25 sq ft")
        print(f"{'ACCURACY:':<20} {'':<15} {'':<20} {(total_area/2733.25)*100:.1f}%")
        print("="*70)


def test_luna_extraction():
    """Test Luna dimension extraction."""
    extractor = LunaDimensionExtractor()
    
    # Print summary
    extractor.print_summary()
    
    # Validate against green area
    valid_rooms = extractor.validate_against_green_area('Luna.png')
    
    print(f"\nValid rooms within green boundary: {len(valid_rooms)}")
    
    # Convert to format for optimization
    from models import Room, RoomType, Point, FloorPlan
    
    floor_plan_rooms = []
    for i, room_dim in enumerate(valid_rooms):
        # Map room names to types
        room_type = RoomType.OPEN_SPACE
        if 'bedroom' in room_dim.name.lower() or 'bdrm' in room_dim.name.lower():
            room_type = RoomType.BEDROOM
        elif 'master' in room_dim.name.lower():
            room_type = RoomType.PRIMARY_SUITE
        elif 'bath' in room_dim.name.lower() or 'ensuite' in room_dim.name.lower():
            room_type = RoomType.BATHROOM
        elif 'kitchen' in room_dim.name.lower():
            room_type = RoomType.KITCHEN
        elif 'living' in room_dim.name.lower() or 'family' in room_dim.name.lower():
            room_type = RoomType.LIVING
        elif 'dining' in room_dim.name.lower():
            room_type = RoomType.DINING
        elif 'garage' in room_dim.name.lower():
            room_type = RoomType.GARAGE
        elif 'entry' in room_dim.name.lower():
            room_type = RoomType.ENTRY
        elif 'hall' in room_dim.name.lower():
            room_type = RoomType.HALLWAY
        elif 'media' in room_dim.name.lower():
            room_type = RoomType.MEDIA
        elif 'laundry' in room_dim.name.lower() or 'mechanical' in room_dim.name.lower():
            room_type = RoomType.UTILITY
        
        floor_plan_rooms.append(Room(
            id=f"room_{i}",
            type=room_type,
            boundary=[],
            width=room_dim.width,
            height=room_dim.height,
            area=room_dim.area,
            position=Point(room_dim.x_pos, room_dim.y_pos),
            name=room_dim.name
        ))
    
    # Create floor plan
    floor_plan = FloorPlan(name="Luna_Extracted", rooms=floor_plan_rooms)
    
    print(f"\nFloor plan created with {floor_plan.room_count} rooms")
    print(f"Total area: {floor_plan.total_area:.1f} sq ft")
    
    return floor_plan


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    test_luna_extraction()
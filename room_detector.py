"""
Room Detection Module
====================
Detects room names and indoor spaces using OCR.
"""

import cv2
import numpy as np
import pytesseract
from typing import List, Dict, Tuple, Optional
import re
import logging

logger = logging.getLogger(__name__)


class RoomDetector:
    """Detects rooms and indoor spaces from floor plan images."""
    
    # Room keywords to identify indoor spaces
    ROOM_KEYWORDS = [
        'bdrm', 'bedroom', 'bed',
        'bath', 'bathroom', 'wc', 'toilet',
        'kitchen', 'kit',
        'living', 'family', 'great',
        'dining', 'eating',
        'media', 'theater', 'rec',
        'office', 'study', 'den',
        'entry', 'foyer', 'hall',
        'closet', 'storage', 'utility',
        'laundry', 'mud',
        'master', 'suite',
        'nook', 'breakfast',
        'pwdr', 'powder'
    ]
    
    # Outdoor space keywords to exclude
    OUTDOOR_KEYWORDS = [
        'patio', 'deck', 'porch',
        'garage', 'carport',
        'balcony', 'terrace',
        'yard', 'garden'
    ]
    
    def __init__(self):
        """Initialize room detector."""
        self.indoor_rooms = []
        self.outdoor_spaces = []
        
    def detect_rooms(self, image: np.ndarray) -> Dict:
        """
        Detect all rooms in the floor plan using OCR.
        
        Args:
            image: Floor plan image
            
        Returns:
            Dictionary with room detection results
        """
        # Convert to grayscale for OCR
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Enhance for OCR
        enhanced = self._enhance_for_ocr(gray)
        
        # Run OCR with bounding boxes
        ocr_data = pytesseract.image_to_data(enhanced, output_type=pytesseract.Output.DICT)
        
        # Parse OCR results
        rooms = self._parse_ocr_results(ocr_data, image.shape)
        
        # Classify rooms
        classified = self._classify_rooms(rooms)
        
        return classified
    
    def _enhance_for_ocr(self, gray: np.ndarray) -> np.ndarray:
        """Enhance image for better OCR results."""
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Denoise
        denoised = cv2.medianBlur(binary, 3)
        
        return denoised
    
    def _parse_ocr_results(self, ocr_data: Dict, image_shape: Tuple) -> List[Dict]:
        """Parse OCR results to extract room information."""
        rooms = []
        height, width = image_shape[:2]
        
        n_boxes = len(ocr_data['text'])
        current_room = []
        
        for i in range(n_boxes):
            text = ocr_data['text'][i].strip()
            if not text:
                # If we have accumulated text, save it as a room
                if current_room:
                    room_text = ' '.join([r['text'] for r in current_room])
                    if len(room_text) > 2:  # Minimum length check
                        # Calculate bounding box for the room name
                        min_x = min(r['x'] for r in current_room)
                        min_y = min(r['y'] for r in current_room)
                        max_x = max(r['x'] + r['width'] for r in current_room)
                        max_y = max(r['y'] + r['height'] for r in current_room)
                        
                        rooms.append({
                            'text': room_text,
                            'bbox': (min_x, min_y, max_x, max_y),
                            'center': ((min_x + max_x) // 2, (min_y + max_y) // 2),
                            'confidence': np.mean([r['conf'] for r in current_room])
                        })
                    current_room = []
                continue
            
            # Get confidence
            conf = int(ocr_data['conf'][i])
            if conf < 30:  # Skip low confidence
                continue
            
            # Get bounding box
            x, y, w, h = (
                ocr_data['left'][i],
                ocr_data['top'][i],
                ocr_data['width'][i],
                ocr_data['height'][i]
            )
            
            # Add to current room
            current_room.append({
                'text': text,
                'x': x,
                'y': y,
                'width': w,
                'height': h,
                'conf': conf
            })
        
        # Don't forget the last room
        if current_room:
            room_text = ' '.join([r['text'] for r in current_room])
            if len(room_text) > 2:
                min_x = min(r['x'] for r in current_room)
                min_y = min(r['y'] for r in current_room)
                max_x = max(r['x'] + r['width'] for r in current_room)
                max_y = max(r['y'] + r['height'] for r in current_room)
                
                rooms.append({
                    'text': room_text,
                    'bbox': (min_x, min_y, max_x, max_y),
                    'center': ((min_x + max_x) // 2, (min_y + max_y) // 2),
                    'confidence': np.mean([r['conf'] for r in current_room])
                })
        
        return rooms
    
    def _classify_rooms(self, rooms: List[Dict]) -> Dict:
        """Classify rooms as indoor or outdoor spaces."""
        indoor = []
        outdoor = []
        unclassified = []
        
        for room in rooms:
            text_lower = room['text'].lower()
            
            # Check for room numbers (e.g., "Bdrm 3", "Room 4")
            has_room_number = bool(re.search(r'\b\d+\b', text_lower))
            
            # Check for indoor keywords
            is_indoor = any(keyword in text_lower for keyword in self.ROOM_KEYWORDS)
            
            # Check for outdoor keywords
            is_outdoor = any(keyword in text_lower for keyword in self.OUTDOOR_KEYWORDS)
            
            # Classify
            if is_indoor and not is_outdoor:
                room['type'] = 'indoor'
                indoor.append(room)
                logger.debug(f"Indoor room detected: {room['text']}")
            elif is_outdoor:
                room['type'] = 'outdoor'
                outdoor.append(room)
                logger.debug(f"Outdoor space detected: {room['text']}")
            else:
                # Check if it might be a room with just a number
                if has_room_number and len(text_lower.split()) <= 3:
                    room['type'] = 'indoor'
                    indoor.append(room)
                    logger.debug(f"Numbered room detected: {room['text']}")
                else:
                    room['type'] = 'unclassified'
                    unclassified.append(room)
        
        return {
            'indoor_rooms': indoor,
            'outdoor_spaces': outdoor,
            'unclassified': unclassified,
            'total_detected': len(rooms)
        }
    
    def get_indoor_mask_from_color(self, image: np.ndarray) -> np.ndarray:
        """
        Get indoor space mask from green color detection.
        
        Args:
            image: Floor plan image (BGR)
            
        Returns:
            Binary mask of indoor spaces
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define green range (adjust based on actual floor plans)
        # Hue range for green: roughly 40-80 degrees (out of 180 in OpenCV)
        lower_green = np.array([35, 30, 30])  # Lower saturation to catch lighter greens
        upper_green = np.array([85, 255, 255])
        
        # Create mask
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Fill holes
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Find largest contour (main floor area)
            largest = max(contours, key=cv2.contourArea)
            mask_filled = np.zeros_like(mask)
            cv2.drawContours(mask_filled, [largest], -1, 255, -1)
            mask = mask_filled
        
        return mask
    
    def validate_indoor_spaces(self, image: np.ndarray, room_detections: Dict) -> Dict:
        """
        Validate indoor spaces using both color and OCR.
        
        Args:
            image: Floor plan image
            room_detections: Room detection results
            
        Returns:
            Validated indoor space information
        """
        # Get color-based mask
        color_mask = self.get_indoor_mask_from_color(image)
        
        # Validate room positions against color mask
        validated_rooms = []
        for room in room_detections['indoor_rooms']:
            center_x, center_y = room['center']
            
            # Check if room center is in green area
            if 0 <= center_y < color_mask.shape[0] and 0 <= center_x < color_mask.shape[1]:
                if color_mask[center_y, center_x] > 0:
                    room['validated'] = True
                    validated_rooms.append(room)
                    logger.debug(f"Validated indoor room: {room['text']}")
                else:
                    room['validated'] = False
                    logger.warning(f"Room '{room['text']}' not in green area")
        
        # Calculate indoor boundary from mask
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get largest contour as main indoor area
            main_contour = max(contours, key=cv2.contourArea)
            
            # Simplify contour
            epsilon = 0.01 * cv2.arcLength(main_contour, True)
            simplified = cv2.approxPolyDP(main_contour, epsilon, True)
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(main_contour)
            
            return {
                'indoor_mask': color_mask,
                'main_contour': simplified,
                'bounding_box': (x, y, w, h),
                'validated_rooms': validated_rooms,
                'area_pixels': cv2.contourArea(main_contour)
            }
        
        return {
            'indoor_mask': color_mask,
            'main_contour': None,
            'bounding_box': None,
            'validated_rooms': validated_rooms,
            'area_pixels': 0
        }
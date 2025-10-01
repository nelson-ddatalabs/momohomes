"""
Floor Plan Processor Module
===========================
Orchestrates the extraction pipeline for processing floor plan images.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from cassette_models import FloorBoundary
from dimension_extractor import DimensionExtractor, Dimension
from boundary_detector import BoundaryDetector
from improved_floor_extractor import ImprovedFloorExtractor
from config_cassette import CassetteConfig

logger = logging.getLogger(__name__)


@dataclass
class ProcessedFloorPlan:
    """Result of floor plan processing."""
    boundary: FloorBoundary
    dimensions: Dict[str, list]
    scale: float  # pixels per foot
    image_path: str
    width_feet: float
    height_feet: float
    area_sqft: float
    
    def summary(self) -> str:
        """Get summary string."""
        return (f"Floor Plan: {self.width_feet:.1f}' x {self.height_feet:.1f}' "
                f"({self.area_sqft:.1f} sq ft)")


class FloorPlanProcessor:
    """Processes floor plan images to extract boundary and dimensions."""
    
    def __init__(self, use_improved=True):
        """Initialize processor."""
        self.config = CassetteConfig()
        self.use_improved = use_improved
        
        if use_improved:
            self.improved_extractor = ImprovedFloorExtractor()
        else:
            self.dimension_extractor = DimensionExtractor()
            self.boundary_detector = BoundaryDetector()
        
    def process(self, image_path: str) -> ProcessedFloorPlan:
        """
        Process a floor plan image to extract all necessary information.
        
        Args:
            image_path: Path to floor plan image
            
        Returns:
            ProcessedFloorPlan with extracted information
        """
        logger.info(f"Processing floor plan: {image_path}")
        
        # Validate image exists
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Use improved extraction if enabled
        if self.use_improved:
            return self._process_improved(image_path)
        else:
            return self._process_legacy(image_path)
    
    def _process_improved(self, image_path: str) -> ProcessedFloorPlan:
        """
        Process using improved extraction system.
        
        Args:
            image_path: Path to floor plan image
            
        Returns:
            ProcessedFloorPlan with extracted information
        """
        # Use improved extraction
        extraction = self.improved_extractor.extract(image_path)
        
        # Validate extraction
        is_valid, warnings = self.improved_extractor.validate_extraction(extraction)
        if not is_valid:
            logger.warning(f"Extraction validation issues: {warnings}")
        
        # Convert dimension details to legacy format if needed
        dimensions = {}
        for side in ['top', 'bottom', 'left', 'right']:
            dimensions[side] = extraction['dimensions']['details'].get(side, [])
        
        # Create result
        result = ProcessedFloorPlan(
            boundary=extraction['indoor_space']['boundary'],
            dimensions=dimensions,
            scale=1.0,  # Already in feet
            image_path=image_path,
            width_feet=extraction['dimensions']['width_feet'],
            height_feet=extraction['dimensions']['height_feet'],
            area_sqft=extraction['dimensions']['area_sqft']
        )
        
        logger.info(f"Processing complete (improved): {result.summary()}")
        logger.info(f"Extraction quality: {extraction['extraction_quality']['quality']} "
                   f"(score: {extraction['extraction_quality']['score']}/100)")
        
        return result
    
    def _process_legacy(self, image_path: str) -> ProcessedFloorPlan:
        """
        Process using legacy extraction (fallback).
        
        Args:
            image_path: Path to floor plan image
            
        Returns:
            ProcessedFloorPlan with extracted information
        """
        # Check if it's Luna floor plan for special handling
        path = Path(image_path)
        is_luna = 'luna' in path.stem.lower()
        
        # Step 1: Extract dimensions
        if is_luna:
            dimensions = self.dimension_extractor.extract_from_luna(image_path)
        else:
            dimensions = self.dimension_extractor.extract_dimensions(image_path)
        
        # Step 2: Calculate scale from dimensions
        scale = self._calculate_scale_from_dimensions(dimensions, image_path)
        
        # Step 3: Detect boundary
        if is_luna:
            boundary = self.boundary_detector.detect_luna_boundary(image_path)
        else:
            boundary = self.boundary_detector.detect_boundary(image_path, scale)
        
        # Step 4: Refine boundary using dimensions
        boundary = self.boundary_detector.refine_boundary(boundary, dimensions)
        
        # Step 5: Validate boundary
        if not self.boundary_detector.validate_boundary(boundary):
            logger.warning("Boundary validation failed, using as-is")
        
        # Step 6: Calculate final metrics
        width_feet = boundary.width
        height_feet = boundary.height
        area_sqft = boundary.area
        
        # Create result
        result = ProcessedFloorPlan(
            boundary=boundary,
            dimensions=dimensions,
            scale=scale,
            image_path=image_path,
            width_feet=width_feet,
            height_feet=height_feet,
            area_sqft=area_sqft
        )
        
        logger.info(f"Processing complete (legacy): {result.summary()}")
        
        return result
    
    def _calculate_scale_from_dimensions(self, dimensions: Dict[str, list], 
                                        image_path: str) -> float:
        """Calculate pixels per foot scale from dimensions."""
        import cv2
        
        # Load image to get size
        image = cv2.imread(image_path)
        if image is None:
            return self.config.IMAGE_PROCESSING['default_pixels_per_foot']
        
        height, width = image.shape[:2]
        
        scales = []
        
        # Calculate from horizontal dimensions
        for edge in ['top', 'bottom']:
            if edge in dimensions and dimensions[edge]:
                total_feet = sum(d.value_feet for d in dimensions[edge])
                if total_feet > 0:
                    scales.append(width / total_feet)
        
        # Calculate from vertical dimensions  
        for edge in ['left', 'right']:
            if edge in dimensions and dimensions[edge]:
                total_feet = sum(d.value_feet for d in dimensions[edge])
                if total_feet > 0:
                    scales.append(height / total_feet)
        
        if scales:
            import numpy as np
            scale = np.median(scales)
            logger.info(f"Calculated scale: {scale:.2f} pixels per foot")
        else:
            scale = self.config.IMAGE_PROCESSING['default_pixels_per_foot']
            logger.warning(f"Using default scale: {scale} pixels per foot")
        
        # Clamp to reasonable range
        min_scale = self.config.IMAGE_PROCESSING['min_pixels_per_foot']
        max_scale = self.config.IMAGE_PROCESSING['max_pixels_per_foot']
        scale = max(min_scale, min(scale, max_scale))
        
        return scale
    
    def process_batch(self, image_paths: list) -> list:
        """
        Process multiple floor plan images.
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List of ProcessedFloorPlan objects
        """
        results = []
        
        for i, path in enumerate(image_paths, 1):
            logger.info(f"Processing {i}/{len(image_paths)}: {path}")
            try:
                result = self.process(path)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {path}: {e}")
                continue
        
        logger.info(f"Batch processing complete: {len(results)}/{len(image_paths)} successful")
        
        return results
    
    def validate_for_cassettes(self, processed: ProcessedFloorPlan) -> Tuple[bool, list]:
        """
        Validate processed floor plan for cassette optimization.
        
        Args:
            processed: Processed floor plan
            
        Returns:
            Tuple of (is_valid, list_of_warnings)
        """
        warnings = []
        
        # Check area limits
        min_area = self.config.VALIDATION['min_floor_area']
        max_area = self.config.VALIDATION['max_floor_area']
        
        if processed.area_sqft < min_area:
            warnings.append(f"Floor area too small: {processed.area_sqft:.1f} < {min_area} sq ft")
        
        if processed.area_sqft > max_area:
            warnings.append(f"Floor area too large: {processed.area_sqft:.1f} > {max_area} sq ft")
        
        # Check dimensions
        if processed.width_feet <= 0 or processed.height_feet <= 0:
            warnings.append("Invalid dimensions detected")
        
        # Check aspect ratio
        if processed.width_feet > 0 and processed.height_feet > 0:
            aspect = max(processed.width_feet, processed.height_feet) / \
                    min(processed.width_feet, processed.height_feet)
            if aspect > 5:
                warnings.append(f"Unusual aspect ratio: {aspect:.1f}")
        
        # Check if area is suitable for cassettes
        smallest_cassette = 16  # 4x4 cassette
        if processed.area_sqft < smallest_cassette:
            warnings.append("Floor area smaller than smallest cassette")
        
        # Check boundary points
        if not processed.boundary.points or len(processed.boundary.points) < 3:
            warnings.append("Invalid boundary detected")
        
        is_valid = len(warnings) == 0
        
        if warnings:
            logger.warning(f"Validation warnings: {warnings}")
        
        return is_valid, warnings
    
    def get_cassette_grid_size(self, processed: ProcessedFloorPlan) -> Tuple[int, int]:
        """
        Calculate optimal grid size for cassette placement.
        
        Args:
            processed: Processed floor plan
            
        Returns:
            Tuple of (columns, rows) for 6x6 cassette grid
        """
        # Base grid on 6x6 cassettes
        cassette_size = 6.0
        
        cols = int(processed.width_feet / cassette_size)
        rows = int(processed.height_feet / cassette_size)
        
        # Ensure at least 1x1 grid
        cols = max(1, cols)
        rows = max(1, rows)
        
        logger.info(f"Cassette grid size: {cols} x {rows} (6x6 cassettes)")
        
        return cols, rows
    
    def estimate_cassette_count(self, processed: ProcessedFloorPlan) -> Dict[str, int]:
        """
        Estimate cassette counts for quick feasibility check.
        
        Args:
            processed: Processed floor plan
            
        Returns:
            Dictionary with estimated counts by cassette type
        """
        estimates = {}
        
        # Get grid size for 6x6 cassettes
        cols, rows = self.get_cassette_grid_size(processed)
        
        # Estimate main cassettes (6x6)
        estimates['6x6'] = cols * rows
        
        # Estimate edge cassettes
        remaining_width = processed.width_feet - (cols * 6)
        remaining_height = processed.height_feet - (rows * 6)
        
        if remaining_width >= 4:
            estimates['4x6'] = rows
        elif remaining_width >= 2:
            estimates['2x6'] = rows
        
        if remaining_height >= 4:
            estimates['6x4'] = cols
        elif remaining_height >= 2:
            estimates['6x2'] = cols
        
        # Corner piece if needed
        if remaining_width >= 2 and remaining_height >= 2:
            if remaining_width >= 4 and remaining_height >= 4:
                estimates['4x4'] = 1
            else:
                estimates['2x4'] = 1
        
        # Calculate totals
        total_cassettes = sum(estimates.values())
        total_area = sum(
            count * self._get_cassette_area(size)
            for size, count in estimates.items()
        )
        coverage = (total_area / processed.area_sqft) * 100 if processed.area_sqft > 0 else 0
        
        logger.info(f"Estimated {total_cassettes} cassettes for {coverage:.1f}% coverage")
        
        return estimates
    
    def _get_cassette_area(self, size_str: str) -> float:
        """Get area for cassette size string."""
        size_map = {
            '6x6': 36, '4x8': 32, '8x4': 32,
            '4x6': 24, '6x4': 24, '4x4': 16,
            '2x4': 8, '4x2': 8, '2x6': 12, '6x2': 12
        }
        return size_map.get(size_str, 0)
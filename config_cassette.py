"""
Cassette System Configuration
==============================
Central configuration for cassette-based floor joist optimization system.
"""

from pathlib import Path
from typing import Dict, List, Tuple


class CassetteConfig:
    """Configuration settings for cassette optimization system."""
    
    # Base directory
    BASE_DIR = Path(__file__).parent
    
    # Data directories
    DATA_DIR = BASE_DIR / "data"
    OUTPUT_DIR = BASE_DIR / "output"
    TEMP_DIR = BASE_DIR / "temp"
    
    # Ensure directories exist
    for dir_path in [DATA_DIR, OUTPUT_DIR, TEMP_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # ============================================================================
    # CASSETTE SPECIFICATIONS
    # ============================================================================
    
    # Weight calculation: 10.5 lbs per square foot
    WEIGHT_PER_SQFT = 10.5
    
    # Maximum weight per cassette for handling (lbs)
    MAX_CASSETTE_WEIGHT = 500.0
    
    # Standard cassette sizes (width, height, joist_count, spacing_inches)
    CASSETTE_SPECS = {
        '6x6': {'width': 6, 'height': 6, 'joist_count': 5, 'spacing': 16},
        '4x8': {'width': 4, 'height': 8, 'joist_count': 3, 'spacing': 16},
        '8x4': {'width': 8, 'height': 4, 'joist_count': 5, 'spacing': 16},
        '4x6': {'width': 4, 'height': 6, 'joist_count': 3, 'spacing': 16},
        '6x4': {'width': 6, 'height': 4, 'joist_count': 4, 'spacing': 16},
        '4x4': {'width': 4, 'height': 4, 'joist_count': 3, 'spacing': 16},
        # Edge fillers
        '2x4': {'width': 2, 'height': 4, 'joist_count': 2, 'spacing': 16},
        '4x2': {'width': 4, 'height': 2, 'joist_count': 3, 'spacing': 16},
        '2x6': {'width': 2, 'height': 6, 'joist_count': 2, 'spacing': 16},
        '6x2': {'width': 6, 'height': 2, 'joist_count': 4, 'spacing': 16},
    }
    
    # Joist specifications (depth_inches, max_span_feet)
    JOIST_SPECS = {
        '2x6': {'depth': 6, 'max_span': 10, 'weight_per_ft': 2.0},
        '2x8': {'depth': 8, 'max_span': 12, 'weight_per_ft': 2.5},
        '2x10': {'depth': 10, 'max_span': 14, 'weight_per_ft': 3.0},
        '2x12': {'depth': 12, 'max_span': 16, 'weight_per_ft': 3.5},
    }
    
    # ============================================================================
    # OPTIMIZATION PARAMETERS
    # ============================================================================
    
    OPTIMIZATION = {
        # Coverage targets
        'min_coverage': 0.94,  # 94% minimum
        'target_coverage': 0.96,  # 96% target
        'ideal_coverage': 0.98,  # 98% ideal
        
        # Custom work limits
        'max_custom_percentage': 0.08,  # Maximum 8% custom work
        'min_custom_area': 4.0,  # Minimum area to consider for custom work (sq ft)
        
        # Cassette placement
        'allow_overhang': True,  # Allow cassettes to extend beyond boundary
        'max_overhang': 0.5,  # Maximum overhang in feet
        'cassette_gap': 0.0,  # Gap between cassettes (0 = no gap)
        
        # Algorithm settings
        'use_main_cassettes_first': True,  # Prioritize larger cassettes
        'use_edge_fillers': True,  # Use small cassettes for edges
        'allow_rotation': True,  # Allow cassette rotation
        'optimization_time_limit': 10.0,  # Maximum time in seconds
        
        # Placement strategies
        'strategies': ['grid', 'staggered', 'hybrid'],
        'default_strategy': 'hybrid',
        
        # Local search parameters
        'local_search_iterations': 10,
        'local_search_enabled': True,
        
        # Genetic algorithm parameters (optional)
        'ga_population_size': 50,
        'ga_generations': 100,
        'ga_mutation_rate': 0.1,
        'ga_crossover_rate': 0.8,
    }
    
    # ============================================================================
    # IMAGE PROCESSING CONFIGURATION
    # ============================================================================
    
    IMAGE_PROCESSING = {
        # Supported formats
        'supported_formats': ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'],
        
        # OCR settings
        'ocr_confidence_threshold': 0.7,
        'dimension_regex_patterns': [
            r"(\d+)'[-\s]?(\d+)\"?",  # Matches: 14'-6" or 14' 6"
            r"(\d+)\.(\d+)'",  # Matches: 14.5'
            r"(\d+)'",  # Matches: 14'
        ],
        
        # Boundary detection
        'boundary_color_ranges': {
            'green': {
                'lower_hsv': [35, 30, 30],
                'upper_hsv': [85, 255, 255],
            },
            'gray': {
                'lower_hsv': [0, 0, 50],
                'upper_hsv': [180, 30, 200],
            }
        },
        
        # Scale detection
        'default_pixels_per_foot': 50,
        'min_pixels_per_foot': 10,
        'max_pixels_per_foot': 200,
        
        # Contour processing
        'min_contour_area': 1000,  # Minimum contour area in pixels
        'epsilon_factor': 0.02,  # For polygon approximation
    }
    
    # ============================================================================
    # VISUALIZATION CONFIGURATION
    # ============================================================================
    
    VISUALIZATION = {
        # Figure settings
        'figure_size': (16, 10),
        'dpi': 150,
        'grid_alpha': 0.3,
        
        # Colors for different cassette types
        'cassette_colors': {
            '6x6': '#FF6B6B',
            '4x8': '#4ECDC4',
            '8x4': '#45B7D1',
            '4x6': '#96E6B3',
            '6x4': '#F7DC6F',
            '4x4': '#BB8FCE',
            '2x4': '#F8B739',
            '4x2': '#85C1E2',
            '2x6': '#F1948A',
            '6x2': '#73C6B6',
        },
        
        # Custom area highlighting
        'custom_area_color': '#FF0000',
        'custom_area_alpha': 0.3,
        
        # Text settings
        'label_font_size': 8,
        'title_font_size': 14,
        'stats_font_size': 10,
        
        # Output formats
        'output_formats': ['png', 'pdf', 'svg'],
        'default_format': 'pdf',
    }
    
    # ============================================================================
    # REPORTING CONFIGURATION
    # ============================================================================
    
    REPORTING = {
        # Report formats
        'formats': ['html', 'pdf', 'json', 'csv'],
        'default_format': 'pdf',
        
        # Date format
        'date_format': '%Y-%m-%d %H:%M:%S',
        
        # Report sections
        'include_sections': {
            'summary': True,
            'cassette_list': True,
            'weight_distribution': True,
            'installation_sequence': True,
            'custom_areas': True,
            'cost_estimate': False,  # Disabled for MVP
        },
        
        # Installation sequence
        'sequence_numbering': 'row_by_row',  # Options: 'row_by_row', 'size_grouped'
        
        # Weight calculations
        'include_weight_warnings': True,
        'weight_warning_threshold': 450,  # Warning if cassette > 450 lbs
    }
    
    # ============================================================================
    # EXPORT CONFIGURATION
    # ============================================================================
    
    EXPORT = {
        # DXF settings
        'dxf_layer_names': {
            'boundary': 'FLOOR_BOUNDARY',
            'cassettes': 'CASSETTES',
            'labels': 'LABELS',
            'dimensions': 'DIMENSIONS',
            'custom_areas': 'CUSTOM_AREAS',
        },
        
        # Excel settings
        'excel_sheets': {
            'summary': 'Summary',
            'cassettes': 'Cassette List',
            'installation': 'Installation Sequence',
            'weights': 'Weight Distribution',
        },
        
        # JSON settings
        'json_indent': 2,
        'json_sort_keys': True,
    }
    
    # ============================================================================
    # PERFORMANCE CONFIGURATION
    # ============================================================================
    
    PERFORMANCE = {
        # Parallel processing
        'use_multiprocessing': False,  # Disabled for MVP simplicity
        'max_workers': 4,
        
        # Caching
        'enable_caching': True,
        'cache_dir': TEMP_DIR / 'cache',
        'cache_ttl': 3600,  # Cache time-to-live in seconds
        
        # Memory limits
        'max_memory_mb': 2048,  # Maximum memory usage in MB
        
        # Profiling
        'enable_profiling': False,  # Disabled for MVP
        'profile_output_dir': TEMP_DIR / 'profiles',
    }
    
    # ============================================================================
    # VALIDATION CONFIGURATION
    # ============================================================================
    
    VALIDATION = {
        # Coverage validation
        'strict_coverage_check': True,
        'coverage_tolerance': 0.01,  # 1% tolerance
        
        # Weight validation
        'validate_weights': True,
        'max_total_weight': 50000,  # Maximum total weight in lbs
        
        # Dimension validation
        'min_floor_area': 100,  # Minimum floor area in sq ft
        'max_floor_area': 10000,  # Maximum floor area in sq ft
        
        # Cassette validation
        'max_cassettes': 500,  # Maximum number of cassettes
        'validate_overlaps': True,
        'overlap_tolerance': 0.01,  # 0.01 ft tolerance for overlaps
    }
    
    @classmethod
    def get_config_dict(cls) -> Dict:
        """Get all configuration as a dictionary."""
        return {
            'cassette_specs': cls.CASSETTE_SPECS,
            'joist_specs': cls.JOIST_SPECS,
            'optimization': cls.OPTIMIZATION,
            'image_processing': cls.IMAGE_PROCESSING,
            'visualization': cls.VISUALIZATION,
            'reporting': cls.REPORTING,
            'export': cls.EXPORT,
            'performance': cls.PERFORMANCE,
            'validation': cls.VALIDATION,
        }
    
    @classmethod
    def validate_config(cls) -> List[str]:
        """Validate configuration settings."""
        warnings = []
        
        # Check weight calculations
        for name, spec in cls.CASSETTE_SPECS.items():
            area = spec['width'] * spec['height']
            weight = area * cls.WEIGHT_PER_SQFT
            if weight > cls.MAX_CASSETTE_WEIGHT:
                warnings.append(
                    f"Cassette {name} exceeds weight limit: "
                    f"{weight:.1f} lbs > {cls.MAX_CASSETTE_WEIGHT} lbs"
                )
        
        # Check directory permissions
        for dir_path in [cls.DATA_DIR, cls.OUTPUT_DIR, cls.TEMP_DIR]:
            if not dir_path.is_dir():
                warnings.append(f"Directory does not exist: {dir_path}")
            elif not dir_path.is_dir():
                warnings.append(f"No write permission for: {dir_path}")
        
        # Check optimization parameters
        if cls.OPTIMIZATION['min_coverage'] > cls.OPTIMIZATION['target_coverage']:
            warnings.append(
                "Minimum coverage exceeds target coverage"
            )
        
        return warnings


# Create singleton instance
cassette_config = CassetteConfig()
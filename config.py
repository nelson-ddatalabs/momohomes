"""
config.py - System Configuration Settings
==========================================
Central configuration for the floor plan optimization system.
"""

import os
from pathlib import Path
from typing import Dict, Any


class Config:
    """System-wide configuration settings."""
    
    # ============================================================================
    # PATH CONFIGURATION
    # ============================================================================
    
    # Base directory
    BASE_DIR = Path(__file__).parent
    
    # Data directories
    DATA_DIR = BASE_DIR / "data"
    INPUT_DIR = DATA_DIR / "input"
    OUTPUT_DIR = DATA_DIR / "output"
    TEMP_DIR = DATA_DIR / "temp"
    MODELS_DIR = DATA_DIR / "models"
    PATTERNS_DIR = DATA_DIR / "patterns"
    
    # Create directories if they don't exist
    for dir_path in [DATA_DIR, INPUT_DIR, OUTPUT_DIR, TEMP_DIR, MODELS_DIR, PATTERNS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # ============================================================================
    # IMAGE PROCESSING CONFIGURATION
    # ============================================================================
    
    # Supported image formats
    SUPPORTED_IMAGE_FORMATS = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
    
    # Image processing parameters
    IMAGE_PROCESSING = {
        'scale_detection': {
            'default_pixels_per_foot': 50,
            'min_pixels_per_foot': 10,
            'max_pixels_per_foot': 200
        },
        'room_detection': {
            'min_room_area_pixels': 1000,
            'min_room_area_sqft': 25,
            'contour_approximation_epsilon': 0.02,
            'morphology_kernel_size': 3
        },
        'preprocessing': {
            'gaussian_blur_kernel': (5, 5),
            'adaptive_threshold_block_size': 11,
            'adaptive_threshold_constant': 2
        },
        'ocr': {
            'language': 'eng',
            'config': '--psm 11'  # Sparse text mode
        }
    }
    
    # ============================================================================
    # PANEL CONFIGURATION
    # ============================================================================
    
    # Panel specifications
    PANEL_SPECS = {
        '6x8': {
            'width': 6,
            'length': 8,
            'area': 48,
            'cost_factor': 1.00,
            'base_price': 48.00,
            'max_span': 12,
            'weight': 150  # pounds
        },
        '6x6': {
            'width': 6,
            'length': 6,
            'area': 36,
            'cost_factor': 1.15,
            'base_price': 41.40,
            'max_span': 10,
            'weight': 112
        },
        '4x6': {
            'width': 4,
            'length': 6,
            'area': 24,
            'cost_factor': 1.35,
            'base_price': 32.40,
            'max_span': 8,
            'weight': 75
        },
        '4x4': {
            'width': 4,
            'length': 4,
            'area': 16,
            'cost_factor': 1.60,
            'base_price': 25.60,
            'max_span': 6,
            'weight': 50
        }
    }
    
    # Panel optimization priorities
    PANEL_PRIORITIES = ['6x8', '6x6', '4x6', '4x4']
    
    # ============================================================================
    # OPTIMIZATION CONFIGURATION
    # ============================================================================
    
    OPTIMIZATION = {
        'strategies': {
            'greedy': {
                'enabled': True,
                'timeout': 10,  # seconds
                'priority': 1
            },
            'dynamic': {
                'enabled': True,
                'timeout': 30,
                'priority': 2,
                'max_room_size': 500  # sq ft, use greedy for larger
            },
            'pattern': {
                'enabled': True,
                'timeout': 5,
                'priority': 0,
                'tolerance': 0.5  # feet
            },
            'hybrid': {
                'enabled': True,
                'timeout': 20,
                'priority': 0
            },
            'genetic': {
                'enabled': True,
                'timeout': 60,
                'population_size': 100,
                'generations': 500,
                'mutation_rate': 0.1,
                'crossover_rate': 0.8,
                'elite_size': 10
            },
            'annealing': {
                'enabled': True,
                'timeout': 45,
                'initial_temp': 1000,
                'cooling_rate': 0.995,
                'min_temp': 1,
                'max_iterations': 10000
            }
        },
        'targets': {
            'min_coverage_ratio': 0.98,  # 98%
            'max_cost_per_sqft': 1.50,
            'min_panel_efficiency': 0.65,  # 65% large panels
            'max_waste_ratio': 0.02  # 2%
        },
        'weights': {
            'coverage': 0.4,
            'cost': 0.3,
            'efficiency': 0.2,
            'simplicity': 0.1
        }
    }
    
    # ============================================================================
    # STRUCTURAL CONFIGURATION
    # ============================================================================
    
    STRUCTURAL = {
        'load_bearing': {
            'exterior_walls': True,
            'center_tolerance': 0.1,  # 10% of building width/height
            'min_wall_thickness': 0.5  # feet
        },
        'span_limits': {
            '6x8': 12,  # feet
            '6x6': 10,
            '4x6': 8,
            '4x4': 6
        },
        'safety_factor': 1.5,
        'deflection_limit': 'L/360'  # L = span length
    }
    
    # ============================================================================
    # ROOM CLASSIFICATION CONFIGURATION
    # ============================================================================
    
    ROOM_CLASSIFICATION = {
        'keywords': {
            'bedroom': ['bedroom', 'bdrm', 'bed', 'br', 'master', 'guest'],
            'bathroom': ['bathroom', 'bath', 'ens', 'ensuite', 'powder', 'wc'],
            'kitchen': ['kitchen', 'kit', 'pantry'],
            'living': ['living', 'liv', 'great', 'family', 'den'],
            'dining': ['dining', 'din', 'breakfast'],
            'hallway': ['hallway', 'hall', 'corridor', 'passage'],
            'closet': ['closet', 'wic', 'storage', 'wardrobe'],
            'entry': ['entry', 'foyer', 'vestibule', 'mud'],
            'office': ['office', 'study', 'library', 'den'],
            'laundry': ['laundry', 'utility', 'mud'],
            'garage': ['garage', 'carport', 'parking']
        },
        'size_ranges': {  # in square feet
            'closet': (0, 40),
            'bathroom': (30, 100),
            'hallway': (20, 150),
            'bedroom': (80, 300),
            'primary_suite': (150, 500),
            'kitchen': (100, 400),
            'living': (150, 600),
            'dining': (100, 300),
            'office': (80, 200),
            'garage': (200, 1000)
        },
        'aspect_ratios': {
            'hallway': 3.0,  # min aspect ratio for hallway
            'normal': 2.5   # max aspect ratio for normal rooms
        }
    }
    
    # ============================================================================
    # VISUALIZATION CONFIGURATION
    # ============================================================================
    
    VISUALIZATION = {
        'colors': {
            'room_types': {
                'bedroom': '#E3F2FD',      # Light blue
                'bathroom': '#E8F5E9',     # Light green
                'kitchen': '#FFF9C4',      # Light yellow
                'living': '#FFEBEE',       # Light red
                'dining': '#F3E5F5',       # Light purple
                'hallway': '#F5F5F5',      # Light gray
                'closet': '#FFF3E0',       # Light orange
                'office': '#E0F2F1',       # Light teal
                'garage': '#ECEFF1',       # Blue gray
                'unknown': '#FFFFFF'       # White
            },
            'panel_sizes': {
                '6x8': '#1B5E20',  # Dark green
                '6x6': '#388E3C',  # Green
                '4x6': '#FBC02D',  # Yellow
                '4x4': '#F57C00'   # Orange
            },
            'structural': {
                'load_bearing': '#D32F2F',  # Red
                'exterior': '#1976D2',      # Blue
                'interior': '#757575'       # Gray
            }
        },
        'figure_size': (20, 10),
        'dpi': 150,
        'grid_alpha': 0.3,
        'panel_alpha': 0.6,
        'room_alpha': 0.7,
        'font_size': {
            'title': 14,
            'label': 10,
            'annotation': 8
        }
    }
    
    # ============================================================================
    # REPORTING CONFIGURATION
    # ============================================================================
    
    REPORTING = {
        'formats': ['txt', 'json', 'html', 'pdf', 'csv'],
        'default_format': 'txt',
        'include_visualizations': True,
        'include_room_details': True,
        'include_panel_list': True,
        'include_cost_breakdown': True,
        'decimal_places': 2,
        'currency_symbol': '$',
        'date_format': '%Y-%m-%d %H:%M:%S'
    }
    
    # ============================================================================
    # PERFORMANCE CONFIGURATION
    # ============================================================================
    
    PERFORMANCE = {
        'multiprocessing': {
            'enabled': True,
            'max_workers': os.cpu_count() or 4,
            'chunk_size': 10
        },
        'caching': {
            'enabled': True,
            'max_cache_size': 1000,  # MB
            'ttl': 3600  # seconds
        },
        'memory_limits': {
            'max_image_size': 100,  # MB
            'max_rooms': 1000,
            'max_panels_per_room': 1000
        },
        'timeouts': {
            'total_optimization': 300,  # seconds
            'per_room': 30,
            'image_processing': 60
        }
    }
    
    # ============================================================================
    # LOGGING CONFIGURATION
    # ============================================================================
    
    LOGGING = {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'date_format': '%Y-%m-%d %H:%M:%S',
        'file': OUTPUT_DIR / 'optimization.log',
        'max_bytes': 10 * 1024 * 1024,  # 10 MB
        'backup_count': 5,
        'console_output': True
    }
    
    # ============================================================================
    # PATTERN LIBRARY CONFIGURATION
    # ============================================================================
    
    PATTERNS = {
        'enabled': True,
        'auto_learn': True,
        'min_confidence': 0.9,
        'max_patterns': 1000,
        'pattern_file': PATTERNS_DIR / 'patterns.json',
        'update_frequency': 100  # Update after every N optimizations
    }
    
    # ============================================================================
    # API CONFIGURATION (if implementing API)
    # ============================================================================
    
    API = {
        'enabled': False,
        'host': '0.0.0.0',
        'port': 5000,
        'debug': False,
        'max_upload_size': 50 * 1024 * 1024,  # 50 MB
        'allowed_origins': ['*'],
        'rate_limit': '100/hour',
        'auth_required': False
    }
    
    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-notation key."""
        keys = key.split('.')
        value = cls
        
        for k in keys:
            if hasattr(value, k):
                value = getattr(value, k)
            elif isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    @classmethod
    def set(cls, key: str, value: Any):
        """Set configuration value by dot-notation key."""
        keys = key.split('.')
        target = cls
        
        for k in keys[:-1]:
            if hasattr(target, k):
                target = getattr(target, k)
            elif isinstance(target, dict) and k in target:
                target = target[k]
            else:
                raise KeyError(f"Configuration key not found: {key}")
        
        final_key = keys[-1]
        if hasattr(target, final_key):
            setattr(target, final_key, value)
        elif isinstance(target, dict):
            target[final_key] = value
        else:
            raise KeyError(f"Cannot set configuration key: {key}")
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {}
        
        for attr in dir(cls):
            if not attr.startswith('_') and attr.isupper():
                value = getattr(cls, attr)
                if not callable(value):
                    result[attr] = value
        
        return result
    
    @classmethod
    def from_file(cls, filepath: str):
        """Load configuration from JSON or YAML file."""
        import json
        
        with open(filepath, 'r') as f:
            if filepath.endswith('.json'):
                config_data = json.load(f)
            elif filepath.endswith(('.yml', '.yaml')):
                import yaml
                config_data = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config file format: {filepath}")
        
        for key, value in config_data.items():
            if hasattr(cls, key):
                setattr(cls, key, value)
    
    @classmethod
    def save_to_file(cls, filepath: str):
        """Save configuration to JSON or YAML file."""
        import json
        
        config_data = cls.to_dict()
        
        with open(filepath, 'w') as f:
            if filepath.endswith('.json'):
                json.dump(config_data, f, indent=2, default=str)
            elif filepath.endswith(('.yml', '.yaml')):
                import yaml
                yaml.safe_dump(config_data, f, default_flow_style=False)
            else:
                raise ValueError(f"Unsupported config file format: {filepath}")


# Create a global config instance
config = Config()

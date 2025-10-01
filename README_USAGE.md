# How to Run the Cassette Optimization System

## Quick Start

### 1. Simplest Way - Use the Main Runner

```bash
# Run on test polygons
poetry run python optimize_cassettes.py bungalow
poetry run python optimize_cassettes.py luna
poetry run python optimize_cassettes.py rectangle
poetry run python optimize_cassettes.py l-shape

# Run on your own files
poetry run python optimize_cassettes.py myfloorplan.json
poetry run python optimize_cassettes.py floorplan.png
```

### 2. Interactive Mode

```bash
# Just run without arguments for interactive menu
poetry run python optimize_cassettes.py
```

### 3. Using the Modular Pipeline Directly

```bash
# Test polygons
poetry run python run_modular_pipeline.py bungalow
poetry run python run_modular_pipeline.py luna

# Custom files
poetry run python run_modular_pipeline.py path/to/polygon.json
poetry run python run_modular_pipeline.py floorplan.png

# Fast mode (skip backtracking and analysis)
poetry run python run_modular_pipeline.py bungalow --fast
```

### 4. Using the Original System (with Edge Detection)

```bash
# Original system with manual edge measurements
poetry run python run_cassette_system.py floorplan.png

# Or use modular pipeline through the same script
poetry run python run_cassette_system.py bungalow --modular
```

## Input Formats

### JSON Format
```json
{
  "polygon": [
    [0, 0],
    [40, 0],
    [40, 30],
    [0, 30]
  ]
}
```

### Text/CSV Format
```
0, 0
40, 0
40, 30
0, 30
```

### Image Files (PNG/JPG)
- Requires manual edge measurement input
- Will launch interactive edge detection system

## Output Files

The system generates:
- `[name]_modular_results.json` - Detailed optimization results
- Console output with coverage analysis
- Optional visualization PNG (if visualizer available)

## Test Polygons Available

- **bungalow**: Complex floor plan, achieves ~79% (max possible 89%)
- **luna**: Large rectangular floor plan
- **rectangle**: Simple 40x30 ft rectangle
- **l-shape**: L-shaped building for testing

## Understanding Results

### Coverage Metrics
- **Current Coverage**: Actual achieved coverage
- **Theoretical Maximum**: Best possible for this polygon
- **Target**: 94% (industry requirement)

### Key Indicators
- ✅ **Meets requirement**: Coverage ≥ 94%
- ❌ **Below requirement**: Coverage < 94%
- ⚠️ **Impossible target**: Theoretical max < 94%

## Example Commands

```bash
# Quick test on rectangle
poetry run python optimize_cassettes.py rectangle

# Test Bungalow floor plan
poetry run python optimize_cassettes.py bungalow

# Load from JSON file
poetry run python optimize_cassettes.py my_floor_plan.json

# Interactive mode
poetry run python optimize_cassettes.py

# Fast mode for quick results
poetry run python run_modular_pipeline.py bungalow --fast
```

## Performance Notes

- Rectangle (40x30): ~79% coverage in < 1 second
- Bungalow: ~79% coverage (89% theoretical max) in ~2 seconds
- Large floor plans (>2000 sq ft): 3-5 seconds with full pipeline

The system correctly identifies when 94% coverage is mathematically impossible and provides specific recommendations for polygon modifications.
# 100% Coverage Cassette Optimization System

## Overview
Production-grade cassette floor joist optimizer targeting 100% coverage with minimum cassette count.

## Key Features
- **Primary Goal**: 100% coverage (currently achieves 87-97% on complex floor plans)
- **Secondary Goal**: Minimize cassette count
- **Cassette Sizes**: 9 sizes from 3.5x3.5 ft (42"x42" minimum) to 6x8 ft (maximum)
- **Constraints**:
  - Maximum weight: 500 lbs per cassette
  - Size increments: 0.5 ft only
  - No overlap, no overhang, edge-to-edge placement

## Usage

### Command Line
```bash
# Use 100% coverage optimizer
poetry run python run_cassette_system.py <input> --hundred

# Examples with pre-defined polygons
poetry run python run_cassette_system.py umbra --hundred
poetry run python run_cassette_system.py bungalow --hundred
poetry run python run_cassette_system.py luna --hundred

# Examples with floor plan images (will ask for measurements)
poetry run python run_cassette_system.py floorplans/Umbra_XL.png --hundred
poetry run python run_cassette_system.py floorplans/Bungalow-Conditioned.png --hundred
poetry run python run_cassette_system.py path/to/your/floorplan.jpg --hundred
```

### Supported Inputs
1. **Pre-defined polygons**: umbra, bungalow, luna, rectangle
2. **Floor plan images**: .png, .jpg, .jpeg
   - Opens visualization window with numbered edges
   - Asks for measurements interactively
   - Processes measurements to create polygon
   - Runs 100% coverage optimizer automatically
3. **JSON files**: Files containing polygon coordinates

## Implementation Details

### Cassette Sizes (9 total)
```
6.0 x 8.0 ft - 48.0 sq ft (Maximum)
5.5 x 7.5 ft - 41.25 sq ft
5.0 x 7.0 ft - 35.0 sq ft
6.0 x 6.0 ft - 36.0 sq ft
5.0 x 5.5 ft - 27.5 sq ft
4.5 x 5.0 ft - 22.5 sq ft
4.0 x 4.5 ft - 18.0 sq ft
3.5 x 4.0 ft - 14.0 sq ft
3.5 x 3.5 ft - 12.25 sq ft (Minimum - 42"x42")
```

### Optimization Strategies
1. **Largest First**: Places largest cassettes first, fills gaps with smaller
2. **Balanced**: Uses medium sizes for better distribution
3. **Gap Filling**: Iteratively fills remaining gaps with smallest fitting cassettes

### File Structure
```
hundred_percent_optimizer.py   # Core optimizer
run_hundred_percent.py         # Runner/integration
run_cassette_system.py        # Main entry point
```

## Results

### Typical Coverage Achieved
- **Umbra XL**: 96.7% coverage (36 cassettes)
- **Bungalow**: 87.2% coverage (23 cassettes)
- **Rectangle (20x15)**: 86.0% coverage (6 cassettes)

### Output Files
- `results_hundred.json`: Optimization results with cassette placements
- `cassette_layout_hundred.png`: Visual layout (if visualization available)

## Limitations
- May not achieve true 100% on all floor plans due to:
  - Minimum cassette size (3.5x3.5 ft)
  - 0.5 ft increment constraint
  - Complex polygon geometries
  - Edge alignment requirements

## Floor Plan Processing

When given a floor plan image (.png, .jpg, .jpeg), the system:
1. Opens a visualization window showing numbered edges
2. Prompts for measurements of each edge
3. Processes measurements to create a polygon
4. Automatically runs the 100% coverage optimizer
5. Saves results and visualization

This integration uses the `CassetteLayoutSystemCardinal` system to:
- Detect edges in the floor plan
- Collect cardinal direction measurements (N, S, E, W)
- Reconstruct the polygon from measurements
- Ensure perfect closure

## Integration
The system is fully integrated with the main cassette system:
- Use `--hundred` flag for 100% coverage optimizer (now supports floor plan images!)
- Use `--modular` flag for modular pipeline
- Default (no flag) uses original optimizer
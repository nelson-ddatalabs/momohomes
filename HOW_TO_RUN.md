# Cassette Floor Joist Optimization System - User Guide

## Quick Start

### Method 1: Interactive Mode (RECOMMENDED)
```bash
cd "/Users/nelsondsouza/Dropbox (Personal)/Mac/Documents/products/momohomes"
poetry run python run_cassette_interactive.py floorplans/YourFloorPlan.png
```

This will:
1. **SHOW YOU A VISUAL WINDOW** with numbered edges and cardinal directions
2. Keep the window open while you measure
3. Highlight each edge as you measure it
4. Provide immediate feedback on closure balance
5. Generate optimized cassette layout

### Method 2: Without Matplotlib Display
```bash
poetry run python run_cassette_interactive.py floorplans/YourFloorPlan.png --no-matplotlib
```
This will use OpenCV or save the image for manual viewing.

### Method 3: Pre-defined Measurements
```bash
poetry run python run_with_measurements.py floorplans/YourFloorPlan.png
```
Edit the measurements in the script first.

## How It Works

### Step 1: Binary Conversion
- Green areas (indoor space) → White
- Everything else → Black

### Step 2: Edge Detection
- Detects building edges
- Classifies each edge as N/S/E/W (cardinal direction)
- Numbers edges clockwise from 0

### Step 3: Interactive Measurement ⭐ NEW!
- **DISPLAYS VISUALIZATION WINDOW** showing:
  - Numbered edges (0, 1, 2, ...)
  - Cardinal directions (N/S/E/W)
  - Color coding (E=Red, W=Magenta, N=Green, S=Blue)
  - Direction arrows
- **HIGHLIGHTS CURRENT EDGE** being measured
- Shows edge information in terminal
- Collects measurements in feet

### Step 4: Balance Verification
- Checks if ΣEast = ΣWest
- Checks if ΣNorth = ΣSouth
- Reports closure error

### Step 5: Cassette Optimization
- Places cassettes within boundaries
- Uses multiple sizes (6x8, 5x8, 6x6, etc.)
- Achieves ≥94% coverage target

### Step 6: Visualization Output
- Saves final layout with cassettes overlaid
- Provides statistics and coverage report

## Important Tips

### For Perfect Polygon Closure
The sum of measurements in opposite directions MUST be equal:
- Total East = Total West
- Total North = Total South

Example for a rectangle:
```
Edge 0: N = 40 ft
Edge 1: E = 78 ft
Edge 2: S = 40 ft  (must equal N)
Edge 3: W = 78 ft  (must equal E)
```

### Understanding the Visualization

When the window opens, you'll see:
- **Numbers** on each edge (0, 1, 2, ...)
- **Letters** showing direction (N, S, E, W)
- **Colors** for quick identification
- **Yellow highlight** on the edge being measured

### Cassette Specifications
- Maximum size: 6' x 8' (48 sq ft)
- Maximum weight: 500 lbs
- Weight factor: 10.4 lbs/sq ft
- Available sizes: 6x8, 5x8, 6x6, 4x8, 4x6, 6x4, 4x4, 3x4, 4x3

## Troubleshooting

### "Matplotlib not available"
Install with: `poetry add matplotlib`

### "Window not showing"
- Check if you're running via SSH (needs X11 forwarding)
- Try `--no-matplotlib` flag to use file output

### "Polygon doesn't close"
- Verify your measurements
- Ensure ΣE = ΣW and ΣN = ΣS
- Check the balance report in the output

### "Coverage below 94%"
- This usually means the polygon didn't close properly
- Fix measurements for proper closure first

## Output Files

After running, check the output directory for:
- `numbered_cardinal_edges.png` - Edge reference
- `cassette_layout_cardinal.png` - Final layout with cassettes
- `results_cardinal.json` - Detailed measurements and statistics
- `binary.png` - Binary conversion result
- `cardinal_edges.png` - Edge detection result

## Example Run

```bash
poetry run python run_cassette_interactive.py floorplans/Luna-Conditioned.png
```

You'll see:
1. A window opens showing the floor plan with numbered edges
2. Terminal prompts for Edge 0 measurement
3. Edge 0 is highlighted in yellow in the window
4. Enter measurement (e.g., "40.5")
5. Edge 1 becomes highlighted
6. Continue until all edges measured
7. System shows balance check
8. Generates optimized layout

## Need Help?

Run the test suite to verify everything works:
```bash
poetry run python test_interactive_system.py
```
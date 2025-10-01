# Fix: 100% Coverage Optimizer Now Asks for Measurements

## Problem
When running with `--hundred` flag on a floor plan image, the system was NOT asking for measurements. Instead, it was failing silently with an EOFError.

### Command that was broken:
```bash
poetry run python run_cassette_system.py floorplans/Umbra_XL.png --hundred
```

## Root Cause

### Broken Flow:
1. `run_cassette_system.py` detected `--hundred` flag
2. Immediately called `run_hundred_percent.main()`
3. `run_hundred_percent` tried to process floor plan internally
4. Called `CassetteLayoutSystemCardinal.process_floor_plan()` programmatically
5. When it tried to call `input()` for measurements, got **EOFError**
6. Exception was caught and returned `None`
7. No polygon extracted, no measurements asked

### Why EOFError?
- `input()` requires an interactive terminal session
- When called programmatically from within another function, there's no terminal input available
- Results in "EOF when reading a line" error

## Solution

Modified `run_cassette_system.py` to handle floor plan images specially when used with `--hundred`:

### New Flow:
1. Detect if input is a floor plan image (.png, .jpg, .jpeg)
2. **FIRST** process the floor plan using `CassetteLayoutSystemCardinal`
   - This opens visualization window
   - Asks for measurements interactively
   - Saves polygon to results
3. **THEN** extract the polygon from saved results
4. Run 100% optimizer on the extracted polygon
5. Save and display results

### Key Code Change:
```python
# Special handling for --hundred with floor plan images
if use_hundred and len(sys.argv) > 1:
    potential_file = sys.argv[1]
    if Path(potential_file).exists() and Path(potential_file).suffix.lower() in ['.png', '.jpg', '.jpeg']:
        # Process floor plan FIRST to get measurements
        system = CassetteLayoutSystemCardinal(output_dir=output_dir)
        result = system.process_floor_plan(floor_plan_path)

        # THEN run 100% optimizer on extracted polygon
        polygon = saved_data['polygon']
        optimizer = HundredPercentOptimizer(polygon)
        results = optimizer.optimize()
```

## Result

Now when you run:
```bash
poetry run python run_cassette_system.py floorplans/Umbra_XL.png --hundred
```

The system will:
1. ✅ Show "Processing floor plan for 100% COVERAGE OPTIMIZER..."
2. ✅ Open visualization window with numbered edges
3. ✅ Ask for measurements: "Enter measurement for Edge 0 (feet):"
4. ✅ Process all measurements
5. ✅ Run 100% coverage optimizer
6. ✅ Display results and save visualization

## Testing
The fix ensures that floor plan images work properly with the `--hundred` flag by processing measurements BEFORE attempting optimization, maintaining the interactive flow that users expect.
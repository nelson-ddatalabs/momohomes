# Cassette-Based Floor Joist Optimization System
## Complete System Documentation

---

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Interactive Measurement Process](#interactive-measurement-process)
4. [Cassette Optimization](#cassette-optimization)
5. [Usage Guide](#usage-guide)
6. [API Reference](#api-reference)
7. [Output Files](#output-files)
8. [Troubleshooting](#troubleshooting)

---

## System Overview

The Cassette-Based Floor Joist Optimization System is a comprehensive solution for optimizing the placement of pre-built floor joist cassettes on residential floor plans. The system combines interactive edge measurement with advanced optimization algorithms to achieve maximum coverage while minimizing custom work.

### Key Features
- **Interactive Edge Measurement**: Guided measurement of floor plan perimeters
- **Automatic Edge Detection**: Binary image processing to detect floor boundaries
- **Flexible Optimization Strategies**: Grid, staggered, and hybrid placement
- **Comprehensive Reporting**: Visual layouts, HTML reports, and JSON data
- **Coverage Analysis**: Detailed gap analysis and custom work calculation

### System Requirements
- Python 3.8+
- OpenCV 4.0+
- NumPy, Matplotlib, Pytesseract
- 4GB RAM minimum
- 500MB disk space

---

## Architecture

### Core Modules

```
momohomes/
├── main_cassette_interactive.py       # Main entry point
├── interactive_edge_measurement.py    # Interactive measurement system
├── measurement_to_cassette_integration.py  # Integration layer
├── cassette_optimizer.py              # Optimization engine
├── cassette_models.py                 # Data models
├── cassette_visualizer.py             # Visualization generator
├── cassette_report_generator.py       # Report generation
├── coverage_analyzer.py               # Coverage analysis
└── config_cassette.py                 # Configuration
```

### Data Flow

```
Floor Plan Image → Edge Detection → Interactive Measurement → 
Polygon Construction → Cassette Optimization → Coverage Analysis → 
Visualization & Reports
```

---

## Interactive Measurement Process

### Step 1: Binary Conversion
The system converts the floor plan to a binary image where:
- **White pixels**: Indoor living space
- **Black pixels**: Outdoor areas and walls

### Step 2: Edge Detection
Automatic detection of all straight edges along the perimeter using:
- Contour detection
- Polygon approximation
- Edge filtering

### Step 3: Edge Verification
```
Detected 12 edges. Is this correct? (y/n)
```
If incorrect, users can:
- **Split edges**: Click to divide one edge into two
- **Merge edges**: Select two adjacent edges to combine

### Step 4: Starting Corner Selection
Users click their preferred starting corner for clockwise measurement.

### Step 5: Guided Measurement
The system highlights each edge in sequence:
- **RED**: Unmeasured edges
- **YELLOW**: Current edge being measured
- **GREEN**: Completed measurements

Users enter measurements in decimal feet (e.g., "32.5").

### Step 6: Validation
The system calculates:
- Total perimeter
- Floor area
- Polygon closure error

---

## Cassette Optimization

### Cassette Specifications

| Size | Dimensions | Weight | Typical Use |
|------|------------|--------|-------------|
| 8x8 | 8' × 8' | 64 lbs | Large open areas |
| 6x8 | 6' × 8' | 48 lbs | Standard rooms |
| 6x6 | 6' × 6' | 36 lbs | Medium spaces |
| 8x4 | 8' × 4' | 32 lbs | Corridors |
| 6x4 | 6' × 4' | 24 lbs | Transitions |
| 4x4 | 4' × 4' | 16 lbs | Small gaps |

### Optimization Strategies

#### Grid Strategy
- Places cassettes in regular grid pattern
- Starts from top-left corner
- Best for rectangular spaces

#### Staggered Strategy
- Offsets alternate rows by half cassette width
- Reduces long continuous seams
- Better structural integrity

#### Hybrid Strategy (Default)
- Combines grid and staggered approaches
- Optimizes based on local geometry
- Achieves highest coverage

### Coverage Targets
- **Minimum**: 93% coverage
- **Target**: 94-95% coverage
- **Maximum gap size**: 8% of total area

---

## Usage Guide

### Basic Usage

```bash
# Interactive measurement with hybrid optimization
poetry run python main_cassette_interactive.py Luna.png

# Specify output directory
poetry run python main_cassette_interactive.py Luna.png --output results/

# Use existing measurements
poetry run python main_cassette_interactive.py Luna.png --use-existing measurements.json

# Different optimization strategy
poetry run python main_cassette_interactive.py Luna.png --strategy grid
```

### Command Line Options

```
Options:
  --strategy {grid,staggered,hybrid}
                        Cassette placement strategy (default: hybrid)
  --output OUTPUT       Output directory for results (default: output/)
  --use-existing FILE   Skip measurement, use existing file
  --target-coverage FLOAT
                        Target coverage percentage (default: 0.94)
  --verbose            Enable verbose logging
  --debug              Enable debug logging
```

### Interactive Measurement Example

```
======================================================================
INTERACTIVE FLOOR PLAN MEASUREMENT
======================================================================

1. EDGE DETECTION
Detected 12 edges.
Is this correct? (y/n): y

2. SELECT STARTING CORNER
Enter corner number: 1

3. EDGE MEASUREMENT
Enter measurements in decimal feet (e.g., 32.5)
Edge 1/12 measurement (feet): 32.5
Edge 2/12 measurement (feet): 7.5
...

✓ All edges measured!
```

---

## API Reference

### InteractiveCassetteSystem

Main system class for complete pipeline.

```python
system = InteractiveCassetteSystem()

results = system.process_floor_plan_interactive(
    image_path="floor_plan.png",
    strategy="hybrid",
    output_dir="output",
    skip_measurement=False,
    measurement_file=None
)
```

### InteractiveEdgeMeasurement

Handles interactive measurement process.

```python
measurement = InteractiveEdgeMeasurement()

# Process floor plan
results = measurement.process_floor_plan("floor_plan.png")

# Save results
measurement.save_results(results, "measurements.json")
```

### MeasurementToCassetteIntegration

Integrates measurements with optimization.

```python
integration = MeasurementToCassetteIntegration()

# Run complete pipeline
results = integration.run_complete_pipeline(
    measurement_file="measurements.json",
    strategy="hybrid",
    output_dir="output"
)
```

---

## Output Files

### 1. Measurement File (JSON)
```json
{
  "edges": [...],
  "vertices": [...],
  "perimeter_feet": 265.75,
  "area_sqft": 2733.25,
  "closure_error_feet": 0.5,
  "is_closed": true
}
```

### 2. Visualization (PNG)
- Visual layout of cassette placement
- Color-coded by cassette size
- Gap highlighting
- Dimension labels

### 3. HTML Report
- Detailed construction report
- Cassette bill of materials
- Installation sequence
- Coverage statistics

### 4. Complete Results (JSON)
```json
{
  "success": true,
  "optimization": {
    "cassette_count": 76,
    "coverage_percentage": 95.2,
    "total_weight": 2736
  },
  "cassettes": {...},
  "coverage": {...}
}
```

---

## Troubleshooting

### Common Issues

#### 1. Edge Detection Problems
**Symptom**: Wrong number of edges detected
**Solution**: 
- Ensure good image quality
- Check color contrast
- Use edge adjustment tools (split/merge)

#### 2. Polygon Doesn't Close
**Symptom**: Large closure error
**Solution**:
- Verify all measurements
- Check edge order is clockwise
- Ensure no missing edges

#### 3. Low Coverage
**Symptom**: Coverage below 93%
**Solution**:
- Try different optimization strategy
- Check floor plan complexity
- Consider custom cassette sizes

#### 4. OCR Not Working
**Symptom**: Can't extract dimensions automatically
**Solution**:
- Use interactive measurement instead
- Ensure Tesseract is installed
- Check image resolution

### Debug Mode

Enable debug logging for detailed information:
```bash
poetry run python main_cassette_interactive.py floor_plan.png --debug
```

### Support Files

- **Binary image**: `luna_binary.png` - Check edge detection
- **Edge visualization**: `detected_edges.png` - Verify edge count
- **Measurement visualization**: `measurement_visualization.png` - Check polygon
- **Debug output**: `debug_output/` - Detailed debugging images

---

## Performance Metrics

### Typical Results (2,700 sq ft floor plan)
- **Processing time**: 5-10 seconds
- **Cassette count**: 75-80 units
- **Coverage achieved**: 94-96%
- **Custom work required**: 4-6%
- **Total weight**: 2,500-3,000 lbs

### Optimization Benchmarks
- Grid strategy: 93-94% coverage
- Staggered strategy: 94-95% coverage
- Hybrid strategy: 95-96% coverage

---

## Best Practices

1. **Image Quality**
   - Use high-resolution floor plans (min 1000x1000)
   - Ensure clear distinction between indoor/outdoor
   - Remove unnecessary annotations

2. **Measurement Accuracy**
   - Double-check critical dimensions
   - Verify total area against expectations
   - Ensure polygon closes properly

3. **Optimization Strategy**
   - Start with hybrid strategy
   - Try grid for simple rectangular plans
   - Use staggered for complex shapes

4. **Validation**
   - Always verify edge count
   - Check coverage percentage
   - Review gap locations

---

## Future Enhancements

- [ ] Automatic dimension extraction from text
- [ ] Support for multi-story buildings
- [ ] Load capacity calculations
- [ ] Cost estimation module
- [ ] 3D visualization
- [ ] Mobile app interface
- [ ] Cloud-based processing
- [ ] Machine learning for edge detection

---

## Contact & Support

For issues, questions, or contributions:
- GitHub: [Project Repository]
- Email: support@momohomes.com
- Documentation: This file

---

*Last Updated: September 2024*
*Version: 1.0.0*
# Visualization Updates for 100% Coverage Optimizer

## Changes Implemented

### 1. Legend with Color-Size Mapping and Counts ✅
- Added a comprehensive legend at the bottom left of the visualization
- Each cassette size has a consistent color throughout the visualization
- Legend shows:
  - Color box for each size
  - Size dimensions (e.g., "6.0x8.0")
  - Count of cassettes for that size (e.g., "3 units")
- Legend has a clean background box with border for clarity

### 2. Statistics Moved to Bottom Left ✅
- Statistics panel now positioned at bottom left, above the legend
- Ensures statistics don't overlap with the floor plan
- Statistics include:
  - Coverage percentage
  - Number of cassettes
  - Total area
  - Gap area (removed as it was redundant)
- Clean background box with border for readability

### 3. Dynamic Title with Floor Plan Name ✅
- Title now shows: "[FLOOR_PLAN_NAME] CASSETTE PLAN"
- Examples:
  - "UMBRA XL CASSETTE PLAN"
  - "BUNGALOW CONDITIONED CASSETTE PLAN"
- Automatically extracts floor plan name from file path
- Cleans up underscores and hyphens for better display

## Technical Implementation

### Files Modified:
1. **hundred_percent_visualizer.py**
   - Updated `create_simple_visualization()` to accept `floor_plan_name` parameter
   - Implemented consistent color mapping per cassette size
   - Added legend generation with counts
   - Repositioned statistics to bottom left
   - Dynamic title generation

2. **run_cassette_system.py**
   - Updated to pass floor plan name to visualization function
   - Extracts name from file path

3. **run_hundred_percent.py**
   - Updated to pass floor plan name to visualization function
   - Handles both image files and pre-defined polygons

### Color Assignment Logic:
- Each unique cassette size gets a consistent color
- Sizes are sorted alphabetically for consistent ordering
- Up to 9 distinct colors available (matching 9 cassette sizes)
- Colors are light/pastel shades for better visibility

### Layout Structure:
```
┌─────────────────────────────────────┐
│  [FLOOR_PLAN_NAME] CASSETTE PLAN    │  <- Dynamic title
│                                      │
│         Floor Plan Area              │
│      (with colored cassettes)        │
│                                      │
│  ┌──────────────┐                   │
│  │ STATISTICS   │                   │  <- Bottom left
│  │ Coverage: X% │                   │
│  │ Cassettes: N │                   │
│  │ Total: X sqft│                   │
│  └──────────────┘                   │
│  ┌──────────────────┐               │
│  │ CASSETTE SIZES   │               │  <- Legend below stats
│  │ □ 3.5x4.0: 2 units│              │
│  │ □ 4.5x5.0: 3 units│              │
│  │ □ 6.0x8.0: 5 units│              │
│  └──────────────────┘               │
└─────────────────────────────────────┘
```

## Benefits:
1. **Clear Visual Communication**: Legend makes it easy to understand cassette distribution
2. **Professional Appearance**: Clean layout with proper positioning
3. **Informative Title**: Immediately identifies which floor plan is shown
4. **No Overlapping**: Stats and legend positioned to avoid floor plan area
5. **Consistent Colors**: Each size has the same color throughout the visualization

## Testing:
Successfully tested with:
- Rectangle test polygon
- Umbra floor plan
- Bungalow floor plan
- Various cassette size distributions
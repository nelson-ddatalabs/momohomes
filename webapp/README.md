# Cassette Floor Plan Optimizer - Web Application

A user-friendly web interface for optimizing cassette and C-channel floor plan layouts.

## 🚀 Quick Start

### 1. Navigate to webapp directory
```bash
cd webapp
```

### 2. Start the application
```bash
poetry run python app.py
```

### 3. Open your browser
Navigate to: **http://127.0.0.1:5000**

## 📋 Features

### ✅ Implemented (Phases 1-3)
- **Image Upload**: Upload floor plan images (PNG, JPG, JPEG up to 10MB)
- **Edge Detection**: Automatic detection and numbering of floor plan edges
- **Measurement Input**: Dynamic form for entering edge measurements in feet
- **Optimization**: Gap redistribution with C-channel filling
- **Visualization**: Dual PNG + SVG output with labeled cassettes and C-channels
- **Results Display**: Interactive results page with statistics
- **Download**: SVG and PNG download options
- **Session Management**: Auto-cleanup of old files (60-minute lifetime)

### 🎨 UI Features
- Modern Bootstrap 5 design
- Responsive layout (mobile-friendly)
- Construction-themed stacking blocks loading animation
- Clean statistics display
- Easy-to-use form validation

## 🔧 How It Works

### User Workflow
```
1. Upload Floor Plan Image
   ↓
2. View Numbered Edges
   ↓
3. Enter Measurements (feet)
   ↓
4. Click "Start Optimization"
   ↓
5. View Results & Download SVG
```

### Technical Flow
```
Upload → Edge Detection → Polygon Creation → Optimization → Visualization → Results
```

## 📁 File Structure

```
webapp/
├── app.py                      # Flask application (main)
├── edge_processor.py           # Edge detection module
├── static/
│   ├── css/style.css          # Custom styles + spinner animation
│   └── js/main.js             # Client-side logic
├── templates/
│   ├── index.html             # Upload & measurement page
│   └── result.html            # Results display page
├── uploads/                    # Temporary uploads (auto-created)
│   └── <session-id>/
│       ├── original_*.png
│       └── edges_labeled.png
└── results/                    # Generated results (auto-created)
    └── <session-id>/
        ├── results.json
        ├── cassette_layout.svg
        └── cassette_layout.png
```

## 🎛️ Configuration

Edit `app.py` to customize:

```python
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # File size limit
app.config['SESSION_LIFETIME'] = 60  # Session lifetime in minutes
```

## 🐛 Troubleshooting

### Port Already in Use
If port 5000 is busy, change in `app.py`:
```python
app.run(debug=True, host='127.0.0.1', port=5001)  # Use different port
```

### Edge Detection Issues
If edges aren't detected correctly:
- Ensure floor plan has clear boundaries
- Try a higher contrast image
- Check image isn't too complex

### Optimization Fails
- Verify measurements are in feet
- Ensure all edge measurements are entered
- Check that measurements create a valid polygon

## 📊 Statistics Displayed

The results page shows:
- **Total Area**: Floor plan area in square feet
- **Coverage**: Percentage of area covered (should be 100%)
- **Cassettes**: Number of cassette units used
- **Cassette Area**: Total area covered by cassettes
- **C-Channel Area**: Total area covered by C-channels

## 🔐 Security Notes

**For production deployment:**
1. Change secret key in `app.py`:
   ```python
   app.secret_key = 'your-secure-random-key-here'
   ```
2. Set `debug=False`
3. Use a production WSGI server (e.g., Gunicorn)
4. Add authentication if needed
5. Configure file upload limits appropriately

## 🚧 Future Enhancements (Phase 4+)

- Enhanced error handling and validation
- Interactive SVG with hover tooltips
- Multiple floor plan format support
- Export to DXF/CAD formats
- Optimization parameter controls
- Upload history and comparison
- PDF report generation

## 📝 Notes

- Sessions auto-expire after 60 minutes
- Old uploads/results are automatically cleaned up
- SVG files are scalable and suitable for CAD import
- Edge detection uses simplified contour algorithm
- Currently supports rectangular floor plans best

## 🆘 Support

For issues or questions:
1. Check the console output for error messages
2. Verify all dependencies are installed: `poetry install`
3. Ensure you're running from the `webapp/` directory
4. Check file permissions for uploads/ and results/ directories

## 📜 License

Part of the MomoHomes cassette floor plan optimization system.

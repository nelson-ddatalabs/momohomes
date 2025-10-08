"""
Flask Web Application for Cassette Floor Plan Optimizer
Allows users to upload floor plans, input measurements, and download optimized SVG layouts
"""

from flask import Flask, render_template, request, jsonify, send_file, session, redirect, url_for
import os
import sys
import uuid
import json
from pathlib import Path
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename

# Add parent directory to path to import optimization modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import edge processor
from edge_processor import EdgeProcessor

app = Flask(__name__)
app.secret_key = 'cassette-optimizer-secret-key-change-in-production'  # Change in production!
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['SESSION_LIFETIME'] = 60  # minutes

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    """Check if uploaded file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def cleanup_old_sessions():
    """Delete uploads and results older than SESSION_LIFETIME minutes"""
    current_time = datetime.now()
    lifetime = timedelta(minutes=app.config['SESSION_LIFETIME'])

    for folder in [app.config['UPLOAD_FOLDER'], app.config['RESULTS_FOLDER']]:
        folder_path = Path(folder)
        if folder_path.exists():
            for session_dir in folder_path.iterdir():
                if session_dir.is_dir():
                    # Check directory modification time
                    mod_time = datetime.fromtimestamp(session_dir.stat().st_mtime)
                    if current_time - mod_time > lifetime:
                        # Delete old session directory
                        import shutil
                        shutil.rmtree(session_dir)
                        print(f"Cleaned up old session: {session_dir}")


@app.route('/')
def index():
    """Landing page with upload form"""
    # Cleanup old sessions on page load
    cleanup_old_sessions()
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and generate edge-labeled image"""

    # Check if file is present
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    # Check if file is selected
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Check if file type is allowed
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: PNG, JPG, JPEG'}), 400

    try:
        # Generate unique session ID
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id

        # Create session directory
        upload_dir = Path(app.config['UPLOAD_FOLDER']) / session_id
        upload_dir.mkdir(parents=True, exist_ok=True)

        # Save uploaded file
        filename = secure_filename(file.filename)
        original_path = upload_dir / f"original_{filename}"
        file.save(str(original_path))

        # Process edges
        edge_processor = EdgeProcessor()
        edges_image_path = upload_dir / "edges_labeled.png"

        result = edge_processor.process_image(
            str(original_path),
            str(edges_image_path)
        )

        if not result['success']:
            return jsonify({'error': f'Edge detection failed: {result["error"]}'}), 500

        # Store edge data in session
        session['edge_count'] = result['edge_count']
        session['edges'] = result['edges']

        # Return success with edge data
        return jsonify({
            'success': True,
            'session_id': session_id,
            'edge_count': result['edge_count'],
            'edges_image_url': f'/uploads/{session_id}/edges_labeled.png'
        })

    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500


@app.route('/uploads/<session_id>/<filename>')
def serve_upload(session_id, filename):
    """Serve uploaded/processed files"""
    file_path = Path(app.config['UPLOAD_FOLDER']) / session_id / filename
    if file_path.exists():
        return send_file(str(file_path))
    return jsonify({'error': 'File not found'}), 404


@app.route('/optimize', methods=['POST'])
def optimize():
    """Handle optimization with measurements"""
    try:
        # Get data from request
        data = request.json
        session_id = data.get('session_id')
        measurements = data.get('measurements')

        if not session_id or not measurements:
            return jsonify({'error': 'Missing session_id or measurements'}), 400

        # Import optimizer
        from gap_redistribution_optimizer import GapRedistributionOptimizer

        # Get edge data from session
        edges = session.get('edges', [])
        if not edges:
            return jsonify({'error': 'No edge data found in session'}), 400

        # Convert measurements to polygon
        # For now, create a simple rectangular polygon based on measurements
        # This is a simplified version - actual implementation would use edge geometry
        polygon = _create_polygon_from_measurements(edges, measurements)

        # Run optimization
        optimizer = GapRedistributionOptimizer(polygon)
        result = optimizer.optimize()

        if not result:
            return jsonify({'error': 'Optimization failed'}), 500

        # Create results directory
        results_dir = Path(app.config['RESULTS_FOLDER']) / session_id
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save results JSON
        results_file = results_dir / 'results.json'
        with open(results_file, 'w') as f:
            json.dump({
                'cassettes': result['cassettes'],
                'c_channels': result.get('c_channels', []),
                'statistics': result['statistics'],
                'polygon': polygon
            }, f, indent=2)

        # Generate visualizations
        from hundred_percent_visualizer import create_simple_visualization

        output_base = str(results_dir / 'cassette_layout')

        # Prepare statistics for visualizer
        stats = result['statistics']
        vis_stats = {
            'coverage': stats.get('coverage_percent', 100.0),
            'total_area': stats['total_area'],
            'covered': stats.get('cassette_area', 0) + stats.get('cchannel_area', 0),
            'cassettes': stats.get('cassette_count', len(result['cassettes'])),
            'cchannel_area': stats.get('cchannel_area', 0),
            'per_cassette_cchannel': True,
            'cchannel_widths_per_cassette': result.get('c_channels_inches', []),
            'cchannel_geometries': result.get('cchannel_geometries', [])
        }

        create_simple_visualization(
            cassettes=result['cassettes'],
            polygon=polygon,
            statistics=vis_stats,
            output_path=output_base,
            floor_plan_name='Floor Plan'
        )

        return jsonify({
            'success': True,
            'session_id': session_id
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Optimization error: {str(e)}'}), 500


def _create_polygon_from_measurements(edges: List[Dict], measurements: Dict) -> List[Tuple[float, float]]:
    """
    Create polygon from edge measurements
    This is a simplified version that creates a polygon from measured edges
    """
    # For now, create a simple rectangular polygon
    # In a full implementation, this would reconstruct the polygon from edge geometry

    # Extract measurements (convert to feet)
    edge_lengths = [float(measurements.get(str(i+1), 10)) for i in range(len(edges))]

    # Simple algorithm: assume first measurement is width, second is height
    # For complex polygons, this would need proper reconstruction
    if len(edge_lengths) >= 4:
        # Assume rectangular: width, height, width, height
        width = edge_lengths[0]
        height = edge_lengths[1]

        return [
            (0, 0),
            (width, 0),
            (width, height),
            (0, height)
        ]
    else:
        # Fallback: create square
        side = edge_lengths[0] if edge_lengths else 30
        return [
            (0, 0),
            (side, 0),
            (side, side),
            (0, side)
        ]


@app.route('/result/<session_id>')
def result(session_id):
    """Display optimization results"""
    try:
        # Load results
        results_file = Path(app.config['RESULTS_FOLDER']) / session_id / 'results.json'
        if not results_file.exists():
            return "Results not found", 404

        with open(results_file, 'r') as f:
            results_data = json.load(f)

        # Check if SVG exists
        svg_file = Path(app.config['RESULTS_FOLDER']) / session_id / 'cassette_layout.svg'
        if not svg_file.exists():
            return "SVG not generated", 500

        return render_template('result.html',
                             session_id=session_id,
                             statistics=results_data['statistics'])

    except Exception as e:
        return f"Error loading results: {str(e)}", 500


@app.route('/results/<session_id>/<filename>')
def serve_result(session_id, filename):
    """Serve result files"""
    file_path = Path(app.config['RESULTS_FOLDER']) / session_id / filename
    if file_path.exists():
        return send_file(str(file_path))
    return jsonify({'error': 'File not found'}), 404


@app.route('/download/<session_id>/svg')
def download_svg(session_id):
    """Download SVG file"""
    svg_file = Path(app.config['RESULTS_FOLDER']) / session_id / 'cassette_layout.svg'
    if svg_file.exists():
        return send_file(str(svg_file), as_attachment=True, download_name='floor_plan_optimized.svg')
    return jsonify({'error': 'SVG not found'}), 404


if __name__ == '__main__':
    # Create directories if they don't exist
    Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)
    Path(app.config['RESULTS_FOLDER']).mkdir(exist_ok=True)

    # Run Flask app
    print("Starting Cassette Optimizer Web App...")
    print("Access at: http://127.0.0.1:5000")
    app.run(debug=True, host='127.0.0.1', port=5000)

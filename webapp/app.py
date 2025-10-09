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
from typing import List, Dict, Tuple

# Add parent directory to path to import optimization modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add alternate parent directory for cardinal modules
alt_parent = "/Users/nelsondsouza/Documents/products/momohomes"
if alt_parent not in sys.path:
    sys.path.insert(0, alt_parent)

# Import edge processor
from edge_processor import EdgeProcessor

# Import cardinal system modules for polygon reconstruction
from cardinal_edge_detector import CardinalEdge
from cardinal_polygon_reconstructor import CardinalPolygonReconstructor
from cardinal_edge_mapper import CardinalEdgeMapper
from smart_edge_merger import SmartEdgeMerger

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

        # Store filename in session for later use
        session['original_filename'] = filename

        # Process edges using cardinal system
        edge_processor = EdgeProcessor()
        edges_image_path = upload_dir / "edges_labeled.png"
        binary_image_path = upload_dir / "binary.png"

        result = edge_processor.process_image(
            str(original_path),
            str(edges_image_path),
            str(binary_image_path)
        )

        if not result['success']:
            return jsonify({'error': f'Edge detection failed: {result["error"]}'}), 500

        # Store edge data in session
        session['edge_count'] = result['edge_count']
        session['edges'] = result['edges']

        # Store cardinal edges as serialized dict for polygon reconstruction
        # Convert numpy int32 to Python int for JSON serialization
        cardinal_edges_data = []
        for edge in result.get('cardinal_edges', []):
            cardinal_edges_data.append({
                'start': (int(edge.start[0]), int(edge.start[1])),
                'end': (int(edge.end[0]), int(edge.end[1])),
                'pixel_length': float(edge.pixel_length),
                'cardinal_direction': edge.cardinal_direction,
                'measurement': None
            })
        session['cardinal_edges'] = cardinal_edges_data

        # Return success with edge data and cardinal directions
        edge_directions = {}
        for i, edge_data in enumerate(cardinal_edges_data):
            edge_directions[str(i + 1)] = edge_data['cardinal_direction']

        return jsonify({
            'success': True,
            'session_id': session_id,
            'edge_count': result['edge_count'],
            'edges_image_url': f'/uploads/{session_id}/edges_labeled.png',
            'edge_directions': edge_directions
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


def validate_polygon_closure(cardinal_edges: List[Dict], measurements: Dict) -> Dict:
    """
    Validate that user measurements form a closed polygon.
    For rectilinear polygons: Sum(East) must equal Sum(West), Sum(North) must equal Sum(South)

    Args:
        cardinal_edges: List of edge data with cardinal directions
        measurements: Dict mapping edge_id to measurement value

    Returns:
        Dict with validation results:
        {
            'closes': bool,
            'east': float,
            'west': float,
            'north': float,
            'south': float,
            'errors': List[str]
        }
    """
    # Initialize directional sums
    sums = {'E': 0.0, 'W': 0.0, 'N': 0.0, 'S': 0.0}

    # Calculate sums from user measurements only
    # Skip edges with zero measurement (they don't exist)
    for i, edge_data in enumerate(cardinal_edges):
        edge_id = str(i + 1)  # User sees 1-indexed edges

        if edge_id in measurements:
            measurement = float(measurements[edge_id])

            # Only count non-zero measurements
            if measurement > 0.01:  # Small threshold for floating point comparison
                direction = edge_data['cardinal_direction']
                sums[direction] += measurement

    # Check closure constraints
    errors = []

    # Check East-West balance
    horizontal_diff = abs(sums['E'] - sums['W'])
    if horizontal_diff > 0.0:  # Exact closure required
        errors.append(
            f"East-West imbalance: East = {sums['E']:.2f} ft, West = {sums['W']:.2f} ft "
            f"(difference: {horizontal_diff:.2f} ft)"
        )

    # Check North-South balance
    vertical_diff = abs(sums['N'] - sums['S'])
    if vertical_diff > 0.0:  # Exact closure required
        errors.append(
            f"North-South imbalance: North = {sums['N']:.2f} ft, South = {sums['S']:.2f} ft "
            f"(difference: {vertical_diff:.2f} ft)"
        )

    # Determine if polygon closes
    closes = len(errors) == 0

    return {
        'closes': closes,
        'east': sums['E'],
        'west': sums['W'],
        'north': sums['N'],
        'south': sums['S'],
        'errors': errors
    }


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

        # Get cardinal edge data from session
        cardinal_edges = session.get('cardinal_edges', [])
        if not cardinal_edges:
            return jsonify({'error': 'No edge data found in session'}), 400

        # VALIDATE POLYGON CLOSURE BEFORE PROCEEDING
        validation_result = validate_polygon_closure(cardinal_edges, measurements)

        if not validation_result['closes']:
            # Polygon doesn't close - block optimization and return error
            error_message = "Polygon does not close. Please check your measurements:\n\n"
            error_message += "\n".join(validation_result['errors'])

            return jsonify({
                'error': error_message,
                'validation': validation_result
            }), 400

        # Convert measurements to polygon using CardinalPolygonReconstructor
        polygon = _create_polygon_from_measurements(cardinal_edges, measurements)

        # Run optimization
        optimizer = GapRedistributionOptimizer(polygon)
        result = optimizer.optimize()

        if not result:
            return jsonify({'error': 'Optimization failed'}), 500

        # Create results directory
        results_dir = Path(app.config['RESULTS_FOLDER']) / session_id
        results_dir.mkdir(parents=True, exist_ok=True)

        # Extract floor plan name from original filename (without extension)
        original_filename = session.get('original_filename', '')
        if original_filename:
            # Remove extension: "Umbra.png" -> "Umbra"
            floor_plan_name = Path(original_filename).stem
        else:
            floor_plan_name = None

        # Save results JSON
        results_file = results_dir / 'results.json'
        with open(results_file, 'w') as f:
            json.dump({
                'cassettes': result['cassettes'],
                'c_channels': result.get('c_channels', []),
                'statistics': result['statistics'],
                'polygon': polygon,
                'floor_plan_name': floor_plan_name
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
            floor_plan_name=floor_plan_name
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
    Create polygon from cardinal edge measurements using CardinalPolygonReconstructor
    This EXACTLY mirrors the flow in cassette_layout_system_cardinal.py

    Zero-measurement edges are filtered out (they don't exist in the actual floor plan)
    """
    # Step 1: Reconstruct CardinalEdge objects from session data
    cardinal_edges = []
    skipped_edges = []

    for i, edge_data in enumerate(edges):
        edge = CardinalEdge(
            start=tuple(edge_data['start']),
            end=tuple(edge_data['end'])
        )
        edge.pixel_length = edge_data['pixel_length']
        edge.cardinal_direction = edge_data['cardinal_direction']

        # Add measurement from user input (convert to float)
        # User sees 1-indexed edges (Edge 1, Edge 2, ...)
        # But internally we work with 0-indexed
        edge_id = i + 1
        if str(edge_id) in measurements:
            edge.measurement = float(measurements[str(edge_id)])
        else:
            edge.measurement = 0.0  # Default for missing measurements

        # FILTER: Skip edges with zero or near-zero measurements (they don't exist)
        if edge.measurement > 0.01:
            cardinal_edges.append(edge)
        else:
            skipped_edges.append(edge_id)
            print(f"Webapp: Skipping Edge {edge_id} (measurement: {edge.measurement:.3f} ft) - edge marked as non-existent")

    if skipped_edges:
        print(f"Webapp: Filtered out {len(skipped_edges)} zero-measurement edges: {skipped_edges}")
        print(f"Webapp: Remaining edges: {len(cardinal_edges)}")

    # Step 2: Build measurement dict (0-indexed for reconstructor)
    # FIX: Reconstructor expects {0: ..., 1: ..., 2: ...} NOT {1: ..., 2: ..., 3: ...}
    measurement_dict = {}
    for i, edge in enumerate(cardinal_edges):
        measurement_dict[i] = edge.measurement  # 0-indexed!

    # Step 3: Smart edge merging (same as cardinal system)
    # Merge edges with zero or minimal measurements
    edge_merger = SmartEdgeMerger(min_edge_length=0.5)
    original_edge_count = len(cardinal_edges)

    cardinal_edges, measurement_dict = edge_merger.merge_edges(cardinal_edges, measurement_dict)

    if len(cardinal_edges) < original_edge_count:
        print(f"Webapp: Edges merged: {original_edge_count} -> {len(cardinal_edges)}")

    # Step 4: Build polygon using CardinalPolygonReconstructor
    reconstructor = CardinalPolygonReconstructor()
    polygon = reconstructor.build_from_cardinal_measurements(cardinal_edges, measurement_dict)

    # Step 5: Normalize to positive coordinates
    polygon = reconstructor.normalize_to_positive()

    # Step 6: Simplify polygon to remove duplicate vertices (same as cardinal system)
    polygon = edge_merger.simplify_polygon(polygon)
    print(f"Webapp: Final polygon has {len(polygon)} vertices")

    # Log reconstruction details
    print(f"Webapp: Polygon area: {reconstructor.area:.1f} sq ft")
    print(f"Webapp: Polygon closure: {reconstructor.is_closed}, Error: {reconstructor.closure_error:.4f} ft")

    return polygon


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
    """Download SVG file with floor plan name"""
    svg_file = Path(app.config['RESULTS_FOLDER']) / session_id / 'cassette_layout.svg'
    if not svg_file.exists():
        return jsonify({'error': 'SVG not found'}), 404

    # Get floor plan name from results
    results_file = Path(app.config['RESULTS_FOLDER']) / session_id / 'results.json'
    floor_plan_name = None
    if results_file.exists():
        try:
            with open(results_file, 'r') as f:
                results_data = json.load(f)
                floor_plan_name = results_data.get('floor_plan_name')
        except:
            pass

    # Format filename
    if floor_plan_name:
        download_filename = f"{floor_plan_name}_Floor_Joist_Cassette_Plan.svg"
    else:
        download_filename = 'Floor_Joist_Cassette_Plan.svg'

    return send_file(str(svg_file), as_attachment=True, download_name=download_filename)


@app.route('/download/<session_id>/png')
def download_png(session_id):
    """Download PNG file with floor plan name"""
    png_file = Path(app.config['RESULTS_FOLDER']) / session_id / 'cassette_layout.png'
    if not png_file.exists():
        return jsonify({'error': 'PNG not found'}), 404

    # Get floor plan name from results
    results_file = Path(app.config['RESULTS_FOLDER']) / session_id / 'results.json'
    floor_plan_name = None
    if results_file.exists():
        try:
            with open(results_file, 'r') as f:
                results_data = json.load(f)
                floor_plan_name = results_data.get('floor_plan_name')
        except:
            pass

    # Format filename
    if floor_plan_name:
        download_filename = f"{floor_plan_name}_Floor_Joist_Cassette_Plan.png"
    else:
        download_filename = 'Floor_Joist_Cassette_Plan.png'

    return send_file(str(png_file), as_attachment=True, download_name=download_filename)


if __name__ == '__main__':
    # Create directories if they don't exist
    Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)
    Path(app.config['RESULTS_FOLDER']).mkdir(exist_ok=True)

    # Run Flask app
    print("Starting Cassette Optimizer Web App...")
    print("Access at: http://127.0.0.1:5001")
    app.run(debug=True, host='127.0.0.1', port=5001)

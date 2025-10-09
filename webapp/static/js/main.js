// Main JavaScript for Cassette Floor Plan Optimizer

// Global variables
let sessionId = null;
let edgeDirections = {}; // Store cardinal directions for each edge

// Wait for DOM to load
document.addEventListener('DOMContentLoaded', function() {
    // Initialize upload form handler
    initUploadForm();
});

/**
 * Initialize upload form submission handler
 */
function initUploadForm() {
    const uploadForm = document.getElementById('upload-form');
    const fileInput = document.getElementById('file-input');
    const uploadBtn = document.getElementById('upload-btn');
    const uploadBtnText = document.getElementById('upload-btn-text');
    const uploadBtnSpinner = document.getElementById('upload-btn-spinner');
    const errorAlert = document.getElementById('error-alert');

    uploadForm.addEventListener('submit', async function(e) {
        e.preventDefault();

        // Validate file selection
        if (!fileInput.files || fileInput.files.length === 0) {
            showError(errorAlert, 'Please select a file to upload');
            return;
        }

        const file = fileInput.files[0];

        // Validate file size (10MB max)
        const maxSize = 10 * 1024 * 1024; // 10MB in bytes
        if (file.size > maxSize) {
            showError(errorAlert, 'File size exceeds 10 MB limit');
            return;
        }

        // Validate file type
        const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg'];
        if (!allowedTypes.includes(file.type)) {
            showError(errorAlert, 'Invalid file type. Please upload PNG, JPG, or JPEG');
            return;
        }

        // Hide error
        errorAlert.classList.add('d-none');

        // Show loading state
        uploadBtn.disabled = true;
        uploadBtnText.textContent = 'Uploading...';
        uploadBtnSpinner.classList.remove('d-none');

        try {
            // Create FormData and append file
            const formData = new FormData();
            formData.append('file', file);

            // Send upload request
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok && data.success) {
                // Store session ID and edge directions
                sessionId = data.session_id;
                edgeDirections = data.edge_directions || {};

                // Show measurement section with edge image
                showMeasurementSection(data.edges_image_url, data.edge_count);
            } else {
                showError(errorAlert, data.error || 'Upload failed');
            }
        } catch (error) {
            showError(errorAlert, 'Network error: ' + error.message);
        } finally {
            // Reset loading state
            uploadBtn.disabled = false;
            uploadBtnText.textContent = 'Upload & Process';
            uploadBtnSpinner.classList.add('d-none');
        }
    });
}

/**
 * Show error message
 */
function showError(alertElement, message) {
    alertElement.textContent = message;
    alertElement.classList.remove('d-none');
}

/**
 * Show success message (temporary)
 */
function showSuccess(message) {
    const uploadSection = document.getElementById('upload-section');
    const successDiv = document.createElement('div');
    successDiv.className = 'alert alert-success mt-3';
    successDiv.textContent = message;
    uploadSection.appendChild(successDiv);

    // Remove after 5 seconds
    setTimeout(() => {
        successDiv.remove();
    }, 5000);
}

/**
 * Show measurement section (Phase 2)
 */
function showMeasurementSection(edgeImageUrl, edgeCount) {
    // Hide upload section
    document.getElementById('upload-section').classList.add('d-none');

    // Show measurement section
    const measurementSection = document.getElementById('measurement-section');
    measurementSection.classList.remove('d-none');

    // Set edge image
    document.getElementById('edge-image').src = edgeImageUrl;

    // Generate measurement inputs
    generateMeasurementInputs(edgeCount);

    // Initialize measurement form handler
    initMeasurementForm();

    // Initialize edge image pan & zoom
    initEdgeImagePanZoom();
}

/**
 * Generate measurement input fields (Phase 2) - Two-column layout
 */
function generateMeasurementInputs(edgeCount) {
    const container = document.getElementById('measurement-inputs');
    container.innerHTML = '';

    for (let i = 1; i <= edgeCount; i++) {
        const inputGroup = document.createElement('div');
        inputGroup.className = 'col-6 mb-2';
        inputGroup.innerHTML = `
            <label for="edge-${i}" class="form-label small">Edge ${i}:</label>
            <input type="number" class="form-control form-control-sm edge-measurement-input" id="edge-${i}" name="edge-${i}"
                   step="0.1" min="0" max="1000" required data-direction="" style="max-width: 120px;">
        `;
        container.appendChild(inputGroup);
    }

    // Add event listeners for live validation
    addClosureValidationListeners();
}

/**
 * Initialize measurement form handler (Phase 2)
 */
function initMeasurementForm() {
    const form = document.getElementById('measurement-form');
    form.addEventListener('submit', async function(e) {
        e.preventDefault();

        // Show loading overlay
        showLoadingOverlay();

        // Collect measurements
        const measurements = {};
        const inputs = form.querySelectorAll('input[type="number"]');
        inputs.forEach(input => {
            const edgeNum = input.id.replace('edge-', '');
            measurements[edgeNum] = parseFloat(input.value);
        });

        try {
            // Send optimization request
            const response = await fetch('/optimize', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    session_id: sessionId,
                    measurements: measurements
                })
            });

            const data = await response.json();

            if (response.ok && data.success) {
                // Redirect to results page
                window.location.href = `/result/${sessionId}`;
            } else {
                hideLoadingOverlay();

                // Check if this is a closure validation error
                const errorAlert = document.getElementById('measurement-error-alert');
                if (data.validation) {
                    // Format closure validation error with details
                    let errorMsg = '<strong>Polygon Closure Error</strong><br><br>';
                    errorMsg += 'The polygon does not close. Please check your measurements:<br><ul>';

                    if (data.validation.errors && data.validation.errors.length > 0) {
                        data.validation.errors.forEach(err => {
                            errorMsg += `<li>${err}</li>`;
                        });
                    }

                    errorMsg += '</ul>';
                    errorMsg += '<small>Current sums:<br>';
                    errorMsg += `East: ${data.validation.east} ft | West: ${data.validation.west} ft<br>`;
                    errorMsg += `North: ${data.validation.north} ft | South: ${data.validation.south} ft</small>`;

                    errorAlert.innerHTML = errorMsg;
                    errorAlert.classList.remove('d-none');
                } else {
                    // Generic error
                    showError(errorAlert, data.error || 'Optimization failed');
                }
            }
        } catch (error) {
            hideLoadingOverlay();
            showError(
                document.getElementById('measurement-error-alert'),
                'Network error: ' + error.message
            );
        }
    });
}

/**
 * Show loading overlay with stacking blocks spinner
 */
function showLoadingOverlay() {
    const overlay = document.getElementById('loading-overlay');
    const spinnerContainer = document.getElementById('spinner-container');

    // Create stacking blocks
    spinnerContainer.innerHTML = `
        <div class="block"></div>
        <div class="block"></div>
        <div class="block"></div>
    `;

    overlay.classList.remove('d-none');
}

/**
 * Hide loading overlay
 */
function hideLoadingOverlay() {
    document.getElementById('loading-overlay').classList.add('d-none');
}

/**
 * Add closure validation event listeners to measurement inputs
 */
function addClosureValidationListeners() {
    const inputs = document.querySelectorAll('.edge-measurement-input');
    inputs.forEach(input => {
        input.addEventListener('input', validateClosureRealtime);
        input.addEventListener('change', validateClosureRealtime);
    });

    // Initial validation
    validateClosureRealtime();
}

/**
 * Real-time closure validation
 */
function validateClosureRealtime() {
    // Calculate directional sums
    const sums = { E: 0.0, W: 0.0, N: 0.0, S: 0.0 };

    const inputs = document.querySelectorAll('.edge-measurement-input');
    inputs.forEach(input => {
        const edgeNum = input.id.replace('edge-', '');
        const measurement = parseFloat(input.value) || 0.0;
        const direction = edgeDirections[edgeNum];

        // Only count non-zero measurements
        if (measurement > 0.01 && direction) {
            sums[direction] += measurement;
        }
    });

    // Update UI with sums
    document.getElementById('closure-east').textContent = sums.E.toFixed(2) + ' ft';
    document.getElementById('closure-west').textContent = sums.W.toFixed(2) + ' ft';
    document.getElementById('closure-north').textContent = sums.N.toFixed(2) + ' ft';
    document.getElementById('closure-south').textContent = sums.S.toFixed(2) + ' ft';

    // Check closure
    const ewMatch = Math.abs(sums.E - sums.W) < 0.001;
    const nsMatch = Math.abs(sums.N - sums.S) < 0.001;
    const closes = ewMatch && nsMatch;

    // Update icons
    const ewIcon = document.getElementById('closure-ew-icon');
    const nsIcon = document.getElementById('closure-ns-icon');

    ewIcon.innerHTML = ewMatch ? '<span class="text-success">✓</span>' : '<span class="text-danger">✗</span>';
    nsIcon.innerHTML = nsMatch ? '<span class="text-success">✓</span>' : '<span class="text-danger">✗</span>';

    // Update status panel
    const statusAlert = document.getElementById('closure-status');
    const statusText = document.getElementById('closure-status-text');

    if (closes) {
        statusAlert.className = 'alert alert-success mb-0';
        statusText.textContent = 'Polygon Closes ✓';
    } else {
        statusAlert.className = 'alert alert-danger mb-0';
        let errorMsg = 'Polygon does not close: ';
        const errors = [];
        if (!ewMatch) {
            const diff = Math.abs(sums.E - sums.W);
            errors.push(`E-W difference: ${diff.toFixed(2)} ft`);
        }
        if (!nsMatch) {
            const diff = Math.abs(sums.N - sums.S);
            errors.push(`N-S difference: ${diff.toFixed(2)} ft`);
        }
        statusText.textContent = errorMsg + errors.join(', ');
    }

    // Enable/disable optimize button
    const optimizeBtn = document.getElementById('optimize-btn');
    if (closes) {
        optimizeBtn.disabled = false;
        optimizeBtn.title = '';
    } else {
        optimizeBtn.disabled = true;
        optimizeBtn.title = 'Polygon must close before optimization';
    }

    return closes;
}

/**
 * Initialize edge image pan & zoom functionality
 */
function initEdgeImagePanZoom() {
    const container = document.getElementById('edge-image-container');
    const image = document.getElementById('edge-image');
    const zoomInBtn = document.getElementById('edge-zoom-in');
    const zoomOutBtn = document.getElementById('edge-zoom-out');
    const zoomResetBtn = document.getElementById('edge-zoom-reset');

    let scale = 1;
    let translateX = 0;
    let translateY = 0;
    let isDragging = false;
    let startX = 0;
    let startY = 0;

    const MIN_SCALE = 0.1;
    const MAX_SCALE = 5;
    const ZOOM_STEP = 0.2;
    const PAN_STEP = 50;

    function updateTransform() {
        image.style.transform = `translate(${translateX}px, ${translateY}px) scale(${scale})`;
        zoomResetBtn.innerHTML = `<span style="font-size: 14px;">${Math.round(scale * 100)}%</span>`;
    }

    function centerImage() {
        const containerRect = container.getBoundingClientRect();
        const imageRect = image.getBoundingClientRect();
        translateX = (containerRect.width - imageRect.width) / 2;
        translateY = (containerRect.height - imageRect.height) / 2;
        updateTransform();
    }

    // Initialize: center the image when loaded
    image.onload = function() {
        centerImage();
    };
    if (image.complete && image.naturalWidth > 0) {
        centerImage();
    }

    // Zoom In
    zoomInBtn.addEventListener('click', () => {
        if (scale < MAX_SCALE) {
            const oldScale = scale;
            scale = Math.min(MAX_SCALE, scale + ZOOM_STEP);
            // Zoom towards center
            const rect = container.getBoundingClientRect();
            const centerX = rect.width / 2;
            const centerY = rect.height / 2;
            translateX = centerX - (centerX - translateX) * (scale / oldScale);
            translateY = centerY - (centerY - translateY) * (scale / oldScale);
            updateTransform();
        }
    });

    // Zoom Out
    zoomOutBtn.addEventListener('click', () => {
        if (scale > MIN_SCALE) {
            const oldScale = scale;
            scale = Math.max(MIN_SCALE, scale - ZOOM_STEP);
            // Zoom towards center
            const rect = container.getBoundingClientRect();
            const centerX = rect.width / 2;
            const centerY = rect.height / 2;
            translateX = centerX - (centerX - translateX) * (scale / oldScale);
            translateY = centerY - (centerY - translateY) * (scale / oldScale);
            updateTransform();
        }
    });

    // Reset Zoom
    zoomResetBtn.addEventListener('click', () => {
        scale = 1;
        centerImage();
    });

    // Mouse Wheel Zoom
    container.addEventListener('wheel', (e) => {
        e.preventDefault();
        const rect = container.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;

        const oldScale = scale;
        if (e.deltaY < 0) {
            scale = Math.min(MAX_SCALE, scale + ZOOM_STEP);
        } else {
            scale = Math.max(MIN_SCALE, scale - ZOOM_STEP);
        }

        // Zoom towards mouse position
        translateX = mouseX - (mouseX - translateX) * (scale / oldScale);
        translateY = mouseY - (mouseY - translateY) * (scale / oldScale);
        updateTransform();
    });

    // Mouse Drag Pan
    container.addEventListener('mousedown', (e) => {
        isDragging = true;
        startX = e.clientX - translateX;
        startY = e.clientY - translateY;
        container.style.cursor = 'grabbing';
    });

    document.addEventListener('mousemove', (e) => {
        if (isDragging) {
            translateX = e.clientX - startX;
            translateY = e.clientY - startY;
            updateTransform();
        }
    });

    document.addEventListener('mouseup', () => {
        if (isDragging) {
            isDragging = false;
            container.style.cursor = 'grab';
        }
    });

    // Arrow Keys Pan
    document.addEventListener('keydown', (e) => {
        if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.key)) {
            // Only pan if measurement section is visible and no input is focused
            if (!document.getElementById('measurement-section').classList.contains('d-none') &&
                document.activeElement.tagName !== 'INPUT') {
                e.preventDefault();
                switch(e.key) {
                    case 'ArrowUp':
                        translateY += PAN_STEP;
                        break;
                    case 'ArrowDown':
                        translateY -= PAN_STEP;
                        break;
                    case 'ArrowLeft':
                        translateX += PAN_STEP;
                        break;
                    case 'ArrowRight':
                        translateX -= PAN_STEP;
                        break;
                }
                updateTransform();
            }
        }
    });

    // Touch support for mobile
    let touchStartDist = 0;
    let touchStartScale = 1;

    container.addEventListener('touchstart', (e) => {
        if (e.touches.length === 2) {
            const dx = e.touches[0].clientX - e.touches[1].clientX;
            const dy = e.touches[0].clientY - e.touches[1].clientY;
            touchStartDist = Math.sqrt(dx * dx + dy * dy);
            touchStartScale = scale;
        } else if (e.touches.length === 1) {
            isDragging = true;
            startX = e.touches[0].clientX - translateX;
            startY = e.touches[0].clientY - translateY;
        }
    });

    container.addEventListener('touchmove', (e) => {
        e.preventDefault();
        if (e.touches.length === 2) {
            const dx = e.touches[0].clientX - e.touches[1].clientX;
            const dy = e.touches[0].clientY - e.touches[1].clientY;
            const dist = Math.sqrt(dx * dx + dy * dy);
            scale = Math.max(MIN_SCALE, Math.min(MAX_SCALE, touchStartScale * (dist / touchStartDist)));
            updateTransform();
        } else if (e.touches.length === 1 && isDragging) {
            translateX = e.touches[0].clientX - startX;
            translateY = e.touches[0].clientY - startY;
            updateTransform();
        }
    });

    container.addEventListener('touchend', () => {
        isDragging = false;
        touchStartDist = 0;
    });
}

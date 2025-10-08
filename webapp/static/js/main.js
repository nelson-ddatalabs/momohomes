// Main JavaScript for Cassette Floor Plan Optimizer

// Global variables
let sessionId = null;

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
                // Store session ID
                sessionId = data.session_id;

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
}

/**
 * Generate measurement input fields (Phase 2)
 */
function generateMeasurementInputs(edgeCount) {
    const container = document.getElementById('measurement-inputs');
    container.innerHTML = '';

    for (let i = 1; i <= edgeCount; i++) {
        const inputGroup = document.createElement('div');
        inputGroup.className = 'mb-3';
        inputGroup.innerHTML = `
            <label for="edge-${i}" class="form-label">Edge ${i} (feet):</label>
            <input type="number" class="form-control" id="edge-${i}" name="edge-${i}"
                   step="0.1" min="1" max="1000" required>
        `;
        container.appendChild(inputGroup);
    }
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
                showError(
                    document.getElementById('measurement-error-alert'),
                    data.error || 'Optimization failed'
                );
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

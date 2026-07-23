// app/static/js/dashboard.js
// Dashboard JavaScript for Leaf ID System

let selectedFile = null;

document.addEventListener('DOMContentLoaded', function() {
    // Setup drop zone
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const analyzeBtn = document.getElementById('analyzeBtn');
    
    // Click to browse
    dropZone.addEventListener('click', function() {
        fileInput.click();
    });
    
    // Drag and drop events
    dropZone.addEventListener('dragover', function(e) {
        e.preventDefault();
        this.classList.add('dragover');
    });
    
    dropZone.addEventListener('dragleave', function(e) {
        e.preventDefault();
        this.classList.remove('dragover');
    });
    
    dropZone.addEventListener('drop', function(e) {
        e.preventDefault();
        this.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });
    
    // File input change
    fileInput.addEventListener('change', function() {
        if (this.files.length > 0) {
            handleFile(this.files[0]);
        }
    });
});

function handleFile(file) {
    // Validate file type
    const validTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp'];
    if (!validTypes.includes(file.type)) {
        showError('Please upload a valid image file (JPG, PNG, WEBP, BMP)');
        return;
    }
    
    // Validate file size (max 50MB)
    if (file.size > 50 * 1024 * 1024) {
        showError('File too large. Please upload an image under 50MB.');
        return;
    }
    
    selectedFile = file;
    
    // Show preview
    const reader = new FileReader();
    reader.onload = function(e) {
        const dropZone = document.getElementById('dropZone');
        const preview = document.createElement('img');
        preview.src = e.target.result;
        preview.style.maxWidth = '200px';
        preview.style.maxHeight = '200px';
        preview.style.borderRadius = '8px';
        preview.style.marginTop = '16px';
        
        // Remove existing preview
        const existing = dropZone.querySelector('img');
        if (existing) existing.remove();
        
        // Add new preview
        dropZone.appendChild(preview);
        dropZone.querySelector('h3').textContent = '✅ ' + file.name;
        dropZone.querySelector('p').textContent = (file.size / 1024).toFixed(1) + ' KB';
    };
    reader.readAsDataURL(file);
    
    // Enable analyze button
    document.getElementById('analyzeBtn').disabled = false;
    hideError();
}

async function analyze() {
    if (!selectedFile) return;
    
    const loading = document.getElementById('loading');
    const results = document.getElementById('results');
    const errorDiv = document.getElementById('errorDiv');
    const btn = document.getElementById('analyzeBtn');
    const metricsDiv = document.getElementById('metrics');
    const imagesDiv = document.getElementById('images');
    
    loading.style.display = 'block';
    results.style.display = 'none';
    errorDiv.style.display = 'none';
    btn.disabled = true;
    
    const formData = new FormData();
    formData.append('file', selectedFile);
    
    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        loading.style.display = 'none';
        btn.disabled = false;
        
        if (response.ok && data.success) {
            displayResults(data);
        } else {
            showError(data.error || 'Analysis failed');
        }
    } catch (error) {
        loading.style.display = 'none';
        btn.disabled = false;
        showError('Connection error: ' + error.message);
    }
}

function displayResults(data) {
    const metricsDiv = document.getElementById('metrics');
    const imagesDiv = document.getElementById('images');
    const results = document.getElementById('results');
    
    results.style.display = 'block';
    
    // Health score color
    const healthScore = data.health_score || 0;
    const healthColor = healthScore >= 80 ? '#2e7d32' : healthScore >= 50 ? '#f9a825' : '#c62828';
    
    // Grade color
    const grade = data.quality_grade || 'F';
    const gradeColor = grade === 'A' ? '#2e7d32' : grade === 'B' ? '#f9a825' : '#c62828';
    
    // Build metrics
    let metricsHtml = `
        <div class="metric-card" style="animation-delay: 0s;">
            <div class="value" style="color: ${healthColor};">${healthScore.toFixed(1)}%</div>
            <div class="label">Health Score</div>
        </div>
        <div class="metric-card" style="animation-delay: 0.1s;">
            <div class="value">${data.nitrogen_status || 'Unknown'}</div>
            <div class="label">Nitrogen Status</div>
        </div>
        <div class="metric-card" style="animation-delay: 0.2s;">
            <div class="value">${data.stress_level || 'Unknown'}</div>
            <div class="label">Stress Level</div>
        </div>
        <div class="metric-card" style="animation-delay: 0.3s;">
            <div class="value" style="color: ${gradeColor};">${grade}</div>
            <div class="label">Quality Grade</div>
        </div>
        <div class="metric-card" style="animation-delay: 0.4s;">
            <div class="value">${(data.processing_time || 0).toFixed(2)}s</div>
            <div class="label">Processing Time</div>
        </div>
    `;
    
    // Add disease metrics if available
    if (data.disease) {
        const statusColor = data.disease.status === 'Healthy' ? '#2e7d32' : '#c62828';
        metricsHtml += `
            <div class="metric-card" style="animation-delay: 0.5s; border-left: 4px solid ${statusColor};">
                <div class="value" style="color: ${statusColor}; font-size: 20px;">${data.disease.status || 'Unknown'}</div>
                <div class="label">Disease Status</div>
            </div>
            <div class="metric-card" style="animation-delay: 0.6s;">
                <div class="value" style="color: ${data.disease.chlorosis > 10 ? '#f9a825' : '#2e7d32'};">${(data.disease.chlorosis || 0).toFixed(1)}%</div>
                <div class="label">Chlorosis</div>
            </div>
            <div class="metric-card" style="animation-delay: 0.7s;">
                <div class="value" style="color: ${data.disease.necrosis > 10 ? '#c62828' : '#2e7d32'};">${(data.disease.necrosis || 0).toFixed(1)}%</div>
                <div class="label">Necrosis</div>
            </div>
            <div class="metric-card" style="animation-delay: 0.8s;">
                <div class="value">${data.disease.spots || 0}</div>
                <div class="label">Spots</div>
            </div>
        `;
        
        // Add recommendations
        if (data.disease.recommendations && data.disease.recommendations.length > 0) {
            metricsHtml += `
                <div style="grid-column: 1 / -1; background: #e3f2fd; border-radius: 12px; padding: 16px 20px; animation: fadeIn 0.5s ease forwards; opacity: 0; animation-delay: 0.9s;">
                    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                        <span style="font-size: 20px;">💡</span>
                        <strong style="color: #0d47a1;">Recommendations</strong>
                    </div>
                    <ul style="margin: 0; padding-left: 20px; color: #333;">
                        ${data.disease.recommendations.map(r => `<li>${r}</li>`).join('')}
                    </ul>
                </div>
            `;
        }
    }
    
    metricsDiv.innerHTML = metricsHtml;
    
    // Build images
    let imagesHtml = '';
    if (data.leaf_image) {
        imagesHtml += `
            <div class="image-card" style="animation-delay: 0.2s;">
                <img src="data:image/jpeg;base64,${data.leaf_image}" alt="Isolated Leaf">
                <p class="caption">🌱 Isolated Leaf</p>
            </div>
        `;
    }
    if (data.mask) {
        imagesHtml += `
            <div class="image-card" style="animation-delay: 0.4s;">
                <img src="data:image/jpeg;base64,${data.mask}" alt="Segmentation Mask">
                <p class="caption">🎭 Segmentation Mask</p>
            </div>
        `;
    }
    imagesDiv.innerHTML = imagesHtml || '<p>No images available</p>';
    
    // Scroll to results
    results.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function showError(message) {
    const errorDiv = document.getElementById('errorDiv');
    document.getElementById('errorMsg').textContent = message;
    errorDiv.style.display = 'flex';
}

function hideError() {
    document.getElementById('errorDiv').style.display = 'none';
}
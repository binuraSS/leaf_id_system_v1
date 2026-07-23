// app/static/js/leaf_ui.js
/**
 * Leaf UI - Interactive leaf analysis interface
 * Provides advanced visualization and interaction features
 */

class LeafUI {
    constructor(options = {}) {
        this.container = options.container || document.body;
        this.theme = options.theme || 'light';
        this.onAnalyze = options.onAnalyze || null;
        this.onUpload = options.onUpload || null;
        
        this.state = {
            image: null,
            mask: null,
            result: null,
            loading: false
        };
        
        this.init();
    }
    
    init() {
        this.createUI();
        this.setupEventListeners();
    }
    
    createUI() {
        // Create main container
        this.uiContainer = document.createElement('div');
        this.uiContainer.className = `leaf-ui leaf-ui-${this.theme}`;
        this.uiContainer.innerHTML = `
            <style>
                .leaf-ui {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }
                
                .leaf-ui-dark {
                    background: #1a1a1a;
                    color: #e0e0e0;
                }
                
                .leaf-ui-light {
                    background: #f5f5f5;
                    color: #1a1a1a;
                }
                
                .leaf-ui-toolbar {
                    display: flex;
                    gap: 12px;
                    margin-bottom: 20px;
                    flex-wrap: wrap;
                    align-items: center;
                }
                
                .leaf-ui-toolbar button {
                    padding: 8px 16px;
                    border: none;
                    border-radius: 8px;
                    cursor: pointer;
                    font-weight: 500;
                    transition: all 0.2s;
                }
                
                .leaf-ui-toolbar button:hover {
                    transform: translateY(-1px);
                }
                
                .leaf-ui-toolbar .btn-primary {
                    background: #2e7d32;
                    color: white;
                }
                
                .leaf-ui-toolbar .btn-primary:hover {
                    background: #1b5e20;
                }
                
                .leaf-ui-toolbar .btn-secondary {
                    background: #e0e0e0;
                    color: #333;
                }
                
                .leaf-ui-toolbar .btn-secondary:hover {
                    background: #ccc;
                }
                
                .leaf-ui-toolbar .btn-danger {
                    background: #c62828;
                    color: white;
                }
                
                .leaf-ui-toolbar .btn-danger:hover {
                    background: #b71c1c;
                }
                
                .leaf-ui-grid {
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 20px;
                }
                
                .leaf-ui-panel {
                    background: rgba(255,255,255,0.05);
                    border-radius: 12px;
                    padding: 16px;
                    border: 1px solid rgba(255,255,255,0.1);
                }
                
                .leaf-ui-dark .leaf-ui-panel {
                    background: rgba(255,255,255,0.05);
                    border-color: rgba(255,255,255,0.1);
                }
                
                .leaf-ui-light .leaf-ui-panel {
                    background: white;
                    border-color: #e0e0e0;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
                }
                
                .leaf-ui-panel h3 {
                    margin: 0 0 12px 0;
                    font-size: 16px;
                    font-weight: 600;
                }
                
                .leaf-ui-panel .panel-content {
                    min-height: 200px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    flex-direction: column;
                }
                
                .leaf-ui-panel img {
                    max-width: 100%;
                    max-height: 400px;
                    border-radius: 8px;
                }
                
                .leaf-ui-dropzone {
                    border: 2px dashed #81c784;
                    border-radius: 12px;
                    padding: 40px;
                    text-align: center;
                    cursor: pointer;
                    transition: all 0.3s;
                }
                
                .leaf-ui-dropzone:hover {
                    border-color: #2e7d32;
                    background: rgba(46,125,50,0.05);
                }
                
                .leaf-ui-dropzone.dragover {
                    border-color: #2e7d32;
                    background: rgba(46,125,50,0.1);
                }
                
                .leaf-ui-dropzone .icon {
                    font-size: 48px;
                    margin-bottom: 12px;
                }
                
                .leaf-ui-dropzone .hint {
                    font-size: 13px;
                    color: #888;
                }
                
                .leaf-ui-metrics {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
                    gap: 12px;
                    margin-top: 12px;
                }
                
                .leaf-ui-metric {
                    background: rgba(255,255,255,0.05);
                    padding: 12px;
                    border-radius: 8px;
                    text-align: center;
                }
                
                .leaf-ui-light .leaf-ui-metric {
                    background: #f5f5f5;
                }
                
                .leaf-ui-metric .value {
                    font-size: 24px;
                    font-weight: 700;
                }
                
                .leaf-ui-metric .label {
                    font-size: 11px;
                    color: #888;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                }
                
                .leaf-ui-metric .value.healthy {
                    color: #2e7d32;
                }
                
                .leaf-ui-metric .value.warning {
                    color: #f9a825;
                }
                
                .leaf-ui-metric .value.danger {
                    color: #c62828;
                }
                
                .leaf-ui-loading {
                    display: none;
                    text-align: center;
                    padding: 40px;
                }
                
                .leaf-ui-loading .spinner {
                    width: 40px;
                    height: 40px;
                    border: 4px solid #e8f5e9;
                    border-top: 4px solid #2e7d32;
                    border-radius: 50%;
                    animation: spin 1s linear infinite;
                    margin: 0 auto 12px;
                }
                
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
                
                .leaf-ui-error {
                    display: none;
                    background: #ffebee;
                    color: #c62828;
                    padding: 16px 20px;
                    border-radius: 8px;
                    margin-top: 12px;
                    border-left: 4px solid #c62828;
                }
                
                .leaf-ui-results {
                    display: none;
                }
                
                @media (max-width: 768px) {
                    .leaf-ui-grid {
                        grid-template-columns: 1fr;
                    }
                    
                    .leaf-ui-toolbar {
                        justify-content: center;
                    }
                }
            </style>
            
            <div class="leaf-ui-toolbar">
                <button class="btn-primary" id="leafUIUpload">📤 Upload Image</button>
                <button class="btn-primary" id="leafUIAnalyze" disabled>🔬 Analyze</button>
                <button class="btn-secondary" id="leafUIReset">🔄 Reset</button>
                <button class="btn-secondary" id="leafUIExport">📊 Export Report</button>
                <input type="file" id="leafUIFileInput" accept="image/*" style="display:none;">
            </div>
            
            <div class="leaf-ui-loading" id="leafUILoading">
                <div class="spinner"></div>
                <p>Analyzing leaf...</p>
                <p style="font-size:13px;color:#888;">This may take a few seconds</p>
            </div>
            
            <div class="leaf-ui-error" id="leafUIError"></div>
            
            <div class="leaf-ui-grid">
                <div class="leaf-ui-panel">
                    <h3>📷 Original Image</h3>
                    <div class="panel-content" id="leafUIOriginal">
                        <div class="leaf-ui-dropzone" id="leafUIDropZone">
                            <div class="icon">🌿</div>
                            <p>Drop your leaf image here</p>
                            <p class="hint">or click to browse</p>
                        </div>
                    </div>
                </div>
                
                <div class="leaf-ui-panel">
                    <h3>🌱 Isolated Leaf</h3>
                    <div class="panel-content" id="leafUIIsolated">
                        <p style="color:#888;">Upload an image to see isolated leaf</p>
                    </div>
                </div>
                
                <div class="leaf-ui-panel">
                    <h3>🎭 Segmentation Mask</h3>
                    <div class="panel-content" id="leafUIMask">
                        <p style="color:#888;">Upload an image to see segmentation</p>
                    </div>
                </div>
                
                <div class="leaf-ui-panel">
                    <h3>📊 Analysis Results</h3>
                    <div class="panel-content" id="leafUIResults">
                        <p style="color:#888;">Results will appear here after analysis</p>
                    </div>
                </div>
            </div>
        `;
        
        this.container.appendChild(this.uiContainer);
    }
    
    setupEventListeners() {
        // Upload button
        const uploadBtn = this.uiContainer.querySelector('#leafUIUpload');
        const fileInput = this.uiContainer.querySelector('#leafUIFileInput');
        const dropZone = this.uiContainer.querySelector('#leafUIDropZone');
        const analyzeBtn = this.uiContainer.querySelector('#leafUIAnalyze');
        const resetBtn = this.uiContainer.querySelector('#leafUIReset');
        const exportBtn = this.uiContainer.querySelector('#leafUIExport');
        
        uploadBtn.addEventListener('click', () => fileInput.click());
        
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleFile(e.target.files[0]);
            }
        });
        
        // Drag and drop
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });
        
        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('dragover');
        });
        
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) {
                this.handleFile(e.dataTransfer.files[0]);
            }
        });
        
        dropZone.addEventListener('click', () => fileInput.click());
        
        // Analyze button
        analyzeBtn.addEventListener('click', () => this.analyze());
        
        // Reset button
        resetBtn.addEventListener('click', () => this.reset());
        
        // Export button
        exportBtn.addEventListener('click', () => this.exportReport());
    }
    
    handleFile(file) {
        const validTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp'];
        if (!validTypes.includes(file.type)) {
            this.showError('Please upload a valid image file');
            return;
        }
        
        if (file.size > 50 * 1024 * 1024) {
            this.showError('File too large. Max 50MB');
            return;
        }
        
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => {
                this.state.image = img;
                this.displayOriginal(img);
                this.uiContainer.querySelector('#leafUIAnalyze').disabled = false;
                this.hideError();
                
                if (this.onUpload) {
                    this.onUpload(img);
                }
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }
    
    displayOriginal(img) {
        const container = this.uiContainer.querySelector('#leafUIOriginal');
        container.innerHTML = `<img src="${img.src}" alt="Original leaf">`;
    }
    
    async analyze() {
        if (!this.state.image) return;
        
        const loading = this.uiContainer.querySelector('#leafUILoading');
        const results = this.uiContainer.querySelector('#leafUIResults');
        const error = this.uiContainer.querySelector('#leafUIError');
        const analyzeBtn = this.uiContainer.querySelector('#leafUIAnalyze');
        
        loading.style.display = 'block';
        results.innerHTML = '<p style="color:#888;">Analyzing...</p>';
        error.style.display = 'none';
        analyzeBtn.disabled = true;
        
        try {
            // Convert image to blob
            const canvas = document.createElement('canvas');
            canvas.width = this.state.image.width;
            canvas.height = this.state.image.height;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(this.state.image, 0, 0);
            const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg'));
            
            // Send to API
            const formData = new FormData();
            formData.append('file', blob, 'leaf.jpg');
            
            const response = await fetch('/api/analyze', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            loading.style.display = 'none';
            analyzeBtn.disabled = false;
            
            if (response.ok && data.success) {
                this.state.result = data;
                this.displayResults(data);
                
                if (this.onAnalyze) {
                    this.onAnalyze(data);
                }
            } else {
                this.showError(data.error || 'Analysis failed');
            }
        } catch (err) {
            loading.style.display = 'none';
            analyzeBtn.disabled = false;
            this.showError('Network error: ' + err.message);
        }
    }
    
    displayResults(data) {
        const container = this.uiContainer.querySelector('#leafUIResults');
        
        // Display isolated leaf
        if (data.leaf_image) {
            const isolated = this.uiContainer.querySelector('#leafUIIsolated');
            isolated.innerHTML = `<img src="data:image/jpeg;base64,${data.leaf_image}" alt="Isolated leaf">`;
        }
        
        // Display mask
        if (data.mask) {
            const maskContainer = this.uiContainer.querySelector('#leafUIMask');
            maskContainer.innerHTML = `<img src="data:image/jpeg;base64,${data.mask}" alt="Segmentation mask">`;
        }
        
        // Build metrics
        let metricsHtml = '<div class="leaf-ui-metrics">';
        
        if (data.health_score !== undefined) {
            const score = data.health_score || 0;
            const cls = score >= 80 ? 'healthy' : score >= 50 ? 'warning' : 'danger';
            metricsHtml += `
                <div class="leaf-ui-metric">
                    <div class="value ${cls}">${score.toFixed(1)}%</div>
                    <div class="label">Health Score</div>
                </div>
            `;
        }
        
        if (data.nitrogen_status) {
            metricsHtml += `
                <div class="leaf-ui-metric">
                    <div class="value">${data.nitrogen_status}</div>
                    <div class="label">Nitrogen</div>
                </div>
            `;
        }
        
        if (data.stress_level) {
            metricsHtml += `
                <div class="leaf-ui-metric">
                    <div class="value">${data.stress_level}</div>
                    <div class="label">Stress</div>
                </div>
            `;
        }
        
        if (data.quality_grade) {
            const cls = data.quality_grade === 'A' || data.quality_grade === 'B' ? 'healthy' : 'warning';
            metricsHtml += `
                <div class="leaf-ui-metric">
                    <div class="value ${cls}">${data.quality_grade}</div>
                    <div class="label">Quality</div>
                </div>
            `;
        }
        
        if (data.disease) {
            const statusColor = data.disease.status === 'Healthy' ? 'healthy' : 'danger';
            metricsHtml += `
                <div class="leaf-ui-metric">
                    <div class="value ${statusColor}">${data.disease.status}</div>
                    <div class="label">Disease</div>
                </div>
            `;
            
            if (data.disease.chlorosis !== undefined) {
                metricsHtml += `
                    <div class="leaf-ui-metric">
                        <div class="value">${data.disease.chlorosis.toFixed(1)}%</div>
                        <div class="label">Chlorosis</div>
                    </div>
                `;
            }
            
            if (data.disease.necrosis !== undefined) {
                metricsHtml += `
                    <div class="leaf-ui-metric">
                        <div class="value">${data.disease.necrosis.toFixed(1)}%</div>
                        <div class="label">Necrosis</div>
                    </div>
                `;
            }
        }
        
        metricsHtml += '</div>';
        
        // Add recommendations
        let recHtml = '';
        if (data.disease && data.disease.recommendations && data.disease.recommendations.length > 0) {
            recHtml = `
                <div style="margin-top: 12px; padding: 12px 16px; background: #e3f2fd; border-radius: 8px;">
                    <strong>💡 Recommendations:</strong>
                    <ul style="margin: 4px 0 0 0; padding-left: 20px;">
                        ${data.disease.recommendations.map(r => `<li>${r}</li>`).join('')}
                    </ul>
                </div>
            `;
        }
        
        container.innerHTML = metricsHtml + recHtml;
        container.style.display = 'block';
    }
    
    showError(message) {
        const error = this.uiContainer.querySelector('#leafUIError');
        error.textContent = '❌ ' + message;
        error.style.display = 'block';
    }
    
    hideError() {
        const error = this.uiContainer.querySelector('#leafUIError');
        error.style.display = 'none';
    }
    
    reset() {
        this.state = {
            image: null,
            mask: null,
            result: null,
            loading: false
        };
        
        // Reset UI
        const original = this.uiContainer.querySelector('#leafUIOriginal');
        const isolated = this.uiContainer.querySelector('#leafUIIsolated');
        const maskContainer = this.uiContainer.querySelector('#leafUIMask');
        const results = this.uiContainer.querySelector('#leafUIResults');
        const analyzeBtn = this.uiContainer.querySelector('#leafUIAnalyze');
        const fileInput = this.uiContainer.querySelector('#leafUIFileInput');
        
        original.innerHTML = `
            <div class="leaf-ui-dropzone" id="leafUIDropZone">
                <div class="icon">🌿</div>
                <p>Drop your leaf image here</p>
                <p class="hint">or click to browse</p>
            </div>
        `;
        isolated.innerHTML = '<p style="color:#888;">Upload an image to see isolated leaf</p>';
        maskContainer.innerHTML = '<p style="color:#888;">Upload an image to see segmentation</p>';
        results.innerHTML = '<p style="color:#888;">Results will appear here after analysis</p>';
        analyzeBtn.disabled = true;
        fileInput.value = '';
        this.hideError();
        
        // Re-bind drop zone
        this.setupEventListeners();
    }
    
    exportReport() {
        if (!this.state.result) {
            this.showError('No results to export. Please analyze an image first.');
            return;
        }
        
        // Build report data
        const reportData = {
            timestamp: Date.now(),
            health: {
                score: this.state.result.health_score,
                nitrogen_status: this.state.result.nitrogen_status,
                stress_level: this.state.result.stress_level
            },
            quality: {
                grade: this.state.result.quality_grade
            },
            disease: this.state.result.disease || {},
            leaf_image: this.state.result.leaf_image,
            mask: this.state.result.mask
        };
        
        // Open report in new window
        const encoded = encodeURIComponent(JSON.stringify(reportData));
        window.open(`/report?data=${encoded}`, '_blank');
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // Check if LeafUI should be auto-initialized
    const container = document.getElementById('leafUI');
    if (container) {
        window.leafUI = new LeafUI({ container });
    }
});

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = LeafUI;
}
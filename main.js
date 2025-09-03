/**
 * Medical Image Analysis System - Frontend JavaScript
 * Handles file upload, analysis progress, and result display
 */

class MedicalImageAnalyzer {
    constructor() {
        this.currentFile = null;
        this.analysisResults = null;
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.setupDragAndDrop();
    }

    setupEventListeners() {
        // File input change
        document.getElementById('fileInput').addEventListener('change', (e) => {
            this.handleFileSelect(e.target.files[0]);
        });

        // Analyze button click
        document.getElementById('analyzeBtn').addEventListener('click', () => {
            this.startAnalysis();
        });

        // Download buttons
        document.getElementById('downloadJson')?.addEventListener('click', () => {
            this.downloadResults('json');
        });

        document.getElementById('downloadCsv')?.addEventListener('click', () => {
            this.downloadResults('csv');
        });

        document.getElementById('downloadImages')?.addEventListener('click', () => {
            this.downloadResults('images');
        });
    }

    setupDragAndDrop() {
        const uploadBox = document.getElementById('uploadBox');

        uploadBox.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadBox.classList.add('dragover');
        });

        uploadBox.addEventListener('dragleave', (e) => {
            e.preventDefault();
            uploadBox.classList.remove('dragover');
        });

        uploadBox.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadBox.classList.remove('dragover');

            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleFileSelect(files[0]);
            }
        });

        // Click to upload
        uploadBox.addEventListener('click', () => {
            document.getElementById('fileInput').click();
        });
    }

    handleFileSelect(file) {
        if (!file) return;

        // Validate file type
        const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/tiff'];
        if (!validTypes.includes(file.type)) {
            this.showError('Please select a valid image file (PNG, JPG, JPEG, TIFF).');
            return;
        }

        // Validate file size (100MB limit)
        const maxSize = 100 * 1024 * 1024; // 100MB
        if (file.size > maxSize) {
            this.showError('File size must be less than 100MB.');
            return;
        }

        this.currentFile = file;
        this.showImagePreview(file);
        this.showAnalysisOptions();
    }

    showImagePreview(file) {
        const reader = new FileReader();

        reader.onload = (e) => {
            const img = document.getElementById('imagePreview');
            img.src = e.target.result;

            // Show image info
            const info = document.getElementById('imageInfo');
            info.innerHTML = `
                <div class="info-item"><strong>File Name:</strong> ${file.name}</div>
                <div class="info-item"><strong>File Size:</strong> ${this.formatFileSize(file.size)}</div>
                <div class="info-item"><strong>File Type:</strong> ${file.type}</div>
            `;

            // Show preview section
            this.showSection('previewSection');
        };

        reader.readAsDataURL(file);
    }

    showAnalysisOptions() {
        this.showSection('analysisOptions');
    }

    async startAnalysis() {
        if (!this.currentFile) {
            this.showError('Please select an image file first.');
            return;
        }

        this.hideAllSections();
        this.showSection('progressSection');
        this.showLoadingOverlay();

        try {
            // Prepare form data
            const formData = new FormData();
            formData.append('file', this.currentFile);
            formData.append('min_cell_size', document.getElementById('minCellSize').value);
            formData.append('max_cell_size', document.getElementById('maxCellSize').value);
            formData.append('analysis_type', document.getElementById('analysisType').value);

            // Start analysis with progress tracking
            const response = await this.uploadWithProgress(formData);

            if (response.ok) {
                const results = await response.json();
                this.analysisResults = results;
                this.displayResults(results);
            } else {
                const error = await response.json();
                this.showError(error.error || 'Analysis failed. Please try again.');
            }
        } catch (error) {
            console.error('Analysis error:', error);
            this.showError('Network error occurred. Please check your connection and try again.');
        } finally {
            this.hideLoadingOverlay();
        }
    }

    async uploadWithProgress(formData) {
        const progressFill = document.getElementById('progressFill');
        const progressStatus = document.getElementById('progressStatus');
        const processingSteps = document.getElementById('processingSteps');

        // Simulate analysis steps
        const steps = [
            'Uploading image...',
            'Loading and preprocessing image...',
            'Performing cell segmentation...',
            'Extracting cellular features...',
            'Classifying cell types...',
            'Performing quantitative analysis...',
            'Generating visualizations...',
            'Finalizing results...'
        ];

        let currentStep = 0;
        const updateProgress = () => {
            if (currentStep < steps.length) {
                const progress = ((currentStep + 1) / steps.length) * 100;
                progressFill.style.width = progress + '%';
                progressStatus.textContent = steps[currentStep];

                // Update processing steps display
                processingSteps.innerHTML = steps.map((step, index) => {
                    let className = '';
                    if (index < currentStep) className = 'completed';
                    else if (index === currentStep) className = 'active';

                    return `<div class="${className}">${step}</div>`;
                }).join('');

                currentStep++;

                if (currentStep <= steps.length) {
                    setTimeout(updateProgress, 1500); // Update every 1.5 seconds
                }
            }
        };

        // Start progress simulation
        updateProgress();

        // Make the actual request
        return fetch('/analyze', {
            method: 'POST',
            body: formData
        });
    }

    displayResults(results) {
        this.hideAllSections();
        this.showSection('resultsSection');

        // Update summary cards
        this.updateSummaryCards(results);

        // Update visualizations
        this.updateVisualizations(results);

        // Update detailed results
        this.updateDetailedResults(results);

        // Add fade-in animation
        document.getElementById('resultsSection').classList.add('fade-in');
    }

    updateSummaryCards(results) {
        const segResults = results.segmentation_results || {};
        const classResults = results.classification_results || {};
        const qualityResults = results.quantitative_analysis?.quality_metrics || {};

        document.getElementById('cellCount').textContent = segResults.total_cells_detected || 0;
        document.getElementById('avgArea').textContent = Math.round(segResults.average_cell_area || 0) + ' pxÂ²';
        document.getElementById('confidence').textContent = Math.round((classResults.classification_confidence || 0) * 100) + '%';
        document.getElementById('qualityScore').textContent = qualityResults.quality_grade || 'N/A';
    }

    updateVisualizations(results) {
        const visualizations = results.visualizations || {};

        if (visualizations.original) {
            document.getElementById('originalImage').src = visualizations.original;
        }

        if (visualizations.segmented) {
            document.getElementById('segmentedImage').src = visualizations.segmented;
        }

        if (visualizations.overlay) {
            document.getElementById('overlayImage').src = visualizations.overlay;
        }
    }

    updateDetailedResults(results) {
        const classificationSummary = document.getElementById('classificationSummary');
        const morphologicalAnalysis = document.getElementById('morphologicalAnalysis');

        // Classification summary
        const classResults = results.classification_results || {};
        const cellCounts = classResults.cell_type_counts || {};

        let classHtml = '<div class="classification-breakdown">';
        Object.keys(cellCounts).forEach(cellType => {
            const count = cellCounts[cellType];
            const percentage = classResults.cell_type_percentages?.[cellType] || 0;
            classHtml += `
                <div class="classification-item">
                    <span class="cell-type">${this.capitalizeFirst(cellType)}:</span>
                    <span class="cell-count">${count} cells (${percentage.toFixed(1)}%)</span>
                </div>
            `;
        });
        classHtml += '</div>';

        if (classResults.dominant_type) {
            classHtml += `<p class="mt-2"><strong>Dominant Cell Type:</strong> ${this.capitalizeFirst(classResults.dominant_type)}</p>`;
        }

        classificationSummary.innerHTML = classHtml;

        // Morphological analysis
        const morphResults = results.quantitative_analysis?.morphological_analysis || {};
        let morphHtml = '<div class="morphological-metrics">';

        Object.keys(morphResults).forEach(feature => {
            const stats = morphResults[feature];
            if (stats && typeof stats === 'object') {
                morphHtml += `
                    <div class="metric-group">
                        <h6>${this.formatFeatureName(feature)}</h6>
                        <div class="metric-stats">
                            <span>Mean: ${stats.mean?.toFixed(2) || 'N/A'}</span>
                            <span>Std: ${stats.std?.toFixed(2) || 'N/A'}</span>
                            <span>Range: ${stats.min?.toFixed(2) || 'N/A'} - ${stats.max?.toFixed(2) || 'N/A'}</span>
                        </div>
                    </div>
                `;
            }
        });
        morphHtml += '</div>';

        morphologicalAnalysis.innerHTML = morphHtml;
    }

    async downloadResults(type) {
        if (!this.analysisResults) {
            this.showError('No results available for download.');
            return;
        }

        try {
            let downloadData;
            let fileName;
            let mimeType;

            switch (type) {
                case 'json':
                    downloadData = JSON.stringify(this.analysisResults, null, 2);
                    fileName = `analysis_results_${Date.now()}.json`;
                    mimeType = 'application/json';
                    break;

                case 'csv':
                    downloadData = this.convertToCSV(this.analysisResults);
                    fileName = `analysis_results_${Date.now()}.csv`;
                    mimeType = 'text/csv';
                    break;

                case 'images':
                    this.downloadImages();
                    return;

                default:
                    throw new Error('Invalid download type');
            }

            // Create and download file
            const blob = new Blob([downloadData], { type: mimeType });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = fileName;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);

        } catch (error) {
            console.error('Download error:', error);
            this.showError('Failed to download results. Please try again.');
        }
    }

    convertToCSV(results) {
        // Create CSV from quantitative analysis
        const data = results.quantitative_analysis || {};
        let csv = 'Metric,Value\n';

        // Add summary statistics
        const summary = data.summary_statistics || {};
        Object.keys(summary).forEach(key => {
            csv += `${this.formatFeatureName(key)},${summary[key]}\n`;
        });

        // Add classification results
        const classification = results.classification_results || {};
        Object.keys(classification.cell_type_counts || {}).forEach(cellType => {
            csv += `${cellType}_count,${classification.cell_type_counts[cellType]}\n`;
        });

        return csv;
    }

    downloadImages() {
        const visualizations = this.analysisResults.visualizations || {};

        Object.keys(visualizations).forEach(imageType => {
            if (visualizations[imageType] && imageType !== 'error') {
                const link = document.createElement('a');
                link.href = visualizations[imageType];
                link.download = `${imageType}_${Date.now()}.png`;
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            }
        });
    }

    // Utility methods
    showSection(sectionId) {
        const section = document.getElementById(sectionId);
        if (section) {
            section.style.display = 'block';
            section.classList.add('fade-in');
        }
    }

    hideSection(sectionId) {
        const section = document.getElementById(sectionId);
        if (section) {
            section.style.display = 'none';
            section.classList.remove('fade-in');
        }
    }

    hideAllSections() {
        const sections = [
            'previewSection', 'analysisOptions', 'progressSection', 
            'resultsSection', 'errorSection'
        ];
        sections.forEach(id => this.hideSection(id));
    }

    showError(message) {
        this.hideAllSections();
        document.getElementById('errorMessage').textContent = message;
        this.showSection('errorSection');
        this.hideLoadingOverlay();
    }

    showLoadingOverlay() {
        document.getElementById('loadingOverlay').style.display = 'flex';
    }

    hideLoadingOverlay() {
        document.getElementById('loadingOverlay').style.display = 'none';
    }

    formatFileSize(bytes) {
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        if (bytes === 0) return '0 Bytes';
        const i = Math.floor(Math.log(bytes) / Math.log(1024));
        return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
    }

    capitalizeFirst(str) {
        return str.charAt(0).toUpperCase() + str.slice(1);
    }

    formatFeatureName(name) {
        return name.replace(/_/g, ' ')
                  .replace(/\b\w/g, l => l.toUpperCase());
    }
}

// Tab functionality
function showTab(tabName) {
    // Hide all tab panes
    const panes = document.querySelectorAll('.tab-pane');
    panes.forEach(pane => {
        pane.classList.remove('active');
    });

    // Remove active class from all buttons
    const buttons = document.querySelectorAll('.tab-btn');
    buttons.forEach(btn => {
        btn.classList.remove('active');
    });

    // Show selected tab pane
    document.getElementById(tabName).classList.add('active');

    // Add active class to clicked button
    event.target.classList.add('active');
}

// Reset analysis
function resetAnalysis() {
    const analyzer = window.medicalAnalyzer;
    analyzer.hideAllSections();
    analyzer.currentFile = null;
    analyzer.analysisResults = null;

    // Reset form
    document.getElementById('fileInput').value = '';
    document.getElementById('minCellSize').value = '50';
    document.getElementById('maxCellSize').value = '5000';
    document.getElementById('analysisType').value = 'comprehensive';

    // Reset progress
    document.getElementById('progressFill').style.width = '0%';
}

// Initialize application
document.addEventListener('DOMContentLoaded', () => {
    window.medicalAnalyzer = new MedicalImageAnalyzer();

    // Add some startup animations
    document.querySelector('.container').classList.add('fade-in');
});

// Handle window resize for responsive charts
window.addEventListener('resize', () => {
    // Resize any charts or visualizations if needed
    // This can be expanded when adding interactive charts
});

// Service worker registration for offline functionality (optional)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/sw.js')
            .then(registration => {
                console.log('SW registered: ', registration);
            })
            .catch(registrationError => {
                console.log('SW registration failed: ', registrationError);
            });
    });
}
"""
Medical Image Analysis System
Features: Cell segmentation, classification, and quantification of high-resolution images
"""

import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import base64
from PIL import Image
import io
import json
from datetime import datetime

from cell_segmentation import CellSegmentationModel
from feature_extractor import FeatureExtractor
from classifier import CellClassifier
from image_preprocessing import ImagePreprocessor
from postprocessing import ResultProcessor

app = Flask(__name__)
app.config['SECRET_KEY'] = 'medical-image-analysis-2025'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Initialize models
segmentation_model = CellSegmentationModel()
feature_extractor = FeatureExtractor()
classifier = CellClassifier()
preprocessor = ImagePreprocessor()
result_processor = ResultProcessor()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tiff', 'tif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_image():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, JPEG, TIFF files'}), 400

        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)

        # Process image
        analysis_results = process_medical_image(filepath, unique_filename)

        return jsonify(analysis_results)

    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

def process_medical_image(filepath, filename):
    results = {
        'filename': filename,
        'timestamp': datetime.now().isoformat(),
        'processing_steps': [],
        'segmentation_results': {},
        'classification_results': {},
        'quantitative_analysis': {},
        'visualizations': {}
    }

    # Step 1: Load and preprocess image
    results['processing_steps'].append('Loading and preprocessing image...')
    image = cv2.imread(filepath)
    if image is None:
        raise ValueError("Could not load image file")

    original_shape = image.shape
    preprocessed_image = preprocessor.enhance_image(image)

    # Step 2: Cell segmentation (use original image, not preprocessed)
    results['processing_steps'].append('Performing cell segmentation...')
    segmented_image, cell_masks, cell_contours = segmentation_model.segment_cells(image)

    # Step 3: Feature extraction
    results['processing_steps'].append('Extracting cellular features...')
    features = feature_extractor.extract_features(preprocessed_image, cell_masks, cell_contours)

    # Step 4: Classification
    results['processing_steps'].append('Classifying cell types...')
    classifications = classifier.classify_cells(features)

    # Step 5: Quantitative analysis
    results['processing_steps'].append('Performing quantitative analysis...')
    quantitative_data = result_processor.analyze_results(
        cell_masks, features, classifications, original_shape
    )

    # Step 6: Generate visualizations
    results['processing_steps'].append('Generating visualizations...')
    visualizations = create_visualizations(
        image, segmented_image, cell_masks, classifications, filename
    )

    # Populate results
    results['segmentation_results'] = {
        'total_cells_detected': len(cell_masks),
        'segmentation_method': 'Watershed with marker-controlled segmentation',
        'average_cell_area': quantitative_data.get('avg_cell_area', 0),
        'cell_size_distribution': quantitative_data.get('size_distribution', {})
    }

    results['classification_results'] = {
        'cell_type_counts': quantitative_data.get('cell_type_counts', {}),
        'classification_confidence': quantitative_data.get('avg_confidence', 0),
        'dominant_cell_type': quantitative_data.get('dominant_type', 'Unknown')
    }

    results['quantitative_analysis'] = quantitative_data
    results['visualizations'] = visualizations

    return results

def create_visualizations(original_image, segmented_image, cell_masks, classifications, filename):
    base_filename = filename.split('.')[0]
    visualizations = {}

    try:
        # Original image
        orig_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{base_filename}_original.png")
        cv2.imwrite(orig_path, original_image)
        visualizations['original'] = image_to_base64(orig_path)

        # Segmented image
        seg_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{base_filename}_segmented.png")
        cv2.imwrite(seg_path, segmented_image)
        visualizations['segmented'] = image_to_base64(seg_path)

        # Cell overlay
        overlay_image = create_cell_overlay(original_image, cell_masks, classifications)
        overlay_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{base_filename}_overlay.png")
        cv2.imwrite(overlay_path, overlay_image)
        visualizations['overlay'] = image_to_base64(overlay_path)

    except Exception as e:
        print(f"Error creating visualizations: {e}")
        visualizations['error'] = str(e)

    return visualizations

def create_cell_overlay(image, cell_masks, classifications):
    overlay = image.copy()
    colors = {
        'normal': (0, 255, 0),     # Green
        'abnormal': (0, 0, 255),   # Red
        'uncertain': (255, 255, 0)  # Yellow
    }

    for i, mask in enumerate(cell_masks):
        if i < len(classifications):
            cell_type = classifications[i]['type']
            color = colors.get(cell_type, (255, 255, 255))

            # Draw contour
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, color, 2)

    return overlay

def image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        return f"data:image/png;base64,{encoded_string}"
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return None

@app.route('/results/<filename>')
def view_results(filename):
    return render_template('results.html', filename=filename)

@app.route('/download/<filename>')
def download_results(filename):
    """Download analysis results as JSON."""
    try:
        results_file = os.path.join(app.config['OUTPUT_FOLDER'], f"{filename}_results.json")
        if os.path.exists(results_file):
            return send_file(results_file, as_attachment=True)
        else:
            return jsonify({'error': 'Results file not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint for deployment monitoring."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': {
            'segmentation': segmentation_model is not None,
            'feature_extractor': feature_extractor is not None,
            'classifier': classifier is not None
        }
    })

if __name__ == '__main__':
    print("Medical Image Analysis System Starting...")
    print("Models loaded successfully")
    print("Starting web server on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
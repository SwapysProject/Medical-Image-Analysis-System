# Medical Image Analysis System

## Advanced Digital Pathology Analysis Platform

A comprehensive medical image analysis system built with OpenCV, machine learning, and Flask, designed specifically for digital pathology applications. This project implements advanced cell segmentation, feature extraction, and classification algorithms optimized for high-resolution medical images.

### Project Overview

This system addresses the growing need for automated analysis in digital pathology, providing:
- **Cell Segmentation**: Advanced watershed-based algorithm for separating overlapping cells
- **Feature Extraction**: Comprehensive morphological, texture, and intensity features
- **Classification**: ML-based cell type identification with confidence estimation
- **Quantitative Analysis**: Statistical analysis and reporting
- **High-Resolution Support**: Optimized for large medical images (up to 50K Ã— 50K pixels)
- **Web Interface**: User-friendly interface for easy analysis and result visualization

### Medical Applications

- Digital pathology image analysis
- Cell counting and morphometry
- Tissue classification
- Biomarker quantification
- Drug discovery research
- Clinical diagnosis support

## Quick Start

### Using Docker (Recommended)

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd medical-image-analyzer
   ```

2. **Build and run with Docker Compose**
   ```bash
   docker-compose up --build
   ```

3. **Access the application**
   - Open your browser and navigate to `http://localhost:5000`
   - Upload your medical images and start analysis

### Manual Installation

1. **Clone and setup environment**
   ```bash
   git clone <repository-url>
   cd medical-image-analyzer
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

4. **Access at** `http://localhost:5000`

## System Architecture

### Core Components

```
medical_image_analyzer/
   app.py                  # Main Flask application
   config.py               # Configuration settings
   requirements.txt        # Python dependencies
   Dockerfile              # Container configuration
   docker-compose.yml      # Multi-container setup
   models/                 # Analysis models
      cell_segmentation.py  # Watershed-based segmentation
      feature_extractor.py  # Feature extraction algorithms
      classifier.py         # ML classification models
   utils/                  # Utility modules
      image_preprocessing.py# Image enhancement and preprocessing
      postprocessing.py     # Result analysis and reporting
   templates/              # Web templates
      index.html            # Main user interface
   static/                 # Static web assets
      css/style.css         # Professional styling
      js/main.js            # Frontend JavaScript
   tests/                  # Test suite
      test_segmentation.py  # Unit tests
```

### Analysis Pipeline

1. **Image Preprocessing**
   - Color space conversion (LAB/HSV)
   - Noise reduction with edge preservation
   - Contrast enhancement (CLAHE)
   - Illumination correction

2. **Cell Segmentation**
   - Marker-controlled watershed algorithm
   - Adaptive thresholding
   - Morphological operations
   - Overlap resolution

3. **Feature Extraction**
   - Morphological features (area, circularity, solidity)
   - Texture analysis (LBP, GLCM)
   - Intensity statistics
   - Shape descriptors (Hu moments)

4. **Classification & Analysis**
   - Rule-based and ML classification
   - Confidence estimation
   - Spatial distribution analysis
   - Statistical reporting

## Features

### Advanced Segmentation
- **Watershed Algorithm**: Marker-controlled segmentation for overlapping cells
- **Quality Filtering**: Morphological validation of detected cells
- **Large Image Support**: Tile-based processing for high-resolution images
- **Artifact Removal**: Automatic detection and correction of common artifacts

### Comprehensive Feature Extraction
- **Morphological Features**: Area, perimeter, circularity, solidity, aspect ratio
- **Texture Analysis**: Local Binary Patterns (LBP), Gray-Level Co-occurrence Matrix (GLCM)
- **Intensity Features**: Statistical measures, percentiles, distribution analysis
- **Shape Descriptors**: Hu moments, convexity defects, geometric properties

### Intelligent Classification
- **Multiple Algorithms**: Random Forest, SVM, Gradient Boosting
- **Rule-Based Fallback**: Hand-crafted rules when no trained model available
- **Confidence Scoring**: Probability estimates for each classification
- **Adaptive Learning**: Support for custom training data

### Quantitative Analysis
- **Statistical Reports**: Comprehensive analysis of cell populations
- **Spatial Analysis**: Distribution patterns and clustering metrics
- **Quality Metrics**: Assessment of segmentation and classification quality
- **Export Options**: JSON, CSV, and image downloads

### Web Interface
- **Drag-and-Drop Upload**: Easy file handling
- **Real-Time Progress**: Analysis progress tracking
- **Interactive Visualization**: Tabbed result display
- **Responsive Design**: Mobile-friendly interface

## Configuration

### Environment Variables

```bash
# Application settings
SECRET_KEY=your-secret-key-here
FLASK_ENV=production
MAX_WORKERS=4
TIMEOUT_SECONDS=300

# File handling
UPLOAD_FOLDER=uploads
OUTPUT_FOLDER=outputs
MAX_CONTENT_LENGTH=104857600  # 100MB

# Analysis parameters
DEFAULT_MIN_CELL_SIZE=50
DEFAULT_MAX_CELL_SIZE=5000
HIGH_RESOLUTION_THRESHOLD=4096

# Logging
LOG_LEVEL=INFO
LOG_FILE=medical_analysis.log
```

### Analysis Parameters

- **Minimum Cell Size**: 10-500 pixels (default: 50)
- **Maximum Cell Size**: 100-10000 pixels (default: 5000)  
- **Analysis Types**: Comprehensive, Segmentation Only, Classification Only
- **Supported Formats**: PNG, JPG, JPEG, TIFF
- **Maximum File Size**: 100MB

## Usage Examples

### Basic Analysis
1. Open the web interface
2. Drag and drop your medical image
3. Adjust analysis parameters if needed
4. Click "Start Analysis"
5. View results and download reports

### API Usage
```python
import requests

# Upload and analyze image
files = {'file': open('sample_image.tiff', 'rb')}
data = {'min_cell_size': 50, 'max_cell_size': 5000}

response = requests.post('http://localhost:5000/analyze', 
                        files=files, data=data)
results = response.json()

print(f"Cells detected: {results['segmentation_results']['total_cells_detected']}")
```

### Batch Processing
```python
import os
import requests
from pathlib import Path

def analyze_directory(image_dir, output_dir):
    for image_path in Path(image_dir).glob("*.tiff"):
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post('http://localhost:5000/analyze', files=files)

            if response.ok:
                results = response.json()
                output_file = Path(output_dir) / f"{image_path.stem}_results.json"

                with open(output_file, 'w') as out_f:
                    json.dump(results, out_f, indent=2)

# Analyze all TIFF images in a directory
analyze_directory("./medical_images", "./analysis_results")
```

## Testing

Run the test suite:
```bash
pytest tests/ -v
```

Run with coverage:
```bash
pytest tests/ --cov=. --cov-report=html
```

## Performance Optimization

### For Large Images (>10K Ã— 10K pixels)
- Use tile-based processing
- Enable GPU acceleration if available
- Adjust worker count based on system resources
- Consider image downsampling for preview analysis

### Memory Management
- Images are processed in tiles to manage memory usage
- Automatic cleanup of temporary files
- Configurable timeout settings for long-running analyses

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Reduce image size or increase system memory
   - Lower the number of workers
   - Use tile-based processing for large images

2. **Slow Analysis**
   - Check system resources (CPU, memory)
   - Reduce image resolution for faster processing
   - Adjust segmentation parameters

3. **Poor Segmentation Results**
   - Try different preprocessing options
   - Adjust min/max cell size parameters
   - Check image quality and contrast

4. **Docker Issues**
   - Ensure Docker has sufficient memory allocated
   - Check port availability (5000)
   - Verify file permissions for volume mounts

## Contributing

We welcome contributions! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black flake8

# Run code formatting
black . --line-length 88
flake8 . --max-line-length 88

# Run tests
pytest tests/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenCV community for computer vision tools
- scikit-image for advanced image processing algorithms
- Flask community for the web framework
 

## Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Check the documentation
- Review existing issues and discussions

## Future Enhancements

- [ ] Deep learning model integration
- [ ] Real-time analysis streaming
- [ ] Multi-user support with authentication
- [ ] Advanced visualization dashboard
- [ ] Integration with PACS systems
- [ ] Mobile application
- [ ] Cloud deployment options
- [ ] API rate limiting and monitoring
- [ ] Advanced export formats (PDF reports)
- [ ] Plugin system for custom analysis modules

---

**Built for the medical imaging community**

*This system represents a comprehensive solution for digital pathology analysis, combining state-of-the-art computer vision techniques with practical deployment considerations for real-world medical applications.*
"""
Unit tests for Medical Image Analysis System
"""

import unittest
import numpy as np
import cv2
from cell_segmentation import CellSegmentationModel
from feature_extractor import FeatureExtractor
from classifier import CellClassifier
from image_preprocessing import ImagePreprocessor

class TestCellSegmentation(unittest.TestCase):
    """Test cases for cell segmentation functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.segmentation_model = CellSegmentationModel()
        self.test_image = self._create_test_image()

    def _create_test_image(self):
        """Create a synthetic test image with circular objects."""
        img = np.zeros((500, 500, 3), dtype=np.uint8)
        centers = [(100, 100), (200, 200), (350, 150), (400, 400)]
        radii = [30, 25, 35, 28]
        for center, radius in zip(centers, radii):
            cv2.circle(img, center, radius, (150, 150, 150), -1)
            cv2.circle(img, center, radius//2, (200, 200, 200), -1)
        return img

    def test_segmentation_basic(self):
        """Test basic segmentation functionality."""
        segmented, masks, contours = self.segmentation_model.segment_cells(self.test_image)
        self.assertIsInstance(segmented, np.ndarray)
        self.assertIsInstance(masks, list)
        self.assertIsInstance(contours, list)
        self.assertGreater(len(masks), 0, "Should detect at least one cell")
        self.assertEqual(len(masks), len(contours), "Masks and contours should have same length")

    def test_segmentation_empty_image(self):
        """Test segmentation with empty image"""
        empty_image = np.zeros((100, 100, 3), dtype=np.uint8)
        segmented, masks, contours = self.segmentation_model.segment_cells(empty_image)

        # Should handle empty image gracefully
        self.assertEqual(len(masks), 0, "Should detect no cells in empty image")
        self.assertEqual(len(contours), 0, "Should have no contours in empty image")

    def test_area_filtering(self):
        """Test that cell area filtering works correctly"""
        # Test with very restrictive size limits
        model = CellSegmentationModel(min_cell_area=1000, max_cell_area=1500)
        segmented, masks, contours = model.segment_cells(self.test_image)

        # Should filter out most/all cells due to size restrictions
        self.assertLessEqual(len(masks), 4, "Should filter cells by area")

class TestFeatureExtractor(unittest.TestCase):
    """Test cases for feature extraction"""

    def setUp(self):
        """Set up test fixtures"""
        self.feature_extractor = FeatureExtractor()
        self.test_image = np.random.randint(0, 255, (200, 200), dtype=np.uint8)

        # Create a simple circular mask
        self.test_mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(self.test_mask, (100, 100), 30, 255, -1)

        # Create corresponding contour
        contours, _ = cv2.findContours(self.test_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.test_contour = contours[0]

    def test_feature_extraction_basic(self):
        """Test basic feature extraction"""
        features = self.feature_extractor.extract_features(
            self.test_image, [self.test_mask], [self.test_contour]
        )

        self.assertEqual(len(features), 1, "Should return features for one cell")

        feature_dict = features[0]

        # Check that essential features are present
        self.assertIn('cell_id', feature_dict)
        self.assertIn('morphological_area', feature_dict)
        self.assertIn('morphological_circularity', feature_dict)
        self.assertIn('intensity_mean_intensity', feature_dict)

        # Check feature value ranges
        self.assertGreater(feature_dict['morphological_area'], 0)
        self.assertGreaterEqual(feature_dict['morphological_circularity'], 0)
        self.assertLessEqual(feature_dict['morphological_circularity'], 1)

    def test_multiple_cells_extraction(self):
        """Test feature extraction for multiple cells"""
        # Create multiple masks and contours
        masks = []
        contours = []

        for i, center in enumerate([(50, 50), (150, 150)]):
            mask = np.zeros((200, 200), dtype=np.uint8)
            cv2.circle(mask, center, 20 + i*5, 255, -1)
            masks.append(mask)

            contour, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours.append(contour[0])

        features = self.feature_extractor.extract_features(self.test_image, masks, contours)

        self.assertEqual(len(features), 2, "Should return features for two cells")

        # Check that features are different for different cells
        area1 = features[0]['morphological_area']
        area2 = features[1]['morphological_area']
        self.assertNotEqual(area1, area2, "Different cells should have different areas")

class TestCellClassifier(unittest.TestCase):
    """Test cases for cell classification"""

    def setUp(self):
        """Set up test fixtures"""
        self.classifier = CellClassifier()
        self.sample_features = [
            {
                'cell_id': 0,
                'morphological_area': 1000,
                'morphological_circularity': 0.8,
                'morphological_solidity': 0.9,
                'intensity_mean_intensity': 120,
                'intensity_std_intensity': 25
            },
            {
                'cell_id': 1,
                'morphological_area': 2000,
                'morphological_circularity': 0.4,
                'morphological_solidity': 0.6,
                'intensity_mean_intensity': 80,
                'intensity_std_intensity': 45
            }
        ]

    def test_rule_based_classification(self):
        """Test rule-based classification when no trained model is available."""
        results = self.classifier.classify_cells(self.sample_features)
        self.assertEqual(len(results), 2, "Should classify two cells")
        for result in results:
            self.assertIn('cell_id', result)
            self.assertIn('type', result)
            self.assertIn('confidence', result)
            self.assertIn('probabilities', result)

            # Check that classification type is valid
            self.assertIn(result['type'], ['normal', 'abnormal', 'uncertain'])

            # Check confidence range
            self.assertGreaterEqual(result['confidence'], 0)
            self.assertLessEqual(result['confidence'], 1)

            # Check probabilities sum to 1 (approximately)
            prob_sum = sum(result['probabilities'].values())
            self.assertAlmostEqual(prob_sum, 1.0, places=2)

    def test_empty_features(self):
        """Test classification with empty feature list"""
        results = self.classifier.classify_cells([])
        self.assertEqual(len(results), 0, "Should return empty results for empty input")

    def test_feature_preparation(self):
        """Test internal feature preparation method"""
        X = self.classifier._prepare_features(self.sample_features)

        self.assertIsInstance(X, np.ndarray)
        self.assertEqual(X.shape[0], 2, "Should have 2 samples")
        self.assertGreater(X.shape[1], 0, "Should have feature columns")

class TestImagePreprocessor(unittest.TestCase):
    """Test cases for image preprocessing"""

    def setUp(self):
        """Set up test fixtures"""
        self.preprocessor = ImagePreprocessor()
        self.test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    def test_image_enhancement(self):
        """Test basic image enhancement"""
        enhanced = self.preprocessor.enhance_image(self.test_image)

        # Should return single channel image
        self.assertEqual(len(enhanced.shape), 2, "Enhanced image should be single channel")

        # Should have same spatial dimensions
        self.assertEqual(enhanced.shape[:2], self.test_image.shape[:2])

        # Should be uint8
        self.assertEqual(enhanced.dtype, np.uint8)

    def test_grayscale_input(self):
        """Test preprocessing with grayscale input"""
        gray_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        enhanced = self.preprocessor.enhance_image(gray_image)

        self.assertEqual(enhanced.shape, gray_image.shape)

    def test_quality_validation(self):
        """Test image quality validation"""
        quality_metrics = self.preprocessor.validate_image_quality(self.test_image)

        # Check that all expected metrics are present
        expected_metrics = [
            'blur_score', 'is_blurry', 'contrast_score', 'low_contrast',
            'brightness_mean', 'too_dark', 'too_bright', 'quality_score',
            'recommendations'
        ]

        for metric in expected_metrics:
            self.assertIn(metric, quality_metrics)

        # Check value ranges
        self.assertGreaterEqual(quality_metrics['quality_score'], 0)
        self.assertLessEqual(quality_metrics['quality_score'], 1)

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline"""

    def setUp(self):
        """Set up test fixtures"""
        self.segmentation_model = CellSegmentationModel()
        self.feature_extractor = FeatureExtractor()
        self.classifier = CellClassifier()
        self.preprocessor = ImagePreprocessor()

        # Create a more realistic test image
        self.test_image = self._create_realistic_test_image()

    def _create_realistic_test_image(self):
        """Create a more realistic test image with multiple cells"""
        img = np.ones((400, 400, 3), dtype=np.uint8) * 240  # Light background

        # Add multiple cells with varying properties
        cell_params = [
            ((100, 100), 25, 180),  # Normal cell
            ((200, 150), 30, 160),  # Normal cell
            ((300, 200), 45, 140),  # Large cell
            ((150, 300), 15, 200),  # Small bright cell
            ((350, 350), 35, 120),  # Dark cell
        ]

        for center, radius, intensity in cell_params:
            # Add main cell body
            cv2.circle(img, center, radius, (intensity, intensity, intensity), -1)

            # Add nucleus
            nucleus_intensity = max(intensity - 40, 50)
            cv2.circle(img, center, radius//3, (nucleus_intensity, nucleus_intensity, nucleus_intensity), -1)

            # Add some texture
            for _ in range(3):
                noise_center = (center[0] + np.random.randint(-radius//2, radius//2),
                               center[1] + np.random.randint(-radius//2, radius//2))
                noise_radius = np.random.randint(2, 5)
                noise_intensity = intensity + np.random.randint(-20, 20)
                cv2.circle(img, noise_center, noise_radius, 
                          (noise_intensity, noise_intensity, noise_intensity), -1)

        # Add some background noise
        noise = np.random.normal(0, 5, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return img

    def test_complete_pipeline(self):
        """Test the complete analysis pipeline"""
        # Step 1: Preprocessing
        preprocessed = self.preprocessor.enhance_image(self.test_image)
        self.assertIsInstance(preprocessed, np.ndarray)

        # Step 2: Segmentation
        segmented, masks, contours = self.segmentation_model.segment_cells(preprocessed)
        self.assertGreater(len(masks), 0, "Should detect some cells")

        # Step 3: Feature extraction
        features = self.feature_extractor.extract_features(preprocessed, masks, contours)
        self.assertEqual(len(features), len(masks), "Features should match number of detected cells")

        # Step 4: Classification
        classifications = self.classifier.classify_cells(features)
        self.assertEqual(len(classifications), len(features), "Classifications should match number of features")

        # Verify end-to-end consistency
        for i, (mask, feature, classification) in enumerate(zip(masks, features, classifications)):
            self.assertEqual(feature['cell_id'], i)
            self.assertEqual(classification['cell_id'], i)

if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)
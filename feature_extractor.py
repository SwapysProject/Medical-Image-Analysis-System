"""
Feature Extractor for Medical Image Analysis
"""
""" 
Comprehensive feature extraction for cell analysis in digital pathology
Extracts morphological, texture, and intensity features for classification
"""

import cv2
import numpy as np
from scipy.spatial.distance import cdist
from skimage import measure
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
import logging

class FeatureExtractor:
    """
    Comprehensive feature extraction for cell analysis in digital pathology
    Extracts morphological, texture, and intensity features for classification
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # LBP parameters for texture analysis
        self.lbp_radius = 3
        self.lbp_n_points = 8 * self.lbp_radius

        # GLCM parameters for texture analysis
        self.glcm_distances = [1, 2, 3]
        self.glcm_angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    def extract_features(self, image, cell_masks, cell_contours):
        """
        Extract comprehensive features from segmented cells

        Args:
            image: Original preprocessed image
            cell_masks: List of binary masks for each cell
            cell_contours: List of contours for each cell

        Returns:
            List of feature dictionaries, one per cell
        """
        features_list = []

        for i, (mask, contour) in enumerate(zip(cell_masks, cell_contours)):
            try:
                # Extract all feature types for this cell
                cell_features = {
                    'cell_id': i,
                    'morphological': self._extract_morphological_features(mask, contour),
                    'texture': self._extract_texture_features(image, mask),
                    'intensity': self._extract_intensity_features(image, mask),
                    'geometric': self._extract_geometric_features(contour),
                    'shape_descriptors': self._extract_shape_descriptors(contour, mask)
                }

                # Flatten features into single dictionary for ML algorithms
                flattened_features = self._flatten_features(cell_features)
                features_list.append(flattened_features)

            except Exception as e:
                self.logger.warning(f"Feature extraction failed for cell {i}: {str(e)}")
                # Add empty feature set for failed extractions
                features_list.append(self._get_default_features(i))

        self.logger.info(f"Features extracted for {len(features_list)} cells")
        return features_list

    def _extract_morphological_features(self, mask, contour):
        """Extract basic morphological features"""
        features = {}

        # Basic area and perimeter
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        features['area'] = area
        features['perimeter'] = perimeter

        # Shape metrics
        if perimeter > 0:
            features['circularity'] = 4 * np.pi * area / (perimeter ** 2)
            features['aspect_ratio'] = self._calculate_aspect_ratio(contour)
            features['roundness'] = (4 * area) / (np.pi * perimeter)
        else:
            features['circularity'] = 0
            features['aspect_ratio'] = 1
            features['roundness'] = 0

        # Convexity features
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)

        if hull_area > 0:
            features['solidity'] = area / hull_area
            features['convexity'] = cv2.arcLength(hull, True) / perimeter if perimeter > 0 else 0
        else:
            features['solidity'] = 0
            features['convexity'] = 0

        # Extent (ratio of contour area to bounding rectangle area)
        if len(contour) >= 5:
            x, y, w, h = cv2.boundingRect(contour)
            rect_area = w * h
            features['extent'] = area / rect_area if rect_area > 0 else 0
            features['bounding_box_ratio'] = w / h if h > 0 else 1
        else:
            features['extent'] = 0
            features['bounding_box_ratio'] = 1

        return features

    def _extract_texture_features(self, image, mask):
        """Extract texture features using LBP and GLCM"""
        features = {}

        # Get region of interest
        roi = cv2.bitwise_and(image, image, mask=mask.astype(np.uint8))

        # Find bounding box for cropping
        coords = np.where(mask > 0)
        if len(coords[0]) == 0:
            return self._get_default_texture_features()

        y_min, y_max = coords[0].min(), coords[0].max() + 1
        x_min, x_max = coords[1].min(), coords[1].max() + 1

        cell_region = roi[y_min:y_max, x_min:x_max]
        cell_mask_crop = mask[y_min:y_max, x_min:x_max]

        if cell_region.size == 0:
            return self._get_default_texture_features()

        # Local Binary Pattern features
        try:
            lbp = local_binary_pattern(cell_region, self.lbp_n_points, self.lbp_radius, method='uniform')
            lbp_masked = lbp[cell_mask_crop > 0]

            if len(lbp_masked) > 0:
                features['lbp_mean'] = np.mean(lbp_masked)
                features['lbp_std'] = np.std(lbp_masked)
                features['lbp_entropy'] = self._calculate_entropy(lbp_masked)
            else:
                features.update(self._get_default_lbp_features())
        except:
            features.update(self._get_default_lbp_features())

        # Gray Level Co-occurrence Matrix features
        try:
            # Normalize image to 0-255 range for GLCM
            normalized_region = cv2.normalize(cell_region, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            # Calculate GLCM
            glcm = graycomatrix(normalized_region, distances=self.glcm_distances, 
                             angles=self.glcm_angles, levels=256, symmetric=True, normed=True)

            # Extract GLCM properties
            features['glcm_contrast'] = np.mean(graycoprops(glcm, 'contrast'))
            features['glcm_dissimilarity'] = np.mean(graycoprops(glcm, 'dissimilarity'))
            features['glcm_homogeneity'] = np.mean(graycoprops(glcm, 'homogeneity'))
            features['glcm_energy'] = np.mean(graycoprops(glcm, 'energy'))
            features['glcm_correlation'] = np.mean(graycoprops(glcm, 'correlation'))

        except:
            features.update(self._get_default_glcm_features())

        return features

    def _extract_intensity_features(self, image, mask):
        """Extract intensity-based features"""
        features = {}

        # Get pixel values within the mask
        masked_pixels = image[mask > 0]

        if len(masked_pixels) == 0:
            return self._get_default_intensity_features()

        # Basic intensity statistics
        features['mean_intensity'] = np.mean(masked_pixels)
        features['std_intensity'] = np.std(masked_pixels)
        features['min_intensity'] = np.min(masked_pixels)
        features['max_intensity'] = np.max(masked_pixels)
        features['intensity_range'] = features['max_intensity'] - features['min_intensity']

        # Percentile features
        features['intensity_p25'] = np.percentile(masked_pixels, 25)
        features['intensity_p50'] = np.percentile(masked_pixels, 50)
        features['intensity_p75'] = np.percentile(masked_pixels, 75)
        features['intensity_iqr'] = features['intensity_p75'] - features['intensity_p25']

        # Skewness and kurtosis (simplified approximations)
        mean_int = features['mean_intensity']
        std_int = features['std_intensity']

        if std_int > 0:
            # Simplified skewness calculation
            deviations = masked_pixels - mean_int
            features['intensity_skewness'] = np.mean((deviations / std_int) ** 3)
            features['intensity_kurtosis'] = np.mean((deviations / std_int) ** 4)
        else:
            features['intensity_skewness'] = 0
            features['intensity_kurtosis'] = 0

        return features

    def _extract_geometric_features(self, contour):
        """Extract geometric features from contour"""
        features = {}

        if len(contour) < 5:
            return self._get_default_geometric_features()

        # Fit ellipse to contour
        try:
            ellipse = cv2.fitEllipse(contour)
            (center_x, center_y), (minor_axis, major_axis), angle = ellipse

            features['ellipse_major_axis'] = major_axis
            features['ellipse_minor_axis'] = minor_axis
            features['ellipse_eccentricity'] = np.sqrt(1 - (minor_axis / major_axis) ** 2) if major_axis > 0 else 0
            features['ellipse_orientation'] = angle

        except:
            features.update(self._get_default_geometric_features())

        # Moments
        moments = cv2.moments(contour)

        if moments['m00'] > 0:
            # Centroids
            features['centroid_x'] = moments['m10'] / moments['m00']
            features['centroid_y'] = moments['m01'] / moments['m00']

            # Hu moments for shape invariance
            hu_moments = cv2.HuMoments(moments)
            for i, hu in enumerate(hu_moments.flatten()):
                features[f'hu_moment_{i+1}'] = -np.sign(hu) * np.log10(np.abs(hu)) if hu != 0 else 0
        else:
            features['centroid_x'] = 0
            features['centroid_y'] = 0
            for i in range(7):
                features[f'hu_moment_{i+1}'] = 0

        return features

    def _extract_shape_descriptors(self, contour, mask):
        """Extract advanced shape descriptors"""
        features = {}

        # Contour-based features
        if len(contour) >= 5:
            # Defects analysis (for convexity)
            hull = cv2.convexHull(contour, returnPoints=False)
            if len(hull) > 3:
                defects = cv2.convexityDefects(contour, hull)
                if defects is not None:
                    features['num_defects'] = len(defects)
                    defect_depths = []
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        defect_depths.append(d / 256.0)  # Convert to actual distance

                    if defect_depths:
                        features['mean_defect_depth'] = np.mean(defect_depths)
                        features['max_defect_depth'] = np.max(defect_depths)
                    else:
                        features['mean_defect_depth'] = 0
                        features['max_defect_depth'] = 0
                else:
                    features['num_defects'] = 0
                    features['mean_defect_depth'] = 0
                    features['max_defect_depth'] = 0
            else:
                features['num_defects'] = 0
                features['mean_defect_depth'] = 0
                features['max_defect_depth'] = 0
        else:
            features['num_defects'] = 0
            features['mean_defect_depth'] = 0
            features['max_defect_depth'] = 0

        # Equivalent diameter
        area = cv2.contourArea(contour)
        features['equivalent_diameter'] = np.sqrt(4 * area / np.pi) if area > 0 else 0

        return features

    def _calculate_aspect_ratio(self, contour):
        """Calculate aspect ratio from minimum area rectangle"""
        if len(contour) < 5:
            return 1.0

        rect = cv2.minAreaRect(contour)
        width, height = rect[1]

        if height > 0:
            return max(width, height) / min(width, height)
        else:
            return 1.0

    def _calculate_entropy(self, values):
        """Calculate entropy of pixel values"""
        if len(values) == 0:
            return 0

        # Create histogram
        hist, _ = np.histogram(values, bins=50)
        hist = hist[hist > 0]  # Remove zero bins

        if len(hist) <= 1:
            return 0

        # Normalize to probabilities
        probs = hist / np.sum(hist)

        # Calculate entropy
        entropy = -np.sum(probs * np.log2(probs))
        return entropy

    def _flatten_features(self, nested_features):
        """Flatten nested feature dictionary"""
        flat_features = {'cell_id': nested_features['cell_id']}

        for category, features in nested_features.items():
            if category == 'cell_id':
                continue

            for feature_name, value in features.items():
                flat_features[f"{category}_{feature_name}"] = value

        return flat_features

    # Default feature methods for error handling
    def _get_default_features(self, cell_id):
        """Return default features when extraction fails"""
        return {
            'cell_id': cell_id,
            **self._get_default_morphological_features(),
            **self._get_default_texture_features(),
            **self._get_default_intensity_features(),
            **self._get_default_geometric_features(),
            **self._get_default_shape_features()
        }

    def _get_default_morphological_features(self):
        return {f"morphological_{key}": 0 for key in [
            'area', 'perimeter', 'circularity', 'aspect_ratio', 'roundness',
            'solidity', 'convexity', 'extent', 'bounding_box_ratio'
        ]}

    def _get_default_texture_features(self):
        features = {}
        features.update(self._get_default_lbp_features())
        features.update(self._get_default_glcm_features())
        return {f"texture_{key}": value for key, value in features.items()}

    def _get_default_lbp_features(self):
        return {'lbp_mean': 0, 'lbp_std': 0, 'lbp_entropy': 0}

    def _get_default_glcm_features(self):
        return {
            'glcm_contrast': 0, 'glcm_dissimilarity': 0, 'glcm_homogeneity': 0,
            'glcm_energy': 0, 'glcm_correlation': 0
        }

    def _get_default_intensity_features(self):
        return {f"intensity_{key}": 0 for key in [
            'mean_intensity', 'std_intensity', 'min_intensity', 'max_intensity',
            'intensity_range', 'intensity_p25', 'intensity_p50', 'intensity_p75',
            'intensity_iqr', 'intensity_skewness', 'intensity_kurtosis'
        ]}

    def _get_default_geometric_features(self):
        features = {
            'ellipse_major_axis': 0, 'ellipse_minor_axis': 0,
            'ellipse_eccentricity': 0, 'ellipse_orientation': 0,
            'centroid_x': 0, 'centroid_y': 0
        }
        for i in range(7):
            features[f'hu_moment_{i+1}'] = 0
        return {f"geometric_{key}": value for key, value in features.items()}

    def _get_default_shape_features(self):
        return {f"shape_descriptors_{key}": 0 for key in [
            'num_defects', 'mean_defect_depth', 'max_defect_depth', 'equivalent_diameter'
        ]}
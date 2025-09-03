"""
Postprocessing Utilities for Medical Image Analysis Results
Handles result aggregation, statistical analysis, and report generation
"""

import numpy as np
import cv2
import pandas as pd
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging

class ResultProcessor:
    """
    Postprocessing and analysis of cell segmentation and classification results
    Generates comprehensive quantitative reports and visualizations
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def analyze_results(self, cell_masks, features, classifications, image_shape):
        """
        Comprehensive analysis of segmentation and classification results

        Args:
            cell_masks: List of cell segmentation masks
            features: List of extracted features for each cell
            classifications: List of classification results
            image_shape: Original image shape (height, width, channels)

        Returns:
            Dictionary containing comprehensive quantitative analysis
        """
        try:
            analysis = {
                'summary_statistics': self._calculate_summary_statistics(cell_masks, features, classifications),
                'morphological_analysis': self._analyze_morphological_features(features),
                'classification_summary': self._analyze_classifications(classifications),
                'spatial_analysis': self._analyze_spatial_distribution(cell_masks, image_shape),
                'quality_metrics': self._calculate_quality_metrics(cell_masks, features, classifications),
                'cell_density': self._calculate_cell_density(cell_masks, image_shape),
                'size_distribution': self._analyze_size_distribution(features),
                'recommendations': self._generate_recommendations(cell_masks, features, classifications)
            }

            self.logger.info("Results analysis completed successfully")
            return analysis

        except Exception as e:
            self.logger.error(f"Results analysis failed: {str(e)}")
            return self._get_default_analysis()

    def _calculate_summary_statistics(self, cell_masks, features, classifications):
        """Calculate basic summary statistics"""
        total_cells = len(cell_masks)

        if total_cells == 0:
            return {
                'total_cells': 0,
                'avg_cell_area': 0,
                'total_tissue_area': 0,
                'cell_coverage_ratio': 0
            }

        # Calculate areas
        cell_areas = [cv2.countNonZero(mask) for mask in cell_masks]
        avg_cell_area = np.mean(cell_areas) if cell_areas else 0
        total_tissue_area = np.sum(cell_areas)

        # Calculate coverage ratio (cells vs total image)
        total_image_pixels = cell_masks[0].shape[0] * cell_masks[0].shape[1] if cell_masks else 1
        cell_coverage_ratio = total_tissue_area / total_image_pixels

        return {
            'total_cells': total_cells,
            'avg_cell_area': float(avg_cell_area),
            'total_tissue_area': int(total_tissue_area),
            'cell_coverage_ratio': float(cell_coverage_ratio),
            'min_cell_area': float(min(cell_areas)) if cell_areas else 0,
            'max_cell_area': float(max(cell_areas)) if cell_areas else 0,
            'std_cell_area': float(np.std(cell_areas)) if cell_areas else 0
        }

    def _analyze_morphological_features(self, features):
        """Analyze morphological characteristics of detected cells"""
        if not features:
            return {}

        # Extract morphological features
        areas = [f.get('morphological_area', 0) for f in features]
        circularities = [f.get('morphological_circularity', 0) for f in features]
        solidities = [f.get('morphological_solidity', 0) for f in features]
        aspect_ratios = [f.get('morphological_aspect_ratio', 1) for f in features]

        # Calculate statistics for each feature
        morphological_stats = {}

        for feature_name, values in [
            ('area', areas),
            ('circularity', circularities),
            ('solidity', solidities),
            ('aspect_ratio', aspect_ratios)
        ]:
            if values:
                morphological_stats[feature_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'median': float(np.median(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'q25': float(np.percentile(values, 25)),
                    'q75': float(np.percentile(values, 75))
                }

        return morphological_stats

    def _analyze_classifications(self, classifications):
        """Analyze classification results and confidence scores"""
        if not classifications:
            return {}

        # Count cell types
        cell_type_counts = {}
        confidence_scores = []

        for result in classifications:
            cell_type = result.get('type', 'uncertain')
            confidence = result.get('confidence', 0)

            cell_type_counts[cell_type] = cell_type_counts.get(cell_type, 0) + 1
            confidence_scores.append(confidence)

        # Calculate confidence statistics
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
        min_confidence = np.min(confidence_scores) if confidence_scores else 0

        # Determine dominant cell type
        dominant_type = max(cell_type_counts.items(), key=lambda x: x[1])[0] if cell_type_counts else 'unknown'

        # Calculate percentages
        total_cells = len(classifications)
        cell_type_percentages = {
            cell_type: (count / total_cells) * 100 
            for cell_type, count in cell_type_counts.items()
        }

        return {
            'cell_type_counts': cell_type_counts,
            'cell_type_percentages': cell_type_percentages,
            'dominant_type': dominant_type,
            'avg_confidence': float(avg_confidence),
            'min_confidence': float(min_confidence),
            'confidence_distribution': {
                'high_confidence': sum(1 for c in confidence_scores if c >= 0.8),
                'medium_confidence': sum(1 for c in confidence_scores if 0.5 <= c < 0.8),
                'low_confidence': sum(1 for c in confidence_scores if c < 0.5)
            }
        }

    def _analyze_spatial_distribution(self, cell_masks, image_shape):
        """Analyze spatial distribution of cells"""
        if not cell_masks:
            return {}

        # Calculate centroids of all cells
        centroids = []
        for mask in cell_masks:
            moments = cv2.moments(mask.astype(np.uint8))
            if moments['m00'] != 0:
                cx = moments['m10'] / moments['m00']
                cy = moments['m01'] / moments['m00']
                centroids.append((cx, cy))

        if not centroids:
            return {}

        centroids = np.array(centroids)

        # Calculate spatial statistics
        center_of_mass = np.mean(centroids, axis=0)

        # Calculate distances from center of mass
        distances_from_center = [
            np.sqrt((x - center_of_mass[0])**2 + (y - center_of_mass[1])**2)
            for x, y in centroids
        ]

        # Calculate nearest neighbor distances
        from scipy.spatial.distance import pdist
        if len(centroids) > 1:
            pairwise_distances = pdist(centroids)
            avg_nearest_neighbor = np.mean(pairwise_distances)
            min_distance = np.min(pairwise_distances)
        else:
            avg_nearest_neighbor = 0
            min_distance = 0

        # Analyze distribution in image quadrants
        h, w = image_shape[:2]
        quadrant_counts = {'Q1': 0, 'Q2': 0, 'Q3': 0, 'Q4': 0}

        for x, y in centroids:
            if x < w/2 and y < h/2:
                quadrant_counts['Q1'] += 1
            elif x >= w/2 and y < h/2:
                quadrant_counts['Q2'] += 1
            elif x < w/2 and y >= h/2:
                quadrant_counts['Q3'] += 1
            else:
                quadrant_counts['Q4'] += 1

        return {
            'center_of_mass': center_of_mass.tolist(),
            'avg_distance_from_center': float(np.mean(distances_from_center)),
            'std_distance_from_center': float(np.std(distances_from_center)),
            'avg_nearest_neighbor_distance': float(avg_nearest_neighbor),
            'min_distance_between_cells': float(min_distance),
            'quadrant_distribution': quadrant_counts,
            'spatial_clustering_score': self._calculate_clustering_score(centroids)
        }

    def _calculate_clustering_score(self, centroids):
        """Calculate a simple clustering score (0 = scattered, 1 = highly clustered)"""
        if len(centroids) < 2:
            return 0.0

        # Use coefficient of variation of nearest neighbor distances as clustering metric
        from scipy.spatial.distance import pdist
        distances = pdist(centroids)
        cv = np.std(distances) / np.mean(distances) if np.mean(distances) > 0 else 0

        # Normalize to 0-1 scale (higher CV means more clustered)
        clustering_score = min(cv / 2.0, 1.0)  # Divide by 2 for normalization

        return float(clustering_score)

    def _calculate_quality_metrics(self, cell_masks, features, classifications):
        """Calculate quality metrics for the analysis"""
        if not cell_masks:
            return {}

        # Segmentation quality metrics
        avg_solidity = np.mean([f.get('morphological_solidity', 0) for f in features]) if features else 0
        avg_circularity = np.mean([f.get('morphological_circularity', 0) for f in features]) if features else 0

        # Classification quality metrics
        avg_confidence = np.mean([c.get('confidence', 0) for c in classifications]) if classifications else 0
        high_confidence_ratio = sum(1 for c in classifications if c.get('confidence', 0) >= 0.8) / len(classifications) if classifications else 0

        # Overall quality score (weighted combination)
        quality_score = (
            0.25 * avg_solidity +
            0.25 * avg_circularity +
            0.3 * avg_confidence +
            0.2 * high_confidence_ratio
        )

        return {
            'segmentation_quality': {
                'avg_solidity': float(avg_solidity),
                'avg_circularity': float(avg_circularity)
            },
            'classification_quality': {
                'avg_confidence': float(avg_confidence),
                'high_confidence_ratio': float(high_confidence_ratio)
            },
            'overall_quality_score': float(quality_score),
            'quality_grade': self._assign_quality_grade(quality_score)
        }

    def _assign_quality_grade(self, score):
        """Assign letter grade based on quality score"""
        if score >= 0.9:
            return 'A'
        elif score >= 0.8:
            return 'B'
        elif score >= 0.7:
            return 'C'
        elif score >= 0.6:
            return 'D'
        else:
            return 'F'

    def _calculate_cell_density(self, cell_masks, image_shape):
        """Calculate cell density metrics"""
        if not cell_masks:
            return {}

        h, w = image_shape[:2]
        total_area_pixels = h * w
        total_area_mm2 = total_area_pixels * (0.001**2)  # Assuming 1 pixel = 1 micron

        cell_count = len(cell_masks)

        return {
            'cells_per_mm2': float(cell_count / total_area_mm2),
            'cells_per_1000_pixels': float(cell_count / (total_area_pixels / 1000)),
            'total_analyzed_area_mm2': float(total_area_mm2)
        }

    def _analyze_size_distribution(self, features):
        """Analyze cell size distribution and classify cells by size"""
        if not features:
            return {}

        areas = [f.get('morphological_area', 0) for f in features]

        if not areas:
            return {}

        # Define size categories (in pixels)
        small_threshold = np.percentile(areas, 33)
        large_threshold = np.percentile(areas, 67)

        size_categories = {
            'small': sum(1 for area in areas if area <= small_threshold),
            'medium': sum(1 for area in areas if small_threshold < area <= large_threshold),
            'large': sum(1 for area in areas if area > large_threshold)
        }

        return {
            'size_distribution': size_categories,
            'size_thresholds': {
                'small_max': float(small_threshold),
                'large_min': float(large_threshold)
            },
            'area_statistics': {
                'mean': float(np.mean(areas)),
                'std': float(np.std(areas)),
                'coefficient_of_variation': float(np.std(areas) / np.mean(areas)) if np.mean(areas) > 0 else 0
            }
        }

    def _generate_recommendations(self, cell_masks, features, classifications):
        """Generate analysis recommendations based on results"""
        recommendations = []

        # Check cell count
        cell_count = len(cell_masks)
        if cell_count == 0:
            recommendations.append("No cells detected. Consider adjusting segmentation parameters or checking image quality.")
        elif cell_count < 10:
            recommendations.append("Low cell count detected. Results may not be statistically significant.")

        # Check classification confidence
        if classifications:
            avg_confidence = np.mean([c.get('confidence', 0) for c in classifications])
            if avg_confidence < 0.6:
                recommendations.append("Low average classification confidence. Consider manual review of results.")

        # Check morphological consistency
        if features:
            areas = [f.get('morphological_area', 0) for f in features]
            cv_area = np.std(areas) / np.mean(areas) if np.mean(areas) > 0 else 0

            if cv_area > 1.0:
                recommendations.append("High variability in cell sizes detected. This may indicate mixed cell populations or segmentation issues.")

        # Check for potential issues
        if classifications:
            uncertain_ratio = sum(1 for c in classifications if c.get('type') == 'uncertain') / len(classifications)
            if uncertain_ratio > 0.3:
                recommendations.append("High proportion of uncertain classifications. Consider additional training data or feature engineering.")

        if not recommendations:
            recommendations.append("Analysis completed successfully. Results appear consistent and reliable.")

        return recommendations

    def _get_default_analysis(self):
        """Return default analysis structure for error cases"""
        return {
            'summary_statistics': {'total_cells': 0, 'avg_cell_area': 0},
            'morphological_analysis': {},
            'classification_summary': {},
            'spatial_analysis': {},
            'quality_metrics': {'overall_quality_score': 0},
            'cell_density': {},
            'size_distribution': {},
            'recommendations': ["Analysis failed. Please check input data and try again."]
        }

    def generate_csv_report(self, features, classifications, output_path):
        """Generate detailed CSV report of all cell measurements"""
        try:
            # Combine features and classifications
            report_data = []

            for i, (feature_dict, class_dict) in enumerate(zip(features, classifications)):
                row = {
                    'cell_id': i + 1,
                    'classification': class_dict.get('type', 'unknown'),
                    'confidence': class_dict.get('confidence', 0)
                }

                # Add all features
                for key, value in feature_dict.items():
                    if key != 'cell_id':
                        row[key] = value

                report_data.append(row)

            # Create DataFrame and save
            df = pd.DataFrame(report_data)
            df.to_csv(output_path, index=False)

            self.logger.info(f"CSV report saved to {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to generate CSV report: {str(e)}")
            return False

    def generate_summary_report(self, analysis_results, output_path):
        """Generate a comprehensive summary report in JSON format"""
        try:
            report = {
                'analysis_timestamp': datetime.now().isoformat(),
                'analysis_results': analysis_results,
                'methodology': {
                    'segmentation': 'Marker-controlled watershed algorithm',
                    'feature_extraction': 'Morphological, texture, and intensity features',
                    'classification': 'Rule-based classification with confidence estimation'
                }
            }

            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)

            self.logger.info(f"Summary report saved to {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to generate summary report: {str(e)}")
            return False
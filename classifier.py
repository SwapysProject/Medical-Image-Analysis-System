"""
Cell Classifier for Digital Pathology
Implements machine learning classification for cell type identification
"""

import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import logging

class CellClassifier:
    """
    Machine learning classifier for automated cell type identification
    Supports multiple algorithms and includes confidence estimation
    """

    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.logger = logging.getLogger(__name__)
        self.is_trained = False

        # Initialize model based on type
        self._initialize_model()

        # Try to load pre-trained model
        self._load_pretrained_model()

    def _initialize_model(self):
        """Initialize the classification model"""
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
        elif self.model_type == 'svm':
            self.model = SVC(
                kernel='rbf',
                probability=True,  # Enable probability estimates
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        self.scaler = StandardScaler()

    def classify_cells(self, features_list):
        """
        Classify cells based on extracted features

        Args:
            features_list: List of feature dictionaries from FeatureExtractor

        Returns:
            List of classification results with confidence scores
        """
        if not features_list:
            return []

        try:
            # If no trained model, use rule-based classification
            if not self.is_trained:
                return self._rule_based_classification(features_list)

            # Prepare features for prediction
            X = self._prepare_features(features_list)

            if X.size == 0:
                return self._get_default_classifications(len(features_list))

            # Make predictions with confidence scores
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)

            results = []
            for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
                results.append({
                    'cell_id': features_list[i].get('cell_id', i),
                    'type': self.class_names[pred],
                    'confidence': float(np.max(probs)),
                    'probabilities': {
                        self.class_names[j]: float(prob) 
                        for j, prob in enumerate(probs)
                    }
                })

            return results

        except Exception as e:
            self.logger.error(f"Classification failed: {str(e)}")
            return self._get_default_classifications(len(features_list))

    def _rule_based_classification(self, features_list):
        """
        Rule-based classification when no trained model is available
        Based on morphological and intensity features
        """
        results = []

        for i, features in enumerate(features_list):
            cell_id = features.get('cell_id', i)

            # Extract key features for rule-based classification
            area = features.get('morphological_area', 0)
            circularity = features.get('morphological_circularity', 0)
            solidity = features.get('morphological_solidity', 0)
            mean_intensity = features.get('intensity_mean_intensity', 0)
            intensity_std = features.get('intensity_std_intensity', 0)

            # Define classification rules based on typical cell characteristics
            classification = self._apply_classification_rules(
                area, circularity, solidity, mean_intensity, intensity_std
            )

            results.append({
                'cell_id': cell_id,
                'type': classification['type'],
                'confidence': classification['confidence'],
                'probabilities': classification['probabilities']
            })

        return results

    def _apply_classification_rules(self, area, circularity, solidity, mean_intensity, intensity_std):
        """Apply hand-crafted rules for cell classification"""

        # Initialize scores for each class
        normal_score = 0
        abnormal_score = 0
        uncertain_score = 0

        # Rule 1: Size-based classification
        if 100 <= area <= 2000:  # Normal cell size range
            normal_score += 0.3
        elif area < 50 or area > 5000:  # Very small or very large
            abnormal_score += 0.4
        else:
            uncertain_score += 0.2

        # Rule 2: Shape-based classification
        if 0.6 <= circularity <= 1.0 and solidity >= 0.8:  # Round and solid
            normal_score += 0.4
        elif circularity < 0.3 or solidity < 0.5:  # Irregular shape
            abnormal_score += 0.4
        else:
            uncertain_score += 0.3

        # Rule 3: Intensity-based classification
        if 50 <= mean_intensity <= 200 and intensity_std < 50:  # Normal intensity
            normal_score += 0.3
        elif mean_intensity < 30 or mean_intensity > 220 or intensity_std > 80:  # Extreme values
            abnormal_score += 0.3
        else:
            uncertain_score += 0.2

        # Normalize scores
        total_score = normal_score + abnormal_score + uncertain_score
        if total_score > 0:
            normal_score /= total_score
            abnormal_score /= total_score
            uncertain_score /= total_score
        else:
            # Default to uncertain if no rules apply
            uncertain_score = 1.0

        # Determine final classification
        scores = [normal_score, abnormal_score, uncertain_score]
        max_score_idx = np.argmax(scores)

        return {
            'type': self.class_names[max_score_idx],
            'confidence': float(scores[max_score_idx]),
            'probabilities': {
                'normal': float(normal_score),
                'abnormal': float(abnormal_score),
                'uncertain': float(uncertain_score)
            }
        }

    def train_model(self, features_list, labels):
        """
        Train the classifier with labeled data

        Args:
            features_list: List of feature dictionaries
            labels: List of corresponding class labels
        """
        try:
            # Prepare features
            X = self._prepare_features(features_list)
            y = np.array(labels)

            if X.size == 0 or len(y) == 0:
                raise ValueError("No valid training data provided")

            # Fit scaler and transform features
            X_scaled = self.scaler.fit_transform(X)

            # Train model
            self.model.fit(X_scaled, y)
            self.is_trained = True

            # Evaluate model with cross-validation
            cv_scores = cross_val_score(self.model, X_scaled, y, cv=5)

            self.logger.info(f"Model trained successfully")
            self.logger.info(f"Cross-validation accuracy: {np.mean(cv_scores):.3f} Â± {np.std(cv_scores):.3f}")

            # Save trained model
            self._save_model()

            return {
                'success': True,
                'cv_mean': float(np.mean(cv_scores)),
                'cv_std': float(np.std(cv_scores)),
                'n_samples': len(y)
            }

        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            return {'success': False, 'error': str(e)}

    def _prepare_features(self, features_list):
        """Prepare feature matrix from feature dictionaries"""
        if not features_list:
            return np.array([])

        # Extract feature names (excluding cell_id)
        if not self.feature_names:
            self.feature_names = [key for key in features_list[0].keys() if key != 'cell_id']

        # Create feature matrix
        X = []
        for features in features_list:
            row = []
            for feature_name in self.feature_names:
                value = features.get(feature_name, 0)
                # Handle any non-numeric values
                if isinstance(value, (int, float)) and not np.isnan(value):
                    row.append(value)
                else:
                    row.append(0)
            X.append(row)

        return np.array(X)

    def _get_default_classifications(self, n_cells):
        """Return default classifications for error cases"""
        results = []
        for i in range(n_cells):
            results.append({
                'cell_id': i,
                'type': 'uncertain',
                'confidence': 0.5,
                'probabilities': {
                    'normal': 0.33,
                    'abnormal': 0.33,
                    'uncertain': 0.34
                }
            })
        return results

    def _save_model(self, filename=None):
        """Save trained model and scaler"""
        if not filename:
            filename = f'cell_classifier_{self.model_type}.pkl'

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'class_names': self.class_names,
            'model_type': self.model_type,
            'is_trained': self.is_trained
        }

        try:
            with open(filename, 'wb') as f:
                pickle.dump(model_data, f)
            self.logger.info(f"Model saved to {filename}")
        except Exception as e:
            self.logger.error(f"Failed to save model: {str(e)}")

    def _load_pretrained_model(self, filename=None):
        """Load pre-trained model if available"""
        if not filename:
            filename = f'cell_classifier_{self.model_type}.pkl'

        if not os.path.exists(filename):
            return False

        try:
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)

            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.class_names = model_data['class_names']
            self.is_trained = model_data['is_trained']

            self.logger.info(f"Pre-trained model loaded from {filename}")
            return True

        except Exception as e:
            self.logger.warning(f"Failed to load pre-trained model: {str(e)}")
            return False

    def get_feature_importance(self):
        """Get feature importance scores (for tree-based models)"""
        if not self.is_trained:
            return {}

        if hasattr(self.model, 'feature_importances_'):
            importance_scores = self.model.feature_importances_
            return {
                name: float(score) 
                for name, score in zip(self.feature_names, importance_scores)
            }
        else:
            return {}

    def evaluate_model(self, features_list, true_labels):
        """Evaluate model performance on test data"""
        if not self.is_trained:
            return {'error': 'Model not trained'}

        try:
            X = self._prepare_features(features_list)
            X_scaled = self.scaler.transform(X)

            predictions = self.model.predict(X_scaled)

            # Generate classification report
            report = classification_report(true_labels, predictions, 
                                        target_names=self.class_names, 
                                        output_dict=True)

            return {
                'classification_report': report,
                'accuracy': float(report['accuracy']),
                'n_samples': len(true_labels)
            }

        except Exception as e:
            return {'error': str(e)}
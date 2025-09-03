"""
Image Preprocessing Utilities for Medical Image Analysis
Handles enhancement, normalization, and quality improvement for digital pathology images
"""

import cv2
import numpy as np
from scipy import ndimage
from skimage import exposure, filters, morphology
import logging

class ImagePreprocessor:
    """
    Advanced preprocessing pipeline for medical images
    Optimized for digital pathology and high-resolution microscopy images
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Default parameters for preprocessing
        self.clahe_clip_limit = 3.0
        self.clahe_tile_grid = (8, 8)
        self.gaussian_kernel_size = 5
        self.gaussian_sigma = 0

    def enhance_image(self, image):
        """
        Main preprocessing pipeline for medical images

        Args:
            image: Input BGR image (numpy array)

        Returns:
            Enhanced single-channel image ready for segmentation
        """
        try:
            # Step 1: Color space conversion for better cell visibility
            enhanced_image = self._convert_color_space(image)

            # Step 2: Noise reduction
            enhanced_image = self._reduce_noise(enhanced_image)

            # Step 3: Contrast enhancement
            enhanced_image = self._enhance_contrast(enhanced_image)

            # Step 4: Illumination correction
            enhanced_image = self._correct_illumination(enhanced_image)

            # Step 5: Final smoothing
            enhanced_image = self._apply_smoothing(enhanced_image)

            self.logger.info("Image preprocessing completed successfully")
            return enhanced_image

        except Exception as e:
            self.logger.error(f"Image preprocessing failed: {str(e)}")
            # Return grayscale version as fallback
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def _convert_color_space(self, image):
        """
        Convert to optimal color space for cell segmentation
        Uses LAB color space which often provides better cell-background separation
        """
        if len(image.shape) == 3:
            # Convert BGR to LAB
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

            # Extract channels
            l_channel, a_channel, b_channel = cv2.split(lab)

            # A channel often provides best cell contrast in pathology images
            # But we can also experiment with other combinations

            # Option 1: Use A channel (green-red component)
            enhanced = a_channel

            # Option 2: Combine channels for better contrast
            # You can uncomment this for different results
            # enhanced = cv2.addWeighted(a_channel, 0.7, b_channel, 0.3, 0)

        else:
            # Already grayscale
            enhanced = image.copy()

        return enhanced

    def _reduce_noise(self, image):
        """
        Apply noise reduction while preserving cell boundaries
        """
        # Bilateral filter preserves edges while reducing noise
        # This is crucial for maintaining cell boundaries
        denoised = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

        # Alternative: Non-local means denoising (more computationally expensive)
        # denoised = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)

        return denoised

    def _enhance_contrast(self, image):
        """
        Enhance local contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        This is particularly important for medical images with varying illumination
        """
        # Create CLAHE object
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_tile_grid
        )

        # Apply CLAHE
        enhanced = clahe.apply(image)

        return enhanced

    def _correct_illumination(self, image):
        """
        Correct uneven illumination common in microscopy images
        """
        # Method 1: Top-hat transform to correct uneven illumination
        # Create morphological kernel
        kernel_size = min(image.shape) // 20  # Adaptive kernel size
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        # Apply top-hat transform
        tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)

        # Subtract from original to get corrected image
        corrected = cv2.subtract(image, tophat)

        # Method 2: Background subtraction using Gaussian blur (alternative approach)
        # background = cv2.GaussianBlur(image, (51, 51), 0)
        # corrected = cv2.subtract(image, background)
        # corrected = cv2.add(corrected, 128)  # Add offset to avoid negative values

        return corrected

    def _apply_smoothing(self, image):
        """
        Apply final smoothing for better segmentation results
        """
        # Gaussian smoothing
        if self.gaussian_sigma == 0:
            # Auto-calculate sigma
            sigma = 0.3 * ((self.gaussian_kernel_size - 1) * 0.5 - 1) + 0.8
        else:
            sigma = self.gaussian_sigma

        smoothed = cv2.GaussianBlur(image, (self.gaussian_kernel_size, self.gaussian_kernel_size), sigma)

        return smoothed

    def preprocess_for_deep_learning(self, image, target_size=(512, 512)):
        """
        Preprocessing specifically for deep learning models
        Includes normalization and resizing
        """
        try:
            # Resize image
            resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

            # Convert to float32 and normalize to [0, 1]
            normalized = resized.astype(np.float32) / 255.0

            # Optionally convert to 3-channel if needed for pretrained models
            if len(normalized.shape) == 2:
                normalized = cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)

            return normalized

        except Exception as e:
            self.logger.error(f"Deep learning preprocessing failed: {str(e)}")
            return None

    def preprocess_large_image(self, image, max_dimension=4096):
        """
        Preprocess very large images by resizing while maintaining aspect ratio
        Important for handling the high-resolution images mentioned in job requirements
        """
        h, w = image.shape[:2]

        # Check if resizing is needed
        if max(h, w) <= max_dimension:
            return self.enhance_image(image)

        # Calculate new dimensions maintaining aspect ratio
        if h > w:
            new_h = max_dimension
            new_w = int(w * (max_dimension / h))
        else:
            new_w = max_dimension
            new_h = int(h * (max_dimension / w))

        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Apply standard preprocessing
        enhanced = self.enhance_image(resized)

        self.logger.info(f"Large image resized from {w}x{h} to {new_w}x{new_h}")

        return enhanced

    def correct_staining_variations(self, image, reference_image=None):
        """
        Correct staining variations in histopathology images
        This is crucial for consistent analysis across different slides
        """
        try:
            if reference_image is not None:
                # Histogram matching for stain normalization
                matched = self._match_histogram(image, reference_image)
                return matched
            else:
                # Standard stain normalization without reference
                # Convert to LAB color space
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)

                # Normalize each channel
                l = cv2.equalizeHist(l)

                # Merge and convert back
                normalized_lab = cv2.merge([l, a, b])
                normalized = cv2.cvtColor(normalized_lab, cv2.COLOR_LAB2BGR)

                return normalized

        except Exception as e:
            self.logger.error(f"Stain correction failed: {str(e)}")
            return image

    def _match_histogram(self, source, reference):
        """
        Match histogram of source image to reference image
        """
        # Convert images to LAB color space
        source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
        reference_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB)

        # Split channels
        source_channels = cv2.split(source_lab)
        reference_channels = cv2.split(reference_lab)

        # Match each channel
        matched_channels = []
        for src_ch, ref_ch in zip(source_channels, reference_channels):
            matched_ch = self._match_channel_histogram(src_ch, ref_ch)
            matched_channels.append(matched_ch)

        # Merge channels and convert back to BGR
        matched_lab = cv2.merge(matched_channels)
        matched_bgr = cv2.cvtColor(matched_lab, cv2.COLOR_LAB2BGR)

        return matched_bgr

    def _match_channel_histogram(self, source_channel, reference_channel):
        """
        Match histogram of a single channel
        """
        # Calculate cumulative distribution functions
        source_hist, _ = np.histogram(source_channel.flatten(), 256, [0, 256])
        reference_hist, _ = np.histogram(reference_channel.flatten(), 256, [0, 256])

        source_cdf = source_hist.cumsum()
        reference_cdf = reference_hist.cumsum()

        # Normalize CDFs
        source_cdf = source_cdf / source_cdf[-1]
        reference_cdf = reference_cdf / reference_cdf[-1]

        # Create lookup table
        lookup_table = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            # Find closest value in reference CDF
            closest_idx = np.argmin(np.abs(reference_cdf - source_cdf[i]))
            lookup_table[i] = closest_idx

        # Apply lookup table
        matched_channel = cv2.LUT(source_channel, lookup_table)

        return matched_channel

    def detect_and_correct_artifacts(self, image):
        """
        Detect and correct common artifacts in medical images
        Such as dust particles, scratches, or staining artifacts
        """
        try:
            # Convert to grayscale for artifact detection
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Detect bright artifacts (dust particles)
            _, bright_artifacts = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

            # Detect dark artifacts
            _, dark_artifacts = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY_INV)

            # Combine artifact masks
            artifacts = cv2.bitwise_or(bright_artifacts, dark_artifacts)

            # Remove small artifacts (noise)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            artifacts = cv2.morphologyEx(artifacts, cv2.MORPH_OPEN, kernel)

            # Inpaint artifacts
            if len(image.shape) == 3:
                corrected = cv2.inpaint(image, artifacts, 3, cv2.INPAINT_TELEA)
            else:
                corrected = cv2.inpaint(image, artifacts, 3, cv2.INPAINT_TELEA)

            return corrected

        except Exception as e:
            self.logger.error(f"Artifact correction failed: {str(e)}")
            return image

    def validate_image_quality(self, image):
        """
        Validate image quality and provide recommendations
        """
        quality_metrics = {}

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Calculate blur metric (Laplacian variance)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        quality_metrics['blur_score'] = float(blur_score)
        quality_metrics['is_blurry'] = blur_score < 100  # Threshold can be adjusted

        # Calculate contrast metric
        contrast_score = gray.std()
        quality_metrics['contrast_score'] = float(contrast_score)
        quality_metrics['low_contrast'] = contrast_score < 30

        # Calculate brightness metrics
        brightness_mean = gray.mean()
        quality_metrics['brightness_mean'] = float(brightness_mean)
        quality_metrics['too_dark'] = brightness_mean < 50
        quality_metrics['too_bright'] = brightness_mean > 200

        # Overall quality assessment
        issues = [
            quality_metrics['is_blurry'],
            quality_metrics['low_contrast'],
            quality_metrics['too_dark'],
            quality_metrics['too_bright']
        ]

        quality_metrics['quality_score'] = 1.0 - (sum(issues) / len(issues))
        quality_metrics['recommendations'] = self._generate_quality_recommendations(quality_metrics)

        return quality_metrics

    def _generate_quality_recommendations(self, metrics):
        """Generate quality improvement recommendations"""
        recommendations = []

        if metrics['is_blurry']:
            recommendations.append("Image appears blurry. Consider using higher resolution or better focusing.")

        if metrics['low_contrast']:
            recommendations.append("Low contrast detected. Consider adjusting illumination or staining protocol.")

        if metrics['too_dark']:
            recommendations.append("Image is too dark. Increase illumination or adjust camera exposure.")

        if metrics['too_bright']:
            recommendations.append("Image is too bright. Reduce illumination or adjust camera exposure.")

        if not recommendations:
            recommendations.append("Image quality appears acceptable for analysis.")

        return recommendations
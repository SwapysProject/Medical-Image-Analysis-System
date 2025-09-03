"""
Cell Segmentation Model for Digital Pathology
Implements advanced watershed-based segmentation for high-resolution medical images
"""

import cv2
import numpy as np
from scipy import ndimage
from skimage.segmentation import watershed
from skimage.filters import gaussian
import logging

class CellSegmentationModel:
    """
    Advanced cell segmentation model using watershed algorithm
    Designed for high-resolution digital pathology images (~50K x 50K)
    """

    def __init__(self, min_cell_area=50, max_cell_area=5000):
        self.min_cell_area = min_cell_area
        self.max_cell_area = max_cell_area
        self.logger = logging.getLogger(__name__)

    def segment_cells(self, image):
        """
    Main segmentation pipeline combining multiple techniques.
        """
        try:
            # Step 1: Preprocessing
            preprocessed = self._preprocess_image(image)

            # Step 2: Generate markers for watershed
            markers, sure_fg, sure_bg = self._generate_markers(preprocessed)

            # Step 3: Apply watershed segmentation
            labels = self._apply_watershed(preprocessed, markers)

            # Step 4: Extract individual cells and clean up results
            cell_masks, cell_contours = self._extract_cells(labels, preprocessed)

            # Step 5: Create visualization
            segmented_image = self._create_segmentation_visualization(image, cell_masks)

            self.logger.info(f"Segmentation completed: {len(cell_masks)} cells detected")

            return segmented_image, cell_masks, cell_contours

        except Exception as e:
            self.logger.error(f"Segmentation failed: {str(e)}")
            empty_segmented = image.copy()
            return empty_segmented, [], []

    def _preprocess_image(self, image):
        """
        Advanced preprocessing for medical images including:
        - Color space conversion for better cell visibility
        - Noise reduction while preserving cell boundaries
        - Contrast enhancement for improved segmentation
        """
        # Convert to LAB color space for better cell separation
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # Use A channel which often shows good cell contrast
        a_channel = lab[:, :, 1]

        # Apply median blur to reduce noise while preserving edges
        denoised = cv2.medianBlur(a_channel, 3)

        # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        # Gaussian blur for smoother segmentation
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

        return blurred

    def _generate_markers(self, image):
        """
        Generate markers for marker-controlled watershed segmentation
        This is crucial for separating touching/overlapping cells
        """
        # Otsu thresholding to separate foreground from background
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Morphological operations to clean up the binary image
        kernel = np.ones((3, 3), np.uint8)

        # Remove noise
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

        # Sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        # Finding sure foreground area using distance transform
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

        # Use adaptive threshold for distance transform to handle varying cell sizes
        max_dist = np.max(dist_transform)
        threshold_ratio = 0.4  # Adjustable parameter
        _, sure_fg = cv2.threshold(dist_transform, threshold_ratio * max_dist, 255, 0)

        # Find unknown region (boundary between cells)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Create markers for watershed
        _, markers = cv2.connectedComponents(sure_fg)

        # Add background marker
        markers = markers + 1

        # Mark unknown regions as 0
        markers[unknown == 255] = 0

        return markers, sure_fg, sure_bg

    def _apply_watershed(self, image, markers):
        """
        Apply watershed algorithm with generated markers
        """
        # Convert single channel to 3-channel for watershed
        image_3ch = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Apply watershed
        labels = watershed(-image, markers, mask=None)

        return labels

    def _extract_cells(self, labels, original_image):
        """
        Extract individual cells from watershed labels and apply quality filters
        """
        cell_masks = []
        cell_contours = []

        # Get unique labels (excluding background)
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels > 1]  # Exclude 0 (boundaries) and 1 (background)

        for label in unique_labels:
            # Create mask for current cell
            mask = (labels == label).astype(np.uint8)

            # Calculate cell area
            cell_area = np.sum(mask)

            # Filter by area - remove cells that are too small or too large
            if self.min_cell_area <= cell_area <= self.max_cell_area:
                # Find contours for this cell
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if contours:
                    # Use the largest contour
                    largest_contour = max(contours, key=cv2.contourArea)

                    # Additional quality checks
                    if self._is_valid_cell(largest_contour, mask):
                        cell_masks.append(mask)
                        cell_contours.append(largest_contour)

        return cell_masks, cell_contours

    def _is_valid_cell(self, contour, mask):
        """
        Quality validation for detected cells based on morphological properties
        """
        # Calculate basic properties
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if perimeter == 0:
            return False

        # Circularity check (cells should be roughly circular)
        circularity = 4 * np.pi * area / (perimeter * perimeter)

        # Convexity check
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        # Define acceptable ranges for cell-like objects
        min_circularity = 0.3  # Allows for some irregularity
        min_solidity = 0.6     # Ensures the object is relatively solid

        return circularity >= min_circularity and solidity >= min_solidity

    def _create_segmentation_visualization(self, original_image, cell_masks):
        """
        Create a colorized visualization of the segmentation results
        """
        # Create overlay on original image
        visualization = original_image.copy()

        # Generate random colors for each cell
        colors = []
        for i in range(len(cell_masks)):
            color = tuple(np.random.randint(50, 255, 3).tolist())
            colors.append(color)

        # Draw each cell with a unique color
        for i, mask in enumerate(cell_masks):
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Draw filled contour with transparency
            overlay = visualization.copy()
            cv2.fillPoly(overlay, contours, colors[i])
            visualization = cv2.addWeighted(visualization, 0.7, overlay, 0.3, 0)

            # Draw contour boundary
            cv2.drawContours(visualization, contours, -1, colors[i], 2)

            # Add cell number
            if contours:
                M = cv2.moments(contours[0])
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(visualization, str(i+1), (cx-10, cy+5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return visualization

    def segment_large_image(self, large_image, tile_size=2048, overlap=256):
        """
        Handle very large images (50K x 50K) by processing in tiles
        This addresses the high-resolution requirements from the job description
        """
        h, w = large_image.shape[:2]
        all_cell_masks = []
        all_cell_contours = []

        # Process image in overlapping tiles
        for y in range(0, h, tile_size - overlap):
            for x in range(0, w, tile_size - overlap):
                # Extract tile
                tile = large_image[y:y+tile_size, x:x+tile_size]

                if tile.size == 0:
                    continue

                # Process tile
                _, tile_masks, tile_contours = self.segment_cells(tile)

                # Adjust coordinates and add to global lists
                for i, (mask, contour) in enumerate(zip(tile_masks, tile_contours)):
                    # Adjust contour coordinates
                    adjusted_contour = contour.copy()
                    adjusted_contour[:, :, 0] += x  # Adjust x coordinates
                    adjusted_contour[:, :, 1] += y  # Adjust y coordinates

                    # Create full-size mask
                    full_mask = np.zeros((h, w), dtype=np.uint8)
                    mask_h, mask_w = mask.shape
                    full_mask[y:y+mask_h, x:x+mask_w] = mask

                    all_cell_masks.append(full_mask)
                    all_cell_contours.append(adjusted_contour)

        # Remove duplicate detections in overlap regions (simplified approach)
        cleaned_masks, cleaned_contours = self._remove_duplicates(all_cell_masks, all_cell_contours)

        # Create final visualization
        final_visualization = self._create_segmentation_visualization(large_image, cleaned_masks)

        return final_visualization, cleaned_masks, cleaned_contours

    def _remove_duplicates(self, masks, contours, iou_threshold=0.5):
        """
        Remove duplicate cell detections in overlap regions using IoU threshold
        """
        if len(masks) <= 1:
            return masks, contours

        keep_indices = []

        for i in range(len(masks)):
            is_duplicate = False

            for j in keep_indices:
                # Calculate IoU between masks
                intersection = np.logical_and(masks[i], masks[j])
                union = np.logical_or(masks[i], masks[j])

                iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0

                if iou > iou_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                keep_indices.append(i)

        # Keep only non-duplicate masks and contours
        cleaned_masks = [masks[i] for i in keep_indices]
        cleaned_contours = [contours[i] for i in keep_indices]

        return cleaned_masks, cleaned_contours
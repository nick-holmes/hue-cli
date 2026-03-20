"""ImageProcessor — loads image, applies smoothing, extracts dominant colors."""

import numpy as np
import logging
from PIL import Image
from sklearn.cluster import KMeans
from skimage import color

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Handles image loading and color quantization"""

    def __init__(self, image_path, width_mm, color_count):
        self.image_path = image_path
        self.width_mm = width_mm
        self.color_count = color_count
        self.image = None
        self.image_lab = None
        self.alpha_mask = None  # Mask for transparent/rounded corners

    def load_and_prepare(self, nozzle_diameter=0.2):
        """Load image and prepare for processing

        CRITICAL: Resolution is set to match nozzle diameter for maximum quality
        - Each pixel = one nozzle width
        - This matches the high-quality example STL approach (0.2mm pixels)
        """
        logger.info(f"Loading image from {self.image_path}")

        try:
            img_orig = Image.open(self.image_path)

            # Check if image has alpha channel
            has_alpha = img_orig.mode in ('RGBA', 'LA') or (img_orig.mode == 'P' and 'transparency' in img_orig.info)

            if has_alpha:
                logger.info("Image has alpha channel - will preserve transparency for rounded corners")
                img_rgba = img_orig.convert('RGBA')
            else:
                img_rgba = img_orig.convert('RGB')

            # UPDATED: Calculate target size based on NOZZLE DIAMETER for maximum detail
            # Each pixel = one nozzle width (e.g., 0.2mm nozzle = 5 pixels per mm)
            # This matches the example STL's approach and maximizes print quality
            pixels_per_mm = 1.0 / nozzle_diameter
            target_width_px = int(self.width_mm * pixels_per_mm)

            logger.info(f"Target resolution: {pixels_per_mm:.1f} pixels/mm (pixel size = {nozzle_diameter}mm = nozzle diameter)")

            # Maintain aspect ratio
            aspect_ratio = img_rgba.height / img_rgba.width
            target_height_px = int(target_width_px * aspect_ratio)

            img_rgba = img_rgba.resize((target_width_px, target_height_px), Image.Resampling.LANCZOS)
            logger.info(f"Resized image to {target_width_px}x{target_height_px} pixels")

            # Convert to numpy array
            img_array = np.array(img_rgba) / 255.0  # Normalize to 0-1

            # Extract alpha channel if present
            if has_alpha:
                self.alpha_mask = img_array[:, :, 3]  # Alpha channel
                self.image = img_array[:, :, :3]  # RGB channels
                logger.info(f"Extracted alpha mask - transparency range: {self.alpha_mask.min():.2f} to {self.alpha_mask.max():.2f}")
            else:
                self.image = img_array[:, :, :3] if img_array.shape[2] >= 3 else img_array
                # Create a full mask (all opaque)
                self.alpha_mask = np.ones((target_height_px, target_width_px))

            # Flip to correct orientation for 3D printing
            # Vertical flip: image Y-axis points down, 3D Y-axis points up
            self.image = np.flipud(self.image)
            self.alpha_mask = np.flipud(self.alpha_mask)

            # Convert to LAB color space
            self.image_lab = color.rgb2lab(self.image)

            return self.image

        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            raise

    def denoise(self, bilateral_sigma_color=0.03, bilateral_sigma_spatial=1):
        """Denoise the image using bilateral filtering.

        Bilateral filter smooths flat color regions while preserving edges
        (keeps text sharp). Applied to self.image in-place; LAB is
        recomputed from the denoised RGB.

        Args:
            bilateral_sigma_color: color similarity threshold (0.03 = light)
            bilateral_sigma_spatial: spatial radius in pixels (1 = tight)
        """
        from skimage.restoration import denoise_bilateral

        alpha_pixels = self.alpha_mask >= 0.5

        denoised = denoise_bilateral(
            self.image, sigma_color=bilateral_sigma_color,
            sigma_spatial=bilateral_sigma_spatial, channel_axis=-1)
        denoised = np.where(alpha_pixels[:, :, np.newaxis], denoised, 0)

        self.image = denoised
        self.image_lab = color.rgb2lab(self.image)

        logger.info(f"Denoised: bilateral "
                     f"(sc={bilateral_sigma_color}, ss={bilateral_sigma_spatial})")

    def quantize_colors(self):
        """Extract dominant colors using K-means clustering

        Ensures extreme colors (very dark, very light) are included for text/highlights.
        """
        logger.info(f"Extracting {self.color_count} dominant colors")

        # Reshape image for K-means
        pixels_lab = self.image_lab.reshape(-1, 3)

        # Filter out transparent pixels so background doesn't waste K-means centroids
        alpha_flat = self.alpha_mask.reshape(-1)
        opaque_mask = alpha_flat >= 0.5
        pixels_lab = pixels_lab[opaque_mask]
        logger.info(f"Clustering {len(pixels_lab)} opaque pixels (filtered {(~opaque_mask).sum()} transparent)")

        # Check for extreme values that should be preserved (text, highlights)
        min_L = pixels_lab[:, 0].min()
        max_L = pixels_lab[:, 0].max()

        # If we have very dark pixels (text), ensure they're represented
        has_very_dark = min_L < 25  # Black/very dark text
        has_very_light = max_L > 90  # White/very light areas

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=self.color_count, random_state=42, n_init=10)
        kmeans.fit(pixels_lab)

        # Get cluster centers (dominant colors in LAB)
        dominant_colors_lab = kmeans.cluster_centers_

        # Force include very dark color if we detected dark pixels but K-means missed them
        if has_very_dark and dominant_colors_lab[:, 0].min() > 25:
            logger.info(f"Forcing very dark color for text (min L={min_L:.1f})")
            # Replace darkest cluster with actual darkest pixels
            very_dark_pixels = pixels_lab[pixels_lab[:, 0] < 25]
            darkest_cluster_idx = np.argmin(dominant_colors_lab[:, 0])
            dominant_colors_lab[darkest_cluster_idx] = np.mean(very_dark_pixels, axis=0)

        # Sort by luminosity (L channel): darkest to lightest
        sorted_indices = np.argsort(dominant_colors_lab[:, 0])
        dominant_colors_lab = dominant_colors_lab[sorted_indices]

        logger.info(f"Extracted colors (L values): {dominant_colors_lab[:, 0]}")

        return dominant_colors_lab, kmeans, sorted_indices

    def auto_determine_color_count(self, min_delta_e=10.0, max_colors=12):
        """Determine optimal number of colors by iterating K-means until new clusters add little.

        Starts at K=2, increments K. For each K+1 result, finds the "new" cluster center
        (highest minimum delta-E to any K center). If that min delta-E < threshold, stop.

        Args:
            min_delta_e: Minimum delta-E a new color must contribute to be worthwhile
            max_colors: Maximum number of colors to try

        Returns:
            Optimal color count (int)
        """
        pixels_lab = self.image_lab.reshape(-1, 3)
        alpha_flat = self.alpha_mask.reshape(-1)
        pixels_lab = pixels_lab[alpha_flat >= 0.5]

        logger.info(f"Auto-determining color count (min delta-E={min_delta_e}, max={max_colors})...")

        prev_centers = None
        optimal_k = 2

        for k in range(2, max_colors + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(pixels_lab)
            centers = kmeans.cluster_centers_

            if prev_centers is not None:
                # Find the "new" center: the one with highest min-distance to previous centers
                best_new_delta_e = 0
                for center in centers:
                    # Min delta-E from this center to any previous center
                    min_de = min(
                        color.deltaE_ciede2000(
                            np.array([[center]]),
                            np.array([[pc]])
                        )[0][0]
                        for pc in prev_centers
                    )
                    best_new_delta_e = max(best_new_delta_e, min_de)

                logger.info(f"  K={k}: new cluster delta-E={best_new_delta_e:.1f}")

                if best_new_delta_e < min_delta_e:
                    logger.info(f"  Stopping at K={k-1} (new color delta-E {best_new_delta_e:.1f} < {min_delta_e})")
                    break

                optimal_k = k
            else:
                optimal_k = k
                logger.info(f"  K={k}: initial clustering")

            prev_centers = centers

        logger.info(f"Optimal color count: {optimal_k}")
        return optimal_k

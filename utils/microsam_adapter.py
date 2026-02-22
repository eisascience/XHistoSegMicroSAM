"""
MicroSAM Adapter for XHaloPathAnalyzer

Provides a MedSAM-compatible interface for MicroSAM to minimize changes to existing code.

Memory Management Notes:
- For images with many objects (500+), memory can be exhausted
- Reduce TILE_SHAPE in .env (e.g., 512,512 or 256,256) 
- Reduce MICROSAM_BATCH_SIZE in .env (e.g., 4 or 8)
- Set MICROSAM_DEVICE=cpu in .env to avoid GPU OOM
- MPS cache is automatically cleared before each prediction
"""

import torch
import numpy as np
from typing import Optional, Tuple
import logging
import os

logger = logging.getLogger(__name__)


# Import MicroSAM predictor and elf availability check
from xhalo.ml import MicroSAMPredictor as BaseMicroSAMPredictor, is_elf_available


class MicroSAMPredictor:
    """
    Adapter class that provides a MedSAM-compatible interface for MicroSAM.
    
    This allows minimal changes to existing app.py code while using MicroSAM backend.
    """
    
    def __init__(
        self,
        model_type: str = "vit_b_histopathology",
        device: Optional[str] = None,
        segmentation_mode: str = "interactive",  # "interactive" or "automatic"
        tile_shape: Tuple[int, int] = (1024, 1024),
        halo: Tuple[int, int] = (256, 256)
    ):
        """
        Initialize MicroSAM predictor with MedSAM-compatible interface.
        
        Args:
            model_type: Model architecture (default: vit_b_histopathology)
            device: Device to use (cuda/mps/cpu), auto-detected if None
            segmentation_mode: "interactive" (with prompts) or "automatic" (no prompts)
            tile_shape: Tile size for tiled inference
            halo: Halo size for tiled inference
        """
        # Check if automatic mode is requested but elf is not available
        if segmentation_mode == "automatic" and not is_elf_available():
            logger.warning(
                "Automatic segmentation mode requires python-elf, which is not available. "
                "Falling back to interactive mode. Use prompt-based modes instead."
            )
            segmentation_mode = "interactive"
        
        self.segmentation_mode = segmentation_mode
        self.device = device
        
        # Get batch size from environment or use default
        try:
            batch_size = int(os.getenv("MICROSAM_BATCH_SIZE", "8"))
            if batch_size < 1 or batch_size > 32:
                logger.warning(f"MICROSAM_BATCH_SIZE={batch_size} is outside recommended range [1-32]. Using default 8.")
                batch_size = 8
            self.batch_size = batch_size
        except (ValueError, TypeError):
            logger.warning("Invalid MICROSAM_BATCH_SIZE value. Using default 8.")
            self.batch_size = 8
        logger.info(f"Using batch_size={self.batch_size}")
        
        # Initialize base MicroSAM predictor
        logger.info(f"Initializing MicroSAM (mode={segmentation_mode}, model={model_type})")
        self.predictor = BaseMicroSAMPredictor(
            model_type=model_type,
            device=device,
            tile_shape=tile_shape,
            halo=halo
        )
        
        logger.info(f"MicroSAM adapter ready (device={self.predictor.device})")
    
    def predict(
        self,
        image: np.ndarray,
        prompt_mode: str = "auto_box",
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        multimask_output: bool = False,
        min_area_ratio: float = 0.01,
        morph_kernel_size: int = 5,
        # Legacy threshold parameters (kept for backward compatibility)
        threshold_mode: str = "otsu",
        threshold_value: float = 0.5,
        box_min_area: int = 100,
        box_max_area: int = 100000,
        box_dilation_radius: int = 0,
        # New nucleus-grade pipeline parameters (B1-B8)
        nucleus_normalize: bool = True,
        nucleus_p_low: float = 1.0,
        nucleus_p_high: float = 99.5,
        nucleus_invert: bool = False,
        nucleus_bg_correction: bool = True,
        nucleus_bg_method: str = "gaussian",
        nucleus_bg_sigma: float = 50.0,
        nucleus_bg_radius: int = 50,
        nucleus_threshold_mode: str = "otsu",
        nucleus_threshold_value: float = 128.0,
        nucleus_adaptive_block_size: int = 51,
        nucleus_adaptive_C: float = 2.0,
        nucleus_foreground_bright: bool = True,
        nucleus_morph_kernel_size: int = 3,
        nucleus_morph_iterations: int = 1,
        nucleus_morph_order: str = "open_close",
        nucleus_use_watershed: bool = True,
        nucleus_seed_min_distance: int = 5,
        nucleus_min_area_px: int = 100,
        nucleus_max_area_px: int = 5000,
        nucleus_prompt_type: str = "points",
        nucleus_bbox_padding: int = 3,
    ) -> np.ndarray:
        """
        Run inference with MedSAM-compatible interface.

        Args:
            image: Input RGB image (H, W, 3)
            prompt_mode: "auto_box" (tissue ROI box), "full_box", "point",
                         or "auto_box_from_threshold" (nucleus-grade pipeline)
            point_coords: Point prompts (N, 2) in (x, y) format
            point_labels: Labels for points (1 = foreground, 0 = background)
            box: Bounding box prompt [x1, y1, x2, y2]
            multimask_output: Whether to use multimasking
            min_area_ratio: Minimum area ratio for auto tissue detection (auto_box mode)
            morph_kernel_size: Kernel size for morphological ops (auto_box mode)
            threshold_mode: Legacy threshold mode for backward compatibility
            threshold_value: Legacy threshold value for backward compatibility
            box_min_area: Legacy minimum box area for backward compatibility
            box_max_area: Legacy maximum box area for backward compatibility
            box_dilation_radius: Legacy dilation radius for backward compatibility
            nucleus_*: Parameters for the nucleus-grade pipeline (B1-B6).
                       See compute_candidates_from_threshold for details.

        Returns:
            Binary mask (H, W) with 0=background, 255=foreground
            OR instance mask (H, W) with 0=background, 1..N=instances
            (for auto_box_from_threshold or automatic mode)
        """
        # Clear MPS cache before prediction to prevent OOM on Apple Silicon
        if self.predictor.device == "mps":
            try:
                torch.mps.empty_cache()
                logger.debug("Cleared MPS cache before prediction")
            except Exception as e:
                logger.warning(f"Failed to clear MPS cache: {e}")

        # Ensure RGB format
        image = self.predictor.ensure_rgb(image)
        h, w = image.shape[:2]

        # Sanity-check: output masks must always be (h, w)
        assert h > 0 and w > 0, f"Invalid image dimensions: {h}x{w}"

        logger.info(f"MicroSAM prediction: mode={self.segmentation_mode}, prompt_mode={prompt_mode}")

        if self.segmentation_mode == "automatic":
            instance_mask = self.predictor.predict_auto_instances(
                image=image,
                segmentation_mode="apg",
                min_size=25,
                tiled=True
            )
            binary_mask = (instance_mask > 0).astype(np.uint8) * 255
            logger.info(f"Automatic segmentation: {np.unique(instance_mask).size - 1} instances")
            assert binary_mask.shape == (h, w), \
                f"Output mask shape {binary_mask.shape} != image shape ({h},{w})"
            return binary_mask

        # Interactive prompted segmentation
        boxes_prompt = None
        points_prompt = None
        labels_prompt = None

        if box is not None:
            boxes_prompt = np.array([box])
            logger.info(f"Using provided box: {box}")

        elif prompt_mode == "auto_box_from_threshold":
            # Use the robust nucleus-grade pipeline (B1-B6)
            pts, pt_lbls, boxes, _dbg = compute_candidates_from_threshold(
                image=image,
                normalize=nucleus_normalize,
                p_low=nucleus_p_low,
                p_high=nucleus_p_high,
                invert_intensity=nucleus_invert,
                bg_correction=nucleus_bg_correction,
                bg_method=nucleus_bg_method,
                bg_sigma=nucleus_bg_sigma,
                bg_radius=nucleus_bg_radius,
                threshold_mode=nucleus_threshold_mode,
                threshold_value=nucleus_threshold_value,
                adaptive_block_size=nucleus_adaptive_block_size,
                adaptive_C=nucleus_adaptive_C,
                foreground_bright=nucleus_foreground_bright,
                morph_kernel_size=nucleus_morph_kernel_size,
                morph_iterations=nucleus_morph_iterations,
                morph_order=nucleus_morph_order,
                use_watershed=nucleus_use_watershed,
                seed_min_distance=nucleus_seed_min_distance,
                min_area_px=nucleus_min_area_px,
                max_area_px=nucleus_max_area_px,
                prompt_type=nucleus_prompt_type,
                bbox_padding=nucleus_bbox_padding,
            )

            if nucleus_prompt_type == "points" and len(pts) > 0:
                points_prompt = pts
                labels_prompt = pt_lbls
                logger.info(f"Generated {len(pts)} centroid point prompts")
            elif nucleus_prompt_type == "boxes" and len(boxes) > 0:
                boxes_prompt = boxes
                logger.info(f"Generated {len(boxes)} boxes from threshold pipeline")
            else:
                logger.warning("No candidates found from nucleus pipeline. Returning empty mask.")
                return np.zeros((h, w), dtype=np.uint8)

        elif prompt_mode == "auto_box":
            # Tissue ROI detection only (requirement A: single box, not nucleus instances)
            from utils.medsam_models import _compute_tissue_bbox
            auto_box = _compute_tissue_bbox(image, min_area_ratio, morph_kernel_size)
            boxes_prompt = np.array([auto_box])
            logger.info(f"Auto-detected tissue ROI box: {auto_box}")

        elif prompt_mode == "full_box":
            full_box = np.array([0, 0, w - 1, h - 1])
            boxes_prompt = np.array([full_box])
            logger.info("Using full image box")

        elif prompt_mode == "point":
            if point_coords is not None and point_labels is not None:
                points_prompt = point_coords[:, np.newaxis, :]
                labels_prompt = point_labels[:, np.newaxis]
                logger.info(f"Using {len(point_coords)} point prompts")
            else:
                center = np.array([[[w // 2, h // 2]]])
                points_prompt = center
                labels_prompt = np.array([[1]])
                logger.info("Using default center point")

        try:
            instance_mask = self.predictor.predict_from_prompts(
                image=image,
                boxes=boxes_prompt,
                points=points_prompt,
                point_labels=labels_prompt,
                multimasking=multimask_output,
                tiled=True
            )

            assert instance_mask.shape == (h, w), \
                f"Instance mask shape {instance_mask.shape} != image shape ({h},{w})"

            if prompt_mode == "auto_box_from_threshold":
                logger.info(f"Instance segmentation complete: {np.unique(instance_mask).size - 1} instances")
                return instance_mask
            else:
                binary_mask = (instance_mask > 0).astype(np.uint8) * 255
                logger.info("Interactive segmentation complete")
                return binary_mask

        except Exception as e:
            logger.error(f"MicroSAM prediction failed: {e}")
            return np.zeros((h, w), dtype=np.uint8)


# For backward compatibility, also export helper functions
def _ensure_rgb_uint8(image: np.ndarray) -> np.ndarray:
    """Ensure image is RGB uint8 format."""
    from utils.medsam_models import _ensure_rgb_uint8 as original_ensure_rgb
    return original_ensure_rgb(image)


def _compute_tissue_bbox(
    image: np.ndarray,
    min_area_ratio: float = 0.01,
    morph_kernel_size: int = 5
) -> np.ndarray:
    """Compute tissue bounding box."""
    from utils.medsam_models import _compute_tissue_bbox as original_compute_bbox
    return original_compute_bbox(image, min_area_ratio, morph_kernel_size)


def compute_candidates_from_threshold(
    image: np.ndarray,
    # B1 - Normalization
    normalize: bool = True,
    p_low: float = 1.0,
    p_high: float = 99.5,
    invert_intensity: bool = False,
    # B2 - Background correction
    bg_correction: bool = True,
    bg_method: str = "gaussian",
    bg_sigma: float = 50.0,
    bg_radius: int = 50,
    # B3 - Thresholding
    threshold_mode: str = "otsu",
    threshold_value: float = 128.0,
    adaptive_block_size: int = 51,
    adaptive_C: float = 2.0,
    foreground_bright: bool = True,
    # B4 - Morphology
    morph_kernel_size: int = 3,
    morph_iterations: int = 1,
    morph_order: str = "open_close",
    # B5 - Watershed
    use_watershed: bool = True,
    seed_min_distance: int = 5,
    min_area_px: int = 100,
    max_area_px: int = 5000,
    # B6 - Prompt type
    prompt_type: str = "points",
    bbox_padding: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Robust nucleus/cell candidate generation pipeline for DAPI or similar channels.

    Implements requirements B1-B6:
    - Percentile normalization (B1)
    - Background correction (B2)
    - Multiple thresholding modes with polarity control (B3)
    - Morphological cleanup (B4)
    - Watershed splitting for touching nuclei (B5)
    - Point or box prompt generation (B6)

    Args:
        image: Input image (H, W) grayscale or (H, W, 3) RGB
        normalize: Percentile-clip and rescale to uint8 (default True)
        p_low: Low percentile for normalization (default 1.0)
        p_high: High percentile for normalization (default 99.5)
        invert_intensity: Invert image intensities before thresholding (default False)
        bg_correction: Apply background correction before thresholding (default True)
        bg_method: Background correction method: "gaussian" or "tophat" (default "gaussian")
        bg_sigma: Sigma for Gaussian background subtraction (default 50.0)
        bg_radius: Radius for morphological top-hat background correction (default 50)
        threshold_mode: "otsu", "manual", "adaptive_gaussian", "adaptive_mean" (default "otsu")
        threshold_value: Manual threshold (0-255 uint8 scale; default 128.0)
        adaptive_block_size: Block size for adaptive thresholding (must be odd; default 51)
        adaptive_C: Constant subtracted from mean for adaptive threshold (default 2.0)
        foreground_bright: True = bright objects are foreground (default True)
        morph_kernel_size: Kernel size for morphological ops (odd int; default 3)
        morph_iterations: Number of morphological iterations (default 1)
        morph_order: "open_close" or "close_open" (default "open_close")
        use_watershed: Split touching nuclei with watershed (default True)
        seed_min_distance: Min distance between watershed seeds (default 5)
        min_area_px: Min candidate area in pixels (default 100)
        max_area_px: Max candidate area in pixels (default 5000)
        prompt_type: "points" (centroids) or "boxes" (bounding boxes; default "points")
        bbox_padding: Padding added to bounding boxes in pixels (default 3)

    Returns:
        points: (N, 1, 2) centroid array [[x, y]] or empty if prompt_type != "points"
        point_labels: (N, 1) array of ones or empty
        boxes: (N, 4) array [x1, y1, x2, y2] or empty if prompt_type != "boxes"
        debug_info: dict with intermediate images for debugging (keys: normalized,
                    bg_corrected, binary_mask, distance, label_image, n_before, n_after)
    """
    import cv2
    from skimage import measure, morphology
    from scipy.ndimage import distance_transform_edt, gaussian_filter
    from skimage.feature import peak_local_max

    debug_info: dict = {}

    # ---- Convert to single-channel grayscale ----------------------------
    if image.ndim == 3:
        if image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image[:, :, 0].copy()
    else:
        gray = image.copy()

    gray = gray.astype(np.float32)

    # ---- B1: Percentile normalization ------------------------------------
    if normalize:
        lo = float(np.percentile(gray, p_low))
        hi = float(np.percentile(gray, p_high))
        if hi > lo:
            gray = np.clip((gray - lo) / (hi - lo + 1e-8), 0.0, 1.0) * 255.0
        else:
            gray = np.clip(gray, 0.0, 255.0)
        logger.info(f"Percentile normalization p{p_low}={lo:.1f}, p{p_high}={hi:.1f}")
    else:
        gray = np.clip(gray, 0.0, 255.0)

    if invert_intensity:
        gray = 255.0 - gray
        logger.info("Intensity inverted")

    debug_info["normalized"] = gray.astype(np.uint8)

    # ---- B2: Background correction ---------------------------------------
    if bg_correction:
        if bg_method == "gaussian":
            bg = gaussian_filter(gray, sigma=bg_sigma)
            corrected = np.clip(gray - bg, 0.0, 255.0)
        else:  # tophat
            from skimage.morphology import white_tophat, disk
            footprint = disk(bg_radius)
            corrected = white_tophat(gray.astype(np.uint8), footprint).astype(np.float32)
        logger.info(f"Background correction: method={bg_method}")
    else:
        corrected = gray

    debug_info["bg_corrected"] = corrected.astype(np.uint8)

    img_u8 = corrected.astype(np.uint8)

    # ---- B3: Thresholding -----------------------------------------------
    if threshold_mode == "otsu":
        _, binary_u8 = cv2.threshold(img_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        logger.info(f"Otsu thresholding applied")
    elif threshold_mode == "manual":
        tval = int(np.clip(threshold_value, 0, 255))
        _, binary_u8 = cv2.threshold(img_u8, tval, 255, cv2.THRESH_BINARY)
        logger.info(f"Manual threshold={tval}")
    elif threshold_mode == "adaptive_gaussian":
        block = adaptive_block_size if adaptive_block_size % 2 == 1 else adaptive_block_size + 1
        binary_u8 = cv2.adaptiveThreshold(
            img_u8, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block, int(adaptive_C)
        )
        logger.info(f"Adaptive Gaussian threshold: block={block}, C={adaptive_C}")
    elif threshold_mode == "adaptive_mean":
        block = adaptive_block_size if adaptive_block_size % 2 == 1 else adaptive_block_size + 1
        binary_u8 = cv2.adaptiveThreshold(
            img_u8, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            block, int(adaptive_C)
        )
        logger.info(f"Adaptive Mean threshold: block={block}, C={adaptive_C}")
    else:
        # "off": all foreground
        binary_u8 = np.ones_like(img_u8, dtype=np.uint8) * 255
        logger.info("Thresholding mode=off, all pixels foreground")

    # Apply foreground polarity: if dark objects are foreground, invert
    if not foreground_bright:
        binary_u8 = 255 - binary_u8
        logger.info("Inverted binary mask (dark-foreground mode)")

    binary = binary_u8 > 0

    debug_info["binary_mask"] = binary_u8

    # ---- B4: Morphological cleanup --------------------------------------
    if morph_kernel_size > 0:
        ksz = morph_kernel_size if morph_kernel_size % 2 == 1 else morph_kernel_size + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
        b = binary_u8
        if morph_order == "open_close":
            b = cv2.morphologyEx(b, cv2.MORPH_OPEN,  kernel, iterations=morph_iterations)
            b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations)
        else:  # close_open
            b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations)
            b = cv2.morphologyEx(b, cv2.MORPH_OPEN,  kernel, iterations=morph_iterations)
        binary = b > 0

    # ---- B5: Watershed splitting or connected components ----------------
    if use_watershed and np.any(binary):
        distance = distance_transform_edt(binary).astype(np.float32)
        debug_info["distance"] = distance

        coords = peak_local_max(distance, min_distance=seed_min_distance, labels=binary)
        markers = np.zeros_like(binary, dtype=np.int32)
        for idx, (r, c) in enumerate(coords, start=1):
            markers[r, c] = idx

        if np.max(markers) > 0:
            from skimage.segmentation import watershed
            label_image = watershed(-distance, markers, mask=binary)
        else:
            label_image = measure.label(binary)
        logger.info(f"Watershed: {np.max(label_image)} candidate regions found")
    else:
        distance = distance_transform_edt(binary).astype(np.float32)
        debug_info["distance"] = distance
        label_image = measure.label(binary)
        logger.info(f"Connected components: {np.max(label_image)} regions found")

    debug_info["label_image"] = label_image.astype(np.int32)

    # ---- Filter by area and build prompts -------------------------------
    h, w = binary.shape
    regions = measure.regionprops(label_image)
    debug_info["n_before"] = len(regions)
    logger.info(f"{len(regions)} candidates before area filtering")

    points_list = []
    boxes_list = []

    for region in regions:
        area = region.area
        if area < min_area_px or area > max_area_px:
            continue
        cy, cx = region.centroid  # (row, col) -> (y, x)
        if prompt_type == "points":
            points_list.append([[int(round(cx)), int(round(cy))]])
        else:
            min_row, min_col, max_row, max_col = region.bbox
            x1 = max(0, min_col - bbox_padding)
            y1 = max(0, min_row - bbox_padding)
            x2 = min(w - 1, max_col + bbox_padding)
            y2 = min(h - 1, max_row + bbox_padding)
            boxes_list.append([x1, y1, x2, y2])

    debug_info["n_after"] = len(points_list) if prompt_type == "points" else len(boxes_list)
    logger.info(f"{debug_info['n_after']} candidates after area filtering (min={min_area_px}, max={max_area_px})")

    if prompt_type == "points":
        if points_list:
            points = np.array(points_list, dtype=np.int32)   # (N, 1, 2)
            point_labels = np.ones((len(points_list), 1), dtype=np.int32)
        else:
            points = np.empty((0, 1, 2), dtype=np.int32)
            point_labels = np.empty((0, 1), dtype=np.int32)
        boxes = np.empty((0, 4), dtype=np.int32)
    else:
        points = np.empty((0, 1, 2), dtype=np.int32)
        point_labels = np.empty((0, 1), dtype=np.int32)
        boxes = np.array(boxes_list, dtype=np.int32) if boxes_list else np.empty((0, 4), dtype=np.int32)

    return points, point_labels, boxes, debug_info


def compute_boxes_from_threshold(
    image: np.ndarray,
    threshold_mode: str = "otsu",
    threshold_value: float = 0.5,
    min_area: int = 100,
    max_area: int = 100000,
    dilation_radius: int = 0,
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate bounding boxes from thresholded image for auto_box_from_threshold mode.

    .. deprecated::
        Use :func:`compute_candidates_from_threshold` with ``prompt_type="boxes"``
        for nucleus-grade instance segmentation. This function is retained for
        backward compatibility only.

    Args:
        image: Input image (H, W) grayscale or (H, W, 3) RGB
        threshold_mode: "manual", "otsu", or "off"
        threshold_value: Threshold value (0-1) if manual mode
        min_area: Minimum area in pixels for components
        max_area: Maximum area in pixels for components
        dilation_radius: Pixels to dilate mask before boxing (to capture full nuclei)
        normalize: Whether to normalize image before thresholding

    Returns:
        boxes: (N, 4) array of boxes [x1, y1, x2, y2]
        binary_mask: (H, W) binary mask of thresholded regions
    """
    from skimage import measure, morphology
    from skimage.filters import threshold_otsu
    import cv2

    # Convert to grayscale if needed
    if image.ndim == 3:
        if image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image[:, :, 0]
    else:
        gray = image.copy()

    if normalize:
        gray = gray.astype(np.float32)
        if gray.max() > 1.0:
            gray = gray / 255.0

    if threshold_mode == "off":
        binary_mask = np.ones_like(gray, dtype=bool)
    elif threshold_mode == "otsu":
        gray_norm = gray if normalize else gray.astype(np.float32) / 255.0
        thresh = threshold_otsu(gray_norm)
        binary_mask = gray_norm > thresh
        logger.info(f"Otsu threshold: {thresh:.3f}")
    else:
        binary_mask = gray > threshold_value
        logger.info(f"Manual threshold: {threshold_value:.3f}")

    if dilation_radius > 0:
        struct = morphology.disk(dilation_radius)
        binary_mask = morphology.binary_dilation(binary_mask, struct)

    labeled_mask = measure.label(binary_mask)
    regions = measure.regionprops(labeled_mask)
    logger.info(f"Found {len(regions)} connected components before filtering")

    boxes = []
    for region in regions:
        area = region.area
        if min_area <= area <= max_area:
            min_row, min_col, max_row, max_col = region.bbox
            boxes.append([min_col, min_row, max_col, max_row])

    boxes = np.array(boxes) if boxes else np.empty((0, 4))
    logger.info(f"Extracted {len(boxes)} boxes after area filtering (min={min_area}, max={max_area})")

    return boxes, binary_mask.astype(np.uint8)

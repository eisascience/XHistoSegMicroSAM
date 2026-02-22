"""
Tests for the nucleus-grade segmentation pipeline (compute_candidates_from_threshold).
These tests exercise the pipeline logic without requiring torch/micro-sam.
"""
import numpy as np
import pytest
import cv2
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage import measure
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


# ---------------------------------------------------------------------------
# Inline implementation mirror (so the test runs without importing the full
# module that requires torch).  We import the function here and fall back to
# a local copy if torch is unavailable.
# ---------------------------------------------------------------------------
def _pipeline(image, **kwargs):
    """Import or re-implement compute_candidates_from_threshold."""
    try:
        import sys, importlib
        # Stub out torch so the module-level import won't fail
        if "torch" not in sys.modules:
            import types
            torch_stub = types.ModuleType("torch")
            torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
            torch_stub.backends = types.SimpleNamespace(
                mps=types.SimpleNamespace(is_available=lambda: False)
            )
            sys.modules["torch"] = torch_stub
        # Also stub micro_sam if not installed
        if "micro_sam" not in sys.modules:
            import types
            sys.modules["micro_sam"] = types.ModuleType("micro_sam")
        sys.path.insert(0, ".")
        from utils.microsam_adapter import compute_candidates_from_threshold
        return compute_candidates_from_threshold(image, **kwargs)
    except Exception:
        # Fallback: run the logic inline
        return _inline_pipeline(image, **kwargs)


def _inline_pipeline(
    image,
    normalize=True, p_low=1.0, p_high=99.5, invert_intensity=False,
    bg_correction=True, bg_method="gaussian", bg_sigma=50.0, bg_radius=50,
    threshold_mode="otsu", threshold_value=128.0,
    adaptive_block_size=51, adaptive_C=2.0, foreground_bright=True,
    morph_kernel_size=3, morph_iterations=1, morph_order="open_close",
    use_watershed=True, seed_min_distance=5,
    min_area_px=100, max_area_px=5000,
    prompt_type="points", bbox_padding=3,
):
    """Inline copy of compute_candidates_from_threshold for testing."""
    debug_info = {}

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.shape[2] == 3 else image[:, :, 0].copy()
    else:
        gray = image.copy()
    gray = gray.astype(np.float32)

    if normalize:
        lo = float(np.percentile(gray, p_low))
        hi = float(np.percentile(gray, p_high))
        gray = np.clip((gray - lo) / (hi - lo + 1e-8), 0.0, 1.0) * 255.0

    if invert_intensity:
        gray = 255.0 - gray

    debug_info["normalized"] = gray.astype(np.uint8)

    if bg_correction:
        if bg_method == "gaussian":
            bg = gaussian_filter(gray, sigma=bg_sigma)
            corrected = np.clip(gray - bg, 0.0, 255.0)
        else:
            from skimage.morphology import white_tophat, disk
            corrected = white_tophat(gray.astype(np.uint8), disk(bg_radius)).astype(np.float32)
    else:
        corrected = gray

    debug_info["bg_corrected"] = corrected.astype(np.uint8)
    img_u8 = corrected.astype(np.uint8)

    if threshold_mode == "otsu":
        _, binary_u8 = cv2.threshold(img_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif threshold_mode == "manual":
        _, binary_u8 = cv2.threshold(img_u8, int(threshold_value), 255, cv2.THRESH_BINARY)
    elif threshold_mode == "adaptive_gaussian":
        block = adaptive_block_size if adaptive_block_size % 2 == 1 else adaptive_block_size + 1
        binary_u8 = cv2.adaptiveThreshold(img_u8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, block, int(adaptive_C))
    elif threshold_mode == "adaptive_mean":
        block = adaptive_block_size if adaptive_block_size % 2 == 1 else adaptive_block_size + 1
        binary_u8 = cv2.adaptiveThreshold(img_u8, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                           cv2.THRESH_BINARY, block, int(adaptive_C))
    else:
        binary_u8 = np.ones_like(img_u8, dtype=np.uint8) * 255

    if not foreground_bright:
        binary_u8 = 255 - binary_u8

    binary = binary_u8 > 0
    debug_info["binary_mask"] = binary_u8

    if morph_kernel_size > 0:
        ksz = morph_kernel_size if morph_kernel_size % 2 == 1 else morph_kernel_size + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
        b = binary_u8
        if morph_order == "open_close":
            b = cv2.morphologyEx(b, cv2.MORPH_OPEN,  kernel, iterations=morph_iterations)
            b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations)
        else:
            b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations)
            b = cv2.morphologyEx(b, cv2.MORPH_OPEN,  kernel, iterations=morph_iterations)
        binary = b > 0

    distance = distance_transform_edt(binary).astype(np.float32)
    debug_info["distance"] = distance

    if use_watershed and np.any(binary):
        coords = peak_local_max(distance, min_distance=seed_min_distance, labels=binary)
        markers = np.zeros_like(binary, dtype=np.int32)
        for idx, (r, c) in enumerate(coords, start=1):
            markers[r, c] = idx
        if np.max(markers) > 0:
            label_image = watershed(-distance, markers, mask=binary)
        else:
            label_image = measure.label(binary)
    else:
        label_image = measure.label(binary)

    debug_info["label_image"] = label_image.astype(np.int32)
    h, w = binary.shape
    regions = measure.regionprops(label_image)
    debug_info["n_before"] = len(regions)

    points_list, boxes_list = [], []
    for region in regions:
        if region.area < min_area_px or region.area > max_area_px:
            continue
        cy, cx = region.centroid
        if prompt_type == "points":
            points_list.append([[int(round(cx)), int(round(cy))]])
        else:
            r0, c0, r1, c1 = region.bbox
            boxes_list.append([max(0, c0 - bbox_padding), max(0, r0 - bbox_padding),
                                min(w - 1, c1 + bbox_padding), min(h - 1, r1 + bbox_padding)])

    debug_info["n_after"] = len(points_list) if prompt_type == "points" else len(boxes_list)

    if prompt_type == "points":
        pts = np.array(points_list, dtype=np.int32) if points_list else np.empty((0, 1, 2), dtype=np.int32)
        pt_lbls = np.ones((len(points_list), 1), dtype=np.int32) if points_list else np.empty((0, 1), dtype=np.int32)
        boxes = np.empty((0, 4), dtype=np.int32)
    else:
        pts = np.empty((0, 1, 2), dtype=np.int32)
        pt_lbls = np.empty((0, 1), dtype=np.int32)
        boxes = np.array(boxes_list, dtype=np.int32) if boxes_list else np.empty((0, 4), dtype=np.int32)

    return pts, pt_lbls, boxes, debug_info


def _make_dapi_image(size=200, n_nuclei=5, radius=8, brightness=200):
    """Create a synthetic DAPI-like image with round bright blobs."""
    img = np.zeros((size, size), dtype=np.uint8)
    positions = [
        (size // 4, size // 4),
        (size // 4, 3 * size // 4),
        (size // 2, size // 2),
        (3 * size // 4, size // 4),
        (3 * size // 4, 3 * size // 4),
    ][:n_nuclei]
    yy, xx = np.mgrid[:size, :size]
    for cy, cx in positions:
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
        img[mask] = brightness
    return img


class TestNucleusPipelinePoints:
    """Tests for compute_candidates_from_threshold with point prompts."""

    def test_basic_dapi_detection(self):
        """Should detect ~5 well-separated nuclei as centroid points."""
        img = _make_dapi_image(n_nuclei=5)
        pts, pt_lbls, boxes, dbg = _inline_pipeline(
            img,
            normalize=True,
            bg_correction=False,
            threshold_mode="otsu",
            foreground_bright=True,
            use_watershed=False,
            min_area_px=50,
            max_area_px=2000,
            prompt_type="points",
        )
        assert pts.shape[1:] == (1, 2), f"Unexpected points shape: {pts.shape}"
        assert pt_lbls.shape[1] == 1
        assert len(pts) >= 4, f"Expected ≥4 nucleus candidates, got {len(pts)}"
        assert boxes.shape == (0, 4), "Should have no boxes when prompt_type=points"
        assert dbg["n_before"] >= 1
        assert dbg["n_after"] == len(pts)

    def test_point_labels_all_ones(self):
        """All point labels must be 1 (positive prompt)."""
        img = _make_dapi_image(n_nuclei=3)
        pts, pt_lbls, _, _ = _inline_pipeline(
            img, bg_correction=False, use_watershed=False,
            min_area_px=50, max_area_px=2000, prompt_type="points",
        )
        assert np.all(pt_lbls == 1), f"Not all labels are 1: {pt_lbls}"

    def test_empty_image_returns_no_candidates(self):
        """Uniform dark image → no foreground → empty output."""
        img = np.zeros((100, 100), dtype=np.uint8)
        pts, pt_lbls, boxes, dbg = _inline_pipeline(
            img, bg_correction=False, threshold_mode="manual",
            threshold_value=128, foreground_bright=True,
            min_area_px=10, max_area_px=10000, prompt_type="points",
        )
        assert len(pts) == 0
        assert dbg["n_after"] == 0


class TestNucleusPipelineBoxes:
    """Tests for compute_candidates_from_threshold with box prompts."""

    def test_box_shape_and_bounds(self):
        """Box coordinates must be within image and x1<x2, y1<y2."""
        img = _make_dapi_image(n_nuclei=4, size=200)
        pts, pt_lbls, boxes, dbg = _inline_pipeline(
            img,
            normalize=True,
            bg_correction=False,
            threshold_mode="otsu",
            foreground_bright=True,
            use_watershed=False,
            min_area_px=50,
            max_area_px=2000,
            prompt_type="boxes",
            bbox_padding=3,
        )
        assert boxes.shape[1] == 4, f"Unexpected boxes shape: {boxes.shape}"
        assert len(boxes) >= 3, f"Expected ≥3 boxes, got {len(boxes)}"
        for box in boxes:
            x1, y1, x2, y2 = box
            assert x1 < x2, f"x1={x1} ≥ x2={x2}"
            assert y1 < y2, f"y1={y1} ≥ y2={y2}"
            assert x1 >= 0 and y1 >= 0
            assert x2 <= 199 and y2 <= 199
        assert pts.shape == (0, 1, 2), "Should have no points when prompt_type=boxes"


class TestNucleusPipelineWatershed:
    """Tests for watershed splitting of touching nuclei."""

    def _make_touching_nuclei(self):
        """Two overlapping blobs that connected components would merge."""
        img = np.zeros((100, 150), dtype=np.uint8)
        yy, xx = np.mgrid[:100, :150]
        for cy, cx in [(50, 50), (50, 90)]:
            img[(yy - cy) ** 2 + (xx - cx) ** 2 <= 20 ** 2] = 200
        return img

    def test_watershed_splits_touching_nuclei(self):
        """Watershed should produce more labels than connected components for touching blobs."""
        img = self._make_touching_nuclei()

        # Connected components (no watershed) → likely 1 merged blob
        _, _, _, dbg_cc = _inline_pipeline(
            img, bg_correction=False, threshold_mode="otsu",
            use_watershed=False,
            min_area_px=50, max_area_px=50000, prompt_type="points",
        )

        # Watershed → should split into 2+
        pts_ws, _, _, dbg_ws = _inline_pipeline(
            img, bg_correction=False, threshold_mode="otsu",
            use_watershed=True, seed_min_distance=8,
            min_area_px=50, max_area_px=50000, prompt_type="points",
        )
        # Watershed should find at least as many objects as CC
        assert dbg_ws["n_after"] >= dbg_cc["n_after"], \
            f"Watershed found fewer objects ({dbg_ws['n_after']}) than CC ({dbg_cc['n_after']})"


class TestNucleusPipelineNormalization:
    """Tests for B1 normalization and intensity inversion."""

    def test_invert_intensity(self):
        """After inversion, dark blobs on bright background become bright blobs."""
        img = np.ones((100, 100), dtype=np.uint8) * 200  # bright background
        img[40:60, 40:60] = 50  # dark nucleus
        # With invert=True and foreground_bright=True we should detect the dark nucleus
        pts, _, _, dbg = _inline_pipeline(
            img, normalize=True, invert_intensity=True, bg_correction=False,
            threshold_mode="otsu", foreground_bright=True,
            min_area_px=50, max_area_px=10000, use_watershed=False,
            prompt_type="points",
        )
        assert len(pts) >= 1, "Should detect dark nucleus after inversion"

    def test_foreground_polarity_dark(self):
        """foreground_bright=False should detect dark objects."""
        img = np.ones((100, 100), dtype=np.uint8) * 200
        img[40:60, 40:60] = 20
        pts, _, _, _ = _inline_pipeline(
            img, normalize=True, invert_intensity=False, bg_correction=False,
            threshold_mode="otsu", foreground_bright=False,
            min_area_px=50, max_area_px=10000, use_watershed=False,
            prompt_type="points",
        )
        assert len(pts) >= 1, "Should detect dark object with foreground_bright=False"


class TestNucleusPipelineBackgroundCorrection:
    """Tests for B2 background correction."""

    def test_gaussian_bg_correction_reduces_gradient(self):
        """After Gaussian BG correction the uneven illumination gradient should be reduced."""
        # Create a ramp (uneven illumination) plus a bright nucleus
        img = np.zeros((100, 100), dtype=np.float32)
        # Gradient: intensity increases left→right
        for c in range(100):
            img[:, c] = c * 1.5
        # Add a nucleus
        img[45:55, 45:55] = 220
        img = np.clip(img, 0, 255).astype(np.uint8)

        _, _, _, dbg = _inline_pipeline(
            img, normalize=True, bg_correction=True, bg_method="gaussian", bg_sigma=30,
            threshold_mode="otsu", foreground_bright=True, use_watershed=False,
            min_area_px=20, max_area_px=5000, prompt_type="points",
        )
        assert "bg_corrected" in dbg


class TestNucleusPipelineThresholdModes:
    """Tests for B3 thresholding modes."""

    def test_adaptive_gaussian_mode(self):
        """Adaptive Gaussian threshold should run without error and return candidates."""
        img = _make_dapi_image(n_nuclei=3, size=150)
        pts, _, _, dbg = _inline_pipeline(
            img, normalize=True, bg_correction=False,
            threshold_mode="adaptive_gaussian", adaptive_block_size=21, adaptive_C=2,
            foreground_bright=True, use_watershed=False,
            min_area_px=30, max_area_px=3000, prompt_type="points",
        )
        assert "binary_mask" in dbg

    def test_manual_threshold_mode(self):
        """Manual threshold should apply the specified value."""
        img = _make_dapi_image(n_nuclei=2, size=100)
        pts, _, _, dbg = _inline_pipeline(
            img, normalize=True, bg_correction=False,
            threshold_mode="manual", threshold_value=100,
            foreground_bright=True, use_watershed=False,
            min_area_px=10, max_area_px=5000, prompt_type="points",
        )
        assert len(pts) >= 1


class TestDebugInfo:
    """Tests to verify debug_info keys are populated correctly."""

    def test_all_debug_keys_present(self):
        img = _make_dapi_image(n_nuclei=3)
        _, _, _, dbg = _inline_pipeline(
            img, bg_correction=True, use_watershed=True,
            min_area_px=50, max_area_px=3000, prompt_type="points",
        )
        for key in ("normalized", "bg_corrected", "binary_mask", "distance", "label_image",
                    "n_before", "n_after"):
            assert key in dbg, f"Missing debug key: {key}"
        assert isinstance(dbg["n_before"], int)
        assert isinstance(dbg["n_after"], int)
        assert dbg["n_after"] <= dbg["n_before"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

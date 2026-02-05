"""
MedSAM Segmentation Module
Provides integration with MedSAM (Medical Segment Anything Model) for image segmentation
"""

import torch
import numpy as np
from typing import Optional, Tuple, List
import logging
from PIL import Image
import cv2

# Try importing from config, fallback to local implementation if import fails
try:
    from config import is_mps_available
except ImportError:
    def is_mps_available() -> bool:
        """Fallback implementation of MPS availability check"""
        return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()

logger = logging.getLogger(__name__)


class MedSAMPredictor:
    """Wrapper for MedSAM model inference"""
    
    def __init__(
        self, 
        model_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize MedSAM predictor
        
        Args:
            model_path: Path to MedSAM model checkpoint
            device: Device to run inference on (cuda/mps/cpu)
        """
        # Auto-detect best available device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif is_mps_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            # Validate device string
            valid_devices = ["cuda", "mps", "cpu"]
            if device not in valid_devices:
                logger.warning(f"Invalid device '{device}' specified. Valid devices: {valid_devices}. Falling back to CPU")
                self.device = "cpu"
            # Validate and fallback if requested device is not available
            elif device == "cuda" and not torch.cuda.is_available():
                # Check for MPS as fallback
                if is_mps_available():
                    logger.warning("CUDA requested but not available, falling back to MPS")
                    self.device = "mps"
                else:
                    logger.warning("CUDA requested but not available, falling back to CPU")
                    self.device = "cpu"
            elif device == "mps" and not is_mps_available():
                logger.warning("MPS requested but not available, falling back to CPU")
                self.device = "cpu"
            else:
                self.device = device
        self.model_path = model_path
        self.model = None
        
        logger.info(f"MedSAM Predictor initialized on device: {self.device}")
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """
        Load MedSAM model from checkpoint
        
        Args:
            model_path: Path to model checkpoint file
        """
        try:
            # Note: In production, this would load actual MedSAM model
            # For now, we'll use a placeholder
            logger.info(f"Loading MedSAM model from {model_path}")
            # self.model = torch.load(model_path, map_location=self.device)
            # self.model.eval()
            self.model_path = model_path
            logger.warning("Using mock MedSAM model - replace with actual model loading")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def preprocess_image(
        self, 
        image: np.ndarray,
        target_size: Tuple[int, int] = (1024, 1024)
    ) -> torch.Tensor:
        """
        Preprocess image for MedSAM inference
        
        Args:
            image: Input image as numpy array (H, W, C)
            target_size: Target size for model input
            
        Returns:
            Preprocessed image tensor
        """
        # Resize image
        if image.shape[:2] != target_size:
            image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
        
        # Normalize to [0, 1]
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0
        
        # Convert to tensor (C, H, W)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor.to(self.device)
    
    def predict(
        self,
        image: np.ndarray,
        box_prompts: Optional[List[List[int]]] = None,
        point_prompts: Optional[List[Tuple[int, int]]] = None
    ) -> np.ndarray:
        """
        Run MedSAM inference on image
        
        Args:
            image: Input image as numpy array (H, W, C)
            box_prompts: Optional list of bounding boxes [x1, y1, x2, y2]
            point_prompts: Optional list of point coordinates (x, y)
            
        Returns:
            Binary segmentation mask as numpy array (H, W)
        """
        if self.model is None:
            logger.warning("Model not loaded, using mock segmentation")
            return self._mock_segmentation(image)
        
        # Preprocess image
        image_tensor = self.preprocess_image(image)
        
        with torch.no_grad():
            # Run inference
            # mask = self.model(image_tensor, box_prompts, point_prompts)
            pass
        
        return self._mock_segmentation(image)
    
    def _mock_segmentation(self, image: np.ndarray) -> np.ndarray:
        """
        Generate mock segmentation for testing
        
        Args:
            image: Input image
            
        Returns:
            Mock binary segmentation mask
        """
        h, w = image.shape[:2]
        
        # Create a simple threshold-based segmentation as mock
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Simple Otsu thresholding
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask.astype(np.uint8)
    
    def predict_tiles(
        self,
        image: np.ndarray,
        tile_size: int = 1024,
        overlap: int = 128
    ) -> np.ndarray:
        """
        Run inference on large image using tiling strategy
        
        Args:
            image: Input image as numpy array (H, W, C)
            tile_size: Size of tiles to process
            overlap: Overlap between tiles to avoid boundary artifacts
            
        Returns:
            Full segmentation mask
        """
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        stride = tile_size - overlap
        
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                # Extract tile
                y_end = min(y + tile_size, h)
                x_end = min(x + tile_size, w)
                tile = image[y:y_end, x:x_end]
                
                # Predict on tile
                tile_mask = self.predict(tile)
                
                # Resize if needed
                if tile_mask.shape != (y_end - y, x_end - x):
                    tile_mask = cv2.resize(
                        tile_mask, 
                        (x_end - x, y_end - y),
                        interpolation=cv2.INTER_NEAREST
                    )
                
                # Merge into full mask
                mask[y:y_end, x:x_end] = np.maximum(mask[y:y_end, x:x_end], tile_mask)
        
        return mask


def segment_tissue(
    image: np.ndarray,
    predictor: Optional[MedSAMPredictor] = None,
    tile_size: int = 1024,
    overlap: int = 128
) -> np.ndarray:
    """
    Segment tissue regions in a pathology image
    
    Args:
        image: Input image as numpy array
        predictor: MedSAM predictor instance (creates new one if None)
        tile_size: Size of tiles for processing
        overlap: Overlap between tiles
        
    Returns:
        Binary segmentation mask
    """
    if predictor is None:
        predictor = MedSAMPredictor()
    
    return predictor.predict_tiles(image, tile_size, overlap)

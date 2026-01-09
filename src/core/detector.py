# src/core/detector.py

"""
Object detection module for sports analytics system.

Provides a clean YOLOv8 wrapper that converts raw frames into model-agnostic
detection dictionaries suitable for downstream tracking and analytics.

Notes:
    - Uses Ultralytics YOLOv8 predict() API for inference
    - Detector is stateless; no memory of previous frames
    - First inference may be slower due to model warm-up
    - COCO class IDs: 0=person, 32=sports ball, etc.
"""

from typing import List, Dict, Optional, Set, Any
import numpy as np
from ultralytics import YOLO


class YOLODetector:
    """
    YOLOv8-based object detector wrapper.
    
    Provides a stateless interface for detecting objects in individual frames
    and returning standardized detection dictionaries without exposing model
    implementation details.
    
    Usage:
        detector = YOLODetector("yolov8n.pt", conf_threshold=0.5, device="cuda")
        detections = detector.detect(frame)
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            print(f"{det['class_name']}: {det['confidence']:.2f}")
    
    Attributes:
        conf_threshold: Minimum confidence threshold for detections
        allowed_classes: Set of allowed class IDs (None allows all)
        device: Device for inference ('cpu', 'cuda', 'mps', etc.)
    """
    
    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.25,
        allowed_classes: Optional[Set[int]] = None,
        device: str = "cpu"
    ):
        """
        Initialize YOLO detector.
        
        Args:
            model_path: Path to YOLO model weights file
            conf_threshold: Minimum confidence threshold (0.0 to 1.0)
            allowed_classes: Set of allowed class IDs (e.g., {0, 32} for person
                           and sports ball). None allows all classes.
            device: Device for inference ('cpu', 'cuda', 'cuda:0', 'mps', etc.)
        
        Raises:
            RuntimeError: If model fails to load
        """
        self.conf_threshold = conf_threshold
        self.allowed_classes = allowed_classes
        self.device = device
        
        self._model = YOLO(model_path)
        self._model.to(device)
    
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects in a single frame.
        
        Args:
            frame: BGR numpy array with shape (H, W, 3) and dtype uint8
        
        Returns:
            List of detection dictionaries, each containing:
                - bbox: tuple of (x1, y1, x2, y2) in pixel coordinates as floats
                - confidence: float confidence score [0.0, 1.0]
                - class_id: int COCO class identifier
                - class_name: str human-readable class name
            
            Returns empty list if no detections meet the criteria.
        
        Raises:
            ValueError: If frame is not a valid numpy array or has wrong shape
        
        Notes:
            - Detections are filtered by conf_threshold and allowed_classes
            - All tensors are explicitly moved to CPU and converted to numpy
            - Detector maintains no state between calls
        """
        if not isinstance(frame, np.ndarray):
            raise ValueError("Frame must be a numpy array")
        
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Frame must have shape (H, W, 3), got {frame.shape}")
        
        results = self._model.predict(
            frame,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False
        )
        
        detections = []
        
        if len(results) == 0:
            return detections
        
        result = results[0]
        
        if result.boxes is None or len(result.boxes) == 0:
            return detections
        
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        
        for box, conf, class_id in zip(boxes, confidences, class_ids):
            if self.allowed_classes is not None and class_id not in self.allowed_classes:
                continue
            
            x1, y1, x2, y2 = box
            
            detection = {
                'bbox': (float(x1), float(y1), float(x2), float(y2)),
                'confidence': float(conf),
                'class_id': int(class_id),
                'class_name': result.names[class_id]
            }
            
            detections.append(detection)
        
        return detections
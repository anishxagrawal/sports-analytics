#src/core/video.py

"""
Video I/O module for sports analytics system.

Provides clean video reading capabilities with frame metadata for downstream
analytics processing. Handles both video files and camera streams.
"""

import cv2
from pathlib import Path
from typing import Iterator, Tuple, Dict, Union, Any


class VideoReader:
    """
    Video reader with frame-level metadata.
    
    Reads video frames from file or camera source and yields them with
    associated metadata including timestamp, fps, and frame dimensions.
    
    Usage:
        with VideoReader("video.mp4") as reader:
            for frame, metadata in reader:
                # Process frame
                pass
    
    Attributes:
        source: Video source (file path or camera index)
        fps: Frames per second
        width: Frame width in pixels
        height: Frame height in pixels
    """
    
    DEFAULT_FPS = 30.0
    
    def __init__(self, source: Union[str, int, Path]):
        """
        Initialize video reader.
        
        Args:
            source: Video file path (str/Path) or camera index (int)
            
        Raises:
            RuntimeError: If video source cannot be opened
            ValueError: If video has invalid dimensions
        """
        self.source = source
        self._cap = cv2.VideoCapture(str(source) if not isinstance(source, int) else source)
        
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {source}")
        
        raw_fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._fps = raw_fps if 0 < raw_fps <= 1000 else self.DEFAULT_FPS
        
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if self._width <= 0 or self._height <= 0:
            self.release()
            raise ValueError(f"Invalid video dimensions: {self._width}x{self._height}")
        
        self._frame_idx = 0
    
    @property
    def fps(self) -> float:
        """Get video frames per second."""
        return self._fps
    
    @property
    def width(self) -> int:
        """Get frame width in pixels."""
        return self._width
    
    @property
    def height(self) -> int:
        """Get frame height in pixels."""
        return self._height
    
    def __iter__(self) -> Iterator[Tuple[Any, Dict[str, Union[int, float]]]]:
        """
        Iterate over video frames with metadata.
        
        Yields:
            Tuple of (frame, metadata):
                frame: numpy.ndarray with shape (height, width, 3) in BGR format
                metadata: Dictionary containing:
                    - frame_idx (int): Zero-based frame index
                    - timestamp (float): Time in seconds
                    - fps (float): Frames per second
                    - width (int): Frame width in pixels
                    - height (int): Frame height in pixels
        """
        self._frame_idx = 0
        
        while True:
            ret, frame = self._cap.read()
            
            if not ret:
                break
            
            metadata = {
                'frame_idx': self._frame_idx,
                'timestamp': self._frame_idx / self._fps,
                'fps': self._fps,
                'width': self._width,
                'height': self._height
            }
            
            yield frame, metadata
            self._frame_idx += 1
    
    def release(self) -> None:
        """Release video capture resources."""
        if hasattr(self, '_cap') and self._cap is not None:
            self._cap.release()
    
    def __enter__(self):
        """Enter context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and release resources."""
        self.release()
        return False
    
    def __del__(self):
        """Destructor to ensure resources are released."""
        self.release()
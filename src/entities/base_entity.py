# src/entities/base_entity.py

"""
Base entity module for sports analytics system.

Provides a sport-agnostic entity class that represents real-world objects
tracked over time in video sequences. Manages position history, lifecycle
states, and track ID associations.
"""

import uuid
from typing import Tuple, List, Optional, Dict, Any
import numpy as np


class BaseEntity:
    """
    Base entity representing a tracked real-world object over time.
    
    Represents any tracked object (player, ball, referee, etc.) with a stable
    identity, position history, and lifecycle management. Sport-agnostic.
    
    Lifecycle: Active → Missing → Inactive
    
    Usage:
        entity = BaseEntity(track_id=1, class_id=0)
        entity.update(bbox=(100, 200, 150, 250), frame_index=0)
        entity.mark_missing()
        if entity.is_active():
            position = entity.get_position()
    """
    
    DEFAULT_MAX_HISTORY = 100
    DEFAULT_MISSING_THRESHOLD = 30
    
    def __init__(
        self,
        track_id: int,
        class_id: int,
        max_history: int = DEFAULT_MAX_HISTORY,
        missing_threshold: int = DEFAULT_MISSING_THRESHOLD
    ):
        """
        Initialize a new entity.
        
        Args:
            track_id: Initial tracker ID
            class_id: Object class identifier from detector
            max_history: Maximum positions to store in history
            missing_threshold: Frames missing before marking inactive
        """
        self.entity_id = str(uuid.uuid4())
        self.track_id = track_id
        self.class_id = class_id
        self.max_history = max_history
        self.missing_threshold = missing_threshold
        
        self.is_active_flag = True
        self.missing_frames = 0
        self.last_seen_frame: Optional[int] = None
        self.last_seen_timestamp: Optional[float] = None
        
        self._position_history: List[Tuple[float, float]] = []
        self._frame_history: List[int] = []
        self._timestamp_history: List[Optional[float]] = []
    
    def update(
        self,
        bbox: Tuple[float, float, float, float],
        frame_index: int,
        timestamp: Optional[float] = None
    ) -> None:
        """
        Update entity with new detection.
        
        Args:
            bbox: Bounding box as (x1, y1, x2, y2)
            frame_index: Current frame index
            timestamp: Optional timestamp in seconds
        """
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        
        self._position_history.append((center_x, center_y))
        self._frame_history.append(frame_index)
        self._timestamp_history.append(timestamp)
        
        if len(self._position_history) > self.max_history:
            self._position_history.pop(0)
            self._frame_history.pop(0)
            self._timestamp_history.pop(0)
        
        self.last_seen_frame = frame_index
        self.last_seen_timestamp = timestamp
        self.missing_frames = 0
        self.is_active_flag = True
    
    def mark_missing(self) -> None:
        """
        Mark entity as missing for current frame.
        
        Increments missing counter. If threshold exceeded, marks inactive.
        """
        self.missing_frames += 1
        
        if self.missing_frames >= self.missing_threshold:
            self.is_active_flag = False
    
    def set_track_id(self, new_track_id: int) -> None:
        """
        Update entity's tracker ID.
        
        Args:
            new_track_id: New tracker ID to assign
        """
        self.track_id = new_track_id
    
    def is_active(self) -> bool:
        """Check if entity is currently active."""
        return self.is_active_flag
    
    def get_position(self) -> Optional[Tuple[float, float]]:
        """
        Get most recent position.
        
        Returns:
            Tuple of (x, y) center coordinates, or None if no history
        """
        if len(self._position_history) == 0:
            return None
        return self._position_history[-1]
    
    def get_trajectory(self, n: Optional[int] = None) -> np.ndarray:
        """
        Get position trajectory as numpy array.
        
        Args:
            n: Optional number of most recent positions. None = all.
        
        Returns:
            Numpy array of shape (N, 2) with (x, y) positions.
        """
        if len(self._position_history) == 0:
            return np.empty((0, 2), dtype=np.float32)
        
        if n is not None and n > 0:
            positions = self._position_history[-n:]
        else:
            positions = self._position_history
        
        return np.array(positions, dtype=np.float32)
    
    def get_frame_history(self) -> List[int]:
        """Get list of frame indices corresponding to position history."""
        return self._frame_history.copy()
    
    def get_timestamp_history(self) -> List[Optional[float]]:
        """Get list of timestamps corresponding to position history."""
        return self._timestamp_history.copy()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert entity to JSON-serializable dictionary.
        
        Returns:
            Dictionary with entity_id, track_id, class_id, position,
            active status, and trajectory length.
        """
        position = self.get_position()
        
        return {
            'entity_id': self.entity_id,
            'track_id': self.track_id,
            'class_id': self.class_id,
            'position': position,
            'active': self.is_active_flag,
            'missing_frames': self.missing_frames,
            'last_seen_frame': self.last_seen_frame,
            'last_seen_timestamp': self.last_seen_timestamp,
            'trajectory_length': len(self._position_history)
        }
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"<BaseEntity id={self.entity_id[:8]}... "
            f"track_id={self.track_id} "
            f"active={self.is_active_flag}>"
        )
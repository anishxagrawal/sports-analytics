# src/entities/player.py

"""
Player entity module for sports analytics system.

Represents human participants in sports videos (players, athletes).
Extends BaseEntity with player-specific semantics.
"""

from typing import Dict, Any, Optional, Tuple
from collections import deque
from .base_entity import BaseEntity
from ..spatial.ground_point import bbox_to_ground_point


class Player(BaseEntity):
    """
    Player entity representing a human participant in sports video.
    
    Thin wrapper around BaseEntity that adds player-specific semantics.
    Inherits all tracking, history, and lifecycle management from BaseEntity.
    
    Usage:
        player = Player(track_id=1, class_id=0)
        player.update(bbox=(100, 200, 150, 250), frame_index=0)
        player.record_ground_position(bbox=(100, 200, 150, 250))
        if player.is_active():
            position = player.get_position()
    """
    
    entity_type = "player"
    
    def __init__(
        self,
        track_id: int,
        class_id: int,
        max_history: int = BaseEntity.DEFAULT_MAX_HISTORY,
        missing_threshold: int = BaseEntity.DEFAULT_MISSING_THRESHOLD
    ):
        """
        Initialize a new player entity.
        
        Args:
            track_id: Initial tracker ID
            class_id: Object class identifier from detector
            max_history: Maximum positions to store in history
            missing_threshold: Frames missing before marking inactive
        """
        super().__init__(
            track_id=track_id,
            class_id=class_id,
            max_history=max_history,
            missing_threshold=missing_threshold
        )
        
        # Ground contact point history (bottom-center of bbox)
        self.ground_positions: deque = deque(maxlen=max_history)
    
    def record_ground_position(self, bbox: Optional[Tuple[float, float, float, float]]) -> None:
        """
        Record ground contact point from bounding box.
        
        Should be called after update() to track ground position separately.
        
        Args:
            bbox: Bounding box (x1, y1, x2, y2) or None
        """
        if bbox is None:
            return
        
        ground_point = bbox_to_ground_point(bbox)
        self.ground_positions.append(ground_point)
    
    def get_ground_position(self) -> Optional[Tuple[float, float]]:
        """
        Get the most recent ground contact point.
        
        Returns:
            (x, y) ground position or None if no history
        """
        return self.ground_positions[-1] if self.ground_positions else None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert player to JSON-serializable dictionary.
        
        Returns:
            Dictionary with all BaseEntity fields plus entity_type.
        """
        data = super().to_dict()
        data['entity_type'] = self.entity_type
        data['ground_positions'] = list(self.ground_positions)
        return data
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"<Player id={self.entity_id[:8]}... "
            f"track_id={self.track_id} "
            f"active={self.is_active_flag}>"
        )
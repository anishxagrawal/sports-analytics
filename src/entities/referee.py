# src/entities/referee.py

"""
Referee entity module for sports analytics system.

Represents non-team human participants in sports videos (referees, officials, umpires).
Extends BaseEntity with referee-specific semantics.
"""

from typing import Dict, Any
from entities.base_entity import BaseEntity


class Referee(BaseEntity):
    """
    Referee entity representing a non-team human participant in sports video.
    
    Thin wrapper around BaseEntity that adds referee-specific semantics.
    Inherits all tracking, history, and lifecycle management from BaseEntity.
    Unlike players, referees have no team assignment or ground position tracking.
    
    Usage:
        referee = Referee(track_id=1, class_id=2)
        referee.update(bbox=(100, 200, 150, 250), frame_index=0)
        if referee.is_active():
            position = referee.get_position()
    """
    
    entity_type = "referee"
    
    def __init__(
        self,
        track_id: int,
        class_id: int,
        max_history: int = BaseEntity.DEFAULT_MAX_HISTORY,
        missing_threshold: int = BaseEntity.DEFAULT_MISSING_THRESHOLD
    ):
        """
        Initialize a new referee entity.
        
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
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert referee to JSON-serializable dictionary.
        
        Returns:
            Dictionary with all BaseEntity fields plus entity_type.
        """
        data = super().to_dict()
        data['entity_type'] = self.entity_type
        return data
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"<Referee id={self.entity_id[:8]}... "
            f"track_id={self.track_id} "
            f"active={self.is_active_flag}>"
        )
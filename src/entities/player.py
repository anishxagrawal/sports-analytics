# src/entities/player.py

"""
Player entity module for sports analytics system.

Represents human participants in sports videos (players, athletes).
Extends BaseEntity with player-specific semantics.
"""

from typing import Dict, Any, Optional, Tuple, Literal
from collections import deque, Counter
from entities.base_entity import BaseEntity
from spatial.ground_point import bbox_to_ground_point


TeamLabel = Literal["A", "B"]


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
    
    # Team assignment thresholds
    MIN_VOTES_TO_COMMIT = 5      # Minimum votes required to assign team
    MIN_VOTE_MARGIN = 3          # Minimum margin between winning and losing team
    SWITCH_VOTE_THRESHOLD = 15   # Votes required to switch teams (much higher)
    
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
        
        # Team assignment state (stable commit with evidence threshold)
        self.team_votes: deque = deque(maxlen=30)  # Extended window for stability
        self.team_id: Optional[TeamLabel] = None
        self._team_committed = False  # Track if team has been committed
    
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
    
    def record_team_vote(self, team_label: TeamLabel) -> None:
        """
        Record a team vote and commit team_id when evidence threshold is met.
        
        Team assignment logic:
        - Initially team_id = None (uncommitted)
        - Commits when: votes >= MIN_VOTES_TO_COMMIT AND margin >= MIN_VOTE_MARGIN
        - Once committed, team_id is stable (switching requires much higher threshold)
        
        Args:
            team_label: Team label ("A" or "B")
        """
        # Store the vote
        self.team_votes.append(team_label)
        
        # Count votes for each team
        if len(self.team_votes) == 0:
            return
        
        vote_counts = Counter(self.team_votes)
        team_a_votes = vote_counts.get("A", 0)
        team_b_votes = vote_counts.get("B", 0)
        
        # Determine leading team and margin
        if team_a_votes > team_b_votes:
            leading_team = "A"
            leading_votes = team_a_votes
            margin = team_a_votes - team_b_votes
        else:
            leading_team = "B"
            leading_votes = team_b_votes
            margin = team_b_votes - team_a_votes
        
        if not self._team_committed:
            # Initial commit: require minimum votes and margin
            if leading_votes >= self.MIN_VOTES_TO_COMMIT and margin >= self.MIN_VOTE_MARGIN:
                self.team_id = leading_team
                self._team_committed = True
        else:
            # Team already committed: only switch with much stronger evidence
            if leading_team != self.team_id:
                # Require higher threshold to switch teams (prevents flipping)
                if leading_votes >= self.SWITCH_VOTE_THRESHOLD and margin >= self.MIN_VOTE_MARGIN * 2:
                    self.team_id = leading_team
    
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
        data['team_id'] = self.team_id
        data['team_votes'] = list(self.team_votes)
        data['team_committed'] = self._team_committed
        return data
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"<Player id={self.entity_id[:8]}... "
            f"track_id={self.track_id} "
            f"team={self.team_id} "
            f"active={self.is_active_flag}>"
        )
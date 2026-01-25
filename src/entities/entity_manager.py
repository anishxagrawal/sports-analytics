# src/entities/entity_manager.py

"""
Entity manager module for sports analytics system.

Manages Player and Referee entities across frames by associating tracker outputs with
persistent entity objects. Handles entity lifecycle and track ID mapping.
"""

from typing import List, Dict, Any, Optional
from .player import Player
from .referee import Referee
from .ball import Ball


class EntityManager:
    """
    Manages Player and Referee entities across video frames.
    
    Connects tracker outputs to persistent entity objects, handling creation,
    updates, and lifecycle management. Maintains stable identities across frames.
    
    Usage:
        manager = EntityManager()
        for frame_tracks in track_stream:
            manager.update(frame_tracks, frame_index=i)
            active_players = manager.get_active_players()
            active_referees = manager.get_active_referees()
    """
    
    # Conservative deletion threshold: must exceed ByteTrack's buffer
    # ByteTrack typically buffers 30 frames, we use 90 (3x safety margin)
    MAX_FRAMES_MISSING = 90
    
    def __init__(self):
        """Initialize entity manager with empty storage."""
        self._players: Dict[int, Player] = {}
        self._last_seen_players: Dict[int, int] = {}
        self._referees: Dict[int, Referee] = {}
        self._last_seen_referees: Dict[int, int] = {}
        self.ball = Ball()
    
    def update(
        self,
        tracks: List[Dict[str, Any]],
        frame_index: int,
        timestamp: Optional[float] = None
    ) -> None:
        """
        Update entity manager with tracker outputs for one frame.
        
        Args:
            tracks: List of track dictionaries with keys:
                - track_id: int tracker identifier
                - bbox: tuple (x1, y1, x2, y2)
                - confidence: float
                - class_id: int (0=player, 2=referee)
            frame_index: Current frame index
            timestamp: Optional timestamp in seconds
        
        Notes:
            - Creates new entities for unseen track_ids
            - Updates existing entities when track_id is seen
            - Marks unseen entities as missing
            - Removes entities missing for > MAX_FRAMES_MISSING frames
        """
        seen_player_track_ids = set()
        seen_referee_track_ids = set()
        
        for track in tracks:
            track_id = track['track_id']
            bbox = track['bbox']
            class_id = track['class_id']
            
            # Process Player class (class_id == 0)
            if class_id == 0:
                seen_player_track_ids.add(track_id)
                
                if track_id not in self._players:
                    self._players[track_id] = Player(
                        track_id=track_id,
                        class_id=class_id
                    )
                
                self._players[track_id].update(
                    bbox=bbox,
                    frame_index=frame_index,
                    timestamp=timestamp
                )
                
                self._last_seen_players[track_id] = frame_index
            
            # Process Referee class (class_id == 2)
            elif class_id == 2:
                seen_referee_track_ids.add(track_id)
                
                if track_id not in self._referees:
                    self._referees[track_id] = Referee(
                        track_id=track_id,
                        class_id=class_id
                    )
                
                self._referees[track_id].update(
                    bbox=bbox,
                    frame_index=frame_index,
                    timestamp=timestamp
                )
                
                self._last_seen_referees[track_id] = frame_index
        
        # Mark unseen players as missing
        for track_id, player in self._players.items():
            if track_id not in seen_player_track_ids:
                player.mark_missing()
        
        # Mark unseen referees as missing
        for track_id, referee in self._referees.items():
            if track_id not in seen_referee_track_ids:
                referee.mark_missing()
        
        # Remove players that have been missing for extended period
        # This prevents stale entities when track_ids are recycled
        # Only delete after significantly longer than tracker buffer
        permanently_lost_player_track_ids = [
            track_id for track_id in self._players.keys()
            if track_id in self._last_seen_players and 
               frame_index - self._last_seen_players[track_id] > self.MAX_FRAMES_MISSING
        ]
        
        for track_id in permanently_lost_player_track_ids:
            del self._players[track_id]
            del self._last_seen_players[track_id]
        
        # Remove referees that have been missing for extended period
        permanently_lost_referee_track_ids = [
            track_id for track_id in self._referees.keys()
            if track_id in self._last_seen_referees and 
               frame_index - self._last_seen_referees[track_id] > self.MAX_FRAMES_MISSING
        ]
        
        for track_id in permanently_lost_referee_track_ids:
            del self._referees[track_id]
            del self._last_seen_referees[track_id]
    
    def update_ball(self, ball_state: Dict[str, Any], frame_index: int) -> None:
        """
        Update ball entity with tracking state.
        
        Args:
            ball_state: Ball state dictionary with keys:
                - position: tuple (x, y) or None
                - velocity: tuple (vx, vy) or None
                - confidence: float or None
                - visible: bool
            frame_index: Current frame index
        """
        if ball_state["position"] is not None:
            self.ball.update(
                position=ball_state["position"],
                frame_index=frame_index,
                confidence=ball_state["confidence"],
                velocity=ball_state["velocity"]
            )
        else:
            self.ball.mark_not_visible()
    
    def get_active_players(self) -> List[Player]:
        """
        Get list of currently active players.
        
        Returns:
            List of Player objects with is_active() == True
        """
        return [
            player for player in self._players.values()
            if player.is_active()
        ]
    
    def get_all_players(self) -> List[Player]:
        """
        Get list of all players (active and inactive).
        
        Returns:
            List of all Player objects
        """
        return list(self._players.values())
    
    def get_active_referees(self) -> List[Referee]:
        """
        Get list of currently active referees.
        
        Returns:
            List of Referee objects with is_active() == True
        """
        return [
            referee for referee in self._referees.values()
            if referee.is_active()
        ]
    
    def get_all_referees(self) -> List[Referee]:
        """
        Get list of all referees (active and inactive).
        
        Returns:
            List of all Referee objects
        """
        return list(self._referees.values())
    
    def get_player_by_track_id(self, track_id: int) -> Optional[Player]:
        """
        Get player by tracker ID.
        
        Args:
            track_id: Tracker identifier
        
        Returns:
            Player object or None if not found
        """
        return self._players.get(track_id)
    
    def to_dict(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Convert all entities to JSON-serializable dictionary.
        
        Returns:
            Dictionary with keys 'players' and 'referees', each containing
            a list of entity dictionaries from their respective to_dict() methods
        """
        return {
            'players': [player.to_dict() for player in self._players.values()],
            'referees': [referee.to_dict() for referee in self._referees.values()]
        }
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        active_players = len(self.get_active_players())
        total_players = len(self._players)
        active_referees = len(self.get_active_referees())
        total_referees = len(self._referees)
        return (
            f"<EntityManager "
            f"players={total_players} active={active_players} "
            f"referees={total_referees} active={active_referees}>"
        )
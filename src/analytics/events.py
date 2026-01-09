# src/analytics/events.py

"""
Event detection module for sports analytics system.

Provides stateless functions for detecting high-level player events from
motion metrics. All functions are pure and do not modify entities.
"""

from typing import List, Dict, Any, Optional
from .motion import compute_speed


# Event detection thresholds (pixels per second)
SPRINT_SPEED_THRESHOLD = 150.0
STOP_SPEED_THRESHOLD = 10.0


def detect_player_events(player, fps: float) -> List[Dict[str, Any]]:
    """
    Detect motion-based events for a single player.
    
    Analyzes player movement and generates event dictionaries for
    significant motion states (sprinting, stopping).
    
    Args:
        player: Player entity with get_trajectory() method
        fps: Video frame rate (frames per second)
    
    Returns:
        List of event dictionaries, each containing:
            - event_type: str ("sprint" or "stop")
            - entity_id: str (player's unique ID)
            - track_id: int (player's tracker ID)
            - timestamp: Optional[float] (last seen timestamp)
            - speed: float (current speed in pixels/second)
        
        Returns empty list if no events detected or insufficient history.
    
    Notes:
        - Sprint event: speed > SPRINT_SPEED_THRESHOLD
        - Stop event: speed < STOP_SPEED_THRESHOLD
        - Requires at least 2 positions in player trajectory
    """
    events = []
    
    if fps <= 0:
        return events
    
    trajectory = player.get_trajectory(n=2)
    if len(trajectory) < 2:
        return events
    
    speed = compute_speed(player, fps)
    
    event_type = None
    
    if speed > SPRINT_SPEED_THRESHOLD:
        event_type = "sprint"
    elif speed < STOP_SPEED_THRESHOLD:
        event_type = "stop"
    
    if event_type is not None:
        event = {
            "event_type": event_type,
            "entity_id": player.entity_id,
            "track_id": player.track_id,
            "timestamp": player.last_seen_timestamp,
            "speed": speed
        }
        events.append(event)
    
    return events
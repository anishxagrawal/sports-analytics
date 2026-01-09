# src/commentary/memory.py

"""
Commentary memory module for sports analytics system.

Manages per-entity commentary state to prevent repetitive narration and
ensure meaningful event commentary timing.
"""

from typing import Dict, Any


class CommentaryMemory:
    """
    Tracks commentary state per entity to control narration timing.
    
    Decides whether an event should be narrated based on:
    - Entity narration history
    - Event type changes
    - Cooldown periods
    
    Usage:
        memory = CommentaryMemory(cooldown_seconds=5.0)
        if memory.should_speak(event, timestamp):
            # Generate and output commentary
            memory.update(event, timestamp)
    """
    
    def __init__(self, cooldown_seconds: float = 5.0):
        """
        Initialize commentary memory.
        
        Args:
            cooldown_seconds: Minimum time between narrations of same event type
        """
        self.cooldown_seconds = cooldown_seconds
        self._entity_states: Dict[str, Dict[str, Any]] = {}
    
    def should_speak(self, event: Dict[str, Any], timestamp: float) -> bool:
        """
        Determine if event should be narrated.
        
        Args:
            event: Event dictionary with at least:
                - event_type: str
                - entity_id: str
            timestamp: Current timestamp in seconds (expected monotonically increasing)
        
        Returns:
            True if event should be narrated, False otherwise
        
        Decision rules:
            - First narration for entity → True
            - Different event_type than last narration → True
            - Same event_type but cooldown elapsed → True
            - Otherwise → False
        """
        entity_id = event.get("entity_id")
        event_type = event.get("event_type")
        
        if entity_id is None or event_type is None:
            return False
        
        if entity_id not in self._entity_states:
            return True
        
        state = self._entity_states[entity_id]
        last_event_type = state.get("last_event_type")
        last_spoken_time = state.get("last_spoken_time")
        
        if event_type != last_event_type:
            return True
        
        if last_spoken_time is None:
            return True
        
        time_elapsed = timestamp - last_spoken_time
        
        if time_elapsed >= self.cooldown_seconds:
            return True
        
        return False
    
    def update(self, event: Dict[str, Any], timestamp: float) -> None:
        """
        Update commentary state after narration.
        
        Args:
            event: Event dictionary with at least:
                - event_type: str
                - entity_id: str
            timestamp: Current timestamp in seconds (expected monotonically increasing)
        
        Notes:
            Should be called only after event has been narrated.
        """
        entity_id = event.get("entity_id")
        event_type = event.get("event_type")
        
        if entity_id is None or event_type is None:
            return
        
        self._entity_states[entity_id] = {
            "last_event_type": event_type,
            "last_spoken_time": timestamp
        }
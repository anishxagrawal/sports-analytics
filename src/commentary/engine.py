# src/commentary/engine.py

"""
Commentary engine module for sports analytics system.

Converts semantic events into narrative intents using commentary memory
to control narration timing and prevent repetition.
"""

from typing import List, Dict, Any, Optional
from .memory import CommentaryMemory


# Event type to narrative intent mapping
EVENT_TO_INTENT = {
    "sprint": "start_run",
    "stop": "slow_down"
}


class CommentaryEngine:
    """
    Processes events and generates narrative intents.
    
    Uses CommentaryMemory to filter events and converts approved events
    into structured narrative intents for downstream narration.
    
    Usage:
        engine = CommentaryEngine(cooldown_seconds=5.0)
        intents = engine.process_events(events, timestamp)
        if intents:
            # Pass intents to narration layer
            pass
    """
    
    def __init__(self, cooldown_seconds: float = 5.0):
        """
        Initialize commentary engine.
        
        Args:
            cooldown_seconds: Minimum time between narrations of same event type
        """
        self.memory = CommentaryMemory(cooldown_seconds=cooldown_seconds)
    
    def process_events(
        self,
        events: List[Dict[str, Any]],
        timestamp: float
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Process events and generate narrative intents.
        
        Args:
            events: List of event dictionaries with at least:
                - event_type: str
                - entity_id: str
            timestamp: Current timestamp in seconds
        
        Returns:
            List of intent dictionaries if any events approved for narration:
                - entity_id: str
                - intent: str
            Returns None if no events approved.
        
        Notes:
            - Only processes events with known intent mappings
            - Uses CommentaryMemory to filter repetitive events
            - Updates memory state for approved events
        """
        intents = []
        
        for event in events:
            event_type = event.get("event_type")
            entity_id = event.get("entity_id")
            
            if event_type is None or entity_id is None:
                continue
            
            if event_type not in EVENT_TO_INTENT:
                continue
            
            if not self.memory.should_speak(event, timestamp):
                continue
            
            intent = EVENT_TO_INTENT[event_type]
            
            intents.append({
                "entity_id": entity_id,
                "intent": intent
            })
            
            self.memory.update(event, timestamp)
        
        if len(intents) == 0:
            return None
        
        return intents
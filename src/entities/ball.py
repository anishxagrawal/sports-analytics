# src/entities/ball.py

from dataclasses import dataclass, field
from typing import Optional, Tuple, List


@dataclass
class Ball:
    """
    Represents a football in a sports analytics system.
    
    This is a pure data/state object that stores the ball's position,
    velocity, visibility, confidence, and trajectory history.
    """
    
    # Current (x, y) position of the ball, None if unknown
    position: Optional[Tuple[float, float]] = None
    
    # Current (vx, vy) velocity of the ball, None if unknown
    velocity: Optional[Tuple[float, float]] = None
    
    # Whether the ball is currently visible
    visible: bool = False
    
    # Confidence score for the last detection (0.0 to 1.0)
    confidence: Optional[float] = None
    
    # Frame index where the ball was last seen (last frame where ball was visible)
    last_seen_frame: Optional[int] = None
    
    # List of recent (x, y) positions forming the ball's trajectory
    trajectory: List[Tuple[float, float]] = field(default_factory=list)
    
    # Maximum number of positions to keep in trajectory history
    max_trajectory_length: int = 30
    
    def update(
        self,
        position: Optional[Tuple[float, float]],
        frame_index: int,
        confidence: Optional[float] = None,
        velocity: Optional[Tuple[float, float]] = None
    ) -> None:
        """
        Update the ball's state with new detection data.
        
        Args:
            position: New (x, y) position or None if not detected
            frame_index: Current frame index
            confidence: Confidence score for this detection
            velocity: Optional (vx, vy) velocity
        """
        if position is not None:
            # Ball was detected - update all state
            self.position = position
            self.visible = True
            self.last_seen_frame = frame_index
            self.confidence = confidence
            
            if velocity is not None:
                self.velocity = velocity
            
            # Add to trajectory history
            self._add_to_trajectory(position)
        else:
            # Ball was not detected - mark as not visible
            self.mark_not_visible()
    
    def mark_not_visible(self) -> None:
        """Mark the ball as not currently visible."""
        self.visible = False
        self.confidence = None
        self.velocity = None
    
    def reset(self) -> None:
        """Reset the ball state to initial values."""
        self.position = None
        self.velocity = None
        self.visible = False
        self.confidence = None
        self.last_seen_frame = None
        self.trajectory.clear()
    
    def _add_to_trajectory(self, position: Tuple[float, float]) -> None:
        """
        Add a position to the trajectory history.
        
        Maintains a fixed maximum length by removing oldest positions.
        
        Args:
            position: (x, y) position to add
        """
        self.trajectory.append(position)
        
        # Remove oldest position if we exceed max length
        if len(self.trajectory) > self.max_trajectory_length:
            self.trajectory.pop(0)
    
    def get_trajectory(self) -> List[Tuple[float, float]]:
        """
        Get the current trajectory history.
        
        Returns:
            List of (x, y) positions
        """
        return self.trajectory.copy()
    
    def is_visible(self) -> bool:
        """
        Check if the ball is currently visible.
        
        Returns:
            True if visible, False otherwise
        """
        return self.visible
    
    def get_position(self) -> Optional[Tuple[float, float]]:
        """
        Get the current position.
        
        Returns:
            (x, y) position or None if unknown
        """
        return self.position
    
    def get_velocity(self) -> Optional[Tuple[float, float]]:
        """
        Get the current velocity.
        
        Returns:
            (vx, vy) velocity or None if unknown
        """
        return self.velocity
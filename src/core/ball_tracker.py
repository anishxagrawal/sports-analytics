# src/core/ball_tracker.py

from typing import Optional, Tuple, Dict


class BallTracker:
    """
    Stabilizes football tracking over time using prediction and smoothing.
    
    Takes raw ball detections per frame and produces a clean, stable ball state.
    """
    
    def __init__(
        self,
        smoothing_factor: float = 0.3,
        max_prediction_frames: int = 10,
        min_confidence: float = 0.3
    ):
        """
        Initialize the ball tracker.
        
        Args:
            smoothing_factor: Weight for new observations (0-1), higher = less smoothing
            max_prediction_frames: Maximum frames to predict without detection
            min_confidence: Minimum confidence to accept a detection
        """
        # Smoothing and prediction parameters
        self.smoothing_factor = smoothing_factor
        self.max_prediction_frames = max_prediction_frames
        self.min_confidence = min_confidence
        
        # Internal state
        self.position: Optional[Tuple[float, float]] = None
        self.velocity: Optional[Tuple[float, float]] = None
        self.last_detection_frame: Optional[int] = None
        
        # Track consecutive frames without detection
        self.frames_without_detection: int = 0
    
    def update(
        self,
        position: Optional[Tuple[float, float]],
        frame_index: int,
        confidence: Optional[float] = None
    ) -> Dict:
        """
        Update tracker with new detection and return current ball state.
        
        Args:
            position: Detected (x, y) position or None if not detected
            frame_index: Current frame number
            confidence: Detection confidence score
            
        Returns:
            Dictionary with keys: position, velocity, confidence, visible
        """
        # Check if detection is valid
        detection_valid = (
            position is not None and
            (confidence is None or confidence >= self.min_confidence)
        )
        
        if detection_valid:
            # Valid detection received
            self._process_detection(position, confidence, frame_index)
            self.frames_without_detection = 0
        else:
            # No detection - predict or mark invisible
            self.frames_without_detection += 1
            self._process_no_detection()
        
        # Build output state
        return self._build_output_state()
    
    def _process_detection(
        self,
        detected_position: Tuple[float, float],
        confidence: Optional[float],
        frame_index: int
    ) -> None:
        """Process a valid ball detection."""
        if self.position is None:
            # First detection - initialize state
            self.position = detected_position
            self.velocity = (0.0, 0.0)
        else:
            # Compute velocity using previous position (before smoothing)
            if self.last_detection_frame is not None:
                frame_delta = frame_index - self.last_detection_frame
                if frame_delta > 0:
                    vx = (detected_position[0] - self.position[0]) / frame_delta
                    vy = (detected_position[1] - self.position[1]) / frame_delta
                    self.velocity = (vx, vy)
            
            # Smooth the position using exponential moving average
            smoothed_x = (
                self.smoothing_factor * detected_position[0] +
                (1 - self.smoothing_factor) * self.position[0]
            )
            smoothed_y = (
                self.smoothing_factor * detected_position[1] +
                (1 - self.smoothing_factor) * self.position[1]
            )
            
            self.position = (smoothed_x, smoothed_y)
        
        self.last_detection_frame = frame_index
    
    def _process_no_detection(self) -> None:
        """Handle case when no detection is available."""
        if self.frames_without_detection >= self.max_prediction_frames:
            # Too many frames without detection - reset state
            self.position = None
            self.velocity = None
        elif self.position is not None and self.velocity is not None:
            # Predict next position using current velocity
            predicted_x = self.position[0] + self.velocity[0]
            predicted_y = self.position[1] + self.velocity[1]
            self.position = (predicted_x, predicted_y)
            
            # Gradually reduce velocity confidence during prediction
            decay_factor = 0.95
            self.velocity = (
                self.velocity[0] * decay_factor,
                self.velocity[1] * decay_factor
            )
    
    def _build_output_state(self) -> Dict:
        """Build the output state dictionary."""
        visible = (
            self.position is not None and
            self.frames_without_detection < self.max_prediction_frames
        )
        
        # Calculate confidence based on recency of detection
        if self.frames_without_detection == 0:
            confidence = 1.0
        elif self.frames_without_detection < self.max_prediction_frames:
            # Decay confidence during prediction
            confidence = 1.0 - (self.frames_without_detection / self.max_prediction_frames)
        else:
            confidence = None
        
        return {
            'position': self.position,
            'velocity': self.velocity,
            'confidence': confidence,
            'visible': visible
        }
    
    def reset(self) -> None:
        """Reset tracker state."""
        self.position = None
        self.velocity = None
        self.last_detection_frame = None
        self.frames_without_detection = 0
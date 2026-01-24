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
        min_confidence: float = 0.12  # lower threshold for airborne/background-heavy cases
    ):
        """
        Initialize the ball tracker.

        Args:
            smoothing_factor: Weight for new observations (0-1), higher = less smoothing
            max_prediction_frames: Maximum frames to predict without detection
            min_confidence: Minimum confidence to accept a detection
        """
        self.smoothing_factor = smoothing_factor
        self.max_prediction_frames = max_prediction_frames
        self.min_confidence = min_confidence

        # Internal state
        self.position: Optional[Tuple[float, float]] = None
        self.velocity: Optional[Tuple[float, float]] = None
        self.last_detection_frame: Optional[int] = None

        # Store last raw detection (for velocity estimation)
        self._last_raw_position: Optional[Tuple[float, float]] = None

        self.frames_without_detection: int = 0

    def update(
        self,
        position: Optional[Tuple[float, float]],
        frame_index: int,
        confidence: Optional[float] = None
    ) -> Dict:
        """
        Update tracker with new detection and return current ball state.
        """
        detection_valid = (
            position is not None and
            (confidence is None or confidence >= self.min_confidence)
        )

        if detection_valid:
            self._process_detection(position, frame_index)
            self.frames_without_detection = 0
        else:
            self.frames_without_detection += 1
            self._process_no_detection()

        return self._build_output_state()

    def _process_detection(
        self,
        detected_position: Tuple[float, float],
        frame_index: int
    ) -> None:
        """Process a valid ball detection."""
        if self.position is None:
            # First detection
            self.position = detected_position
            self.velocity = (0.0, 0.0)
        else:
            # Compute velocity from raw detections (not smoothed state)
            if self._last_raw_position is not None and self.last_detection_frame is not None:
                frame_delta = frame_index - self.last_detection_frame
                if frame_delta > 0:
                    vx = (detected_position[0] - self._last_raw_position[0]) / frame_delta
                    vy = (detected_position[1] - self._last_raw_position[1]) / frame_delta
                    self.velocity = (vx, vy)

            # Smooth position (EMA)
            smoothed_x = (
                self.smoothing_factor * detected_position[0] +
                (1 - self.smoothing_factor) * self.position[0]
            )
            smoothed_y = (
                self.smoothing_factor * detected_position[1] +
                (1 - self.smoothing_factor) * self.position[1]
            )
            self.position = (smoothed_x, smoothed_y)

        self._last_raw_position = detected_position
        self.last_detection_frame = frame_index

    def _process_no_detection(self) -> None:
        """Handle case when no detection is available."""
        if self.frames_without_detection >= self.max_prediction_frames:
            self.position = None
            self.velocity = None
            self._last_raw_position = None
        elif self.position is not None and self.velocity is not None:
            # Predict forward
            self.position = (
                self.position[0] + self.velocity[0],
                self.position[1] + self.velocity[1]
            )

            # Decay velocity over time
            decay = 0.95
            self.velocity = (self.velocity[0] * decay, self.velocity[1] * decay)

    def _build_output_state(self) -> Dict:
        visible = (
            self.position is not None and
            self.frames_without_detection < self.max_prediction_frames
        )

        if self.frames_without_detection == 0:
            confidence = 1.0
        elif self.frames_without_detection < self.max_prediction_frames:
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
        self._last_raw_position = None
        self.frames_without_detection = 0
# src/core/ball_tracker.py

from typing import Optional, Tuple, Dict
import numpy as np
import cv2


class BallTracker:
    """
    Stabilizes football tracking over time using prediction and smoothing.

    Takes raw ball detections per frame and produces a clean, stable ball state.
    Uses motion-based recovery when detections are unavailable.
    """

    def __init__(
        self,
        smoothing_factor: float = 0.3,
        max_prediction_frames: int = 10,
        min_confidence: float = 0.12,  # lower threshold for airborne/background-heavy cases
        motion_smoothing_factor: float = 0.15,  # lower smoothing for motion detections
        motion_min_blob_area: int = 10,
        motion_max_blob_area: int = 400,
        motion_base_roi_size: int = 80,
        motion_diff_threshold: int = 25,
        motion_score_threshold: float = 0.5
    ):
        """
        Initialize the ball tracker.

        Args:
            smoothing_factor: Weight for new observations (0-1), higher = less smoothing
            max_prediction_frames: Maximum frames to predict without detection
            min_confidence: Minimum confidence to accept a detection
            motion_smoothing_factor: Weight for motion-based detections (lower than YOLO)
            motion_min_blob_area: Minimum blob area to consider
            motion_max_blob_area: Maximum blob area to consider
            motion_base_roi_size: Base ROI size for motion search
            motion_diff_threshold: Threshold for frame differencing
            motion_score_threshold: Minimum score to accept motion candidate
        """
        self.smoothing_factor = smoothing_factor
        self.max_prediction_frames = max_prediction_frames
        self.min_confidence = min_confidence
        
        # Motion recovery parameters
        self.motion_smoothing_factor = motion_smoothing_factor
        self.motion_min_blob_area = motion_min_blob_area
        self.motion_max_blob_area = motion_max_blob_area
        self.motion_base_roi_size = motion_base_roi_size
        self.motion_diff_threshold = motion_diff_threshold
        self.motion_score_threshold = motion_score_threshold

        # Internal state
        self.position: Optional[Tuple[float, float]] = None
        self.velocity: Optional[Tuple[float, float]] = None
        self.last_detection_frame: Optional[int] = None

        # Store last raw detection (for velocity estimation)
        self._last_raw_position: Optional[Tuple[float, float]] = None

        self.frames_without_detection: int = 0
        
        # Motion recovery state
        self._prev_gray_frame: Optional[np.ndarray] = None

    def update(
        self,
        position: Optional[Tuple[float, float]],
        frame_index: int,
        confidence: Optional[float] = None,
        gray_frame: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Update tracker with new detection and return current ball state.
        
        Args:
            position: Detected ball position (x, y) or None
            frame_index: Current frame index
            confidence: Detection confidence or None
            gray_frame: Grayscale frame for motion-based recovery
        
        Returns:
            Dictionary with ball state (position, velocity, confidence, visible)
        """
        detection_valid = (
            position is not None and
            (confidence is None or confidence >= self.min_confidence)
        )

        if detection_valid:
            self._process_detection(position, frame_index)
            self.frames_without_detection = 0
        else:
            # Attempt motion-based recovery before falling back to prediction
            motion_position = self._try_motion_recovery(gray_frame, frame_index)
            
            if motion_position is not None:
                self._process_motion_detection(motion_position, frame_index)
                # Reduce frames_without_detection slightly (partial recovery)
                self.frames_without_detection = max(0, self.frames_without_detection - 1)
            else:
                self.frames_without_detection += 1
                self._process_no_detection()

        # Store current frame for next motion comparison
        if gray_frame is not None:
            self._prev_gray_frame = gray_frame.copy()

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

    def _process_motion_detection(
        self,
        motion_position: Tuple[float, float],
        frame_index: int
    ) -> None:
        """
        Process a motion-based ball detection.
        
        Uses lower smoothing weight than YOLO detections to maintain stability.
        """
        if self.position is None:
            # Should not happen (motion recovery requires prior state)
            self.position = motion_position
            self.velocity = (0.0, 0.0)
        else:
            # Compute velocity from motion detection
            if self._last_raw_position is not None and self.last_detection_frame is not None:
                frame_delta = frame_index - self.last_detection_frame
                if frame_delta > 0:
                    vx = (motion_position[0] - self._last_raw_position[0]) / frame_delta
                    vy = (motion_position[1] - self._last_raw_position[1]) / frame_delta
                    # Smooth velocity update
                    self.velocity = (
                        0.5 * vx + 0.5 * self.velocity[0],
                        0.5 * vy + 0.5 * self.velocity[1]
                    )

            # Smooth position with lower weight (motion is less reliable)
            smoothed_x = (
                self.motion_smoothing_factor * motion_position[0] +
                (1 - self.motion_smoothing_factor) * self.position[0]
            )
            smoothed_y = (
                self.motion_smoothing_factor * motion_position[1] +
                (1 - self.motion_smoothing_factor) * self.position[1]
            )
            self.position = (smoothed_x, smoothed_y)

        self._last_raw_position = motion_position
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

    def _try_motion_recovery(
        self,
        gray_frame: Optional[np.ndarray],
        frame_index: int
    ) -> Optional[Tuple[float, float]]:
        """
        Attempt to recover ball position using motion-only evidence.
        
        Uses frame differencing in a local ROI around predicted position.
        
        Args:
            gray_frame: Current grayscale frame
            frame_index: Current frame index
        
        Returns:
            Motion-based position (x, y) or None if no valid candidate found
        """
        # Prerequisites for motion recovery
        if (gray_frame is None or 
            self._prev_gray_frame is None or
            self.position is None or
            self.velocity is None):
            return None
        
        # Predict ball position
        predicted_x = self.position[0] + self.velocity[0]
        predicted_y = self.position[1] + self.velocity[1]
        
        # Compute adaptive ROI size based on speed and frames without detection
        speed = np.sqrt(self.velocity[0]**2 + self.velocity[1]**2)
        speed_scale = 1.0 + min(speed / 10.0, 2.0)  # Scale up to 3x for fast motion
        time_scale = 1.0 + (self.frames_without_detection / self.max_prediction_frames)
        roi_size = int(self.motion_base_roi_size * speed_scale * time_scale)
        
        # Extract ROI bounds
        h, w = gray_frame.shape
        x1 = int(max(0, predicted_x - roi_size // 2))
        y1 = int(max(0, predicted_y - roi_size // 2))
        x2 = int(min(w, predicted_x + roi_size // 2))
        y2 = int(min(h, predicted_y + roi_size // 2))
        
        # Check valid ROI
        if x2 <= x1 or y2 <= y1:
            return None
        
        # Extract ROI from both frames
        prev_roi = self._prev_gray_frame[y1:y2, x1:x2]
        curr_roi = gray_frame[y1:y2, x1:x2]
        
        if prev_roi.shape != curr_roi.shape or prev_roi.size == 0:
            return None
        
        # Compute frame difference
        diff = cv2.absdiff(prev_roi, curr_roi)
        _, motion_mask = cv2.threshold(diff, self.motion_diff_threshold, 255, cv2.THRESH_BINARY)
        
        # Extract connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            motion_mask, connectivity=8
        )
        
        # Find best motion candidate
        best_score = -1
        best_position = None
        
        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            
            # Filter by blob size
            if area < self.motion_min_blob_area or area > self.motion_max_blob_area:
                continue
            
            # Get blob centroid in frame coordinates
            cx_roi, cy_roi = centroids[i]
            cx = x1 + cx_roi
            cy = y1 + cy_roi
            
            # Compute score based on proximity, compactness, and velocity alignment
            score = self._score_motion_candidate(
                position=(cx, cy),
                predicted_position=(predicted_x, predicted_y),
                area=area,
                stats=stats[i]
            )
            
            if score > best_score:
                best_score = score
                best_position = (cx, cy)
        
        # Return best candidate if it exceeds threshold
        if best_score >= self.motion_score_threshold:
            return best_position
        
        return None

    def _score_motion_candidate(
        self,
        position: Tuple[float, float],
        predicted_position: Tuple[float, float],
        area: int,
        stats: np.ndarray
    ) -> float:
        """
        Score a motion blob candidate using simple physics.
        
        Args:
            position: Blob centroid position (x, y)
            predicted_position: Predicted ball position (x, y)
            area: Blob area in pixels
            stats: Connected component stats array
        
        Returns:
            Score value (higher is better)
        """
        # Proximity to predicted position (normalized by ROI size)
        dx = position[0] - predicted_position[0]
        dy = position[1] - predicted_position[1]
        distance = np.sqrt(dx**2 + dy**2)
        max_distance = self.motion_base_roi_size
        proximity_score = max(0, 1.0 - distance / max_distance)
        
        # Compactness (how circular is the blob)
        width = stats[cv2.CC_STAT_WIDTH]
        height = stats[cv2.CC_STAT_HEIGHT]
        aspect_ratio = min(width, height) / (max(width, height) + 1e-6)
        compactness_score = aspect_ratio
        
        # Velocity direction alignment
        velocity_score = 1.0
        if self.velocity is not None:
            vx, vy = self.velocity
            velocity_mag = np.sqrt(vx**2 + vy**2)
            if velocity_mag > 1.0:  # Only check if there's significant velocity
                # Normalize displacement and velocity
                displacement = (dx / (distance + 1e-6), dy / (distance + 1e-6))
                velocity_norm = (vx / velocity_mag, vy / velocity_mag)
                # Dot product (alignment)
                alignment = displacement[0] * velocity_norm[0] + displacement[1] * velocity_norm[1]
                velocity_score = max(0, alignment)  # 0 to 1
        
        # Weighted combination
        score = (
            0.5 * proximity_score +
            0.2 * compactness_score +
            0.3 * velocity_score
        )
        
        return score

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
        self._prev_gray_frame = None
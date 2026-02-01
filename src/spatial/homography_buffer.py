# src/spatial/homography_buffer.py

"""
Temporal homography stabilization and smoothing.

Smooths per-frame homographies over time while:
- Detecting camera cuts
- Handling invalid frames gracefully
- Maintaining temporal coherence
"""

import numpy as np
from typing import Optional, Tuple, Deque
from collections import deque
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class HomographyObservation:
    """Single observation in the temporal buffer."""
    
    H: np.ndarray  # 3×3 homography
    confidence: float  # [0, 1]
    frame_idx: int


class TemporalHomographyBuffer:
    """
    Buffers and smooths homographies over time.
    
    Strategy:
    - Maintain sliding window of recent homographies
    - Apply EMA (exponential moving average) smoothing
    - Detect camera cuts (sudden discontinuities)
    - Fallback to identity or last valid H when frame is invalid
    
    Usage:
        buffer = TemporalHomographyBuffer(window_size=5)
        
        for frame_idx, (H, conf) in enumerate(homographies):
            H_smoothed = buffer.add_observation(H, conf, frame_idx)
    """
    
    def __init__(
        self,
        window_size: int = 5,
        smoothing_alpha: float = 0.3,
        cut_detection_threshold: float = 0.3
    ):
        """
        Initialize temporal buffer.
        
        Args:
            window_size: Number of frames to keep in buffer
            smoothing_alpha: EMA smoothing factor [0, 1]
                - 0: Pure smoothing (slow response)
                - 1: No smoothing (noisy but responsive)
            cut_detection_threshold: Threshold for camera cut detection
                (rotation angle in radians)
        """
        
        self.window_size = window_size
        self.smoothing_alpha = smoothing_alpha
        self.cut_detection_threshold = cut_detection_threshold
        
        self.buffer: Deque[HomographyObservation] = deque(maxlen=window_size)
        self.smoothed_H: Optional[np.ndarray] = None
        self.last_valid_H: Optional[np.ndarray] = None
        self.last_valid_frame_idx: Optional[int] = None
    
    def add_observation(
        self,
        H: Optional[np.ndarray],
        confidence: float,
        frame_idx: int
    ) -> np.ndarray:
        """
        Add observation and return stabilized homography.
        
        Handles:
        - Invalid frames (H is None): uses last valid H or identity
        - Camera cuts: resets buffer and uses identity
        - Normal frames: adds to buffer and smooths
        
        Args:
            H: Estimated homography (3×3) or None if invalid
            confidence: Confidence in this estimate [0, 1]
            frame_idx: Frame number
        
        Returns:
            Stabilized 3×3 homography to use for this frame
        """
        
        # Tier 1: Check for invalid frame
        if H is None or confidence < 0.3:
            return self._handle_invalid_frame(frame_idx)
        
        # Tier 2: Check for camera cut
        if self.last_valid_H is not None:
            if self._is_camera_cut(H, self.last_valid_H):
                logger.info(f"Camera cut detected at frame {frame_idx}")
                self.buffer.clear()
                self.smoothed_H = np.eye(3)
                self.last_valid_H = H
                self.last_valid_frame_idx = frame_idx
                return np.eye(3)
        
        # Tier 3: Normal frame - add to buffer
        self.buffer.append(HomographyObservation(H, confidence, frame_idx))
        self.last_valid_H = H
        self.last_valid_frame_idx = frame_idx
        
        # Apply temporal smoothing
        H_stabilized = self._smooth_buffer()
        self.smoothed_H = H_stabilized
        
        return H_stabilized
    
    def _handle_invalid_frame(self, frame_idx: int) -> np.ndarray:
        """
        Handle frame with invalid homography.
        
        Fallback strategy:
        1. Use last smoothed H if available
        2. Use last valid H
        3. Use identity
        
        Args:
            frame_idx: Current frame number
        
        Returns:
            Fallback homography
        """
        
        if self.smoothed_H is not None:
            return self.smoothed_H
        
        if self.last_valid_H is not None:
            return self.last_valid_H
        
        return np.eye(3)
    
    def _is_camera_cut(
        self,
        H_curr: np.ndarray,
        H_prev: np.ndarray
    ) -> bool:
        """
        Detect sudden camera motion (cut or reset).
        
        Computes rotation angle from homography change.
        
        Args:
            H_curr, H_prev: Consecutive homographies
        
        Returns:
            True if cut detected
        """
        
        try:
            # Relative homography
            H_delta = H_curr @ np.linalg.inv(H_prev)
            
            # Extract rotation via SVD
            U, S, Vt = np.linalg.svd(H_delta[:2, :2])
            R_approx = U @ Vt
            
            # Rotation angle
            trace = np.trace(R_approx)
            rotation_angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
            
            return rotation_angle > self.cut_detection_threshold
        
        except Exception as e:
            logger.warning(f"Error in cut detection: {e}")
            return False
    
    def _smooth_buffer(self) -> np.ndarray:
        """
        Smooth homographies in buffer using weighted EMA.
        
        Weight by confidence: high-confidence observations have more influence.
        
        Returns:
            Smoothed 3×3 homography
        """
        
        if len(self.buffer) == 0:
            if self.smoothed_H is not None:
                return self.smoothed_H
            return np.eye(3)
        
        if len(self.buffer) == 1:
            return self.buffer[0].H
        
        # Compute weighted average
        obs_list = list(self.buffer)
        weights = np.array([obs.confidence for obs in obs_list])
        weights /= weights.sum()  # Normalize
        
        # Simple linear blend in matrix space
        # (Good approximation for small changes)
        H_blend = np.zeros((3, 3))
        for obs, w in zip(obs_list, weights):
            H_blend += w * obs.H
        
        # EMA update to smoothed estimate
        if self.smoothed_H is not None:
            H_final = (self.smoothing_alpha * H_blend +
                      (1 - self.smoothing_alpha) * self.smoothed_H)
        else:
            H_final = H_blend
        
        return H_final
    
    def reset(self) -> None:
        """
        Reset buffer state.
        
        Call when starting new video sequence or after significant event.
        """
        
        self.buffer.clear()
        self.smoothed_H = None
        self.last_valid_H = None
        self.last_valid_frame_idx = None
    
    def extrapolate_H(self) -> np.ndarray:
        """
        Extrapolate homography using motion model.
        
        Assumes smooth camera motion; predicts next H based on
        motion trend in buffer.
        
        Returns:
            Extrapolated homography
        """
        
        if len(self.buffer) < 2:
            return self.last_valid_H if self.last_valid_H is not None else np.eye(3)
        
        # Simple linear extrapolation of deltas
        obs_list = list(self.buffer)
        H_curr = obs_list[-1].H
        H_prev = obs_list[-2].H
        
        # Compute delta
        H_delta = H_curr @ np.linalg.inv(H_prev)
        
        # Extrapolate
        H_extrap = H_curr @ H_delta
        
        return H_extrap
    
    def get_statistics(self) -> dict:
        """
        Get buffer statistics for debugging.
        
        Returns:
            Dictionary with buffer info
        """
        
        if len(self.buffer) == 0:
            return {
                'buffer_size': 0,
                'avg_confidence': 0.0,
                'has_smoothed': self.smoothed_H is not None
            }
        
        confidences = [obs.confidence for obs in self.buffer]
        
        return {
            'buffer_size': len(self.buffer),
            'avg_confidence': np.mean(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences),
            'has_smoothed': self.smoothed_H is not None,
            'frames': [obs.frame_idx for obs in self.buffer]
        }
    
    def __repr__(self) -> str:
        """String representation."""
        stats = self.get_statistics()
        return (
            f"<TemporalHomographyBuffer "
            f"size={stats['buffer_size']}/{self.window_size} "
            f"avg_conf={stats['avg_confidence']:.2f}>"
        )


class KalmanHomographyFilter:
    """
    Kalman filter for homography smoothing (alternative to EMA).
    
    More sophisticated but requires tuning of process/measurement noise.
    
    Optional: Use this for more advanced filtering.
    """
    
    def __init__(
        self,
        process_noise: float = 0.01,
        measurement_noise: float = 0.1
    ):
        """
        Initialize Kalman filter for homography components.
        
        Args:
            process_noise: Process noise covariance
            measurement_noise: Measurement noise covariance
        """
        
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        
        # State: [h11, h12, h13, h21, h22, h23, h31, h32] (h33=1)
        self.state = np.zeros(8)
        self.covariance = np.eye(8)
        self.initialized = False
    
    def update(self, H_obs: np.ndarray) -> np.ndarray:
        """
        Kalman filter update step.
        
        Args:
            H_obs: Observed homography (3×3)
        
        Returns:
            Filtered homography (3×3)
        """
        
        # Extract state from homography
        z = H_obs[:, :].flatten()[:8]
        
        if not self.initialized:
            self.state = z
            self.initialized = True
            return H_obs
        
        # Predict
        x_pred = self.state
        P_pred = self.covariance + self.process_noise * np.eye(8)
        
        # Update
        y = z - x_pred
        S = P_pred + self.measurement_noise * np.eye(8)
        K = P_pred @ np.linalg.inv(S)
        
        self.state = x_pred + K @ y
        self.covariance = (np.eye(8) - K) @ P_pred
        
        # Reconstruct homography
        H_filtered = np.eye(3)
        H_filtered[:, :].flatten()[:8] = self.state
        
        return H_filtered

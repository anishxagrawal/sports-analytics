# src/spatial/world_velocity.py

"""
Real-world velocity and speed computation from world coordinates.

Computes velocity in m/s from world-space position history with:
- Outlier detection and clipping
- Temporal filtering (EMA)
- Quality-based weighting
- Sanity checks
"""

import numpy as np
from typing import Tuple, Optional, Deque
from collections import deque
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class VelocityMeasurement:
    """Single velocity measurement."""
    
    velocity: Tuple[float, float]  # (vx, vy) in m/s
    speed: float  # magnitude in m/s
    quality: float  # [0, 1] confidence
    frame_idx: int


class WorldVelocityEstimator:
    """
    Estimates real-world velocity from world-space position history.
    
    Incorporates:
    - Position history buffer
    - Temporal filtering (EMA)
    - Outlier detection
    - Quality weighting
    
    Usage:
        velocity_est = WorldVelocityEstimator(fps=30.0)
        
        for frame_idx, world_pos in enumerate(positions):
            vx, vy, speed = velocity_est.update(
                world_pos, frame_idx, quality=0.8
            )
    """
    
    # Constants
    MAX_PLAYER_SPEED = 15.0  # m/s (players max ~10-12)
    MAX_BALL_SPEED = 40.0    # m/s (ball can be faster)
    
    def __init__(
        self,
        fps: float,
        smoothing_alpha: float = 0.3,
        max_history: int = 30,
        entity_type: str = 'player'
    ):
        """
        Initialize velocity estimator.
        
        Args:
            fps: Video frame rate (frames per second)
            smoothing_alpha: EMA smoothing factor [0, 1]
            max_history: Maximum position history to keep
            entity_type: 'player' or 'ball' (affects speed thresholds)
        """
        
        self.fps = fps
        self.dt = 1.0 / fps
        self.smoothing_alpha = smoothing_alpha
        self.entity_type = entity_type
        
        # Position history: deque of (world_x, world_y, frame_idx, quality)
        self.position_history: Deque[Tuple[float, float, int, float]] = deque(maxlen=max_history)
        
        # Velocity history
        self.velocity_history: Deque[VelocityMeasurement] = deque(maxlen=10)
        
        # Smoothed velocity estimate
        self.velocity_ema: Optional[Tuple[float, float]] = None
        self.speed_ema: float = 0.0
        
        # Set speed threshold based on entity type
        if entity_type == 'ball':
            self.max_speed = self.MAX_BALL_SPEED
        else:
            self.max_speed = self.MAX_PLAYER_SPEED
    
    def update(
        self,
        world_position: Tuple[float, float],
        frame_idx: int,
        quality: float = 1.0
    ) -> Tuple[float, float, float]:
        """
        Update with new world position measurement.
        
        Args:
            world_position: (X, Y) in meters
            frame_idx: Current frame number
            quality: Confidence in this measurement [0, 1]
        
        Returns:
            (vx, vy, speed) in m/s
        """
        
        if not (0 <= quality <= 1):
            quality = np.clip(quality, 0, 1)
        
        # Add to history
        self.position_history.append((world_position[0], world_position[1], frame_idx, quality))
        
        # Compute velocity if sufficient history
        vx, vy, speed = self._compute_velocity()
        
        if vx is not None:
            # Sanity check: clip unrealistic speeds
            vx, vy, speed = self._apply_sanity_checks(vx, vy, speed)
            
            # Apply smoothing
            vx, vy, speed = self._smooth_velocity(vx, vy, speed)
            
            # Record measurement
            self.velocity_history.append(
                VelocityMeasurement((vx, vy), speed, quality, frame_idx)
            )
            
            return vx, vy, speed
        
        # Return last estimate or zero
        if self.velocity_ema is not None:
            return (*self.velocity_ema, self.speed_ema)
        
        return (0.0, 0.0, 0.0)
    
    def _compute_velocity(self) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Compute velocity from position history.
        
        Uses last two high-quality measurements separated by at least
        a few frames (not just consecutive frames, to improve robustness).
        
        Returns:
            (vx, vy, speed) or (None, None, None) if insufficient data
        """
        
        if len(self.position_history) < 2:
            return None, None, None
        
        # Find two recent high-quality measurements
        pos_curr = None
        pos_prev = None
        
        # Current is the most recent
        if self.position_history[-1][3] > 0.3:  # quality > 0.3
            pos_curr = self.position_history[-1]
        else:
            # Find last good measurement
            for i in range(len(self.position_history) - 1, -1, -1):
                if self.position_history[i][3] > 0.3:
                    pos_curr = self.position_history[i]
                    break
        
        if pos_curr is None:
            return None, None, None
        
        # Find previous measurement from 2-5 frames ago (avoid single-frame noise)
        target_frame = pos_curr[2] - 3  # 3 frames back
        
        best_prev = None
        best_diff = float('inf')
        
        for i in range(len(self.position_history) - 1):
            pos = self.position_history[i]
            if pos[3] > 0.3:  # quality > 0.3
                frame_diff = abs(pos[2] - target_frame)
                if frame_diff < best_diff:
                    best_diff = frame_diff
                    best_prev = pos
        
        if best_prev is None:
            return None, None, None
        
        # Compute velocity
        X_curr, Y_curr, frame_curr, qual_curr = pos_curr
        X_prev, Y_prev, frame_prev, qual_prev = best_prev
        
        dt = (frame_curr - frame_prev) / self.fps
        
        if dt <= 0:
            return None, None, None
        
        dx = X_curr - X_prev
        dy = Y_curr - Y_prev
        
        vx = dx / dt
        vy = dy / dt
        speed = np.sqrt(vx**2 + vy**2)
        
        return vx, vy, speed
    
    def _apply_sanity_checks(
        self,
        vx: float,
        vy: float,
        speed: float
    ) -> Tuple[float, float, float]:
        """
        Apply sanity checks and clip unrealistic velocities.
        
        Args:
            vx, vy, speed: Computed velocity components and magnitude
        
        Returns:
            Clipped (vx, vy, speed)
        """
        
        # Clip speed to maximum
        if speed > self.max_speed:
            logger.debug(
                f"Velocity clipped: {speed:.2f} m/s → {self.max_speed:.2f} m/s "
                f"(entity_type={self.entity_type})"
            )
            
            # Scale down components proportionally
            if speed > 0:
                scale = self.max_speed / speed
                vx *= scale
                vy *= scale
                speed = self.max_speed
        
        return vx, vy, speed
    
    def _smooth_velocity(
        self,
        vx: float,
        vy: float,
        speed: float
    ) -> Tuple[float, float, float]:
        """
        Apply exponential moving average smoothing.
        
        Args:
            vx, vy, speed: Raw velocity
        
        Returns:
            Smoothed (vx, vy, speed)
        """
        
        if self.velocity_ema is None:
            self.velocity_ema = (vx, vy)
            self.speed_ema = speed
        else:
            vx_smooth = (
                self.smoothing_alpha * vx +
                (1 - self.smoothing_alpha) * self.velocity_ema[0]
            )
            vy_smooth = (
                self.smoothing_alpha * vy +
                (1 - self.smoothing_alpha) * self.velocity_ema[1]
            )
            
            self.velocity_ema = (vx_smooth, vy_smooth)
            
            speed_smooth = (
                self.smoothing_alpha * speed +
                (1 - self.smoothing_alpha) * self.speed_ema
            )
            
            self.speed_ema = speed_smooth
            
            vx, vy, speed = vx_smooth, vy_smooth, speed_smooth
        
        return vx, vy, speed
    
    def get_velocity(self) -> Tuple[float, float, float]:
        """
        Get latest smoothed velocity estimate.
        
        Returns:
            (vx, vy, speed) or (0, 0, 0) if no estimate
        """
        
        if self.velocity_ema is not None:
            return (*self.velocity_ema, self.speed_ema)
        
        return (0.0, 0.0, 0.0)
    
    def get_statistics(self) -> dict:
        """
        Get velocity statistics for debugging.
        
        Returns:
            Dictionary with stats
        """
        
        if len(self.velocity_history) == 0:
            return {
                'num_measurements': 0,
                'avg_speed': 0.0,
                'max_speed': 0.0,
                'current_speed': 0.0
            }
        
        speeds = [m.speed for m in self.velocity_history]
        
        return {
            'num_measurements': len(self.velocity_history),
            'avg_speed': np.mean(speeds),
            'max_speed': np.max(speeds),
            'current_speed': self.speed_ema,
            'history_size': len(self.position_history)
        }
    
    def reset(self) -> None:
        """
        Reset all state.
        
        Call when starting new sequence or resetting entity tracking.
        """
        
        self.position_history.clear()
        self.velocity_history.clear()
        self.velocity_ema = None
        self.speed_ema = 0.0
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"<WorldVelocityEstimator "
            f"speed={self.speed_ema:.2f}m/s "
            f"history={len(self.position_history)}>"
        )


class VelocityAccumulator:
    """
    Accumulates velocities from multiple entities for statistical analysis.
    
    Optional: Use for aggregated statistics (e.g., avg team speed).
    """
    
    def __init__(self):
        """Initialize accumulator."""
        self.measurements = []
    
    def add(self, velocity: Tuple[float, float], speed: float, entity_id: str) -> None:
        """
        Add velocity measurement.
        
        Args:
            velocity: (vx, vy) in m/s
            speed: Magnitude in m/s
            entity_id: Identifier for entity
        """
        
        self.measurements.append({
            'velocity': velocity,
            'speed': speed,
            'entity_id': entity_id
        })
    
    def get_statistics(self) -> dict:
        """
        Get aggregated statistics.
        
        Returns:
            Dictionary with stats
        """
        
        if len(self.measurements) == 0:
            return {}
        
        speeds = np.array([m['speed'] for m in self.measurements])
        
        return {
            'count': len(self.measurements),
            'avg_speed': np.mean(speeds),
            'median_speed': np.median(speeds),
            'max_speed': np.max(speeds),
            'min_speed': np.min(speeds),
            'std_speed': np.std(speeds)
        }
    
    def reset(self) -> None:
        """Clear all measurements."""
        self.measurements.clear()

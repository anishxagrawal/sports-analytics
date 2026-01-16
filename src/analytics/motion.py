# src/analytics/motion.py

"""
Motion analytics module for sports analytics system.

Provides stateless functions for computing movement metrics from Player
entity position histories. All functions are pure and do not modify entities.
"""

from typing import Optional, Tuple
import numpy as np
from .smoothing import smooth_trajectory_ema


def compute_speed(player, fps: float, use_smoothing: bool = False) -> float:
    """
    Compute current speed of a player.
    
    Uses the last two positions from player trajectory to estimate
    instantaneous speed in pixels per second.
    
    Args:
        player: Player entity with get_trajectory() method
        fps: Video frame rate (frames per second)
        use_smoothing: If True, apply EMA smoothing to ground positions
    
    Returns:
        Speed in pixels per second, or 0.0 if insufficient history
    
    Notes:
        - Requires at least 2 positions in trajectory
        - Speed is Euclidean distance / time_delta
        - Time delta is computed as 1/fps (one frame interval)
    """
    if fps <= 0:
        return 0.0
    
    # Prefer ground positions if available, fallback to trajectory
    if hasattr(player, 'ground_positions') and len(player.ground_positions) >= 2:
        positions = list(player.ground_positions)
        
        # Apply EMA smoothing if requested
        if use_smoothing:
            positions = smooth_trajectory_ema(positions, alpha=0.3)
        
        pos_prev = positions[-2]
        pos_curr = positions[-1]
    else:
        trajectory = player.get_trajectory(n=2)
        
        if len(trajectory) < 2:
            return 0.0
        
        pos_prev = trajectory[-2]
        pos_curr = trajectory[-1]
    
    dx = pos_curr[0] - pos_prev[0]
    dy = pos_curr[1] - pos_prev[1]
    
    distance = np.sqrt(dx**2 + dy**2)
    time_delta = 1.0 / fps
    
    speed = distance / time_delta
    
    return float(speed)


def compute_direction(
    player,
    normalize: bool = True,
    use_smoothing: bool = False
) -> Optional[Tuple[float, float]]:
    """
    Compute current movement direction of a player.
    
    Uses the last two positions from player trajectory to compute
    direction vector (dx, dy).
    
    Args:
        player: Player entity with get_trajectory() method
        normalize: If True, return unit vector; if False, return raw displacement
        use_smoothing: If True, apply EMA smoothing to ground positions
    
    Returns:
        Tuple of (dx, dy) as direction vector, or None if insufficient history
    
    Notes:
        - Requires at least 2 positions in trajectory
        - If normalize=True and player is stationary, returns (0.0, 0.0)
        - Direction points from previous position to current position
    """
    # Prefer ground positions if available, fallback to trajectory
    if hasattr(player, 'ground_positions') and len(player.ground_positions) >= 2:
        positions = list(player.ground_positions)
        
        # Apply EMA smoothing if requested
        if use_smoothing:
            positions = smooth_trajectory_ema(positions, alpha=0.3)
        
        pos_prev = positions[-2]
        pos_curr = positions[-1]
    else:
        trajectory = player.get_trajectory(n=2)
        
        if len(trajectory) < 2:
            return None
        
        pos_prev = trajectory[-2]
        pos_curr = trajectory[-1]
    
    dx = pos_curr[0] - pos_prev[0]
    dy = pos_curr[1] - pos_prev[1]
    
    if not normalize:
        return (float(dx), float(dy))
    
    magnitude = np.sqrt(dx**2 + dy**2)
    
    if magnitude < 1e-6:
        return (0.0, 0.0)
    
    dx_norm = dx / magnitude
    dy_norm = dy / magnitude
    
    return (float(dx_norm), float(dy_norm))